use crate::{
    ipc::{Receiver, Sender},
    proto::{
        ArtifactChunk, ArtifactMetric, CheckpointMetadata, CheckpointRequest, MetricsRequest,
        MetricsResponse, MetricsStreamMetadata, ScalarMetric, checkpoint_request, metrics_request,
        tensor_lane_client::TensorLaneClient,
    },
};
use anyhow::{Context, anyhow, bail, ensure};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    path::PathBuf,
    sync::Mutex,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::{
    fs::File,
    io::AsyncReadExt,
    net::{
        UnixListener, UnixStream,
        unix::{OwnedReadHalf, OwnedWriteHalf},
    },
    sync::mpsc,
    task::JoinSet,
};
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::{sync::CancellationToken, task::AbortOnDropHandle};
use tonic::transport::Channel;

const CHUNK_SIZE: usize = 1024 * 1024;
const FINISH_TIMEOUT: Duration = Duration::from_secs(300);

#[derive(Serialize, Deserialize)]
enum Upload {
    Metric {
        step: u64,
        name: String,
        value: f32,
    },
    Artifact {
        step: u64,
        path: PathBuf,
        name: String,
        content_type: String,
    },
    Checkpoint {
        step: u64,
        path: PathBuf,
    },
    Flush,
}

type Reply = Result<(), String>;

struct Connection {
    sender: Sender<Upload, OwnedWriteHalf>,
    receiver: Receiver<Reply, OwnedReadHalf>,
    dirty: bool,
}

impl Connection {
    async fn flush(&mut self) -> anyhow::Result<()> {
        if !self.dirty {
            return Ok(());
        }
        let sent = self.sender.send(&Upload::Flush).await;
        match self.receiver.recv().await? {
            Some(reply) => {
                reply.map_err(|error| anyhow!(error))?;
                self.dirty = false;
                Ok(())
            }
            None => {
                sent?;
                bail!("upload connection closed");
            }
        }
    }
}

#[pyclass]
pub struct UploadClient {
    connection: Mutex<Option<Connection>>,
    runtime: tokio::runtime::Runtime,
}

#[pymethods]
impl UploadClient {
    #[new]
    fn new(py: Python<'_>, path: PathBuf) -> anyhow::Result<Self> {
        py.allow_threads(|| {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?;
            let socket = runtime.block_on(UnixStream::connect(path))?;
            let (read, write) = socket.into_split();
            Ok(Self {
                connection: Mutex::new(Some(Connection {
                    sender: Sender::new(write),
                    receiver: Receiver::new(read),
                    dirty: false,
                })),
                runtime,
            })
        })
    }

    fn metric(&self, py: Python<'_>, step: u64, name: String, value: f32) -> anyhow::Result<()> {
        self.send(py, Upload::Metric { step, name, value })
    }

    fn metric_artifact(
        &self,
        py: Python<'_>,
        step: u64,
        path: PathBuf,
        name: String,
        content_type: String,
    ) -> anyhow::Result<()> {
        self.send(
            py,
            Upload::Artifact {
                step,
                path,
                name,
                content_type,
            },
        )
    }

    fn checkpoint(&self, py: Python<'_>, step: u64, path: PathBuf) -> anyhow::Result<()> {
        self.send(py, Upload::Checkpoint { step, path })
    }

    #[pyo3(signature = (timeout=300.0))]
    fn flush(&self, py: Python<'_>, timeout: f64) -> anyhow::Result<()> {
        let timeout = Duration::try_from_secs_f64(timeout)?;
        py.allow_threads(|| {
            let mut connection = self
                .connection
                .lock()
                .map_err(|_| anyhow!("upload lock poisoned"))?;
            let result = self.runtime.block_on(async {
                let connection = connection.as_mut().context("upload client is closed")?;
                tokio::time::timeout(timeout, connection.flush())
                    .await
                    .context("upload flush timed out")?
            });
            if result.is_err() {
                connection.take();
            }
            result
        })
    }

    fn close(&self, py: Python<'_>) -> anyhow::Result<()> {
        py.allow_threads(|| {
            let connection = self
                .connection
                .lock()
                .map_err(|_| anyhow!("upload lock poisoned"))?
                .take();
            if let Some(mut connection) = connection {
                self.runtime.block_on(async {
                    tokio::time::timeout(FINISH_TIMEOUT, connection.flush()).await
                })??;
            }
            Ok(())
        })
    }
}

impl UploadClient {
    fn send(&self, py: Python<'_>, message: Upload) -> anyhow::Result<()> {
        py.allow_threads(|| {
            let mut connection = self
                .connection
                .lock()
                .map_err(|_| anyhow!("upload lock poisoned"))?;
            let connection = connection.as_mut().context("upload client is closed")?;
            self.runtime.block_on(connection.sender.send(&message))?;
            connection.dirty = true;
            Ok(())
        })
    }
}

pub async fn serve(
    listener: UnixListener,
    grpc: TensorLaneClient<Channel>,
    run_id: String,
    stopping: CancellationToken,
) -> anyhow::Result<()> {
    let mut connections = JoinSet::new();
    loop {
        tokio::select! {
            _ = stopping.cancelled() => break,
            accepted = listener.accept() => {
                let (socket, _) = accepted?;
                let grpc = grpc.clone();
                let run_id = run_id.clone();
                let stopping = stopping.clone();
                connections.spawn(async move {
                    if let Err(error) = receive(socket, grpc, run_id, stopping).await {
                        eprintln!("TensorLane upload failed: {error:#}");
                    }
                });
            }
            Some(result) = connections.join_next() => { result?; }
        }
    }
    drop(listener);
    tokio::time::timeout(FINISH_TIMEOUT, async {
        while let Some(result) = connections.join_next().await {
            result?;
        }
        anyhow::Ok(())
    })
    .await
    .context("uploads did not finish during shutdown")??;
    Ok(())
}

async fn receive(
    socket: UnixStream,
    grpc: TensorLaneClient<Channel>,
    run_id: String,
    stopping: CancellationToken,
) -> anyhow::Result<()> {
    let (read, write) = socket.into_split();
    let mut reader = Receiver::<Upload, _>::new(read);
    let mut replies = Sender::<Reply, _>::new(write);
    let (jobs, mut queue) = mpsc::unbounded_channel();
    let reading = async {
        loop {
            let message = tokio::select! {
                biased;
                message = reader.recv() => message?,
                _ = stopping.cancelled() => break,
            };
            let Some(message) = message else {
                break;
            };
            jobs.send(message)?;
        }
        drop(jobs);
        anyhow::Ok(())
    };
    let processing = async {
        let mut metrics: Option<Metrics> = None;
        while let Some(job) = queue.recv().await {
            match job {
                Upload::Checkpoint { step, path } => {
                    checkpoint(grpc.clone(), &run_id, step, path).await?
                }
                Upload::Flush => {
                    if let Some(metrics) = metrics.take() {
                        metrics.finish().await?;
                    }
                    replies.send(&Ok(())).await?;
                }
                Upload::Metric { step, name, value } => {
                    let stream =
                        metrics.get_or_insert_with(|| Metrics::new(grpc.clone(), run_id.clone()));
                    stream
                        .send(metrics_request::Payload::Metric(ScalarMetric {
                            step,
                            name,
                            value,
                            timestamp_unix_ms: timestamp()?,
                        }))
                        .await?;
                }
                Upload::Artifact {
                    step,
                    path,
                    name,
                    content_type,
                } => {
                    let stream =
                        metrics.get_or_insert_with(|| Metrics::new(grpc.clone(), run_id.clone()));
                    stream.artifact(step, path, name, content_type).await?;
                }
            }
        }
        if let Some(metrics) = metrics {
            metrics.finish().await?;
        }
        anyhow::Ok(())
    };
    let result = tokio::try_join!(reading, processing).map(|_| ());
    if let Err(error) = &result {
        let _ = replies.send(&Err(format!("{error:#}"))).await;
    }
    result
}

fn timestamp() -> anyhow::Result<i64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_millis()
        .try_into()?)
}

struct Metrics {
    sender: mpsc::Sender<MetricsRequest>,
    request: AbortOnDropHandle<Result<tonic::Response<MetricsResponse>, tonic::Status>>,
}

impl Metrics {
    fn new(mut grpc: TensorLaneClient<Channel>, run_id: String) -> Self {
        let (sender, receiver) = mpsc::channel(4);
        let stream = tokio_stream::StreamExt::chain(
            tokio_stream::once(MetricsRequest {
                payload: Some(metrics_request::Payload::Metadata(MetricsStreamMetadata {
                    run_id,
                })),
            }),
            ReceiverStream::new(receiver),
        );
        let request =
            AbortOnDropHandle::new(tokio::spawn(async move { grpc.metrics(stream).await }));
        Self { sender, request }
    }

    async fn send(&mut self, payload: metrics_request::Payload) -> anyhow::Result<()> {
        tokio::select! {
            biased;
            response = &mut self.request => {
                response??;
                bail!("metrics RPC ended before the stream finished");
            }
            sent = self.sender.send(MetricsRequest { payload: Some(payload) }) => {
                if sent.is_err() {
                    (&mut self.request).await??;
                    bail!("metrics RPC stopped receiving");
                }
            }
        }
        Ok(())
    }

    async fn artifact(
        &mut self,
        step: u64,
        path: PathBuf,
        name: String,
        content_type: String,
    ) -> anyhow::Result<()> {
        let mut file = File::open(&path)
            .await
            .with_context(|| format!("opening artifact {}", path.display()))?;
        let metadata = file.metadata().await?;
        ensure!(metadata.is_file(), "artifact path must be a file");
        let size_bytes = metadata.len();
        self.send(metrics_request::Payload::Artifact(ArtifactMetric {
            step,
            name,
            content_type,
            size_bytes,
            timestamp_unix_ms: timestamp()?,
        }))
        .await?;
        let mut total = 0;
        loop {
            let mut bytes = vec![0; CHUNK_SIZE];
            let count = file.read(&mut bytes).await?;
            if count == 0 {
                break;
            }
            total += count as u64;
            ensure!(total <= size_bytes, "artifact grew while uploading");
            bytes.truncate(count);
            self.send(metrics_request::Payload::ArtifactChunk(ArtifactChunk {
                data: bytes,
            }))
            .await?;
        }
        ensure!(total == size_bytes, "artifact shrank while uploading");
        Ok(())
    }

    async fn finish(self) -> anyhow::Result<()> {
        drop(self.sender);
        tokio::time::timeout(FINISH_TIMEOUT, self.request)
            .await
            .context("metrics RPC finish timed out")???;
        Ok(())
    }
}

async fn checkpoint(
    mut grpc: TensorLaneClient<Channel>,
    run_id: &str,
    step: u64,
    path: PathBuf,
) -> anyhow::Result<()> {
    let mut file = File::open(&path)
        .await
        .with_context(|| format!("opening checkpoint {}", path.display()))?;
    ensure!(
        file.metadata().await?.is_file(),
        "checkpoint path must be a file"
    );
    let (sender, receiver) = mpsc::channel(4);
    let sending = async {
        sender
            .send(CheckpointRequest {
                payload: Some(checkpoint_request::Payload::Metadata(CheckpointMetadata {
                    run_id: run_id.to_owned(),
                    step,
                })),
            })
            .await?;
        loop {
            let mut bytes = vec![0; CHUNK_SIZE];
            let count = file.read(&mut bytes).await?;
            if count == 0 {
                break;
            }
            bytes.truncate(count);
            sender
                .send(CheckpointRequest {
                    payload: Some(checkpoint_request::Payload::Chunk(bytes)),
                })
                .await?;
        }
        drop(sender);
        anyhow::Ok(())
    };
    let request = async {
        grpc.checkpoint(ReceiverStream::new(receiver)).await?;
        anyhow::Ok(())
    };
    tokio::time::timeout(FINISH_TIMEOUT, async { tokio::try_join!(request, sending) })
        .await
        .context("checkpoint upload timed out")??;
    Ok(())
}
