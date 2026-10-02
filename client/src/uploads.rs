use crate::{
    ipc::{Receiver, Sender},
    proto::{
        ArtifactChunk, ArtifactMetric, MetricsRequest, MetricsResponse, MetricsStreamMetadata,
        SaveAssetMetadata, SaveAssetRequest, ScalarMetric, metrics_request, save_asset_request,
        tensor_lane_client::TensorLaneClient,
    },
};
use anyhow::{Context, anyhow, bail, ensure};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    io::{Seek, SeekFrom},
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::{
    fs::File,
    io::AsyncReadExt,
    net::{
        UnixListener, UnixStream,
        unix::{OwnedReadHalf, OwnedWriteHalf},
    },
    sync::{mpsc, oneshot},
    task::JoinSet,
};
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::{sync::CancellationToken, task::AbortOnDropHandle};
use tonic::transport::Channel;

const CHUNK_SIZE: usize = 1024 * 1024;

#[derive(Serialize, Deserialize)]
enum Upload {
    Automatic,
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
    SaveAsset {
        asset_id: String,
        name: String,
        step: u64,
        path: PathBuf,
        kind: String,
        asset_type: Option<String>,
        metadata_json: String,
        content_type: String,
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

enum Command {
    Send(Upload),
    Flush(oneshot::Sender<Reply>),
    Close(oneshot::Sender<Reply>),
}

#[pyclass]
pub struct UploadClient {
    sender: Mutex<Option<mpsc::UnboundedSender<Command>>>,
    failure: Arc<Mutex<Option<String>>>,
    runtime: tokio::runtime::Runtime,
}

#[pymethods]
impl UploadClient {
    #[new]
    #[pyo3(signature = (path, automatic=false))]
    fn new(py: Python<'_>, path: PathBuf, automatic: bool) -> anyhow::Result<Self> {
        py.allow_threads(|| {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(1)
                .enable_all()
                .build()?;
            let socket = runtime.block_on(UnixStream::connect(path))?;
            let (read, write) = socket.into_split();
            let (sender, mut commands) = mpsc::unbounded_channel();
            let failure = Arc::new(Mutex::new(None));
            let failed = failure.clone();
            runtime.spawn(async move {
                let mut connection = Connection {
                    sender: Sender::new(write),
                    receiver: Receiver::new(read),
                    dirty: false,
                };
                let result = async {
                    if automatic {
                        connection.sender.send(&Upload::Automatic).await?;
                    }
                    while let Some(command) = commands.recv().await {
                        match command {
                            Command::Send(message) => {
                                connection.sender.send(&message).await?;
                                connection.dirty = true;
                            }
                            Command::Flush(reply) | Command::Close(reply) => {
                                let result = connection
                                    .flush()
                                    .await
                                    .map_err(|error| format!("{error:#}"));
                                let failed = result.is_err();
                                let _ = reply.send(result.clone());
                                if failed {
                                    return Err(anyhow!(result.unwrap_err()));
                                }
                            }
                        }
                    }
                    anyhow::Ok(())
                }
                .await;
                if let Err(error) = result {
                    *failed.lock().unwrap() = Some(format!("{error:#}"));
                }
            });
            Ok(Self {
                sender: Mutex::new(Some(sender)),
                failure,
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

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (name, path, step=0, kind="file".to_owned(), asset_type=None, metadata_json="{}".to_owned(), content_type="application/octet-stream".to_owned()))]
    fn save_asset(
        &self,
        py: Python<'_>,
        name: String,
        path: PathBuf,
        step: u64,
        kind: String,
        asset_type: Option<String>,
        metadata_json: String,
        content_type: String,
    ) -> anyhow::Result<String> {
        ensure!(!name.is_empty(), "asset name must not be empty");
        ensure!(
            kind == "file" || kind == "checkpoint",
            "asset kind must be checkpoint or file"
        );
        let _: serde_json::Map<String, serde_json::Value> = serde_json::from_str(&metadata_json)?;
        let asset_id = uuid::Uuid::new_v4().to_string();
        self.send(
            py,
            Upload::SaveAsset {
                asset_id: asset_id.clone(),
                name,
                step,
                path,
                kind,
                asset_type,
                metadata_json,
                content_type,
            },
        )?;
        Ok(asset_id)
    }

    #[pyo3(signature = (timeout=None))]
    fn flush(&self, py: Python<'_>, timeout: Option<f64>) -> anyhow::Result<()> {
        let timeout = timeout.map(Duration::try_from_secs_f64).transpose()?;
        let (reply, received) = oneshot::channel();
        self.enqueue(Command::Flush(reply))?;
        py.allow_threads(|| {
            self.runtime.block_on(async {
                let result = match timeout {
                    Some(timeout) => tokio::time::timeout(timeout, received)
                        .await
                        .context("upload flush timed out")?,
                    None => received.await,
                };
                let result = result
                    .map_err(|_| self.error())
                    .and_then(|reply| reply.map_err(|error| anyhow!(error)));
                if result.is_err() {
                    self.sender.lock().unwrap().take();
                }
                result
            })
        })
    }

    fn close(&self, py: Python<'_>) -> anyhow::Result<()> {
        let (reply, received) = oneshot::channel();
        let sender = self
            .sender
            .lock()
            .map_err(|_| anyhow!("upload lock poisoned"))?
            .take();
        if let Some(sender) = sender {
            sender
                .send(Command::Close(reply))
                .map_err(|_| self.error())?;
            drop(sender);
            py.allow_threads(|| {
                self.runtime.block_on(async {
                    received
                        .await
                        .map_err(|_| self.error())?
                        .map_err(|error| anyhow!(error))
                })
            })?;
        }
        Ok(())
    }
}

impl UploadClient {
    fn error(&self) -> anyhow::Error {
        anyhow!(
            self.failure
                .lock()
                .ok()
                .and_then(|error| error.clone())
                .unwrap_or_else(|| "upload connection closed".into())
        )
    }
    fn enqueue(&self, command: Command) -> anyhow::Result<()> {
        self.sender
            .lock()
            .map_err(|_| anyhow!("upload lock poisoned"))?
            .as_ref()
            .context("upload client is closed")?
            .send(command)
            .map_err(|_| self.error())
    }
    fn send(&self, _py: Python<'_>, message: Upload) -> anyhow::Result<()> {
        self.enqueue(Command::Send(message))
    }
}

pub async fn serve(
    listener: UnixListener,
    grpc: TensorLaneClient<Channel>,
    run_id: String,
    stopping: CancellationToken,
) -> anyhow::Result<()> {
    let mut connections = JoinSet::new();
    let mut failure = None;
    loop {
        tokio::select! {
            _ = stopping.cancelled() => break,
            accepted = listener.accept() => {
                let (socket, _) = accepted?;
                let grpc = grpc.clone();
                let run_id = run_id.clone();
                let stopping = stopping.clone();
                connections.spawn(async move {
                    let result = receive(socket, grpc, run_id, stopping).await;
                    if let Err(error) = &result {
                        eprintln!("TensorLane upload failed: {error:#}");
                    }
                    result
                });
            }
            Some(result) = connections.join_next() => {
                if let Err(error) = result? { failure = Some(error); }
            }
        }
    }
    drop(listener);
    while let Some(result) = connections.join_next().await {
        if let Err(error) = result? {
            failure = Some(error);
        }
    }
    failure.map_or(Ok(()), Err)
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
    let mut automatic = false;
    let processing = async {
        let mut metrics: Option<Metrics> = None;
        while let Some(job) = queue.recv().await {
            match job {
                Upload::Automatic => automatic = true,
                Upload::SaveAsset {
                    asset_id,
                    name,
                    step,
                    path,
                    kind,
                    asset_type,
                    metadata_json,
                    content_type,
                } => {
                    save_asset(
                        grpc.clone(),
                        SaveAssetMetadata {
                            run_id: run_id.clone(),
                            asset_id,
                            name,
                            step,
                            kind,
                            asset_type,
                            metadata_json,
                            content_type,
                        },
                        path,
                    )
                    .await?
                }
                Upload::Flush => {
                    if let Some(metrics) = metrics.take() {
                        metrics.finish().await?;
                    }
                    replies.send(&Ok(())).await?;
                }
                Upload::Metric { step, name, value } => {
                    let stream = metrics.get_or_insert_with(|| {
                        Metrics::new(grpc.clone(), run_id.clone(), automatic)
                    });
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
                    let stream = metrics.get_or_insert_with(|| {
                        Metrics::new(grpc.clone(), run_id.clone(), automatic)
                    });
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
    if automatic {
        if let Err(error) = &result {
            eprintln!("TensorLane performance metrics failed: {error:#}");
        }
        Ok(())
    } else {
        result
    }
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
    fn new(mut grpc: TensorLaneClient<Channel>, run_id: String, automatic: bool) -> Self {
        let (sender, receiver) = mpsc::channel(4);
        let stream = tokio_stream::StreamExt::chain(
            tokio_stream::once(MetricsRequest {
                payload: Some(metrics_request::Payload::Metadata(MetricsStreamMetadata {
                    run_id,
                    automatic,
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
        let (mut file, is_directory) = upload_source(path.clone())
            .await
            .with_context(|| format!("opening artifact {}", path.display()))?;
        let metadata = file.metadata().await?;
        let size_bytes = metadata.len();
        self.send(metrics_request::Payload::Artifact(ArtifactMetric {
            step,
            name,
            content_type: if is_directory {
                "application/x-tar".into()
            } else {
                content_type
            },
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
        self.request.await??;
        Ok(())
    }
}

async fn save_asset(
    mut grpc: TensorLaneClient<Channel>,
    mut metadata: SaveAssetMetadata,
    path: PathBuf,
) -> anyhow::Result<()> {
    let expected_id = metadata.asset_id.clone();
    let (mut file, is_directory) = upload_source(path.clone())
        .await
        .with_context(|| format!("opening asset {}", path.display()))?;
    if is_directory {
        metadata.content_type = "application/x-tar".into();
    }
    let (sender, receiver) = mpsc::channel(4);
    let sending = async {
        sender
            .send(SaveAssetRequest {
                payload: Some(save_asset_request::Payload::Metadata(metadata)),
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
                .send(SaveAssetRequest {
                    payload: Some(save_asset_request::Payload::Chunk(bytes)),
                })
                .await?;
        }
        drop(sender);
        Ok::<_, anyhow::Error>(())
    };
    let request = async {
        let response = grpc
            .save_asset(ReceiverStream::new(receiver))
            .await?
            .into_inner();
        ensure!(
            response.asset_id == expected_id,
            "server returned an unexpected asset ID"
        );
        Ok::<_, anyhow::Error>(())
    };
    tokio::try_join!(request, sending)?;
    Ok(())
}

async fn upload_source(path: PathBuf) -> anyhow::Result<(File, bool)> {
    let (file, is_directory) = tokio::task::spawn_blocking(move || {
        let metadata = std::fs::symlink_metadata(&path)?;
        ensure!(
            metadata.is_file() || metadata.is_dir(),
            "upload path must be a file or directory"
        );
        if metadata.is_file() {
            return anyhow::Ok((std::fs::File::open(path)?, false));
        }
        let name = path.file_name().context("upload path must have a name")?;
        let mut archive = tar::Builder::new(tempfile::tempfile()?);
        archive.follow_symlinks(false);
        archive.append_dir_all(name, &path)?;
        let mut file = archive.into_inner()?;
        file.seek(SeekFrom::Start(0))?;
        anyhow::Ok((file, true))
    })
    .await
    .context("preparing upload failed")??;
    Ok((File::from_std(file), is_directory))
}
