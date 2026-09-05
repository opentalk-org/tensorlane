use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Context;
use flume::{Receiver, Sender};
use pyo3::prelude::*;
use tokio::io::AsyncReadExt;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_stream::wrappers::ReceiverStream;

use crate::proto::tensor_lane_client::TensorLaneClient;
use crate::proto::{
    ArtifactChunk, ArtifactMetric, MetricsRequest, MetricsStreamMetadata, ScalarMetric,
    metrics_request,
};

const CHUNK_BYTES: usize = 2 * 1024 * 1024;

enum MetricCommand {
    Scalar(ScalarMetric),
    Artifact {
        path: PathBuf,
        name: String,
        step: u64,
        content_type: Option<String>,
        timestamp_unix_ms: i64,
    },
    Directory {
        path: PathBuf,
        name: String,
        step: u64,
        timestamp_unix_ms: i64,
    },
}

struct MetricsInner {
    sender: Mutex<Option<Sender<MetricCommand>>>,
    join: Mutex<Option<JoinHandle<anyhow::Result<()>>>>,
}

#[pyclass(name = "Metrics")]
#[derive(Clone)]
pub struct NativeMetrics {
    inner: Arc<MetricsInner>,
    runtime: Arc<tokio::runtime::Runtime>,
}

#[pymethods]
impl NativeMetrics {
    fn log_metric(
        &self,
        name: String,
        value: f32,
        step: u64,
        timestamp_unix_ms: Option<i64>,
    ) -> anyhow::Result<()> {
        self.send(MetricCommand::Scalar(ScalarMetric {
            step,
            timestamp_unix_ms: timestamp_unix_ms.unwrap_or_else(now_unix_ms),
            name,
            value,
        }))
    }

    fn log_artifact(
        &self,
        path: PathBuf,
        name: String,
        step: u64,
        content_type: Option<String>,
        timestamp_unix_ms: Option<i64>,
    ) -> anyhow::Result<()> {
        self.send(MetricCommand::Artifact {
            path,
            name,
            step,
            content_type,
            timestamp_unix_ms: timestamp_unix_ms.unwrap_or_else(now_unix_ms),
        })
    }

    fn log_artifacts(&self, path: PathBuf, name: String, step: u64) -> anyhow::Result<()> {
        self.send(MetricCommand::Directory {
            path,
            name,
            step,
            timestamp_unix_ms: now_unix_ms(),
        })
    }

    fn close(&self, py: Python<'_>) -> anyhow::Result<()> {
        py.allow_threads(|| self.inner.close(&self.runtime))
    }
}

impl NativeMetrics {
    pub fn spawn(
        runtime: Arc<tokio::runtime::Runtime>,
        client: TensorLaneClient<tonic::transport::Channel>,
        training_id: String,
    ) -> Self {
        let (sender, receiver) = flume::unbounded();
        let join = runtime.spawn(worker(client, training_id, receiver));
        Self {
            inner: Arc::new(MetricsInner {
                sender: Mutex::new(Some(sender)),
                join: Mutex::new(Some(join)),
            }),
            runtime,
        }
    }

    pub fn close_native(&self) -> anyhow::Result<()> {
        self.inner.close(&self.runtime)
    }

    fn send(&self, command: MetricCommand) -> anyhow::Result<()> {
        let sender = self
            .inner
            .sender
            .lock()
            .map_err(|_| anyhow::anyhow!("metrics lock is poisoned"))?;
        let sender = sender.as_ref().context("metrics stream is closed")?;
        sender
            .send(command)
            .map_err(|error| anyhow::anyhow!("{error}"))
    }
}

impl MetricsInner {
    fn close(&self, runtime: &tokio::runtime::Runtime) -> anyhow::Result<()> {
        self.sender
            .lock()
            .map_err(|_| anyhow::anyhow!("metrics lock is poisoned"))?
            .take();
        let join = self
            .join
            .lock()
            .map_err(|_| anyhow::anyhow!("metrics task lock is poisoned"))?
            .take();
        if let Some(join) = join {
            runtime.block_on(join).context("joining metrics worker")??;
        }
        Ok(())
    }
}

async fn worker(
    mut client: TensorLaneClient<tonic::transport::Channel>,
    training_id: String,
    commands: Receiver<MetricCommand>,
) -> anyhow::Result<()> {
    let (request_sender, request_receiver) = mpsc::channel(16);
    let producer = tokio::spawn(produce(training_id, commands, request_sender));
    let rpc = client.metrics(ReceiverStream::new(request_receiver)).await;
    producer.await.context("joining metrics producer")??;
    rpc?;
    Ok(())
}

async fn produce(
    run_id: String,
    commands: Receiver<MetricCommand>,
    sender: mpsc::Sender<MetricsRequest>,
) -> anyhow::Result<()> {
    sender
        .send(MetricsRequest {
            payload: Some(metrics_request::Payload::Metadata(MetricsStreamMetadata {
                run_id,
            })),
        })
        .await?;
    while let Ok(command) = commands.recv_async().await {
        match command {
            MetricCommand::Scalar(metric) => {
                sender
                    .send(MetricsRequest {
                        payload: Some(metrics_request::Payload::Metric(metric)),
                    })
                    .await?;
            }
            MetricCommand::Artifact {
                path,
                name,
                step,
                content_type,
                timestamp_unix_ms,
            } => {
                send_artifact(&sender, path, name, step, content_type, timestamp_unix_ms).await?;
            }
            MetricCommand::Directory {
                path,
                name,
                step,
                timestamp_unix_ms,
            } => {
                for (file, relative) in directory_files(&path)? {
                    send_artifact(
                        &sender,
                        file,
                        format!("{name}/{relative}"),
                        step,
                        None,
                        timestamp_unix_ms,
                    )
                    .await?;
                }
            }
        }
    }
    Ok(())
}

async fn send_artifact(
    sender: &mpsc::Sender<MetricsRequest>,
    path: PathBuf,
    name: String,
    step: u64,
    content_type: Option<String>,
    timestamp_unix_ms: i64,
) -> anyhow::Result<()> {
    let size = tokio::fs::metadata(&path).await?.len();
    let content_type = content_type.unwrap_or_else(|| {
        mime_guess::from_path(&path)
            .first_or_octet_stream()
            .essence_str()
            .to_owned()
    });
    sender
        .send(MetricsRequest {
            payload: Some(metrics_request::Payload::Artifact(ArtifactMetric {
                step,
                timestamp_unix_ms,
                name,
                content_type,
                size_bytes: size,
            })),
        })
        .await?;
    let mut file = tokio::fs::File::open(path).await?;
    loop {
        let mut chunk = vec![0_u8; CHUNK_BYTES];
        let read = file.read(&mut chunk).await?;
        if read == 0 {
            break;
        }
        chunk.truncate(read);
        sender
            .send(MetricsRequest {
                payload: Some(metrics_request::Payload::ArtifactChunk(ArtifactChunk {
                    data: chunk.into(),
                })),
            })
            .await?;
    }
    Ok(())
}

fn directory_files(root: &Path) -> anyhow::Result<Vec<(PathBuf, String)>> {
    let mut pending = vec![root.to_path_buf()];
    let mut files = Vec::new();
    while let Some(directory) = pending.pop() {
        for entry in std::fs::read_dir(&directory)? {
            let path = entry?.path();
            if path.is_dir() {
                pending.push(path);
            } else {
                let relative = path
                    .strip_prefix(root)?
                    .to_string_lossy()
                    .replace(std::path::MAIN_SEPARATOR, "/");
                files.push((path, relative));
            }
        }
    }
    files.sort_by(|left, right| left.1.cmp(&right.1));
    Ok(files)
}

fn now_unix_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis() as i64)
}
