use crate::ipc::{Receiver, Sender};
use anyhow::{Context, anyhow, ensure};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::{
    net::UnixStream,
    sync::{mpsc, oneshot},
};

#[derive(Serialize, Deserialize)]
pub(crate) enum Upload {
    Automatic,
    Metric {
        step: u64,
        name: String,
        value: f32,
        timestamp_unix_ms: i64,
    },
    Artifact {
        step: u64,
        timestamp_unix_ms: i64,
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

pub(crate) type Reply = Result<(), String>;

enum Command {
    Send(Upload),
    Flush(oneshot::Sender<Reply>),
}

#[pyclass]
pub struct UploadClient {
    sender: Mutex<Option<mpsc::Sender<Command>>>,
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
            let (sender, commands) = mpsc::channel(1024);
            let failure = Arc::new(Mutex::new(None));
            runtime.spawn(forward(socket, commands, automatic, failure.clone()));
            Ok(Self {
                sender: Mutex::new(Some(sender)),
                failure,
                runtime,
            })
        })
    }

    #[pyo3(signature = (step, name, value, timeout=None))]
    fn metric(
        &self,
        py: Python<'_>,
        step: u64,
        name: String,
        value: f32,
        timeout: Option<f64>,
    ) -> anyhow::Result<()> {
        let timestamp_unix_ms = timestamp()?;
        let metric = Upload::Metric {
            step,
            name,
            value,
            timestamp_unix_ms,
        };
        self.send(py, metric, timeout)
    }

    fn metric_artifact(
        &self,
        py: Python<'_>,
        step: u64,
        path: PathBuf,
        name: String,
        content_type: String,
    ) -> anyhow::Result<()> {
        let timestamp_unix_ms = timestamp()?;
        let artifact = Upload::Artifact {
            step,
            timestamp_unix_ms,
            path,
            name,
            content_type,
        };
        self.send(py, artifact, None)
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
        let asset = Upload::SaveAsset {
            asset_id: asset_id.clone(),
            name,
            step,
            path,
            kind,
            asset_type,
            metadata_json,
            content_type,
        };
        self.send(py, asset, None)?;
        Ok(asset_id)
    }

    #[pyo3(signature = (timeout=None))]
    fn flush(&self, py: Python<'_>, timeout: Option<f64>) -> anyhow::Result<()> {
        let sender = self
            .sender
            .lock()
            .unwrap()
            .clone()
            .context("upload client is closed")?;
        self.wait(py, sender, timeout)
    }

    fn close(&self, py: Python<'_>) -> anyhow::Result<()> {
        let Some(sender) = self.sender.lock().unwrap().take() else {
            return Ok(());
        };
        self.wait(py, sender, None)
    }
}

impl UploadClient {
    fn send(&self, py: Python<'_>, message: Upload, timeout: Option<f64>) -> anyhow::Result<()> {
        let timeout = timeout.map(Duration::try_from_secs_f64).transpose()?;
        let sender = self
            .sender
            .lock()
            .unwrap()
            .clone()
            .context("upload client is closed")?;
        let sent = py.allow_threads(|| match timeout {
            None => Ok(sender.blocking_send(Command::Send(message))),
            Some(timeout) => self
                .runtime
                .block_on(async {
                    tokio::time::timeout(timeout, sender.send(Command::Send(message))).await
                })
                .context("upload enqueue timed out"),
        })?;
        sent.map_err(|_| self.error())
    }

    fn wait(
        &self,
        py: Python<'_>,
        sender: mpsc::Sender<Command>,
        timeout: Option<f64>,
    ) -> anyhow::Result<()> {
        let timeout = timeout.map(Duration::try_from_secs_f64).transpose()?;
        let (reply, received) = oneshot::channel();
        py.allow_threads(|| {
            self.runtime.block_on(async {
                let flushed = async {
                    sender
                        .send(Command::Flush(reply))
                        .await
                        .map_err(|_| self.error())?;
                    drop(sender);
                    received
                        .await
                        .map_err(|_| self.error())?
                        .map_err(anyhow::Error::msg)
                };
                match timeout {
                    Some(timeout) => tokio::time::timeout(timeout, flushed)
                        .await
                        .context("upload flush timed out")?,
                    None => flushed.await,
                }
            })
        })
    }

    fn error(&self) -> anyhow::Error {
        let failure = self.failure.lock().unwrap().clone();
        anyhow!(failure.unwrap_or_else(|| "upload connection closed".into()))
    }
}

async fn forward(
    socket: UnixStream,
    mut commands: mpsc::Receiver<Command>,
    automatic: bool,
    failure: Arc<Mutex<Option<String>>>,
) {
    let (read, write) = socket.into_split();
    let mut sender = Sender::<Upload, _>::new(write);
    let mut receiver = Receiver::<Reply, _>::new(read);
    let mut dirty = false;
    if automatic && let Err(error) = sender.send(&Upload::Automatic).await {
        *failure.lock().unwrap() = Some(format!("{error:#}"));
        return;
    }
    while let Some(command) = commands.recv().await {
        let reply = match command {
            Command::Send(message) => {
                if let Err(error) = sender.send(&message).await {
                    *failure.lock().unwrap() = Some(format!("{error:#}"));
                    return;
                }
                dirty = true;
                continue;
            }
            Command::Flush(reply) => reply,
        };
        if !dirty {
            let _ = reply.send(Ok(()));
            continue;
        }
        let sent = sender.send(&Upload::Flush).await;
        let result = match receiver.recv().await {
            Ok(Some(result)) => result,
            Ok(None) => Err(sent
                .err()
                .map_or("upload connection closed".into(), |e| format!("{e:#}"))),
            Err(error) => Err(format!("{error:#}")),
        };
        dirty = result.is_err();
        let _ = reply.send(result);
    }
}

fn timestamp() -> anyhow::Result<i64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_millis()
        .try_into()?)
}
