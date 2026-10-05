use crate::ipc::{Receiver, Sender};
use anyhow::{Context, anyhow, bail, ensure};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio::{
    net::{
        UnixStream,
        unix::{OwnedReadHalf, OwnedWriteHalf},
    },
    sync::{mpsc, oneshot},
};

#[derive(Serialize, Deserialize)]
pub(crate) enum Upload {
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

pub(crate) type Reply = Result<(), String>;

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
            let (read, write) = socket.into_split();
            let (sender, mut commands) = mpsc::channel(1024);
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
                            Command::Flush(reply) => {
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

    #[pyo3(signature = (step, name, value, timeout=None))]
    fn metric(
        &self,
        py: Python<'_>,
        step: u64,
        name: String,
        value: f32,
        timeout: Option<f64>,
    ) -> anyhow::Result<()> {
        let command = Command::Send(Upload::Metric { step, name, value });
        let Some(timeout) = timeout else {
            return py.allow_threads(|| self.enqueue(command));
        };
        let timeout = Duration::try_from_secs_f64(timeout)?;
        let sender = self.command_sender()?;
        py.allow_threads(|| {
            self.runtime.block_on(async {
                tokio::time::timeout(timeout, sender.send(command))
                    .await
                    .context("metric enqueue timed out")?
                    .map_err(|_| self.error())
            })
        })
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
        let sender = self.command_sender()?;
        let (reply, received) = oneshot::channel();
        py.allow_threads(|| {
            self.runtime.block_on(async {
                let operation = async {
                    sender
                        .send(Command::Flush(reply))
                        .await
                        .map_err(|_| self.error())?;
                    received
                        .await
                        .map_err(|_| self.error())?
                        .map_err(|error| anyhow!(error))
                };
                let result = match timeout {
                    Some(timeout) => tokio::time::timeout(timeout, operation)
                        .await
                        .context("upload flush timed out")?,
                    None => operation.await,
                };
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
            py.allow_threads(|| {
                sender
                    .blocking_send(Command::Flush(reply))
                    .map_err(|_| self.error())
            })?;
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
    fn command_sender(&self) -> anyhow::Result<mpsc::Sender<Command>> {
        self.sender
            .lock()
            .map_err(|_| anyhow!("upload lock poisoned"))?
            .as_ref()
            .context("upload client is closed")
            .cloned()
    }
    fn enqueue(&self, command: Command) -> anyhow::Result<()> {
        self.command_sender()?
            .blocking_send(command)
            .map_err(|_| self.error())
    }
    fn send(&self, py: Python<'_>, message: Upload) -> anyhow::Result<()> {
        py.allow_threads(|| self.enqueue(Command::Send(message)))
    }
}
