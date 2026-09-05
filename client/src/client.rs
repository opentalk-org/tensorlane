use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, bail};
use flume::Sender;
use pyo3::prelude::*;
use tokio::task::JoinHandle;
use tonic::transport::{Channel, Endpoint};

use crate::assets;
use crate::checkpoints::{self, CheckpointJob};
use crate::data::{self, DataTask, NativeDataStream};
use crate::metrics::NativeMetrics;
use crate::proto::tensor_lane_client::TensorLaneClient;
use crate::proto::{EndRequest, InitRequest};

const MAX_MESSAGE_BYTES: usize = 67_136_000;

struct ClientState {
    checkpoint_sender: Option<Sender<CheckpointJob>>,
    checkpoint_join: Option<JoinHandle<anyhow::Result<()>>>,
    data_tasks: Vec<DataTask>,
    metrics: Option<NativeMetrics>,
    closed: bool,
}

#[pyclass(name = "Client")]
pub struct NativeClient {
    runtime: Arc<tokio::runtime::Runtime>,
    grpc: TensorLaneClient<Channel>,
    run_id: String,
    train_config: String,
    state: Mutex<ClientState>,
}

#[pymethods]
impl NativeClient {
    #[new]
    #[pyo3(signature = (run_id, addr="localhost:8181"))]
    fn new(py: Python<'_>, run_id: String, addr: &str) -> anyhow::Result<Self> {
        py.allow_threads(|| Self::connect(run_id, addr))
    }

    #[getter]
    fn run_id(&self) -> &str {
        &self.run_id
    }

    #[getter]
    fn train_config(&self) -> &str {
        &self.train_config
    }

    #[pyo3(signature = (validation=false, prefetch=4, modality_id=0, pin_memory=false))]
    fn batches(
        &self,
        validation: bool,
        prefetch: usize,
        modality_id: i64,
        pin_memory: bool,
    ) -> anyhow::Result<NativeDataStream> {
        let mut state = self.lock_state()?;
        if state.closed {
            bail!("tensorlane client is closed");
        }
        let (stream, task) = data::spawn(
            &self.runtime,
            self.grpc.clone(),
            self.run_id.clone(),
            validation,
            prefetch,
            modality_id,
            pin_memory,
        )?;
        state.data_tasks.push(task);
        Ok(stream)
    }

    fn download_asset(
        &self,
        py: Python<'_>,
        name: String,
        destination: PathBuf,
    ) -> anyhow::Result<PathBuf> {
        self.ensure_open()?;
        let future = assets::download(self.grpc.clone(), self.run_id.clone(), name, destination);
        py.allow_threads(|| self.runtime.block_on(future))
    }

    fn upload_checkpoint(&self, step: u64, source: PathBuf) -> anyhow::Result<()> {
        let state = self.lock_state()?;
        if state.closed {
            bail!("tensorlane client is closed");
        }
        let sender = state
            .checkpoint_sender
            .as_ref()
            .ok_or_else(|| anyhow!("checkpoint worker is closed"))?;
        sender
            .send(CheckpointJob { step, source })
            .map_err(Into::into)
    }

    fn metrics(&self) -> anyhow::Result<NativeMetrics> {
        let mut state = self.lock_state()?;
        if state.closed {
            bail!("tensorlane client is closed");
        }
        if let Some(metrics) = &state.metrics {
            return Ok(metrics.clone());
        }
        let metrics =
            NativeMetrics::spawn(self.runtime.clone(), self.grpc.clone(), self.run_id.clone());
        state.metrics = Some(metrics.clone());
        Ok(metrics)
    }

    fn close(&self, py: Python<'_>) -> anyhow::Result<()> {
        py.allow_threads(|| self.close_native())
    }
}

impl NativeClient {
    fn connect(run_id: String, addr: &str) -> anyhow::Result<Self> {
        let runtime = Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .thread_name("tensorlane-client")
                .build()?,
        );
        let endpoint = Endpoint::from_shared(format!("http://{addr}"))?;
        let channel = runtime.block_on(endpoint.connect())?;
        let mut grpc = TensorLaneClient::new(channel)
            .max_decoding_message_size(MAX_MESSAGE_BYTES)
            .max_encoding_message_size(MAX_MESSAGE_BYTES);
        let initialized = runtime
            .block_on(grpc.init(InitRequest { run_id }))?
            .into_inner();
        let (checkpoint_sender, checkpoint_receiver) = flume::unbounded();
        let checkpoint_join = runtime.spawn(checkpoints::worker(
            grpc.clone(),
            initialized.run_id.clone(),
            checkpoint_receiver,
        ));
        Ok(Self {
            runtime,
            grpc,
            run_id: initialized.run_id,
            train_config: initialized.train_config,
            state: Mutex::new(ClientState {
                checkpoint_sender: Some(checkpoint_sender),
                checkpoint_join: Some(checkpoint_join),
                data_tasks: Vec::new(),
                metrics: None,
                closed: false,
            }),
        })
    }

    fn close_native(&self) -> anyhow::Result<()> {
        let (checkpoint_sender, checkpoint_join, data_tasks, metrics) = {
            let mut state = self
                .state
                .lock()
                .map_err(|_| anyhow::anyhow!("client lock is poisoned"))?;
            if state.closed {
                return Ok(());
            }
            state.closed = true;
            (
                state.checkpoint_sender.take(),
                state.checkpoint_join.take(),
                std::mem::take(&mut state.data_tasks),
                state.metrics.take(),
            )
        };
        for task in &data_tasks {
            task.cancellation.cancel();
        }
        drop(checkpoint_sender);
        let mut errors = Vec::new();
        if let Some(metrics) = metrics
            && let Err(error) = metrics.close_native()
        {
            errors.push(format!("closing metrics stream: {error:#}"));
        }
        self.runtime.block_on(async {
            for task in data_tasks {
                match task.join.await {
                    Ok(Ok(())) => {}
                    Ok(Err(error)) => errors.push(format!("data stream failed: {error:#}")),
                    Err(error) => errors.push(format!("joining data stream: {error:#}")),
                }
            }
            if let Some(join) = checkpoint_join {
                match join.await {
                    Ok(Ok(())) => {}
                    Ok(Err(error)) => {
                        errors.push(format!("checkpoint upload failed: {error:#}"));
                    }
                    Err(error) => errors.push(format!("joining checkpoint worker: {error:#}")),
                }
            }
        });
        let mut grpc = self.grpc.clone();
        if let Err(error) = self.runtime.block_on(grpc.end(EndRequest {
            run_id: self.run_id.clone(),
        })) {
            errors.push(format!("ending training: {error:#}"));
        }
        if errors.is_empty() {
            Ok(())
        } else {
            anyhow::bail!(errors.join("; "))
        }
    }

    fn ensure_open(&self) -> anyhow::Result<()> {
        let state = self.lock_state()?;
        if state.closed {
            bail!("tensorlane client is closed")
        } else {
            Ok(())
        }
    }

    fn lock_state(&self) -> anyhow::Result<std::sync::MutexGuard<'_, ClientState>> {
        self.state
            .lock()
            .map_err(|_| anyhow::anyhow!("client lock is poisoned"))
    }
}
