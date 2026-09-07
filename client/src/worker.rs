use std::thread::{self, JoinHandle};
use std::time::Duration;

use anyhow::{Context, anyhow};
use tokio::sync::{mpsc, oneshot};
use tonic::transport::{Channel, Endpoint};

use crate::data::DataStream;
use crate::proto::{
    DataRequest, DataResponse, InitRequest, InitResponse, Split,
    tensor_lane_client::TensorLaneClient,
};

#[cfg(test)]
#[path = "worker_tests.rs"]
mod tests;

struct Session {
    grpc: TensorLaneClient<Channel>,
    run_id: String,
    training_data: DataStream,
    validation_data: DataStream,
}

pub(crate) enum Command {
    NextBatch {
        validation: bool,
        reply: oneshot::Sender<anyhow::Result<Option<DataResponse>>>,
    },
}

async fn connect(run_id: String, addr: String) -> anyhow::Result<(Session, InitResponse)> {
    let url = if addr.contains("://") {
        addr
    } else {
        format!("http://{addr}")
    };
    let endpoint = Endpoint::from_shared(url.clone())
        .context("invalid TensorLane address")?
        .connect_timeout(Duration::from_secs(10));
    let channel = endpoint
        .connect()
        .await
        .with_context(|| format!("connecting to {url}"))?;
    let mut grpc = TensorLaneClient::new(channel).max_decoding_message_size(67_136_000);
    let response = tokio::time::timeout(Duration::from_secs(30), grpc.init(InitRequest { run_id }))
        .await
        .context("TensorLane Init timed out")?
        .context("TensorLane Init failed")?
        .into_inner();
    Ok((
        Session {
            grpc,
            run_id: response.run_id.clone(),
            training_data: DataStream::default(),
            validation_data: DataStream::default(),
        },
        response,
    ))
}

async fn run(mut commands: mpsc::UnboundedReceiver<Command>, mut session: Session) {
    while let Some(command) = commands.recv().await {
        match command {
            Command::NextBatch { validation, reply } => {
                let request = DataRequest {
                    run_id: session.run_id.clone(),
                    split: if validation {
                        Split::Validation
                    } else {
                        Split::Training
                    } as i32,
                };
                let stream = if validation {
                    &mut session.validation_data
                } else {
                    &mut session.training_data
                };
                let result = stream.next(&mut session.grpc, request).await;
                let _ = reply.send(result);
            }
        }
    }
}

pub(crate) struct Worker {
    sender: Option<mpsc::UnboundedSender<Command>>,
    thread: Option<JoinHandle<()>>,
}

impl Worker {
    pub(crate) fn start(run_id: String, addr: String) -> anyhow::Result<(Self, InitResponse)> {
        let (sender, receiver) = mpsc::unbounded_channel();
        let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(1);
        let thread = thread::Builder::new()
            .name("tensorlane-client".into())
            .spawn(move || {
                let runtime = match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(runtime) => runtime,
                    Err(error) => {
                        let _ = ready_tx.send(Err(anyhow!(error)));
                        return;
                    }
                };
                runtime.block_on(async move {
                    match connect(run_id, addr).await {
                        Ok((session, initialized)) => {
                            if ready_tx.send(Ok(initialized)).is_ok() {
                                run(receiver, session).await;
                            }
                        }
                        Err(error) => {
                            let _ = ready_tx.send(Err(error));
                        }
                    }
                });
            })
            .context("starting TensorLane worker thread")?;
        let worker = Self {
            sender: Some(sender),
            thread: Some(thread),
        };
        let initialized = ready_rx.recv().context("worker stopped during startup")??;
        Ok((worker, initialized))
    }

    pub(crate) fn send(&self, command: Command) -> anyhow::Result<()> {
        self.sender
            .as_ref()
            .context("TensorLane client is closed")?
            .send(command)
            .map_err(|_| anyhow!("TensorLane worker stopped"))
    }

    pub(crate) fn shutdown(&mut self) -> anyhow::Result<()> {
        self.sender.take();
        if let Some(thread) = self.thread.take() {
            thread
                .join()
                .map_err(|_| anyhow!("TensorLane worker panicked"))?;
        }
        Ok(())
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}
