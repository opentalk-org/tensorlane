use crate::proto::{DataRequest, Split, tensor_lane_client::TensorLaneClient};
use crate::semaphore::BatchBudget;
use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};
use std::{
    sync::Arc,
    thread::{self, JoinHandle},
    time::Duration,
};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tonic::transport::Channel;

#[derive(Serialize, Deserialize)]
pub enum Work {
    Sample {
        batch: (u64, usize),
        index: usize,
        wave: Vec<u8>,
        text: Vec<u8>,
        duration: f64,
        speaker_id: i64,
        language_id: i32,
    },
    End,
}

struct Requests {
    budget: Arc<BatchBudget>,
    thread: Option<JoinHandle<anyhow::Result<()>>>,
}

impl Requests {
    fn finish(&mut self) -> anyhow::Result<()> {
        self.budget.cancel();
        if let Some(thread) = self.thread.take() {
            thread
                .join()
                .map_err(|_| anyhow::anyhow!("request producer panicked"))??;
        }
        Ok(())
    }
}

impl Drop for Requests {
    fn drop(&mut self) {
        let _ = self.finish();
    }
}

pub async fn prefetch(
    mut grpc: TensorLaneClient<Channel>,
    run_id: String,
    budget: Arc<BatchBudget>,
    work: mpsc::UnboundedSender<Work>,
) -> anyhow::Result<()> {
    let (requests, receiver) = mpsc::unbounded_channel();
    let waiting = budget.clone();
    let mut request_task = Requests {
        budget,
        thread: Some(
            thread::Builder::new()
                .name("tensorlane-requests".into())
                .spawn(move || {
                    while waiting.acquire()? {
                        if requests
                            .send(DataRequest {
                                run_id: run_id.clone(),
                                split: Split::Training as i32,
                            })
                            .is_err()
                        {
                            break;
                        }
                    }
                    anyhow::Ok(())
                })?,
        ),
    };
    let mut stream = tokio::time::timeout(
        Duration::from_secs(120),
        grpc.data(UnboundedReceiverStream::new(receiver)),
    )
    .await
    .context("opening Data stream timed out")??
    .into_inner();
    let mut batch_id = 0u64;
    loop {
        let Some(response) = stream.message().await.context("receiving training batch")? else {
            break;
        };
        let batch_size = response.batch.len();
        ensure!(
            batch_size > 0,
            "batch {batch_id}: empty batches are unsupported"
        );
        for (index, sample) in response.batch.into_iter().enumerate() {
            ensure!(
                sample.wave.len() % 2 == 0,
                "batch {batch_id} sample {index}: invalid int16 PCM"
            );
            ensure!(
                sample.text.len() % 8 == 0,
                "batch {batch_id} sample {index}: invalid int64 tokens"
            );
            work.send(Work::Sample {
                batch: (batch_id, batch_size),
                index,
                wave: sample.wave,
                text: sample.text,
                duration: sample.duration,
                speaker_id: sample.speaker_id,
                language_id: sample.language_id,
            })?;
        }
        batch_id += 1;
    }
    request_task.finish()?;
    work.send(Work::End)?;
    Ok(())
}
