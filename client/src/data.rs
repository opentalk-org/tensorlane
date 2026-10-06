use crate::semaphore::BatchBudget;
use anyhow::{Context, ensure};
use futures_util::StreamExt;
use prost::Message;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use std::{
    sync::Arc,
    thread::{self, JoinHandle},
};
use tensorlane_protocol::{DataResponse, Sample};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;

#[derive(Serialize, Deserialize)]
pub enum Work {
    Batch {
        stream: String,
        batch: (u64, usize),
        query_batch_idx: u64,
        timings: [f64; 3],
        samples: Vec<(usize, Sample)>,
        memory_units: usize,
    },
    End {
        stream: String,
    },
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
    http: crate::transport::HttpClient,
    run_id: String,
    stream_name: String,
    budget: Arc<BatchBudget>,
    work: mpsc::UnboundedSender<Work>,
    start: u64,
) -> anyhow::Result<()> {
    let (requests, receiver) = mpsc::unbounded_channel();
    let waiting = budget.clone();
    let mut request_task = Requests {
        budget,
        thread: Some(
            thread::Builder::new()
                .name("tensorlane-requests".into())
                .spawn(move || {
                    let mut sequence = start;
                    while waiting.acquire()? {
                        if requests.send(sequence).is_err() {
                            break;
                        }
                        sequence += 1;
                    }
                    Ok(())
                })?,
        ),
    };
    let memory = request_task.budget.memory.clone();
    let batches = UnboundedReceiverStream::new(receiver)
        .map(|sequence| {
            let http = http.clone();
            let run_id = run_id.clone();
            let stream_name = stream_name.clone();
            let memory = memory.clone();
            async move {
                let started = std::time::Instant::now();
                let (status, bytes, lease) = http
                    .batch(
                        &[
                            "runs",
                            &run_id,
                            "streams",
                            &stream_name,
                            "batches",
                            &sequence.to_string(),
                        ],
                        memory,
                        sequence - start,
                    )
                    .await
                    .context("receiving data batch")?;
                anyhow::Ok((
                    sequence,
                    status,
                    bytes,
                    lease,
                    started.elapsed().as_secs_f64(),
                ))
            }
        })
        .buffered(request_task.budget.capacity);
    futures_util::pin_mut!(batches);
    while let Some(batch) = batches.next().await {
        let (expected_id, status, bytes, lease, receive_seconds) = batch?;
        if status == StatusCode::NO_CONTENT {
            break;
        }
        let response =
            DataResponse::decode(bytes::Bytes::from(bytes)).context("invalid data batch")?;
        let timings = [
            response.load_seconds,
            response.server_wait_seconds,
            receive_seconds,
        ];
        ensure!(
            response.stream == stream_name && response.batch_id == expected_id,
            "unexpected stream or batch ID"
        );
        let batch_size = response.batch.len();
        ensure!(batch_size > 0, "empty batches are unsupported");
        work.send(Work::Batch {
            stream: stream_name.clone(),
            batch: (response.batch_id, batch_size),
            query_batch_idx: response.query_batch_idx,
            timings,
            samples: response.batch.into_iter().enumerate().collect(),
            memory_units: lease.units,
        })
        .context("transform worker disconnected")?;
        lease.transfer();
    }
    request_task.finish()?;
    work.send(Work::End {
        stream: stream_name.clone(),
    })
    .context("transform worker disconnected")?;
    Ok(())
}
