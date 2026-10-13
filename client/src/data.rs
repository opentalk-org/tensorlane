use crate::{
    semaphore::{BatchBudget, MemoryBudget, MemoryLease},
    transport::HttpClient,
};
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
    http: HttpClient,
    run_id: String,
    stream: String,
    budget: Arc<BatchBudget>,
    work: mpsc::UnboundedSender<Work>,
    start: u64,
) -> anyhow::Result<()> {
    let (requests, receiver) = mpsc::unbounded_channel();
    let waiting = budget.clone();
    let mut request_task = Requests {
        budget: budget.clone(),
        thread: Some(
            thread::Builder::new()
                .name("tensorlane-requests".into())
                .spawn(move || {
                    let mut sequence = start;
                    while waiting.acquire()? && requests.send(sequence).is_ok() {
                        sequence += 1;
                    }
                    Ok(())
                })?,
        ),
    };
    let batches = UnboundedReceiverStream::new(receiver)
        .map(|sequence| {
            fetch(
                &http,
                &run_id,
                &stream,
                budget.memory.clone(),
                sequence,
                start,
            )
        })
        .buffered(budget.capacity);
    futures_util::pin_mut!(batches);
    while let Some(batch) = batches.next().await {
        let (status, bytes, lease, receive_seconds) = batch?;
        if status == StatusCode::NO_CONTENT {
            break;
        }
        let response =
            DataResponse::decode(bytes::Bytes::from(bytes)).context("invalid data batch")?;
        ensure!(!response.batch.is_empty(), "empty batches are unsupported");
        work.send(Work::Batch {
            stream: stream.clone(),
            batch: (response.batch_id, response.batch.len()),
            query_batch_idx: response.query_batch_idx,
            timings: [
                response.load_seconds,
                response.server_wait_seconds,
                receive_seconds,
            ],
            samples: response.batch.into_iter().enumerate().collect(),
            memory_units: lease.units,
        })
        .context("transform worker disconnected")?;
        lease.transfer();
    }
    request_task.finish()?;
    work.send(Work::End {
        stream: stream.clone(),
    })
    .context("transform worker disconnected")?;
    Ok(())
}

async fn fetch(
    http: &HttpClient,
    run_id: &str,
    stream: &str,
    memory: Arc<MemoryBudget>,
    sequence: u64,
    start: u64,
) -> anyhow::Result<(StatusCode, Vec<u8>, MemoryLease, f64)> {
    let started = std::time::Instant::now();
    let sequence_id = sequence.to_string();
    let path = ["runs", run_id, "streams", stream, "batches", &sequence_id];
    let (status, bytes, lease) = http
        .batch(&path, memory, sequence - start)
        .await
        .context("receiving data batch")?;
    Ok((status, bytes, lease, started.elapsed().as_secs_f64()))
}
