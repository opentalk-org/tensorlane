use crate::semaphore::BatchBudget;
use anyhow::{Context, ensure};
use prost::Message;
use reqwest::{Method, StatusCode};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::Arc,
    thread::{self, JoinHandle},
};
use tensorlane_protocol::DataResponse;
use tokio::sync::mpsc;

#[derive(Serialize, Deserialize)]
pub enum Work {
    Sample {
        stream: String,
        batch: (u64, usize),
        query_batch_idx: u64,
        timings: [f64; 3],
        index: usize,
        sample_id: String,
        metadata_json: String,
        blobs: HashMap<String, Vec<u8>>,
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
                        if requests.send(()).is_err() {
                            break;
                        }
                    }
                    Ok(())
                })?,
        ),
    };
    let mut receiver = receiver;
    let mut expected_id = 0;
    while receiver.recv().await.is_some() {
        let started = std::time::Instant::now();
        let (status, _, bytes) = http
            .request(
                Method::GET,
                &[
                    "runs",
                    &run_id,
                    "streams",
                    &stream_name,
                    "batches",
                    &expected_id.to_string(),
                ],
                None,
                &[],
                crate::MAX_BATCH_BYTES,
            )
            .await
            .context("receiving data batch")?;
        if status == StatusCode::NO_CONTENT {
            break;
        }
        let response = DataResponse::decode(bytes.as_slice()).context("invalid data batch")?;
        let timings = [
            response.load_seconds,
            response.server_wait_seconds,
            started.elapsed().as_secs_f64(),
        ];
        ensure!(
            response.stream == stream_name && response.batch_id == expected_id,
            "unexpected stream or batch ID"
        );
        let batch_size = response.batch.len();
        ensure!(batch_size > 0, "empty batches are unsupported");
        for (index, sample) in response.batch.into_iter().enumerate() {
            work.send(Work::Sample {
                stream: stream_name.clone(),
                batch: (response.batch_id, batch_size),
                query_batch_idx: response.query_batch_idx,
                timings,
                index,
                sample_id: sample.sample_id,
                metadata_json: sample.metadata_json,
                blobs: sample.blobs,
            })
            .context("transform worker disconnected")?;
        }
        expected_id += 1;
    }
    request_task.finish()?;
    work.send(Work::End {
        stream: stream_name,
    })
    .context("transform worker disconnected")?;
    Ok(())
}
