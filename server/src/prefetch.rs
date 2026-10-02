use crate::{MAX_BATCH_BYTES, loader::Loader, sampling::Sampler};
use anyhow::{Result, ensure};
use prost::Message;
use std::{path::PathBuf, sync::Arc};
use tokio::{fs, sync::mpsc};
use tokio_util::{future::FutureExt, sync::CancellationToken, task::TaskTracker};
use tracing::Instrument;

pub type LoadedBatch = crate::proto::DataResponse;
struct CachedBatch {
    path: PathBuf,
}
const CACHED_BATCHES: usize = 20;

pub struct Prefetcher {
    rx: mpsc::Receiver<Result<CachedBatch>>,
    cancel: CancellationToken,
    tasks: TaskTracker,
}
impl Prefetcher {
    pub fn spawn(
        mut sampler: Box<dyn Sampler>,
        loader: Arc<dyn Loader>,
        cache: PathBuf,
        parent: CancellationToken,
        span: tracing::Span,
        stream: String,
    ) -> Self {
        let (tx, rx) = mpsc::channel(CACHED_BATCHES);
        let cancel = parent.child_token();
        let tasks = TaskTracker::new();
        tasks.spawn({
            let cancel = cancel.clone();
            async move {
                let mut batch_id = 0;
                loop {
                    let Some(Ok(slot)) = tx.reserve().with_cancellation_token(&cancel).await else {
                        break;
                    };
                    let result: Option<Result<Option<CachedBatch>>> = async {
                        let Some(plan) = sampler.next_batch()? else {
                            return Ok(None);
                        };
                        let started = std::time::Instant::now();
                        let mut response = LoadedBatch {
                            stream: stream.clone(),
                            batch_id,
                            query_batch_idx: plan.query_batch_idx,
                            batch: Vec::with_capacity(plan.samples.len()),
                            load_seconds: 0.0,
                            server_wait_seconds: 0.0,
                        };
                        for sample in plan.samples {
                            response.batch.push(loader.load_sample(sample).await?);
                            ensure!(
                                response.encoded_len() <= MAX_BATCH_BYTES,
                                "encoded batch exceeds 64 MiB"
                            );
                        }
                        response.load_seconds = started.elapsed().as_secs_f64();
                        ensure!(
                            response.encoded_len() + 9 <= MAX_BATCH_BYTES,
                            "encoded batch exceeds 64 MiB"
                        );
                        let path = cache.join(format!("{}.batch", uuid::Uuid::new_v4()));
                        let part = path.with_extension("part");
                        let write = async {
                            fs::write(&part, response.encode_to_vec()).await?;
                            fs::rename(&part, &path).await?;
                            Ok::<_, std::io::Error>(())
                        }
                        .await;
                        if write.is_err() {
                            let _ = fs::remove_file(&part).await;
                        }
                        write?;
                        batch_id += 1;
                        Ok(Some(CachedBatch { path }))
                    }
                    .with_cancellation_token(&cancel)
                    .await;
                    match result {
                        Some(Ok(Some(batch))) => slot.send(Ok(batch)),
                        Some(Err(error)) => {
                            slot.send(Err(error));
                            break;
                        }
                        _ => break,
                    }
                }
            }
            .instrument(span)
        });
        Self { rx, cancel, tasks }
    }
    pub async fn next_batch(&mut self) -> Result<Option<LoadedBatch>> {
        let started = std::time::Instant::now();
        let Some(batch) = self.rx.recv().await else {
            return Ok(None);
        };
        let path = batch?.path;
        let result = fs::read(&path).await;
        let _ = fs::remove_file(&path).await;
        let mut response = LoadedBatch::decode(result?.as_slice())?;
        response.server_wait_seconds = started.elapsed().as_secs_f64();
        Ok(Some(response))
    }
    pub async fn finish(self) {
        self.tasks.close();
        self.cancel.cancel();
        self.tasks.wait().await;
    }
}
