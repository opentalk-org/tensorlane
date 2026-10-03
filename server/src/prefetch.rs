use crate::{MAX_BATCH_BYTES, loader::Loader, sampling::Sampler};
use anyhow::{Result, ensure};
use futures::{StreamExt, TryStreamExt};
use prost::Message;
use std::{path::PathBuf, sync::Arc};
use tokio::{fs, sync::mpsc};
use tokio_util::{future::FutureExt, sync::CancellationToken, task::TaskTracker};
use tracing::Instrument;

pub type LoadedBatch = crate::proto::DataResponse;
struct CachedBatch {
    path: PathBuf,
    index: usize,
    available: mpsc::Sender<usize>,
}
impl Drop for CachedBatch {
    fn drop(&mut self) {
        let _ = self.available.try_send(self.index);
    }
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
        let (available, mut slots) = mpsc::channel(CACHED_BATCHES);
        for index in 0..CACHED_BATCHES {
            available.try_send(index).expect("empty slot queue");
        }
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
                        let Some(index) = slots.recv().await else {
                            return Ok(None);
                        };
                        let cached = CachedBatch {
                            path: cache.join(format!("{index}.batch")),
                            index,
                            available: available.clone(),
                        };
                        let Some(plan) = sampler.next_batch().await? else {
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
                        let mut samples = futures::stream::iter(
                            plan.samples
                                .into_iter()
                                .map(|sample| loader.load_sample(sample)),
                        )
                        .buffered(16);
                        while let Some(sample) = samples.try_next().await? {
                            response.batch.push(sample);
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
                        fs::write(&cached.path, response.encode_to_vec()).await?;
                        batch_id += 1;
                        Ok(Some(cached))
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
        let batch = batch?;
        let bytes = fs::read(&batch.path).await?;
        let mut response = LoadedBatch::decode(bytes.as_slice())?;
        response.server_wait_seconds = started.elapsed().as_secs_f64();
        Ok(Some(response))
    }
    pub async fn finish(self) {
        self.tasks.close();
        self.cancel.cancel();
        self.tasks.wait().await;
    }
}

#[cfg(test)]
mod concurrency_tests {
    use super::*;
    use crate::sampling::{BatchPlan, BlobRef, Sample};
    use std::collections::{BTreeMap, HashMap};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct OneBatch(Option<BatchPlan>);

    impl Sampler for OneBatch {
        fn next_batch(&mut self) -> futures::future::BoxFuture<'_, Result<Option<BatchPlan>>> {
            Box::pin(std::future::ready(Ok(self.0.take())))
        }
    }

    struct DelayedLoader {
        active: AtomicUsize,
        peak: AtomicUsize,
    }

    #[async_trait::async_trait]
    impl Loader for DelayedLoader {
        async fn load(&self, _: &BlobRef) -> Result<bytes::Bytes> {
            unreachable!()
        }

        async fn load_sample(&self, sample: Sample) -> Result<crate::proto::Sample> {
            let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak.fetch_max(active, Ordering::SeqCst);
            let index: u64 = sample.sample_id.parse()?;
            tokio::time::sleep(std::time::Duration::from_millis(32 - index)).await;
            self.active.fetch_sub(1, Ordering::SeqCst);
            Ok(crate::proto::Sample {
                sample_id: sample.sample_id,
                metadata_json: sample.metadata_json,
                blobs: HashMap::new(),
            })
        }
    }

    #[tokio::test]
    async fn concurrent_samples_preserve_order_and_bound_concurrency() -> Result<()> {
        let cache = tempfile::tempdir()?;
        let samples = (0..32)
            .map(|index| Sample {
                sample_id: index.to_string(),
                metadata_json: "{}".to_string(),
                blobs: BTreeMap::new(),
            })
            .collect();
        let loader = Arc::new(DelayedLoader {
            active: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
        });
        let mut prefetch = Prefetcher::spawn(
            Box::new(OneBatch(Some(BatchPlan {
                query_batch_idx: 7,
                samples,
            }))),
            loader.clone(),
            cache.path().to_path_buf(),
            CancellationToken::new(),
            tracing::Span::none(),
            "training".to_string(),
        );
        let batch = prefetch.next_batch().await?.expect("one batch");
        assert_eq!(batch.query_batch_idx, 7);
        assert_eq!(batch.batch.len(), 32);
        for (index, sample) in batch.batch.iter().enumerate() {
            assert_eq!(sample.sample_id, index.to_string());
        }
        assert!((2..=16).contains(&loader.peak.load(Ordering::SeqCst)));
        assert!(prefetch.next_batch().await?.is_none());
        prefetch.finish().await;
        Ok(())
    }
}
