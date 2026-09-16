use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use bytes::Bytes;
use tokio::{
    fs,
    sync::{OwnedSemaphorePermit, Semaphore, mpsc},
    time::{Instant, MissedTickBehavior, interval_at},
};
use tokio_util::{future::FutureExt, sync::CancellationToken, task::TaskTracker};
use tracing::{Instrument, debug, error, info, warn};

use crate::{
    loader::Loader,
    sampling::{Sample, Sampler},
};

struct PrefetchedSample {
    sample: Sample,
    path: PathBuf,
}

type PrefetchedBatch = Vec<PrefetchedSample>;

pub struct LoadedSample {
    pub wave: Bytes,
    pub duration: f64,
    pub speaker_id: i64,
    pub language_id: i32,
    pub text: Bytes,
}

pub type LoadedBatch = Vec<LoadedSample>;

impl From<LoadedSample> for crate::proto::Sample {
    fn from(sample: LoadedSample) -> Self {
        Self {
            wave: sample.wave,
            duration: sample.duration,
            speaker_id: sample.speaker_id,
            language_id: sample.language_id,
            text: sample.text,
        }
    }
}

const CACHED_BATCHES: usize = 20;
const CACHE_LOG_INTERVAL: Duration = Duration::from_secs(60);

pub struct Prefetcher {
    rx: mpsc::UnboundedReceiver<(anyhow::Result<PrefetchedBatch>, OwnedSemaphorePermit)>,
    cancel_token: CancellationToken,
    tasks: TaskTracker,
}

impl Prefetcher {
    pub fn spawn(
        mut sampler: Box<dyn Sampler>,
        loader: Arc<dyn Loader>,
        cache_dir: PathBuf,
        cancel_token: CancellationToken,
        span: tracing::Span,
    ) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        let slots = Arc::new(Semaphore::new(CACHED_BATCHES));

        let tasks = TaskTracker::new();
        let cancel_token = cancel_token.child_token();
        tasks.spawn({
            let cancel_token = cancel_token.clone();
            let cache_dir = cache_dir.clone();
            let slots = slots.clone();
            log_cache(cancel_token, cache_dir, slots)
        });
        tasks.spawn({
            let cancel_token = cancel_token.clone();
            async move {
                'outer: loop {
                    let permit = match slots
                        .clone()
                        .acquire_owned()
                        .with_cancellation_token(&cancel_token)
                        .await
                    {
                        Some(Err(_)) | None => break,
                        Some(Ok(permit)) => permit,
                    };

                    let result = loop {
                        let loaded = match sampler.next_batch() {
                            Ok(Some(batch)) => match load_batch(&loader, &cache_dir, batch)
                                .with_cancellation_token(&cancel_token)
                                .await
                            {
                                None => break 'outer,
                                Some(v) => v,
                            },
                            Ok(None) => {
                                debug!("schedule exhausted");
                                break 'outer;
                            }
                            Err(err) => Err(err),
                        };

                        match loaded {
                            Ok(None) => continue,
                            Ok(Some(batch)) => break Ok(batch),
                            Err(err) => {
                                error!(error = format!("{err:#}"), "prefetching batch failed");
                                break Err(err);
                            }
                        }
                    };
                    if tx.send((result, permit)).is_err() {
                        break;
                    }
                }
                debug!("prefetcher stopped");
            }
            .instrument(span)
        });

        Self {
            rx,
            cancel_token,
            tasks,
        }
    }

    pub async fn next_batch(&mut self) -> anyhow::Result<Option<LoadedBatch>> {
        match self.rx.recv().await {
            Some((batch, _permit)) => {
                futures::future::try_join_all(batch?.into_iter().map(read_sample))
                    .await
                    .map(Some)
            }
            None => Ok(None),
        }
    }

    /// Cancels all of the running prefeching tasks.
    pub async fn finish(self) {
        self.tasks.close();
        self.cancel_token.cancel();
        self.tasks.wait().await;
    }
}

async fn cache_bytes(cache_dir: &Path) -> std::io::Result<u64> {
    let mut entries = fs::read_dir(cache_dir).await?;
    let mut bytes = 0;
    while let Some(entry) = entries.next_entry().await? {
        match entry.metadata().await {
            Ok(metadata) if metadata.is_file() => bytes += metadata.len(),
            Ok(_) => {}
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => return Err(err),
        }
    }
    Ok(bytes)
}

async fn log_cache(cancel_token: CancellationToken, cache_dir: PathBuf, slots: Arc<Semaphore>) {
    let mut interval = interval_at(Instant::now() + CACHE_LOG_INTERVAL, CACHE_LOG_INTERVAL);
    interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
    loop {
        if interval
            .tick()
            .with_cancellation_token(&cancel_token)
            .await
            .is_none()
        {
            break;
        }
        match cache_bytes(&cache_dir)
            .with_cancellation_token(&cancel_token)
            .await
        {
            None => break,
            Some(Ok(cached_bytes)) => info!(
                cached_bytes,
                cached_batches = CACHED_BATCHES - slots.available_permits(),
                cache_dir = %cache_dir.display(),
                "prefetch cache usage"
            ),
            Some(Err(err)) => warn!(
                error = format!("{err:#}"),
                cache_dir = %cache_dir.display(),
                "failed to measure prefetch cache"
            ),
        }
    }
}

async fn read_sample(sample: PrefetchedSample) -> anyhow::Result<LoadedSample> {
    let wave = fs::read(&sample.path).await?.into();
    fs::remove_file(&sample.path).await?;
    let meta = sample.sample;
    Ok(LoadedSample {
        wave,
        duration: meta.duration,
        speaker_id: meta.speaker_id as i64,
        language_id: meta.language_id,
        text: meta.text,
    })
}

async fn load_batch(
    loader: &Arc<dyn Loader>,
    cache_dir: &Path,
    batch: Vec<Sample>,
) -> anyhow::Result<Option<PrefetchedBatch>> {
    debug!(samples = batch.len(), "loading batch");

    let mut loaded_batch: Vec<PrefetchedSample> = vec![];
    if let Some(batch) = loader.load_batch(batch).await? {
        for (sample, wave) in batch {
            let path = cache_dir.join(format!("{}-{}.raw", sample.audio_id, uuid::Uuid::new_v4()));
            fs::write(&path, &wave).await?;
            loaded_batch.push(PrefetchedSample { sample, path });
        }
    } else {
        return Ok(None);
    }

    Ok(Some(loaded_batch))
}
