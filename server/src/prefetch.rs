use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use bytes::Bytes;
use tokio::{fs, sync::mpsc};
use tokio_util::{future::FutureExt, sync::CancellationToken, task::TaskTracker};
use tracing::{Instrument, debug, warn};

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

const CACHED_BATCHES: usize = 5;

pub struct Prefetcher {
    rx: mpsc::Receiver<anyhow::Result<PrefetchedBatch>>,
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
        let (tx, rx) = mpsc::channel(CACHED_BATCHES);

        let tasks = TaskTracker::new();
        let cancel_token = cancel_token.child_token();
        tasks.spawn({
            let cancel_token = cancel_token.clone();
            async move {
                'outer: loop {
                    let permit = match tx.reserve().with_cancellation_token(&cancel_token).await {
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
                            Err(err) => break Err(err),
                        };

                        match loaded {
                            Ok(None) => {
                                warn!("batch loading failed, skipping batch");
                                continue;
                            }
                            Ok(Some(batch)) => break Ok(batch),
                            Err(err) => {
                                warn!(error = format!("{err:#}"), "batch loading failed, skipping batch");
                                continue;
                            }
                        }
                    };
                    permit.send(result);
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
            Some(batch) => futures::future::try_join_all(batch?.into_iter().map(read_sample))
                .await
                .map(Some),
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
