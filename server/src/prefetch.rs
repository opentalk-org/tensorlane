use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use bytes::Bytes;
use tokio::{fs, sync::mpsc};
use tokio_util::sync::CancellationToken;
use tracing::{Instrument, debug, error, warn};

use crate::{
    loader::Loader,
    sampling::{self, Sample, Sampler},
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

impl From<(sampling::Sample, Bytes)> for LoadedSample {
    fn from(value: (sampling::Sample, Bytes)) -> Self {
        Self {
            wave: value.1,
            duration: value.0.duration,
            speaker_id: value.0.speaker_id as i64,
            language_id: value.0.language_id,
            text: value.0.text,
        }
    }
}

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
}

impl Prefetcher {
    pub fn spawn(
        mut sampler: Box<dyn Sampler>,
        loader: Arc<dyn Loader>,
        cache_dir: &'static Path,
        cancel_token: CancellationToken,
        span: tracing::Span,
    ) -> Self {
        let (tx, rx) = mpsc::channel(CACHED_BATCHES);

        let cancel_token = cancel_token.child_token();
        tokio::spawn({
            let cancel_token = cancel_token.clone();
            async move {
                loop {
                    let permit = tokio::select! {
                        biased;
                        () = cancel_token.cancelled() => break,
                        permit = tx.reserve() => match permit {
                            Ok(permit) => permit,
                            Err(_) => break,
                        },
                    };

                    let loaded = match sampler.next_batch() {
                        Ok(Some(batch)) => load_batch(&loader, cache_dir, batch).await,
                        Ok(None) => {
                            debug!("schedule exhausted");
                            break;
                        }
                        Err(err) => Err(err),
                    };
                    match &loaded {
                        Ok(batch) => debug!(samples = batch.len(), "batch ready"),
                        Err(err) => {
                            error!(error = format!("{err:#}"), "prefetching batch failed")
                        }
                    }
                    let failed = loaded.is_err();
                    permit.send(loaded);
                    if failed {
                        break;
                    }
                }
                debug!("prefetcher stopped");
            }
            .instrument(span)
        });

        Self { rx, cancel_token }
    }

    pub async fn next_batch(&mut self) -> anyhow::Result<Option<LoadedBatch>> {
        match self.rx.recv().await {
            Some(batch) => futures::future::try_join_all(batch?.into_iter().map(read_sample))
                .await
                .map(Some),
            None => Ok(None),
        }
    }

    // best-effort removal of every cached file still in flight; consumes self
    pub async fn drain(mut self) {
        self.cancel_token.cancel();

        while let Some(batch) = self.rx.recv().await {
            let Ok(batch) = batch else { continue };
            for sample in batch {
                if let Err(err) = fs::remove_file(&sample.path).await {
                    warn!(
                        error = format!("{err:#}"),
                        path = %sample.path.display(),
                        "failed to drain cache file"
                    );
                }
            }
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
    cache_dir: &'static Path,
    batch: Vec<Sample>,
) -> anyhow::Result<PrefetchedBatch> {
    debug!(samples = batch.len(), "loading batch");

    let mut loaded_batch: Vec<PrefetchedSample> = vec![];
    for (sample, wave) in loader.load_batch(batch).await? {
        let path = cache_dir.join(format!("{}-{}.raw", sample.audio_id, uuid::Uuid::new_v4()));
        fs::write(&path, &wave).await?;
        loaded_batch.push(PrefetchedSample { sample, path });
    }

    Ok(loaded_batch)
}
