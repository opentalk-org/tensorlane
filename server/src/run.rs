use std::{path::Path, pin::Pin, sync::Arc};

use anyhow::bail;
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;
use tracing::{Instrument, debug, info, info_span};
use uuid::Uuid;

use crate::{
    db::{
        TRAINING_SEED_SALT, VALIDATION_SEED_SALT, fetch_training_samples, fetch_validation_samples,
        synthetic_rows,
    },
    loader::Loader,
    prefetch::{LoadedBatch, LoadedSample, Prefetcher},
    sampling::{HistogramSampler, Sampler, ScheduledSampler, bins_from_rows},
};

const SYNTHETIC_TRAINING_SAMPLES: usize = 256;

#[derive(Clone, Deserialize, Serialize)]
pub struct DataConfig {
    pub dataset_id: Uuid,
    pub seed: u64,
    pub max_text_tokens: i32,
    #[serde(default)]
    pub plbert_languages: Vec<String>,
    /// Asset names are the contract with the training side.
    #[serde(default)]
    pub assets: std::collections::HashMap<String, AssetConfig>,
    pub validation: ValidationConfig,
    pub training: Vec<SequenceConfig>,
}

#[derive(Clone, Deserialize, Serialize)]
pub struct AssetConfig {
    pub object: String,
    pub entrypoint: Option<String>,
}

#[derive(Clone, Deserialize, Serialize)]
pub struct ValidationConfig {
    pub samples: i64,
    pub max_seconds: f32,
}

#[derive(Clone, Deserialize, Serialize)]
pub struct SequenceConfig {
    pub batches: u64,
    pub max_seconds: f32,
}

impl DataConfig {
    pub fn training_max_seconds(&self) -> f32 {
        self.training
            .iter()
            .map(|s| s.max_seconds)
            .fold(0.0, f32::max)
    }
}

enum Batches {
    Prefetched(Prefetcher),
    OnDemand(Pin<Box<dyn Stream<Item = anyhow::Result<LoadedBatch>> + Send>>),
}

fn on_demand_batches(mut sampler: Box<dyn Sampler>, loader: Arc<dyn Loader>) -> Batches {
    Batches::OnDemand(Box::pin(async_stream::stream! {
        loop {
            match sampler.next_batch() {
                Ok(Some(batch)) => {
                    let batch = loader
                        .load_batch(batch)
                        .await
                        .map(|batch| batch.into_iter().map(LoadedSample::from).collect());
                    let failed = batch.is_err();
                    yield batch;
                    if failed {
                        break;
                    }
                }
                Ok(None) => break,
                Err(err) => {
                    yield Err(err);
                    break;
                }
            }
        }
    }))
}

pub struct RunState {
    pub id: Uuid,
    cancel_token: CancellationToken,
    validation_batches: Batches,
    training_batches: Batches,
}

impl RunState {
    pub async fn new(
        id: Uuid,
        database: &clickhouse::Client,
        loader: Arc<dyn Loader>,
        cache_dir: &'static Path,
        config: &DataConfig,
        synthetic: bool,
    ) -> anyhow::Result<Self> {
        info!(run = %id, dataset = %config.dataset_id, synthetic, "initializing run");

        let (validation_rows, training_rows) = if synthetic {
            let language = config
                .plbert_languages
                .first()
                .map(String::as_str)
                .unwrap_or("en");
            let validation_rows = synthetic_rows(
                config.validation.max_seconds as f64,
                config.seed,
                language,
                config.validation.samples as usize,
                VALIDATION_SEED_SALT,
            );
            let training_rows = synthetic_rows(
                config.training_max_seconds() as f64,
                config.seed,
                language,
                SYNTHETIC_TRAINING_SAMPLES,
                TRAINING_SEED_SALT,
            );
            info!(
                validation = validation_rows.len(),
                training = training_rows.len(),
                "generated synthetic rows"
            );
            (validation_rows, training_rows)
        } else {
            let validation_rows = fetch_validation_samples(database, config).await?;
            info!(
                rows = validation_rows.len(),
                requested = config.validation.samples,
                "fetched validation rows"
            );

            let validation_ids: Vec<Uuid> = validation_rows.iter().map(|r| r.audio_id).collect();

            let training_rows = fetch_training_samples(database, &validation_ids, config).await?;
            info!(rows = training_rows.len(), "fetched training rows");
            (validation_rows, training_rows)
        };

        let validation_bins = bins_from_rows(validation_rows, &config.plbert_languages)?;
        let training_bins = bins_from_rows(training_rows, &config.plbert_languages)?;

        // validation is one endlessly-looping set, so the plain histogram
        // sampler serves it; training follows the batch schedule
        let validation_sampler: Box<dyn Sampler> = Box::new(HistogramSampler::new(
            validation_bins,
            config.validation.max_seconds as f64,
            config.seed,
        ));
        let training_sampler: Box<dyn Sampler> = Box::new(ScheduledSampler::new(
            training_bins,
            &config.training,
            config.seed,
        ));

        let cancel_token = CancellationToken::new();
        let (training_batches, validation_batches) = if synthetic {
            (
                on_demand_batches(training_sampler, loader.clone()),
                on_demand_batches(validation_sampler, loader),
            )
        } else {
            (
                Batches::Prefetched(Prefetcher::spawn(
                    training_sampler,
                    loader.clone(),
                    cache_dir,
                    cancel_token.clone(),
                    info_span!("prefetcher", run = %id, split = "training"),
                )),
                Batches::Prefetched(Prefetcher::spawn(
                    validation_sampler,
                    loader,
                    cache_dir,
                    cancel_token.clone(),
                    info_span!("prefetcher", run = %id, split = "validation"),
                )),
            )
        };

        Ok(RunState {
            id,
            cancel_token,
            training_batches,
            validation_batches,
        })
    }

    pub async fn next_batch(&mut self, validation: bool) -> anyhow::Result<Option<LoadedBatch>> {
        let batches = if validation {
            &mut self.validation_batches
        } else {
            &mut self.training_batches
        };
        match batches {
            Batches::Prefetched(prefetcher) => prefetcher.next_batch().await,
            Batches::OnDemand(stream) => match stream.next().await {
                Some(batch) => batch.map(Some),
                None => Ok(None),
            },
        }
    }

    pub async fn finish(self) {
        self.cancel_token.cancel();
        let prefetchers = match (self.validation_batches, self.training_batches) {
            (Batches::Prefetched(p1), Batches::Prefetched(p2)) => vec![p1, p2],
            (Batches::Prefetched(p), _) | (_, Batches::Prefetched(p)) => vec![p],
            _ => return,
        };
        futures::future::join_all(prefetchers.into_iter().map(Prefetcher::drain)).await;
    }
}

enum Command {
    NextBatch {
        validation: bool,
        reply: oneshot::Sender<anyhow::Result<Option<LoadedBatch>>>,
    },
    Finish {
        reply: oneshot::Sender<()>,
    },
}

#[derive(Clone)]
pub struct RunHandle {
    tx: mpsc::Sender<Command>,
}

impl RunHandle {
    pub fn spawn(run: RunState) -> Self {
        let id = run.id;
        let (tx, rx) = mpsc::channel(1);
        tokio::spawn(serve_commands(run, rx).instrument(info_span!("run", run = %id)));
        Self { tx }
    }

    pub async fn next_batch(&self, validation: bool) -> anyhow::Result<Option<LoadedBatch>> {
        let (reply, response) = oneshot::channel();
        if self
            .tx
            .send(Command::NextBatch { validation, reply })
            .await
            .is_err()
        {
            bail!("run ended");
        }
        match response.await {
            Ok(batch) => batch,
            Err(_) => bail!("run ended"),
        }
    }

    pub async fn finish(self) {
        let (reply, response) = oneshot::channel();
        if self.tx.send(Command::Finish { reply }).await.is_ok() {
            // wait for the drain to complete; an error means the actor is already gone
            let _ = response.await;
        }
    }
}

async fn serve_commands(mut run: RunState, mut rx: mpsc::Receiver<Command>) {
    while let Some(cmd) = rx.recv().await {
        match cmd {
            Command::NextBatch { validation, reply } => {
                let _ = reply.send(run.next_batch(validation).await);
            }
            Command::Finish { reply } => {
                run.finish().await;
                let _ = reply.send(());
                debug!("run finished");
                return;
            }
        }
    }

    // all handles dropped without an End: still stop prefetchers and drain the cache
    run.finish().await;
    debug!("run finished after handles dropped");
}
