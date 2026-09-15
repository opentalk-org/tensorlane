use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use tokio::{
    fs,
    io::AsyncWriteExt,
    sync::{
        RwLock,
        mpsc::{self, Sender},
        oneshot,
    },
    task::JoinHandle,
};
use tokio_util::sync::CancellationToken;
use tracing::{Instrument, debug, error, info, info_span};
use uuid::Uuid;

use crate::{
    db::{fetch_training_samples, fetch_validation_samples},
    loader::Loader,
    prefetch::{LoadedBatch, Prefetcher},
    run_repo::{RunRepo, RunStatus},
    sampling::{HistogramSampler, Sampler, ScheduledSampler, bins_from_rows},
};

#[derive(Clone, Deserialize, Serialize)]
pub struct DataConfig {
    pub dataset_id: Uuid,
    pub asset_type: String,
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

pub struct RunState {
    pub id: Uuid,
    cancel_token: CancellationToken,
    validation_batches: Prefetcher,
    training_batches: Prefetcher,
    run_cache_dir: PathBuf,
}

struct BatchRequest {
    validation: bool,
    reply: oneshot::Sender<Result<Option<LoadedBatch>>>,
}

impl RunState {
    pub async fn new(
        id: Uuid,
        database: &clickhouse::Client,
        loader: Arc<dyn Loader>,
        cache_dir: &Path,
        config: &DataConfig,
    ) -> Result<Self> {
        info!(run = %id, dataset = %config.dataset_id, "initializing run");

        let validation_rows = fetch_validation_samples(database, config).await?;
        info!(
            rows = validation_rows.len(),
            requested = config.validation.samples,
            "fetched validation rows"
        );

        let validation_ids: Vec<Uuid> = validation_rows.iter().map(|r| r.audio_id).collect();

        let training_rows = fetch_training_samples(database, &validation_ids, config).await?;
        info!(rows = training_rows.len(), "fetched training rows");

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

        let cache_dir = cache_dir.join(id.to_string());
        let training_dir = cache_dir.join("data/training");
        let validation_dir = cache_dir.join("data/validation");
        fs::create_dir_all(&training_dir).await?;
        fs::create_dir_all(&validation_dir).await?;
        let cancel_token = CancellationToken::new();
        let training_batches = Prefetcher::spawn(
            training_sampler,
            loader.clone(),
            training_dir,
            cancel_token.clone(),
            info_span!("prefetcher", run = %id, split = "training"),
        );
        let validation_batches = Prefetcher::spawn(
            validation_sampler,
            loader,
            validation_dir,
            cancel_token.clone(),
            info_span!("prefetcher", run = %id, split = "validation"),
        );

        Ok(RunState {
            id,
            cancel_token,
            training_batches,
            validation_batches,
            run_cache_dir: cache_dir,
        })
    }

    pub async fn next_batch(&mut self, validation: bool) -> Result<Option<LoadedBatch>> {
        let batches = if validation {
            &mut self.validation_batches
        } else {
            &mut self.training_batches
        };
        batches.next_batch().await
    }

    async fn handle_requests(mut self, mut rx: mpsc::Receiver<BatchRequest>) {
        loop {
            let BatchRequest { validation, reply } = tokio::select! {
                biased;
                () = self.cancel_token.cancelled() => break,
                req = rx.recv() => match req {
                    Some(req) => req,
                    None => break,
                },
            };
            let cancel_token = self.cancel_token.clone();
            let batch = tokio::select! {
                biased;
                () = cancel_token.cancelled() => break,
                batch = self.next_batch(validation) => batch,
            };
            let _ = reply.send(batch);
        }

        tokio::join!(
            self.validation_batches.finish(),
            self.training_batches.finish(),
        );
        if let Err(err) = fs::remove_dir_all(&self.run_cache_dir).await
            && err.kind() != std::io::ErrorKind::NotFound
        {
            error!(run = %self.id, error = %err, path = %self.run_cache_dir.display(), "removing run cache failed");
        }
        debug!("run finished");
    }
}

#[derive(Clone)]
pub struct RunExecutor {
    runs: Arc<RwLock<HashMap<Uuid, Run>>>,
    repo: RunRepo,
    data_source: clickhouse::Client,
    loader: Arc<dyn Loader>,
    cache_dir: &'static Path,
    assets: AssetStore,
}

pub struct Run {
    tx: Sender<BatchRequest>,
    cancel: CancellationToken,
    handle: JoinHandle<()>,
    config: DataConfig,
}

pub struct RunInitialization {
    pub train_config: String,
    pub assets: Vec<String>,
}

impl RunExecutor {
    pub fn new(
        repo: RunRepo,
        data_source: clickhouse::Client,
        loader: Arc<dyn Loader>,
        root_cache_dir: &'static Path,
        s3_client: aws_sdk_s3::Client,
        bucket: &'static str,
    ) -> Self {
        Self {
            runs: Default::default(),
            repo,
            data_source,
            cache_dir: root_cache_dir,
            loader,
            assets: AssetStore {
                s3_client,
                bucket,
                root: root_cache_dir,
            },
        }
    }

    /// Starts a new run and returns its train config.
    pub async fn start(&self, id: Uuid) -> Result<RunInitialization> {
        if self.runs.read().await.contains_key(&id) {
            bail!("run is already active");
        }
        let run_record = self
            .repo
            .get(id)
            .await?
            .ok_or_else(|| anyhow!("run not found"))?;
        let train_config = serde_json::to_string(&run_record.train_config)?;
        let run_span = info_span!("run", run = %id);

        let state = RunState::new(
            id,
            &self.data_source,
            self.loader.clone(),
            self.cache_dir,
            &run_record.data_config,
        )
        .instrument(run_span)
        .await?;

        let res = self.repo.append_status(id, RunStatus::Running).await;
        if res.is_err() {
            if let Err(err) = fs::remove_dir_all(&state.run_cache_dir).await
                && err.kind() != std::io::ErrorKind::NotFound
            {
                error!(run = %id, error = %err, path = %state.run_cache_dir.display(), "removing run cache failed");
            }
            res?;
        }

        futures::future::join_all(
            run_record
                .data_config
                .assets
                .iter()
                .map(|(name, asset)| self.assets.ensure(id, name, &asset.object)),
        )
        .await;

        let cancel = state.cancel_token.clone();

        let (tx, rx) = mpsc::channel(1);
        let handle = tokio::spawn(state.handle_requests(rx));

        let mut runs = self.runs.write().await;
        runs.insert(
            id,
            Run {
                tx,
                cancel,
                handle,
                config: run_record.data_config.clone(),
            },
        );

        Ok(RunInitialization {
            train_config,
            assets: run_record.data_config.assets.keys().cloned().collect(),
        })
    }

    pub async fn next_batch(&self, id: Uuid, validation: bool) -> Result<Option<LoadedBatch>> {
        let runs = self.runs.read().await;
        let run = runs.get(&id).ok_or_else(|| anyhow!("unknown run"))?;

        let (tx, rx) = oneshot::channel();
        run.tx
            .send(BatchRequest {
                validation,
                reply: tx,
            })
            .await?;

        rx.await?
    }

    pub async fn finish(&self, id: Uuid) -> Result<()> {
        let mut runs = self.runs.write().await;
        let Some(run) = runs.remove(&id) else {
            bail!("unknown run");
        };
        drop(runs);

        run.cancel.cancel();
        run.handle.await?;

        self.repo.append_status(id, RunStatus::Succeeded).await?;

        Ok(())
    }

    pub async fn asset(&self, id: Uuid, name: &str) -> Result<(PathBuf, Option<String>)> {
        let runs = self.runs.read().await;

        let Some(run) = runs.get(&id) else {
            bail!("unknown run");
        };
        let Some(asset) = run.config.assets.get(name) else {
            bail!("unknown asset");
        };

        self.assets.ensure(id, name, &asset.object).await?;
        let path = self.assets.path(id, name);
        let entrypoint = asset.entrypoint.clone();

        Ok((path, entrypoint))
    }

    pub async fn asset_type(&self, id: Uuid) -> Result<String> {
        let runs = self.runs.read().await;
        let Some(run) = runs.get(&id) else {
            bail!("unknown run");
        };

        return Ok(run.config.asset_type.clone());
    }

    pub async fn is_running(&self, id: Uuid) -> bool {
        let runs = self.runs.read().await;
        return runs.contains_key(&id);
    }

    /// Awaits and removes all of the currently active runs.
    pub async fn drain_and_wait(&self) -> Result<()> {
        let mut runs = self.runs.write().await;

        info!(runs = runs.len(), "waiting for active runs before shutdown");

        for (_, run) in runs.drain() {
            run.handle.await?;
        }

        Ok(())
    }
}

#[derive(Clone)]
struct AssetStore {
    s3_client: aws_sdk_s3::Client,
    bucket: &'static str,
    root: &'static Path,
}

impl AssetStore {
    pub fn path(&self, run_id: Uuid, name: &str) -> PathBuf {
        self.root.join(run_id.to_string()).join("assets").join(name)
    }

    pub async fn ensure(&self, run_id: Uuid, name: &str, key: &str) -> anyhow::Result<PathBuf> {
        let run_dir = self.root.join(run_id.to_string()).join("assets");
        fs::create_dir_all(&run_dir).await?;
        let path = run_dir.join(name);
        if fs::try_exists(&path).await? {
            return Ok(path);
        }

        let part = run_dir.join(format!("{name}.part"));
        info!(run = %run_id, asset = name, key, "downloading asset");
        let mut object = self
            .s3_client
            .get_object()
            .bucket(self.bucket)
            .key(key)
            .send()
            .await
            .with_context(|| format!("fetching asset {name} from {key}"))?;
        let mut file = fs::File::create(&part).await?;
        while let Some(bytes) = object.body.try_next().await? {
            file.write_all(&bytes).await?;
        }
        file.sync_all().await?;
        fs::rename(&part, &path).await?;
        Ok(path)
    }
}
