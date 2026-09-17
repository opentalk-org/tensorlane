use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{Context, Result, anyhow, bail};
use tokio::{
    fs,
    io::AsyncWriteExt,
    sync::{
        Mutex, RwLock,
        mpsc::{self, Sender},
        oneshot,
    },
    task::JoinHandle,
};
use tokio_util::{
    sync::CancellationToken,
    task::{TaskTracker, task_tracker::TaskTrackerToken},
};
use tracing::{Instrument, info, info_span};
use uuid::Uuid;

use crate::{
    loader::Loader,
    prefetch::LoadedBatch,
    run_repo::{RunRepo, RunStatus},
};

mod config;
mod state;
pub use config::DataConfig;
use state::{BatchRequest, RunState};

#[derive(Clone)]
pub struct RunExecutor {
    runs: Arc<RwLock<HashMap<Uuid, Run>>>,
    lifecycle: Arc<Mutex<TaskTracker>>,
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
    _lifetime: TaskTrackerToken,
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
            lifecycle: Default::default(),
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
        let lifetime = self.admit().await?;
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

        let setup: Result<()> = async {
            self.repo.append_status(id, RunStatus::Running).await?;
            let assets = futures::future::join_all(
                run_record
                    .data_config
                    .assets
                    .iter()
                    .map(|(name, asset)| self.assets.ensure(id, name, &asset.object)),
            )
            .await;
            for asset in assets {
                asset?;
            }
            Ok(())
        }
        .await;

        if let Err(err) = setup {
            state.finish().await;
            return Err(err);
        }

        let cancel = state.cancel_token.clone();

        let (tx, rx) = mpsc::channel(1);
        let actor_lifetime = lifetime.clone();
        let handle = tokio::spawn(async move {
            let _lifetime = actor_lifetime;
            state.handle_requests(rx).await;
        });

        let mut runs = self.runs.write().await;
        runs.insert(
            id,
            Run {
                tx,
                cancel,
                handle,
                config: run_record.data_config.clone(),
                _lifetime: lifetime,
            },
        );

        Ok(RunInitialization {
            train_config,
            assets: run_record.data_config.assets.keys().cloned().collect(),
        })
    }

    pub async fn next_batch(&self, id: Uuid, validation: bool) -> Result<Option<LoadedBatch>> {
        let sender = {
            let runs = self.runs.read().await;
            runs.get(&id)
                .ok_or_else(|| anyhow!("unknown run"))?
                .tx
                .clone()
        };

        let (tx, rx) = oneshot::channel();
        sender
            .send(BatchRequest {
                validation,
                reply: tx,
            })
            .await?;

        rx.await?
    }

    /// Finish a single run.
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

    async fn admit(&self) -> Result<TaskTrackerToken> {
        let lifecycle = self.lifecycle.lock().await;
        if lifecycle.is_closed() {
            bail!("server is shutting down; new runs are not accepted");
        }
        Ok(lifecycle.token())
    }

    pub async fn shutdown(&self) {
        let lifecycle = {
            let lifecycle = self.lifecycle.lock().await;
            lifecycle.close();
            lifecycle.clone()
        };
        info!(
            pending = lifecycle.len(),
            "waiting for active runs before shutdown"
        );
        lifecycle.wait().await;
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
