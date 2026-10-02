use crate::{
    asset_repo::{AssetRecord, AssetRepo, kind_name, kind_value},
    loader::Loader,
    prefetch::LoadedBatch,
    proto::{AssetMetadata, SaveAssetMetadata},
    run_repo::{RunRepo, RunStatus},
    uploads::UploadStore,
};
use anyhow::{Context, Result, ensure};
use futures::{StreamExt, TryStreamExt, stream};
use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};
use tokio::{
    fs,
    io::AsyncWriteExt,
    sync::{Mutex, RwLock, mpsc, oneshot},
    task::JoinHandle,
};
use tokio_util::{
    sync::CancellationToken,
    task::{TaskTracker, task_tracker::TaskTrackerToken},
};
use uuid::Uuid;
mod config;
mod state;
pub use config::Config;
use state::{BatchRequest, RunState};

type AssetHead = Arc<Mutex<Option<AssetRecord>>>;
#[derive(Clone)]
pub struct RunExecutor {
    runs: Arc<RwLock<HashMap<Uuid, Run>>>,
    starting: Arc<Mutex<HashSet<Uuid>>>,
    save_ids: Arc<Mutex<HashMap<Uuid, std::sync::Weak<Mutex<()>>>>>,
    lifecycle: Arc<Mutex<TaskTracker>>,
    repo: RunRepo,
    database: clickhouse::Client,
    loader: Arc<dyn Loader>,
    cache: &'static Path,
    s3: aws_sdk_s3::Client,
    bucket: &'static str,
    asset_repo: AssetRepo,
}
struct Run {
    senders: HashMap<String, mpsc::Sender<BatchRequest>>,
    cancel: CancellationToken,
    handle: JoinHandle<()>,
    inputs: HashMap<String, (PathBuf, AssetMetadata)>,
    heads: Arc<Mutex<HashMap<String, AssetHead>>>,
    asset_type: Option<String>,
    saves: TaskTracker,
    last_seen: tokio::time::Instant,
    failed: Arc<AtomicBool>,
    _lifetime: TaskTrackerToken,
}
pub struct RunInitialization {
    pub config: String,
    pub assets: Vec<String>,
    pub streams: Vec<String>,
}
pub struct SaveContext {
    pub run_id: Uuid,
    pub failed: Arc<AtomicBool>,
    head: AssetHead,
    asset_type: Option<String>,
    _lifetime: TaskTrackerToken,
}
impl RunExecutor {
    pub fn new(
        repo: RunRepo,
        database: clickhouse::Client,
        loader: Arc<dyn Loader>,
        cache: &'static Path,
        s3: aws_sdk_s3::Client,
        bucket: &'static str,
    ) -> Self {
        Self {
            runs: Default::default(),
            starting: Default::default(),
            save_ids: Default::default(),
            lifecycle: Default::default(),
            repo,
            asset_repo: AssetRepo::new(database.clone()),
            database,
            loader,
            cache,
            s3,
            bucket,
        }
    }
    pub async fn start(&self, id: Uuid) -> Result<RunInitialization> {
        let (lifetime, _initializing) = {
            let tracker = self.lifecycle.lock().await;
            ensure!(
                !tracker.is_closed(),
                "server is shutting down; new runs are not accepted"
            );
            (tracker.token(), tracker.token())
        };
        {
            let mut starting = self.starting.lock().await;
            ensure!(
                !starting.contains(&id) && !self.runs.read().await.contains_key(&id),
                "run is already active"
            );
            starting.insert(id);
        }
        let result = self.initialize(id, lifetime).await;
        self.starting.lock().await.remove(&id);
        if result.is_err() {
            let _ = fs::remove_dir_all(self.cache.join(id.to_string())).await;
            let _ = self.repo.append_status(id, RunStatus::Failed).await;
        }
        result
    }
    async fn initialize(&self, id: Uuid, lifetime: TaskTrackerToken) -> Result<RunInitialization> {
        let record = self.repo.get(id).await?.context("run not found")?;
        let config = Config::parse(&record.config)?;
        let mut downloads = Vec::new();
        let mut heads = HashMap::new();
        self.repo.append_status(id, RunStatus::Running).await?;
        for (index, (name, input)) in config.assets.iter().enumerate() {
            let registered = if let Some(id) = input.asset_id {
                Some(
                    self.asset_repo
                        .get(id)
                        .await?
                        .context("input asset not found or deleted")?,
                )
            } else {
                None
            };
            let object = registered
                .as_ref()
                .map(|record| record.path.as_str())
                .or(input.object.as_deref())
                .unwrap()
                .to_owned();
            let metadata = AssetMetadata {
                entrypoint: input.entrypoint.clone(),
                asset_id: registered.as_ref().map(|asset| asset.id.to_string()),
                metadata_json: registered
                    .as_ref()
                    .map(|asset| asset.metadata.clone())
                    .unwrap_or_else(|| "{}".into()),
                kind: registered
                    .as_ref()
                    .map(|asset| kind_name(asset.kind))
                    .transpose()?
                    .unwrap_or("file")
                    .into(),
                asset_type: registered
                    .as_ref()
                    .map(|asset| asset.asset_type.clone())
                    .unwrap_or_else(|| "generic".into()),
            };
            let directory = self
                .cache
                .join(id.to_string())
                .join("assets")
                .join(index.to_string());
            downloads.push(async move {
                fs::create_dir_all(&directory).await?;
                let path = directory.join("data");
                let part = directory.join("download.part");
                let mut body = self
                    .s3
                    .get_object()
                    .bucket(self.bucket)
                    .key(object)
                    .send()
                    .await?
                    .body;
                let mut file = fs::File::create(&part).await?;
                while let Some(bytes) = body.try_next().await? {
                    file.write_all(&bytes).await?;
                }
                file.sync_all().await?;
                drop(file);
                fs::rename(part, &path).await?;
                anyhow::Ok((name.clone(), (path, metadata)))
            });
            heads.insert(name.clone(), Arc::new(Mutex::new(registered)));
        }
        let inputs = stream::iter(downloads)
            .buffer_unordered(4)
            .try_collect::<Vec<_>>()
            .await?
            .into_iter()
            .collect();
        for asset in self.asset_repo.for_run(id, None).await? {
            heads.insert(asset.name.clone(), Arc::new(Mutex::new(Some(asset))));
        }
        let state =
            RunState::new(id, &self.database, self.loader.clone(), self.cache, &config).await?;
        let cancel = state.cancel.clone();
        let (senders, handle) = state.start();
        let streams = config.queries.keys().cloned().collect();
        let assets = config.assets.keys().cloned().collect();
        self.runs.write().await.insert(
            id,
            Run {
                senders,
                cancel,
                handle,
                inputs,
                heads: Arc::new(Mutex::new(heads)),
                asset_type: config.asset_type,
                saves: TaskTracker::new(),
                last_seen: tokio::time::Instant::now(),
                failed: Arc::new(AtomicBool::new(false)),
                _lifetime: lifetime,
            },
        );
        let executor = self.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(10)).await;
                let expired = {
                    let mut runs = executor.runs.write().await;
                    let Some(run) = runs.get(&id) else { return };
                    if run.last_seen.elapsed() < Duration::from_secs(60) {
                        continue;
                    }
                    runs.remove(&id).unwrap()
                };
                if let Err(error) = executor.complete(id, expired, RunStatus::Failed).await {
                    tracing::error!(%id, %error, "expiring disconnected run failed");
                }
                return;
            }
        });
        Ok(RunInitialization {
            config: serde_json::to_string(&record.config)?,
            assets,
            streams,
        })
    }
    pub async fn next_batch(&self, id: Uuid, stream: &str) -> Result<Option<LoadedBatch>> {
        let sender = self
            .runs
            .read()
            .await
            .get(&id)
            .context("unknown run")?
            .senders
            .get(stream)
            .context("unknown stream")?
            .clone();
        let (reply, result) = oneshot::channel();
        let outcome = async {
            sender.send(BatchRequest { reply }).await?;
            result.await?
        }
        .await;
        if outcome.is_err() {
            self.mark_failed(id).await;
        }
        outcome
    }
    pub async fn heartbeat(&self, id: Uuid) -> Result<()> {
        let mut runs = self.runs.write().await;
        runs.get_mut(&id).context("unknown run")?.last_seen = tokio::time::Instant::now();
        Ok(())
    }
    pub async fn mark_failed(&self, id: Uuid) {
        if let Some(run) = self.runs.read().await.get(&id) {
            run.failed.store(true, Ordering::Release);
        }
    }
    pub async fn finish(&self, id: Uuid, status: RunStatus) -> Result<()> {
        let run = self.runs.write().await.remove(&id);
        match run {
            Some(run) => {
                let executor = self.clone();
                tokio::spawn(async move { executor.complete(id, run, status).await })
                    .await
                    .context("run completion task panicked")?
            }
            None => {
                let record = self.repo.get(id).await?.context("unknown run")?;
                ensure!(
                    matches!(
                        record.status,
                        Some(RunStatus::Succeeded | RunStatus::Failed | RunStatus::Cancelled)
                    ),
                    "run is not active"
                );
                Ok(())
            }
        }
    }
    async fn complete(&self, id: Uuid, run: Run, status: RunStatus) -> Result<()> {
        run.saves.close();
        run.cancel.cancel();
        let joined = run.handle.await;
        run.saves.wait().await;
        let status = if joined.is_err() || run.failed.load(Ordering::Acquire) {
            RunStatus::Failed
        } else {
            status
        };
        self.repo.append_status(id, status).await?;
        joined?;
        Ok(())
    }
    pub async fn asset(&self, id: Uuid, name: &str) -> Result<(PathBuf, AssetMetadata)> {
        Ok(self
            .runs
            .read()
            .await
            .get(&id)
            .context("unknown run")?
            .inputs
            .get(name)
            .context("unknown asset")?
            .clone())
    }
    pub async fn admit_save(&self, run_id: Uuid, name: &str) -> Result<SaveContext> {
        ensure!(!name.is_empty(), "asset name must not be empty");
        let runs = self.runs.read().await;
        let run = runs.get(&run_id).context("unknown run")?;
        let head = run
            .heads
            .lock()
            .await
            .entry(name.to_owned())
            .or_insert_with(|| Arc::new(Mutex::new(None)))
            .clone();
        Ok(SaveContext {
            run_id,
            failed: run.failed.clone(),
            head,
            asset_type: run.asset_type.clone(),
            _lifetime: run.saves.token(),
        })
    }
    pub async fn save_asset(
        &self,
        context: SaveContext,
        metadata: SaveAssetMetadata,
        path: &Path,
        size: u64,
        hash: [u8; 64],
        uploads: &UploadStore,
    ) -> Result<Uuid> {
        let outcome = self
            .commit_asset(&context, metadata, path, size, hash, uploads)
            .await;
        if outcome.is_err() {
            context.failed.store(true, Ordering::Release);
        }
        outcome
    }
    async fn commit_asset(
        &self,
        context: &SaveContext,
        metadata: SaveAssetMetadata,
        path: &Path,
        size: u64,
        hash: [u8; 64],
        uploads: &UploadStore,
    ) -> Result<Uuid> {
        let id: Uuid = metadata.asset_id.parse().context("invalid asset ID")?;
        ensure!(!id.is_nil(), "asset ID must not be nil");
        let kind = kind_value(&metadata.kind)?;
        let _: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(&metadata.metadata_json)
                .context("asset metadata must be a JSON object")?;
        let id_lock = {
            let mut locks = self.save_ids.lock().await;
            locks.retain(|_, lock| lock.strong_count() > 0);
            match locks.get(&id).and_then(std::sync::Weak::upgrade) {
                Some(lock) => lock,
                None => {
                    let lock = Arc::new(Mutex::new(()));
                    locks.insert(id, Arc::downgrade(&lock));
                    lock
                }
            }
        };
        let _id_guard = id_lock.lock().await;
        let mut head = context.head.lock().await;
        if let Some(existing) = self.asset_repo.get(id).await? {
            ensure!(
                existing.run_id == context.run_id
                    && existing.name == metadata.name
                    && existing.content_hash == hash
                    && existing.kind == kind
                    && existing.step == metadata.step
                    && existing.metadata == metadata.metadata_json
                    && metadata
                        .asset_type
                        .as_ref()
                        .is_none_or(|value| value == &existing.asset_type),
                "conflicting retry of asset ID"
            );
            return Ok(id);
        }
        let now = time::OffsetDateTime::now_utc();
        let mut updated_at = now.replace_nanosecond(now.nanosecond() / 1000 * 1000)?;
        if let Some(parent) = head.as_ref()
            && updated_at <= parent.updated_at
        {
            updated_at = parent.updated_at + time::Duration::microseconds(1);
        }
        let record = AssetRecord {
            id,
            updated_at,
            kind,
            name: metadata.name,
            step: metadata.step,
            path: uploads.asset_key(id),
            size,
            content_hash: hash,
            asset_type: metadata
                .asset_type
                .or_else(|| context.asset_type.clone())
                .or_else(|| head.as_ref().map(|asset| asset.asset_type.clone()))
                .unwrap_or_else(|| "generic".into()),
            metadata: metadata.metadata_json,
            run_id: context.run_id,
            ancestor_asset_id: head.as_ref().map(|asset| asset.id).unwrap_or(Uuid::nil()),
            deleted: false,
        };
        let content_type = if metadata.content_type.is_empty() {
            "application/x-tar"
        } else {
            &metadata.content_type
        };
        if let Err(error) = uploads.save_asset(&record, path, content_type).await {
            let committed = self.asset_repo.get(id).await?;
            if !committed.as_ref().is_some_and(|asset| {
                asset.content_hash == hash && asset.ancestor_asset_id == record.ancestor_asset_id
            }) {
                return Err(error);
            }
        }
        *head = Some(record);
        Ok(id)
    }
    pub async fn is_running(&self, id: Uuid) -> bool {
        self.runs.read().await.contains_key(&id)
    }
    pub async fn shutdown(&self) {
        let tracker = {
            let tracker = self.lifecycle.lock().await;
            tracker.close();
            tracker.clone()
        };
        tracker.wait().await;
    }
}
