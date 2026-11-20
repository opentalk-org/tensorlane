use anyhow::{Context, Result, anyhow, ensure};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{Arc, Mutex, Weak},
    time::{Duration, Instant},
};
use tensorlane_protocol::{AssetDownload, AssetMetadata, Checkpoint, InitResponse};
use tokio::{fs, sync::Semaphore};
use tokio_util::{sync::CancellationToken, task::TaskTracker};
use uuid::Uuid;

use crate::{
    asset_repo::kind_name,
    loader::{Loader, S3Loader},
    run_config::Config,
    run_repo::{RunRepo, RunStatus},
    shared_cache::{Lock, write_atomic},
};

pub(super) const LOAD_MEMORY_UNIT: usize = 1024;
type LoadingPools = HashMap<(Uuid, String), Weak<Semaphore>>;
/// How long an active run's config is served from memory before ClickHouse is
/// asked again: a cancellation written elsewhere takes effect within this time.
const ACTIVE_TTL: Duration = Duration::from_secs(30);

#[derive(Clone)]
pub struct Runtime {
    pub repo: RunRepo,
    pub database: clickhouse::Client,
    pub s3: aws_sdk_s3::Client,
    pub bucket: &'static str,
    pub cache: Arc<PathBuf>,
    pub budget: Arc<crate::cache_limits::Budget>,
    pub shutdown: CancellationToken,
    pub tasks: TaskTracker,
    pub(super) checkpoint_prefix: &'static str,
    pub(super) metrics_prefix: &'static str,
    pub asset_slots: Arc<Semaphore>,
    pub upload_slots: Arc<Semaphore>,
    pub(super) loader: Arc<dyn Loader>,
    load_memory: Arc<Mutex<LoadingPools>>,
    active_runs: Arc<Mutex<HashMap<Uuid, (Instant, Config)>>>,
}

pub enum Batch {
    Pending,
    End,
    Ready(PathBuf),
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct AssetSource {
    pub object: String,
    pub download: AssetDownload,
}

impl Runtime {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        database: clickhouse::Client,
        s3: aws_sdk_s3::Client,
        bucket: &'static str,
        cache: PathBuf,
        budget: Arc<crate::cache_limits::Budget>,
        shutdown: CancellationToken,
        checkpoint_prefix: &'static str,
        metrics_prefix: &'static str,
    ) -> Result<Self> {
        let checkpoint_prefix = checkpoint_prefix.trim_matches('/');
        let metrics_prefix = metrics_prefix.trim_matches('/');
        ensure!(!checkpoint_prefix.is_empty(), "checkpoint prefix is empty");
        ensure!(!metrics_prefix.is_empty(), "metrics prefix is empty");
        Ok(Self {
            loader: Arc::new(S3Loader::new(s3.clone(), bucket)),
            repo: RunRepo::new(database.clone()),
            database,
            s3,
            bucket,
            cache: Arc::new(cache),
            budget,
            shutdown,
            tasks: TaskTracker::new(),
            checkpoint_prefix,
            metrics_prefix,
            asset_slots: Arc::new(Semaphore::new(128)),
            upload_slots: Arc::new(Semaphore::new(2)),
            load_memory: Arc::new(Mutex::new(HashMap::new())),
            active_runs: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    pub fn run_dir(&self, run: Uuid) -> PathBuf {
        self.cache.join("runs").join(run.to_string())
    }

    pub fn asset_key(&self, id: Uuid) -> String {
        format!("{}/{id}", self.checkpoint_prefix)
    }

    pub(super) fn loading_memory(
        &self,
        run: Uuid,
        stream: &str,
        bytes: usize,
    ) -> Result<Arc<Semaphore>> {
        let mut streams = self
            .load_memory
            .lock()
            .map_err(|_| anyhow!("loading memory lock poisoned"))?;
        streams.retain(|_, budget| budget.strong_count() > 0);
        let key = (run, stream.to_owned());
        if let Some(budget) = streams.get(&key).and_then(Weak::upgrade) {
            return Ok(budget);
        }
        let budget = Arc::new(Semaphore::new(bytes.div_ceil(LOAD_MEMORY_UNIT)));
        streams.insert(key, Arc::downgrade(&budget));
        Ok(budget)
    }

    pub async fn initialize(&self, id: Uuid) -> Result<InitResponse> {
        ensure!(!self.shutdown.is_cancelled(), "server is shutting down");
        let record = self.repo.get(id).await?.context("run not found")?;
        let config = Config::parse(&record.config)?;
        let checkpoint = self
            .repo
            .run_assets(id, None)
            .await?
            .into_iter()
            .rfind(|asset| asset.kind == 1)
            .map(|asset| Checkpoint {
                name: asset.name,
                asset_id: asset.id.to_string(),
            });
        let mut assets: Vec<_> = config.assets.keys().cloned().collect();
        if let Some(checkpoint) = &checkpoint
            && !assets.contains(&checkpoint.name)
        {
            assets.push(checkpoint.name.clone());
        }
        if record.status != Some(RunStatus::Running) {
            self.repo.append_status(id, RunStatus::Running).await?;
        }
        Ok(InitResponse {
            run_id: id.to_string(),
            config: serde_json::to_string(&record.config)?,
            assets,
            streams: config.queries.into_keys().collect(),
            checkpoint,
        })
    }

    pub async fn active(&self, id: Uuid) -> Result<Config> {
        if let Some((checked, config)) = self.active_runs.lock().unwrap().get(&id)
            && checked.elapsed() < ACTIVE_TTL
        {
            return Ok(config.clone());
        }
        let record = self.repo.get(id).await?.context("run not found")?;
        ensure!(
            record.status != Some(RunStatus::Cancelled),
            "run is cancelled"
        );
        let config = Config::parse(&record.config)?;
        let mut active_runs = self.active_runs.lock().unwrap();
        active_runs.retain(|_, (checked, _)| checked.elapsed() < ACTIVE_TTL);
        active_runs.insert(id, (Instant::now(), config.clone()));
        Ok(config)
    }

    pub async fn end(&self, id: Uuid, failed: bool) -> Result<()> {
        self.active_runs.lock().unwrap().remove(&id);
        let record = self.repo.get(id).await?.context("run not found")?;
        if matches!(
            record.status,
            Some(RunStatus::Succeeded | RunStatus::Cancelled)
        ) {
            return Ok(());
        }
        let status = if failed {
            RunStatus::Failed
        } else {
            RunStatus::Succeeded
        };
        self.repo.append_status(id, status).await
    }

    pub async fn asset(&self, id: Uuid, name: &str) -> Result<AssetSource> {
        let config = self.active(id).await?;
        let dir = self.run_dir(id).join("inputs");
        fs::create_dir_all(&dir).await?;
        let path = dir
            .join(hex::encode(Sha256::digest(name.as_bytes())))
            .with_extension("json");
        if let Ok(bytes) = fs::read(&path).await {
            return Ok(serde_json::from_slice(&bytes)?);
        }
        let _lock = Lock::acquire(&path.with_extension("lock")).await?;
        if let Ok(bytes) = fs::read(&path).await {
            return Ok(serde_json::from_slice(&bytes)?);
        }
        let input = match config.assets.get(name) {
            Some(input) => input.clone(),
            None => crate::run_config::AssetConfig {
                asset_id: Some(name.parse().context("unknown input asset")?),
                object: None,
                entrypoint: None,
            },
        };
        let (object, metadata, sha256) = match input.asset_id {
            Some(asset_id) => {
                let record = self
                    .repo
                    .get_asset(asset_id)
                    .await?
                    .context("input asset not found")?;
                let metadata = AssetMetadata {
                    entrypoint: input.entrypoint,
                    asset_id: Some(asset_id.to_string()),
                    metadata_json: record.metadata,
                    kind: kind_name(record.kind)?.into(),
                    asset_type: record.asset_type,
                };
                let sha256 = String::from_utf8(record.content_hash.to_vec())?;
                (record.path, metadata, Some(sha256))
            }
            None => {
                let metadata = AssetMetadata {
                    entrypoint: input.entrypoint,
                    asset_id: None,
                    metadata_json: "{}".into(),
                    kind: "file".into(),
                    asset_type: String::new(),
                };
                (
                    input.object.context("missing asset object")?,
                    metadata,
                    None,
                )
            }
        };
        let head = self
            .s3
            .head_object()
            .bucket(self.bucket)
            .key(&object)
            .send()
            .await?;
        let source = AssetSource {
            object,
            download: AssetDownload {
                metadata,
                size: u64::try_from(head.content_length().context("asset is missing size")?)?,
                etag: head.e_tag().context("asset is missing ETag")?.to_owned(),
                sha256,
            },
        };
        write_atomic(&path, &serde_json::to_vec(&source)?).await?;
        Ok(source)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn loading_memory_is_shared_only_within_the_same_run_and_stream() -> Result<()> {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        let config = aws_sdk_s3::config::Builder::new()
            .behavior_version(aws_sdk_s3::config::BehaviorVersion::latest())
            .region(aws_sdk_s3::config::Region::new("test"))
            .build();
        let runtime = Runtime::new(
            clickhouse::Client::default(),
            aws_sdk_s3::Client::from_conf(config),
            "test",
            PathBuf::from("unused"),
            crate::cache_limits::Budget::new(PathBuf::from("unused"), u64::MAX),
            CancellationToken::new(),
            "checkpoints",
            "metrics",
        )?;
        let first_id = Uuid::new_v4();
        let first = runtime.loading_memory(first_id, "images", 100 * LOAD_MEMORY_UNIT)?;
        let same = runtime
            .clone()
            .loading_memory(first_id, "images", 100 * LOAD_MEMORY_UNIT)?;
        assert!(Arc::ptr_eq(&first, &same));
        let second = runtime.loading_memory(first_id, "labels", 100 * LOAD_MEMORY_UNIT)?;
        let third = runtime.loading_memory(first_id, "audio", 100 * LOAD_MEMORY_UNIT)?;
        let other_run = runtime.loading_memory(Uuid::new_v4(), "images", 100 * LOAD_MEMORY_UNIT)?;
        let occupied = first.clone().acquire_many_owned(100).await?;
        assert_eq!(same.available_permits(), 0);
        let second_permit = second.clone().try_acquire_many_owned(100)?;
        let third_permit = third.clone().try_acquire_many_owned(100)?;
        let other_run_permit = other_run.clone().try_acquire_many_owned(100)?;
        assert_eq!(second.available_permits(), 0);
        assert_eq!(third.available_permits(), 0);
        assert_eq!(other_run.available_permits(), 0);
        drop(occupied);
        assert_eq!(same.available_permits(), 100);
        assert_eq!(second.available_permits(), 0);
        drop((second_permit, third_permit, other_run_permit));
        Ok(())
    }

    #[tokio::test]
    async fn active_configs_are_cached_until_expiry_or_end() -> Result<()> {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        let config = aws_sdk_s3::config::Builder::new()
            .behavior_version(aws_sdk_s3::config::BehaviorVersion::latest())
            .region(aws_sdk_s3::config::Region::new("test"))
            .build();
        let runtime = Runtime::new(
            clickhouse::Client::default().with_url("http://127.0.0.1:9"),
            aws_sdk_s3::Client::from_conf(config),
            "test",
            PathBuf::from("unused"),
            crate::cache_limits::Budget::new(PathBuf::from("unused"), u64::MAX),
            CancellationToken::new(),
            "checkpoints",
            "metrics",
        )?;
        let document = serde_json::json!({"queries": {"training": {"sql": "SELECT 1"}}});
        let config = Config::parse(document.as_object().unwrap())?;
        let (fresh, expired) = (Uuid::new_v4(), Uuid::new_v4());
        runtime.active_runs.lock().unwrap().extend([
            (fresh, (Instant::now(), config.clone())),
            (expired, (Instant::now() - ACTIVE_TTL, config)),
        ]);
        assert!(
            runtime
                .active(fresh)
                .await?
                .queries
                .contains_key("training")
        );
        assert!(runtime.active(expired).await.is_err());
        assert!(runtime.end(fresh, false).await.is_err());
        assert!(runtime.active(fresh).await.is_err());
        Ok(())
    }
}
