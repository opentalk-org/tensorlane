use anyhow::{Context, Result, ensure};
use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{Arc, Mutex, Weak},
};
use tensorlane_protocol::InitResponse;
use tokio::{fs, sync::Semaphore};
use tokio_util::{sync::CancellationToken, task::TaskTracker};
use uuid::Uuid;

use crate::{
    loader::{Loader, S3Loader},
    run_config::Config,
    run_repo::{RunRepo, RunStatus},
    shared_cache::{Lock, write_atomic},
};

pub(super) const LOAD_MEMORY_UNIT: usize = 1024;
type LoadingPools = HashMap<(Uuid, String), Weak<Semaphore>>;

#[derive(Clone)]
pub struct Runtime {
    pub repo: RunRepo,
    pub database: clickhouse::Client,
    pub s3: aws_sdk_s3::Client,
    pub bucket: &'static str,
    pub cache: Arc<PathBuf>,
    pub shutdown: CancellationToken,
    pub tasks: TaskTracker,
    pub(super) checkpoint_prefix: &'static str,
    pub(super) metrics_prefix: &'static str,
    pub asset_slots: Arc<Semaphore>,
    pub upload_slots: Arc<Semaphore>,
    pub(super) loader: Arc<dyn Loader>,
    load_memory: Arc<Mutex<LoadingPools>>,
}

pub enum Batch {
    Pending,
    End,
    Ready(PathBuf),
}

impl Runtime {
    pub fn new(
        database: clickhouse::Client,
        s3: aws_sdk_s3::Client,
        bucket: &'static str,
        cache: PathBuf,
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
            shutdown,
            tasks: TaskTracker::new(),
            checkpoint_prefix,
            metrics_prefix,
            asset_slots: Arc::new(Semaphore::new(8)),
            upload_slots: Arc::new(Semaphore::new(2)),
            load_memory: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    pub fn run_dir(&self, run: Uuid) -> PathBuf {
        self.cache.join("runs").join(run.to_string())
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
            .map_err(|_| anyhow::anyhow!("loading memory lock poisoned"))?;
        streams.retain(|_, budget| budget.strong_count() > 0);
        let key = (run, stream.to_owned());
        if let Some(budget) = streams.get(&key).and_then(Weak::upgrade) {
            return Ok(budget);
        }
        let budget = Arc::new(Semaphore::new(bytes.div_ceil(LOAD_MEMORY_UNIT)));
        streams.insert(key, Arc::downgrade(&budget));
        Ok(budget)
    }

    pub async fn initialize(&self, id: Uuid, session: Uuid) -> Result<InitResponse> {
        ensure!(!self.shutdown.is_cancelled(), "server is shutting down");
        let record = self.repo.get(id).await?.context("run not found")?;
        ensure!(
            matches!(record.status, Some(RunStatus::Queued | RunStatus::Running)),
            "run is terminal"
        );
        let config = Config::parse(&record.config)?;
        let existing = self.repo.session(id).await?;
        if let Some(existing) = &existing {
            ensure!(
                existing.session_id == session,
                "run belongs to another client session"
            );
        } else {
            ensure!(
                record.status == Some(RunStatus::Queued),
                "run is already started"
            );
        }
        let started_at = time::OffsetDateTime::now_utc().max(
            record
                .status_timestamp
                .context("run has no status timestamp")?
                + time::Duration::nanoseconds(1),
        );
        let owner = self
            .create_state(
                &format!("runs/{id}/session"),
                &(session, i64::try_from(started_at.unix_timestamp_nanos())?),
            )
            .await?;
        ensure!(owner.0 == session, "run belongs to another client session");
        self.repo.renew_session(id, session).await?;
        if record.status != Some(RunStatus::Running) {
            self.repo.start(id, owner.1).await?;
        }
        Ok(InitResponse {
            run_id: id.to_string(),
            config: serde_json::to_string(&record.config)?,
            assets: config.assets.keys().cloned().collect(),
            streams: config.queries.keys().cloned().collect(),
        })
    }

    pub async fn active(&self, id: Uuid, session: Uuid) -> Result<Config> {
        let record = self.repo.get(id).await?.context("run not found")?;
        ensure!(
            record.status == Some(RunStatus::Running),
            "run is not running"
        );
        let owner = self
            .repo
            .session(id)
            .await?
            .context("run is not initialized")?;
        ensure!(
            owner.session_id == session,
            "run belongs to another client session"
        );
        Config::parse(&record.config)
    }

    pub async fn heartbeat(&self, id: Uuid, session: Uuid) -> Result<()> {
        self.active(id, session).await?;
        self.repo.renew_session(id, session).await
    }

    pub async fn end(&self, id: Uuid, session: Uuid, failed: bool) -> Result<()> {
        let record = self.repo.get(id).await?.context("run not found")?;
        let owner = self
            .repo
            .session(id)
            .await?
            .context("run is not initialized")?;
        ensure!(
            owner.session_id == session,
            "run belongs to another client session"
        );
        if matches!(
            record.status,
            Some(RunStatus::Succeeded | RunStatus::Failed | RunStatus::Cancelled)
        ) {
            return Ok(());
        }
        let status = self
            .create_state(
                &format!("runs/{id}/end"),
                &if failed {
                    RunStatus::Failed
                } else {
                    RunStatus::Succeeded
                },
            )
            .await?;
        let (_, started_at): (Uuid, i64) = self
            .read_state(&format!("runs/{id}/session"))
            .await?
            .context("run is not initialized")?;
        self.repo.append_status(id, status, started_at).await
    }

    pub async fn asset(&self, id: Uuid, session: Uuid, name: &str) -> Result<AssetSource> {
        let config = self.active(id, session).await?;
        use sha2::{Digest, Sha256};
        let dir = self.run_dir(id).join("inputs");
        fs::create_dir_all(&dir).await?;
        let path = dir
            .join(hex::encode(Sha256::digest(name.as_bytes())))
            .with_extension("json");
        let _lock = Lock::acquire(&path.with_extension("lock")).await?;
        if fs::try_exists(&path).await? {
            return Ok(serde_json::from_slice(&fs::read(&path).await?)?);
        }
        let input = config.assets.get(name).context("unknown input asset")?;
        let record = match input.asset_id {
            Some(id) => Some(
                self.repo
                    .get_asset(id)
                    .await?
                    .context("input asset not found")?,
            ),
            None => None,
        };
        let object = record
            .as_ref()
            .map(|r| r.path.clone())
            .or_else(|| input.object.clone())
            .context("missing asset object")?;
        let head = self
            .s3
            .head_object()
            .bucket(self.bucket)
            .key(&object)
            .send()
            .await?;
        let etag = head.e_tag().context("asset is missing ETag")?.to_owned();
        let size = u64::try_from(head.content_length().context("asset is missing size")?)?;
        let metadata = tensorlane_protocol::AssetMetadata {
            entrypoint: input.entrypoint.clone(),
            asset_id: input.asset_id.map(|id| id.to_string()),
            metadata_json: record
                .as_ref()
                .map(|r| r.metadata.clone())
                .unwrap_or_else(|| "{}".into()),
            kind: record
                .as_ref()
                .map(|r| crate::asset_repo::kind_name(r.kind))
                .transpose()?
                .unwrap_or("file")
                .into(),
            asset_type: record
                .as_ref()
                .map(|r| r.asset_type.clone())
                .unwrap_or_default(),
        };
        let sha256 = record
            .as_ref()
            .map(|r| String::from_utf8(r.content_hash.to_vec()))
            .transpose()?;
        let source = AssetSource {
            object,
            download: tensorlane_protocol::AssetDownload {
                metadata,
                size,
                etag,
                sha256,
            },
        };
        write_atomic(&path, &serde_json::to_vec(&source)?).await?;
        Ok(source)
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct AssetSource {
    pub object: String,
    pub download: tensorlane_protocol::AssetDownload,
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
}
