use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;

use anyhow::Context;
use bytes::BytesMut;
use clickhouse::Client;
use futures::Stream;
use tokio::fs;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::RwLock;
use tokio::sync::mpsc::UnboundedSender;
use tonic::{Status, Streaming};
use tracing::{debug, error, info};
use uuid::Uuid;

use crate::loader::Loader;
use crate::proto::{
    self, AssetResponse, CheckpointRequest, DataRequest, DataResponse, Split, asset_response,
    checkpoint_request,
};
use crate::run::{DataConfig, RunHandle, RunState};
use crate::run_manager::{RunManager, RunStatus};

const ASSET_CHUNK_BYTES: usize = 2 * 1024 * 1024;

#[derive(Clone)]
pub struct ActiveRun {
    pub handle: RunHandle,
    pub config: Arc<DataConfig>,
}

pub type ActiveRuns = Arc<RwLock<HashMap<Uuid, ActiveRun>>>;

pub fn parse_run_id(value: &str) -> Result<Uuid, Status> {
    Uuid::parse_str(value).map_err(|_| Status::invalid_argument("invalid run ID"))
}

pub struct InitializedRun {
    pub active: ActiveRun,
    pub train_config: String,
}

pub struct AssetStore {
    s3_client: aws_sdk_s3::Client,
    bucket: &'static str,
    root: &'static Path,
    synthetic: bool,
}

impl AssetStore {
    pub fn new(
        s3_client: aws_sdk_s3::Client,
        bucket: &'static str,
        root: &'static Path,
        synthetic: bool,
    ) -> Self {
        Self {
            s3_client,
            bucket,
            root,
            synthetic,
        }
    }

    pub async fn ensure(&self, run_id: Uuid, name: &str, key: &str) -> anyhow::Result<PathBuf> {
        let run_dir = self.root.join(run_id.to_string());
        fs::create_dir_all(&run_dir).await?;
        let path = run_dir.join(name);
        if fs::try_exists(&path).await? {
            return Ok(path);
        }

        let part = run_dir.join(format!("{name}.part"));
        info!(run = %run_id, asset = name, key, "downloading asset");
        if self.synthetic {
            fs::write(&part, format!("synthetic asset {name}\n").repeat(1024)).await?;
        } else {
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
        }
        fs::rename(&part, &path).await?;
        Ok(path)
    }
}

pub async fn initialize(
    run_id: Uuid,
    run_manager: &RunManager,
    assets: &AssetStore,
    database: &Client,
    loader: Arc<dyn Loader>,
    cache_dir: &'static Path,
    synthetic: bool,
) -> Result<InitializedRun, Status> {
    let run = run_manager
        .get(run_id)
        .await
        .map_err(|err| {
            error!(run = %run_id, error = format!("{err:#}"), "loading run failed");
            Status::internal(format!("{err:#}"))
        })?
        .ok_or_else(|| Status::not_found("unknown run"))?;
    let train_config = serde_json::to_string(&run.train_config)
        .map_err(|err| Status::internal(format!("{err:#}")))?;
    let config = Arc::new(run.data_config);
    run_manager
        .append_status(run_id, RunStatus::Running)
        .await
        .map_err(|err| {
            error!(run = %run_id, error = format!("{err:#}"), "recording running status failed");
            Status::internal(format!("{err:#}"))
        })?;
    let initialized: anyhow::Result<RunHandle> = async {
        futures::future::try_join_all(
            config
                .assets
                .iter()
                .map(|(name, asset)| assets.ensure(run_id, name, &asset.object)),
        )
        .await?;
        let run = RunState::new(run_id, database, loader, cache_dir, &config, synthetic).await?;
        Ok(RunHandle::spawn(run))
    }
    .await;
    match initialized {
        Ok(handle) => Ok(InitializedRun {
            active: ActiveRun { handle, config },
            train_config,
        }),
        Err(err) => {
            error!(run = %run_id, error = format!("{err:#}"), "run initialization failed");
            if let Err(status_err) = run_manager.append_status(run_id, RunStatus::Failed).await {
                error!(run = %run_id, error = format!("{status_err:#}"), "recording failed status failed");
            }
            Err(Status::internal(format!("{err:#}")))
        }
    }
}

pub async fn data_handler(
    active_runs: ActiveRuns,
    req_stream: &mut Streaming<DataRequest>,
    resp_stream: &UnboundedSender<Result<DataResponse, Status>>,
) -> anyhow::Result<()> {
    while let Some(req) = req_stream.message().await? {
        let run_id = parse_run_id(&req.run_id)?;
        debug!(run = %run_id, split = ?req.split(), "data request");
        let active = active_runs
            .read()
            .await
            .get(&run_id)
            .cloned()
            .context("unknown run")?;
        let Some(loaded_batch) = active
            .handle
            .next_batch(req.split() == Split::Validation)
            .await?
        else {
            debug!(run = %run_id, split = ?req.split(), "data stream exhausted");
            return Ok(());
        };
        let batch = loaded_batch.into_iter().map(proto::Sample::from).collect();
        resp_stream.send(Ok(DataResponse { batch }))?;
    }
    Ok(())
}

pub fn asset_stream(
    path: PathBuf,
    entrypoint: Option<String>,
) -> Pin<Box<dyn Stream<Item = Result<AssetResponse, Status>> + Send>> {
    Box::pin(async_stream::stream! {
        yield Ok(AssetResponse {
            payload: Some(asset_response::Payload::Metadata(proto::AssetMetadata {
                entrypoint,
            })),
        });
        let mut file = match fs::File::open(&path).await {
            Ok(file) => file,
            Err(err) => {
                yield Err(Status::internal(format!("{err:#}")));
                return;
            }
        };
        loop {
            let mut buf = BytesMut::with_capacity(ASSET_CHUNK_BYTES);
            match file.read_buf(&mut buf).await {
                Ok(0) => break,
                Ok(_) => yield Ok(AssetResponse {
                    payload: Some(asset_response::Payload::Chunk(buf.freeze())),
                }),
                Err(err) => {
                    yield Err(Status::internal(format!("{err:#}")));
                    break;
                }
            }
        }
    })
}

pub async fn receive_checkpoint(
    root: &Path,
    run_id: Uuid,
    step: u64,
    stream: &mut Streaming<CheckpointRequest>,
) -> anyhow::Result<(PathBuf, u64)> {
    let dir = root.join(run_id.to_string());
    fs::create_dir_all(&dir).await?;
    let path = dir.join(format!("step_{step:09}.tar"));
    let part = dir.join(format!("step_{step:09}.tar.part"));
    let mut file = fs::File::create(&part).await?;
    let mut bytes = 0;
    while let Some(request) = stream.message().await? {
        match request.payload {
            Some(checkpoint_request::Payload::Chunk(chunk)) => {
                bytes += chunk.len() as u64;
                file.write_all(&chunk).await?;
            }
            _ => anyhow::bail!("expected checkpoint chunks after the metadata"),
        }
    }
    file.sync_all().await?;
    fs::rename(&part, &path).await?;
    Ok((path, bytes))
}
