use std::net::{Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;

use crate::loader::S3Loader;
use crate::proto::{
    AssetRequest, AssetResponse, CheckpointRequest, CheckpointResponse, DataRequest, DataResponse,
    EndRequest, EndResponse, InitRequest, InitResponse, MetricsRequest, MetricsResponse,
    checkpoint_request, metrics_request,
    tensor_lane_server::{TensorLane as TensorLaneService, TensorLaneServer},
};
use crate::proto::{Split, asset_response};
use crate::run::RunExecutor;
use crate::run_repo::RunRepo;
use crate::uploads::UploadStore;
use crate::{metrics, proto};
use bytes::BytesMut;
use clickhouse::Client;
use futures::Stream;
use sha2::{Digest, Sha256};
use tokio::fs;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::mpsc::{self, UnboundedSender};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_util::sync::CancellationToken;
use tonic::transport::Server;
use tonic::{Request, Response, Status, Streaming};
use tracing::{debug, error, info};
use uuid::Uuid;

const ASSET_CHUNK_BYTES: usize = 2 * 1024 * 1024;

#[derive(Clone)]
struct TensorLane {
    database: Client,
    uploads: UploadStore,
    shutdown: CancellationToken,
    runs: RunExecutor,
}

impl TensorLane {
    fn new(
        s3_client: aws_sdk_s3::Client,
        database: Client,
        run_repo: RunRepo,
        bucket: &'static str,
        cache_dir: &'static Path,
        uploads: UploadStore,
        shutdown: CancellationToken,
    ) -> Self {
        let runs = RunExecutor::new(
            run_repo.clone(),
            database.clone(),
            Arc::new(S3Loader::new(s3_client.clone(), bucket)),
            cache_dir,
            s3_client,
            bucket,
        );
        Self {
            runs,
            uploads,
            database,
            shutdown,
        }
    }

    async fn wait(&self) {
        self.shutdown.cancelled().await;
        if let Err(err) = self.runs.drain_and_wait().await {
            error!(error = format!("{err:#}"), "failed to await all runs");
        }
        info!("active runs finished");
    }
}

#[tonic::async_trait]
impl TensorLaneService for TensorLane {
    async fn init(&self, request: Request<InitRequest>) -> Result<Response<InitResponse>, Status> {
        if self.shutdown.is_cancelled() {
            return Err(Status::unavailable(
                "server is shutting down; new runs are not accepted",
            ));
        }
        let run_id = parse_run_id(&request.into_inner().run_id)?;
        debug!(run = %run_id, "init request");
        let init = self
            .runs
            .start(run_id)
            .await
            .map_err(|err| Status::internal(format!("{err:#}")))?;
        info!(run = %run_id, "run initialized");

        Ok(Response::new(InitResponse {
            run_id: run_id.to_string(),
            train_config: init.train_config,
            assets: init.assets,
        }))
    }

    type DataStream = UnboundedReceiverStream<Result<DataResponse, Status>>;

    async fn data(
        &self,
        request: Request<Streaming<DataRequest>>,
    ) -> Result<Response<Self::DataStream>, Status> {
        let mut stream = request.into_inner();

        let (out_tx, out_rx) = mpsc::unbounded_channel();
        tokio::spawn({
            let runs = self.runs.clone();
            async move {
                if let Err(err) = data_handler(runs, &mut stream, &out_tx).await {
                    error!(error = format!("{err:#}"), "data stream failed");
                    let _ = out_tx.send(Err(Status::internal(format!("{err:#}"))));
                }
            }
        });

        Ok(UnboundedReceiverStream::new(out_rx).into())
    }

    type AssetStream = Pin<Box<dyn Stream<Item = Result<AssetResponse, Status>> + Send>>;

    async fn asset(
        &self,
        request: Request<AssetRequest>,
    ) -> Result<Response<Self::AssetStream>, Status> {
        let request = request.into_inner();
        let run_id = parse_run_id(&request.run_id)?;
        let (path, entrypoint) = self
            .runs
            .asset(run_id, &request.name)
            .await
            .map_err(|err| Status::internal(format!("{err:#}")))?;
        Ok(Response::new(asset_stream(path, entrypoint)))
    }

    async fn checkpoint(
        &self,
        request: Request<Streaming<CheckpointRequest>>,
    ) -> Result<Response<CheckpointResponse>, Status> {
        let mut stream = request.into_inner();

        let metadata = match stream.message().await?.and_then(|r| r.payload) {
            Some(checkpoint_request::Payload::Metadata(metadata)) => metadata,
            _ => {
                return Err(Status::invalid_argument(
                    "first checkpoint message must be metadata",
                ));
            }
        };
        let run_id = parse_run_id(&metadata.run_id)?;
        let asset_type = self
            .runs
            .asset_type(run_id)
            .await
            .map_err(|err| Status::internal(format!("{err:#}")))?;
        info!(run = %run_id, step = metadata.step, "receiving checkpoint");

        let checkpoint_id = uuid::Uuid::new_v4();
        let path = self.uploads.staging_path(checkpoint_id);
        let result = receive_checkpoint(&path, &mut stream).await;

        match result {
            Ok((bytes, content_hash)) => {
                self.uploads.checkpoint(
                    checkpoint_id,
                    run_id,
                    metadata.step,
                    bytes,
                    content_hash,
                    asset_type,
                );
                info!(
                    checkpoint = %checkpoint_id,
                    run = %run_id,
                    step = metadata.step,
                    bytes,
                    path = %path.display(),
                    "checkpoint staged"
                );
                Ok(Response::new(CheckpointResponse {}))
            }
            Err(err) => {
                error!(error = format!("{err:#}"), "staging checkpoint failed");
                Err(Status::internal(format!("{err:#}")))
            }
        }
    }

    async fn metrics(
        &self,
        request: Request<Streaming<MetricsRequest>>,
    ) -> Result<Response<MetricsResponse>, Status> {
        let mut stream = request.into_inner();
        let metadata = match stream.message().await?.and_then(|request| request.payload) {
            Some(metrics_request::Payload::Metadata(metadata)) => metadata,
            _ => {
                return Err(Status::invalid_argument(
                    "first metrics message must be stream metadata",
                ));
            }
        };
        let run_id = parse_run_id(&metadata.run_id)?;
        if !self.runs.is_running(run_id).await {
            return Err(Status::not_found("unknown run"));
        }
        info!(run = %run_id, "receiving metrics");

        match metrics::receive(&self.database, &self.uploads, run_id, stream).await {
            Ok(response) => {
                info!(
                    run = %run_id,
                    metrics = response.metrics_received,
                    array_metrics = response.array_metrics_received,
                    artifacts = response.artifacts_received,
                    artifact_bytes = response.artifact_bytes_received,
                    "metrics stream accepted"
                );
                Ok(Response::new(response))
            }
            Err(status) => {
                error!(
                    run = %run_id,
                    error = %status,
                    "receiving metrics failed"
                );
                Err(status)
            }
        }
    }

    async fn end(&self, request: Request<EndRequest>) -> Result<Response<EndResponse>, Status> {
        let run_id = parse_run_id(&request.into_inner().run_id)?;
        info!(run = %run_id, "ending run");
        self.runs
            .finish(run_id)
            .await
            .map_err(|err| Status::internal(format!("{err:#}")))?;

        Ok(Response::new(EndResponse {}))
    }
}

pub fn parse_run_id(value: &str) -> Result<Uuid, Status> {
    Uuid::parse_str(value).map_err(|_| Status::invalid_argument("invalid run ID"))
}

pub async fn data_handler(
    runs: RunExecutor,
    req_stream: &mut Streaming<DataRequest>,
    resp_stream: &UnboundedSender<Result<DataResponse, Status>>,
) -> anyhow::Result<()> {
    while let Some(req) = req_stream.message().await? {
        let run_id = parse_run_id(&req.run_id)?;
        debug!(run = %run_id, split = ?req.split(), "data request");
        let Some(loaded_batch) = runs
            .next_batch(run_id, req.split() == Split::Validation)
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
    path: &Path,
    stream: &mut Streaming<CheckpointRequest>,
) -> anyhow::Result<(u64, String)> {
    let part = path.with_extension("part");
    let mut file = fs::File::create(&part).await?;
    let mut bytes = 0;
    let mut hasher = Sha256::new();
    while let Some(request) = stream.message().await? {
        match request.payload {
            Some(checkpoint_request::Payload::Chunk(chunk)) => {
                bytes += chunk.len() as u64;
                hasher.update(&chunk);
                file.write_all(&chunk).await?;
            }
            _ => anyhow::bail!("expected checkpoint chunks after the metadata"),
        }
    }
    file.sync_all().await?;
    fs::rename(&part, &path).await?;
    Ok((bytes, format!("{:x}", hasher.finalize())))
}

pub async fn serve(
    port: u16,
    s3_client: aws_sdk_s3::Client,
    database: Client,
    run_repo: RunRepo,
    bucket: &'static str,
    cache_dir: &'static Path,
    uploads_dir: &'static Path,
    checkpoint_prefix: &'static str,
    metrics_prefix: &'static str,
    shutdown: CancellationToken,
) -> anyhow::Result<()> {
    info!("listening on 0.0.0.0:{port}");

    let uploads = UploadStore::new(
        s3_client.clone(),
        database.clone(),
        bucket,
        checkpoint_prefix,
        metrics_prefix,
        uploads_dir,
    )?;
    let service = TensorLane::new(
        s3_client,
        database,
        run_repo,
        bucket,
        cache_dir,
        uploads.clone(),
        shutdown,
    );
    let result = Server::builder()
        .add_service(TensorLaneServer::new(service.clone()))
        .serve_with_shutdown(
            SocketAddr::from((Ipv4Addr::new(0, 0, 0, 0), port)),
            service.wait(),
        )
        .await;

    uploads.finish().await;
    result?;

    Ok(())
}
