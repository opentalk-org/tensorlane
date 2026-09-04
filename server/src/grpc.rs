use std::net::{Ipv4Addr, SocketAddr};
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;

use crate::grpc_support::{self, ActiveRuns, AssetStore};
use crate::loader::{Loader, S3Loader, SyntheticLoader};
use crate::metrics;
use crate::proto::{
    AssetRequest, AssetResponse, CheckpointRequest, CheckpointResponse, DataRequest, DataResponse,
    EndRequest, EndResponse, InitRequest, InitResponse, MetricsRequest, MetricsResponse,
    checkpoint_request,
    give_me_data_server::{GiveMeData as GiveMeDataService, GiveMeDataServer},
    metrics_request,
};
use crate::run_manager::{RunManager, RunStatus};
use clickhouse::Client;
use futures::Stream;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_util::sync::CancellationToken;
use tonic::transport::Server;
use tonic::{Request, Response, Status, Streaming};
use tracing::{debug, error, info};

struct GiveMeData {
    database: Client,
    run_manager: RunManager,
    loader: Arc<dyn Loader>,
    assets: AssetStore,
    cache_dir: &'static Path,
    checkpoint_dir: &'static Path,
    metrics_dir: &'static Path,
    synthetic: bool,
    active_runs: ActiveRuns,
}

impl GiveMeData {
    fn new(
        s3_client: aws_sdk_s3::Client,
        database: Client,
        run_manager: RunManager,
        bucket: &'static str,
        cache_dir: &'static Path,
        assets_dir: &'static Path,
        checkpoint_dir: &'static Path,
        metrics_dir: &'static Path,
        synthetic: bool,
    ) -> Self {
        let loader: Arc<dyn Loader> = if synthetic {
            Arc::new(SyntheticLoader)
        } else {
            Arc::new(S3Loader::new(s3_client.clone(), bucket))
        };
        GiveMeData {
            active_runs: Default::default(),
            loader,
            assets: AssetStore::new(s3_client, bucket, assets_dir, synthetic),
            cache_dir,
            checkpoint_dir,
            metrics_dir,
            synthetic,
            database,
            run_manager,
        }
    }
}

#[tonic::async_trait]
impl GiveMeDataService for GiveMeData {
    async fn init(&self, request: Request<InitRequest>) -> Result<Response<InitResponse>, Status> {
        let run_id = grpc_support::parse_run_id(&request.into_inner().run_id)?;
        debug!(run = %run_id, "init request");
        let initialized = grpc_support::initialize(
            run_id,
            &self.run_manager,
            &self.assets,
            &self.database,
            self.loader.clone(),
            self.cache_dir,
            self.synthetic,
        )
        .await?;
        self.active_runs
            .write()
            .await
            .insert(run_id, initialized.active);
        info!(run = %run_id, "run initialized");

        Ok(Response::new(InitResponse {
            run_id: run_id.to_string(),
            train_config: initialized.train_config,
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
            let active_runs = self.active_runs.clone();
            async move {
                if let Err(err) =
                    grpc_support::data_handler(active_runs, &mut stream, &out_tx).await
                {
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
        let run_id = grpc_support::parse_run_id(&request.run_id)?;
        let active = self
            .active_runs
            .read()
            .await
            .get(&run_id)
            .cloned()
            .ok_or_else(|| Status::not_found("unknown run"))?;
        let asset = active
            .config
            .assets
            .get(&request.name)
            .ok_or_else(|| Status::not_found(format!("unknown asset {:?}", request.name)))?;
        info!(run = %run_id, asset = %request.name, "asset requested");

        let path = self
            .assets
            .ensure(run_id, &request.name, &asset.object)
            .await
            .map_err(|err| {
                error!(error = format!("{err:#}"), asset = %request.name, "asset fetch failed");
                Status::internal(format!("{err:#}"))
            })?;
        let entrypoint = asset.entrypoint.clone();
        Ok(Response::new(grpc_support::asset_stream(path, entrypoint)))
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
        let run_id = grpc_support::parse_run_id(&metadata.run_id)?;
        if !self.active_runs.read().await.contains_key(&run_id) {
            return Err(Status::not_found("unknown run"));
        }
        info!(run = %run_id, step = metadata.step, "receiving checkpoint");

        let result = grpc_support::receive_checkpoint(
            self.checkpoint_dir,
            run_id,
            metadata.step,
            &mut stream,
        )
        .await;

        match result {
            Ok((path, bytes)) => {
                info!(
                    run = %run_id,
                    step = metadata.step,
                    bytes,
                    path = %path.display(),
                    "checkpoint stored"
                );
                Ok(Response::new(CheckpointResponse {}))
            }
            Err(err) => {
                error!(error = format!("{err:#}"), "storing checkpoint failed");
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
        let run_id = grpc_support::parse_run_id(&metadata.run_id)?;
        if !self.active_runs.read().await.contains_key(&run_id) {
            return Err(Status::not_found("unknown run"));
        }
        info!(run = %run_id, "receiving metrics");

        match metrics::receive(&self.database, self.metrics_dir, run_id, stream).await {
            Ok(response) => {
                info!(
                    run = %run_id,
                    metrics = response.metrics_received,
                    array_metrics = response.array_metrics_received,
                    artifacts = response.artifacts_received,
                    artifact_bytes = response.artifact_bytes_received,
                    "metrics stored"
                );
                Ok(Response::new(response))
            }
            Err(status) => {
                error!(
                    run = %run_id,
                    error = %status,
                    "storing metrics failed"
                );
                Err(status)
            }
        }
    }

    async fn end(&self, request: Request<EndRequest>) -> Result<Response<EndResponse>, Status> {
        let run_id = grpc_support::parse_run_id(&request.into_inner().run_id)?;
        info!(run = %run_id, "ending run");
        let removed = self.active_runs.write().await.remove(&run_id);

        match removed {
            None => return Err(Status::not_found("unknown run")),
            Some(active) => active.handle.finish().await,
        }
        self.run_manager
            .append_status(run_id, RunStatus::Succeeded)
            .await
            .map_err(|err| {
                error!(run = %run_id, error = format!("{err:#}"), "finishing run failed");
                Status::internal(format!("{err:#}"))
            })?;

        Ok(Response::new(EndResponse {}))
    }
}

pub async fn serve(
    port: u16,
    s3_client: aws_sdk_s3::Client,
    database: Client,
    run_manager: RunManager,
    bucket: &'static str,
    cache_dir: &'static Path,
    assets_dir: &'static Path,
    checkpoint_dir: &'static Path,
    metrics_dir: &'static Path,
    synthetic: bool,
    shutdown: CancellationToken,
) -> anyhow::Result<()> {
    if synthetic {
        info!("serving synthetic runs");
    }
    info!("listening on 0.0.0.0:{port}");

    Server::builder()
        .add_service(GiveMeDataServer::new(GiveMeData::new(
            s3_client,
            database,
            run_manager,
            bucket,
            cache_dir,
            assets_dir,
            checkpoint_dir,
            metrics_dir,
            synthetic,
        )))
        .serve_with_shutdown(
            SocketAddr::from((Ipv4Addr::new(0, 0, 0, 0), port)),
            shutdown.cancelled_owned(),
        )
        .await?;

    Ok(())
}
