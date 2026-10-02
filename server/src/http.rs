use std::net::{Ipv4Addr, SocketAddr};

use axum::{
    Json, Router,
    extract::{
        Path, Query, State,
        rejection::{JsonRejection, PathRejection},
    },
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::get,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use time::OffsetDateTime;
use tokio_util::sync::CancellationToken;
use tower_http::trace::TraceLayer;
use tracing::{error, info};
use uuid::Uuid;

use crate::{
    run::Config,
    run_repo::{Run, RunRepo, RunStatus},
};

#[derive(Deserialize)]
struct CreateRunRequest {
    project_id: Uuid,
    name: String,
    config: Map<String, Value>,
}

#[derive(Serialize)]
struct CreateRunResponse {
    run_id: Uuid,
    status: RunStatus,
}

#[derive(Serialize)]
struct ErrorResponse {
    message: String,
}

struct AppError {
    status: StatusCode,
    error: anyhow::Error,
}

impl AppError {
    fn new(status: StatusCode, error: impl Into<anyhow::Error>) -> Self {
        Self {
            status,
            error: error.into(),
        }
    }
}

impl<E> From<E> for AppError
where
    E: Into<anyhow::Error>,
{
    fn from(error: E) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, error)
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        if self.status.is_server_error() {
            error!(error = format!("{:#}", self.error), "HTTP request failed");
        }
        (
            self.status,
            Json(ErrorResponse {
                message: self.error.to_string(),
            }),
        )
            .into_response()
    }
}

#[derive(Serialize)]
struct RunResponse {
    run_id: Uuid,
    project_id: Uuid,
    name: String,
    config: Map<String, Value>,
    status: Option<RunStatus>,
    #[serde(with = "time::serde::rfc3339::option")]
    status_timestamp: Option<OffsetDateTime>,
}

pub async fn serve(
    port: u16,
    run_repo: RunRepo,
    shutdown: CancellationToken,
) -> anyhow::Result<()> {
    let app = Router::new()
        .route("/runs", get(list_runs).post(create_run))
        .route("/runs/{run_id}", get(get_run))
        .route("/runs/{run_id}/assets", get(run_assets))
        .route("/assets/{asset_id}", get(get_asset))
        .fallback(async || AppError::new(StatusCode::NOT_FOUND, anyhow::anyhow!("Route not found")))
        .method_not_allowed_fallback(async || {
            AppError::new(
                StatusCode::METHOD_NOT_ALLOWED,
                anyhow::anyhow!("Method not allowed"),
            )
        })
        .layer(TraceLayer::new_for_http())
        .with_state(run_repo);
    let address = SocketAddr::from((Ipv4Addr::UNSPECIFIED, port));
    let listener = tokio::net::TcpListener::bind(address).await?;
    info!(%address, "HTTP server listening");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown.cancelled_owned())
        .await?;
    Ok(())
}

async fn create_run(
    State(run_repo): State<RunRepo>,
    request: Result<Json<CreateRunRequest>, JsonRejection>,
) -> Result<(StatusCode, Json<CreateRunResponse>), AppError> {
    let Json(request) = request.map_err(|err| AppError::new(err.status(), err))?;
    Config::parse(&request.config).map_err(|err| AppError::new(StatusCode::BAD_REQUEST, err))?;
    let run_id = run_repo
        .create(request.project_id, &request.name, &request.config)
        .await?;
    info!(run = %run_id, "run created");
    Ok((
        StatusCode::CREATED,
        Json(CreateRunResponse {
            run_id,
            status: RunStatus::Queued,
        }),
    ))
}

async fn get_run(
    State(run_repo): State<RunRepo>,
    run_id: Result<Path<Uuid>, PathRejection>,
) -> Result<Json<RunResponse>, AppError> {
    let Path(run_id) = run_id.map_err(|err| AppError::new(err.status(), err))?;
    let run = run_repo.get(run_id).await?;

    match run {
        Some(run) => Ok(Json(run.into())),
        None => Err(AppError::new(
            StatusCode::NOT_FOUND,
            anyhow::anyhow!("Run not found"),
        )),
    }
}

async fn list_runs(State(run_repo): State<RunRepo>) -> Result<Json<Vec<RunResponse>>, AppError> {
    let runs = run_repo.list().await?;
    Ok(Json(runs.into_iter().map(Into::into).collect()))
}

impl From<Run> for RunResponse {
    fn from(run: Run) -> Self {
        Self {
            run_id: run.id,
            project_id: run.project_id,
            name: run.name,
            config: run.config,
            status: run.status,
            status_timestamp: run.status_timestamp,
        }
    }
}

#[derive(Deserialize)]
struct AssetFilter {
    name: Option<String>,
}
async fn get_asset(
    State(repo): State<RunRepo>,
    Path(id): Path<Uuid>,
) -> Result<Json<crate::asset_repo::AssetInfo>, AppError> {
    let asset =
        repo.assets().get(id).await?.ok_or_else(|| {
            AppError::new(StatusCode::NOT_FOUND, anyhow::anyhow!("Asset not found"))
        })?;
    Ok(Json(asset.info()?))
}
async fn run_assets(
    State(repo): State<RunRepo>,
    Path(id): Path<Uuid>,
    Query(filter): Query<AssetFilter>,
) -> Result<Json<Vec<crate::asset_repo::AssetInfo>>, AppError> {
    if repo.get(id).await?.is_none() {
        return Err(AppError::new(
            StatusCode::NOT_FOUND,
            anyhow::anyhow!("Run not found"),
        ));
    }
    let assets = repo
        .assets()
        .for_run(id, filter.name.as_deref())
        .await?
        .into_iter()
        .map(|asset| asset.info())
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok(Json(assets))
}
