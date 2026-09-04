use std::net::{Ipv4Addr, SocketAddr};

use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    routing::get,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use time::OffsetDateTime;
use tokio_util::sync::CancellationToken;
use tracing::{error, info};
use uuid::Uuid;

use crate::{
    run::DataConfig,
    run_manager::{Run, RunManager, RunStatus},
};

#[derive(Deserialize)]
struct CreateRunRequest {
    project_id: Uuid,
    name: String,
    data_config: DataConfig,
    train_config: Map<String, Value>,
}

#[derive(Serialize)]
struct CreateRunResponse {
    run_id: Uuid,
    status: RunStatus,
}

#[derive(Serialize)]
struct RunResponse {
    run_id: Uuid,
    project_id: Uuid,
    name: String,
    data_config: DataConfig,
    train_config: Map<String, Value>,
    status: Option<RunStatus>,
    #[serde(with = "time::serde::rfc3339::option")]
    status_timestamp: Option<OffsetDateTime>,
}

pub async fn serve(
    port: u16,
    run_manager: RunManager,
    shutdown: CancellationToken,
) -> anyhow::Result<()> {
    let app = Router::new()
        .route("/runs", get(list_runs).post(create_run))
        .route("/runs/{run_id}", get(get_run))
        .with_state(run_manager);
    let address = SocketAddr::from((Ipv4Addr::UNSPECIFIED, port));
    let listener = tokio::net::TcpListener::bind(address).await?;
    info!(%address, "HTTP server listening");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown.cancelled_owned())
        .await?;
    Ok(())
}

async fn create_run(
    State(run_manager): State<RunManager>,
    Json(request): Json<CreateRunRequest>,
) -> Result<(StatusCode, Json<CreateRunResponse>), StatusCode> {
    let run_id = run_manager
        .create(
            request.project_id,
            &request.name,
            &request.data_config,
            &request.train_config,
        )
        .await
        .map_err(|err| {
            error!(error = format!("{err:#}"), "creating run failed");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
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
    State(run_manager): State<RunManager>,
    Path(run_id): Path<Uuid>,
) -> Result<Json<RunResponse>, StatusCode> {
    let run = run_manager.get(run_id).await.map_err(|err| {
        error!(
            run = %run_id,
            error = format!("{err:#}"),
            "getting run failed"
        );
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    match run {
        Some(run) => Ok(Json(run.into())),
        None => Err(StatusCode::NOT_FOUND),
    }
}

async fn list_runs(
    State(run_manager): State<RunManager>,
) -> Result<Json<Vec<RunResponse>>, StatusCode> {
    let runs = run_manager.list().await.map_err(|err| {
        error!(error = format!("{err:#}"), "listing runs failed");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    Ok(Json(runs.into_iter().map(Into::into).collect()))
}

impl From<Run> for RunResponse {
    fn from(run: Run) -> Self {
        Self {
            run_id: run.id,
            project_id: run.project_id,
            name: run.name,
            data_config: run.data_config,
            train_config: run.train_config,
            status: run.status,
            status_timestamp: run.status_timestamp,
        }
    }
}
