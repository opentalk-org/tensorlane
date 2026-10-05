mod runtime;
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
use tokio_util::sync::CancellationToken;
use tower_http::trace::TraceLayer;
use tracing::{error, info};
use uuid::Uuid;

use crate::{
    run_config::Config,
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

pub(super) struct AppError {
    status: StatusCode,
    error: anyhow::Error,
}

impl AppError {
    pub(super) fn new(status: StatusCode, error: impl Into<anyhow::Error>) -> Self {
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
        if self.status == StatusCode::INTERNAL_SERVER_ERROR {
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

pub async fn serve(
    port: u16,
    auth: crate::auth::Auth,
    shutdown: CancellationToken,
    runtime: crate::runtime::Runtime,
) -> anyhow::Result<()> {
    let app = router(runtime.repo.clone(), auth.clone())
        .merge(runtime::router(runtime, auth))
        .layer(tower::limit::ConcurrencyLimitLayer::new(64));
    let address = SocketAddr::from((Ipv4Addr::UNSPECIFIED, port));
    let listener = tokio::net::TcpListener::bind(address).await?;
    info!(%address, "HTTP server listening");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown.cancelled_owned())
        .await?;
    Ok(())
}

fn router(run_repo: RunRepo, auth: crate::auth::Auth) -> Router {
    Router::new()
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
        .layer(axum::middleware::from_fn_with_state(
            auth,
            crate::auth::http,
        ))
        .with_state(run_repo)
}

async fn create_run(
    State(run_repo): State<RunRepo>,
    request: Result<Json<CreateRunRequest>, JsonRejection>,
) -> Result<(StatusCode, Json<CreateRunResponse>), AppError> {
    let Json(request) = request.map_err(|err| AppError::new(err.status(), err))?;
    if !request.config.get("queries").is_some_and(Value::is_object) {
        return Err(AppError::new(
            StatusCode::BAD_REQUEST,
            anyhow::anyhow!("config.queries must map stream names to query objects"),
        ));
    }
    Config::parse(&request.config).map_err(|err| AppError::new(StatusCode::BAD_REQUEST, err))?;
    let run_id = run_repo
        .create(request.project_id, &request.name, &request.config)
        .await?
        .ok_or_else(|| {
            AppError::new(StatusCode::NOT_FOUND, anyhow::anyhow!("Project not found"))
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
    State(run_repo): State<RunRepo>,
    run_id: Result<Path<Uuid>, PathRejection>,
) -> Result<Json<Run>, AppError> {
    let Path(run_id) = run_id.map_err(|err| AppError::new(err.status(), err))?;
    let run = run_repo.get(run_id).await?;

    match run {
        Some(run) => Ok(Json(run)),
        None => Err(AppError::new(
            StatusCode::NOT_FOUND,
            anyhow::anyhow!("Run not found"),
        )),
    }
}

async fn list_runs(State(run_repo): State<RunRepo>) -> Result<Json<Vec<Run>>, AppError> {
    let runs = run_repo.list().await?;
    Ok(Json(runs))
}

#[derive(Deserialize)]
struct AssetFilter {
    name: Option<String>,
}
async fn get_asset(
    State(repo): State<RunRepo>,
    Path(id): Path<Uuid>,
) -> Result<Json<crate::asset_repo::AssetInfo>, AppError> {
    let asset = repo
        .get_asset(id)
        .await?
        .ok_or_else(|| AppError::new(StatusCode::NOT_FOUND, anyhow::anyhow!("Asset not found")))?;
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
        .run_assets(id, filter.name.as_deref())
        .await?
        .into_iter()
        .map(|asset| asset.info())
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok(Json(assets))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    const KEY: &str = "0123456789abcdef0123456789abcdef";

    #[tokio::test]
    async fn anonymous_http_is_allowed_without_a_configured_key() {
        let app = router(
            RunRepo::new(clickhouse::Client::default()),
            crate::auth::Auth::new(None).unwrap(),
        );
        for (path, expected) in [
            ("/runs/invalid", StatusCode::BAD_REQUEST),
            ("/unknown", StatusCode::NOT_FOUND),
        ] {
            let request = Request::builder().uri(path).body(Body::empty()).unwrap();
            assert_eq!(
                app.clone().oneshot(request).await.unwrap().status(),
                expected
            );
        }
    }

    #[tokio::test]
    async fn authentication_covers_routes_fallbacks_and_wrong_methods() {
        let app = router(
            RunRepo::new(clickhouse::Client::default()),
            crate::auth::Auth::new(Some(KEY)).unwrap(),
        );
        for (method, path) in [
            ("GET", "/runs"),
            ("POST", "/runs"),
            ("GET", "/runs/invalid"),
            ("GET", "/runs/invalid/assets"),
            ("GET", "/assets/invalid"),
            ("GET", "/unknown"),
            ("DELETE", "/runs"),
            ("OPTIONS", "/runs"),
        ] {
            for value in [None, Some("Bearer wrong")] {
                let mut request = Request::builder().method(method).uri(path);
                if let Some(value) = value {
                    request = request.header("authorization", value);
                }
                let response = app
                    .clone()
                    .oneshot(request.body(Body::empty()).unwrap())
                    .await
                    .unwrap();
                assert_eq!(
                    response.status(),
                    StatusCode::UNAUTHORIZED,
                    "{method} {path}"
                );
                assert_eq!(response.headers()["www-authenticate"], "Bearer");
            }
        }
        let request = Request::builder()
            .uri("/runs/invalid")
            .header("authorization", format!("Bearer {KEY}"))
            .body(Body::empty())
            .unwrap();
        assert_eq!(
            app.clone().oneshot(request).await.unwrap().status(),
            StatusCode::BAD_REQUEST
        );
        let request = Request::builder()
            .uri("/unknown")
            .header("authorization", format!("Bearer {KEY}"))
            .body(Body::empty())
            .unwrap();
        assert_eq!(
            app.clone().oneshot(request).await.unwrap().status(),
            StatusCode::NOT_FOUND
        );
        let request = Request::builder()
            .uri("/unknown")
            .header("authorization", format!("Bearer {KEY}"))
            .header("authorization", "Bearer wrong")
            .body(Body::empty())
            .unwrap();
        assert_eq!(
            app.oneshot(request).await.unwrap().status(),
            StatusCode::UNAUTHORIZED
        );
    }
}
