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
use tower_http::trace::TraceLayer;
use tracing::{error, info};
use uuid::Uuid;

use crate::{
    asset_repo::{AssetInfo, AssetRecord},
    auth::Auth,
    run_config::Config,
    run_repo::{Run, RunRepo, RunStatus},
    runtime::Runtime,
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

#[derive(Deserialize)]
struct AssetFilter {
    name: Option<String>,
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

pub async fn serve(port: u16, auth: Auth, runtime: Runtime) -> anyhow::Result<()> {
    let address = SocketAddr::from((Ipv4Addr::UNSPECIFIED, port));
    let listener = tokio::net::TcpListener::bind(address).await?;
    info!(%address, "HTTP server listening");
    let shutdown = runtime.shutdown.clone();
    axum::serve(listener, app(runtime, auth))
        .with_graceful_shutdown(shutdown.cancelled_owned())
        .await?;
    Ok(())
}

fn app(runtime: Runtime, auth: Auth) -> Router {
    router(runtime.repo.clone(), auth.clone())
        .merge(runtime::router(runtime.clone(), auth))
        .merge(crate::health::router(runtime))
}

fn router(run_repo: RunRepo, auth: Auth) -> Router {
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
        .layer(axum::extract::DefaultBodyLimit::disable())
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
    let run = run_repo
        .get(run_id)
        .await?
        .ok_or_else(|| AppError::new(StatusCode::NOT_FOUND, anyhow::anyhow!("Run not found")))?;
    Ok(Json(run))
}

async fn list_runs(State(run_repo): State<RunRepo>) -> Result<Json<Vec<Run>>, AppError> {
    Ok(Json(run_repo.list().await?))
}

async fn get_asset(
    State(repo): State<RunRepo>,
    Path(id): Path<Uuid>,
) -> Result<Json<AssetInfo>, AppError> {
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
) -> Result<Json<Vec<AssetInfo>>, AppError> {
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
        .map(AssetRecord::info)
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok(Json(assets))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, http::Request};
    use tokio_util::sync::CancellationToken;
    use tower::ServiceExt;

    const KEY: &str = "0123456789abcdef0123456789abcdef";

    #[tokio::test]
    async fn probes_are_public_while_application_routes_remain_protected() -> anyhow::Result<()> {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        let s3 = aws_sdk_s3::Client::from_conf(
            aws_sdk_s3::config::Builder::new()
                .behavior_version(aws_sdk_s3::config::BehaviorVersion::latest())
                .region(aws_sdk_s3::config::Region::new("test"))
                .endpoint_url("http://127.0.0.1:9")
                .credentials_provider(aws_sdk_s3::config::Credentials::new(
                    "test", "test", None, None, "test",
                ))
                .build(),
        );
        let cache = tempfile::tempdir()?;
        let shutdown = CancellationToken::new();
        let runtime = Runtime::new(
            clickhouse::Client::default().with_url("http://127.0.0.1:9"),
            s3,
            "test",
            cache.path().to_owned(),
            crate::cache_limits::Budget::new(cache.path().to_owned(), u64::MAX),
            shutdown.clone(),
            "checkpoints",
            "metrics",
        )?;
        let app = app(runtime, Auth::new(Some(KEY))?);
        for (path, expected) in [
            ("/healthz", StatusCode::OK),
            ("/readyz", StatusCode::SERVICE_UNAVAILABLE),
            ("/runs", StatusCode::UNAUTHORIZED),
            ("/unknown", StatusCode::UNAUTHORIZED),
        ] {
            let response = app
                .clone()
                .oneshot(Request::builder().uri(path).body(Body::empty())?)
                .await?;
            assert_eq!(response.status(), expected, "{path}");
        }
        shutdown.cancel();
        for (path, expected) in [
            ("/healthz", StatusCode::OK),
            ("/readyz", StatusCode::SERVICE_UNAVAILABLE),
        ] {
            let response = app
                .clone()
                .oneshot(Request::builder().uri(path).body(Body::empty())?)
                .await?;
            assert_eq!(response.status(), expected, "{path}");
        }
        Ok(())
    }

    #[tokio::test]
    async fn anonymous_http_is_allowed_without_a_configured_key() {
        let app = router(
            RunRepo::new(clickhouse::Client::default()),
            Auth::new(None).unwrap(),
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
            Auth::new(Some(KEY)).unwrap(),
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
        let bearer = format!("Bearer {KEY}");
        for (path, values, expected) in [
            (
                "/runs/invalid",
                vec![bearer.as_str()],
                StatusCode::BAD_REQUEST,
            ),
            ("/unknown", vec![bearer.as_str()], StatusCode::NOT_FOUND),
            (
                "/unknown",
                vec![bearer.as_str(), "Bearer wrong"],
                StatusCode::UNAUTHORIZED,
            ),
        ] {
            let mut request = Request::builder().uri(path);
            for value in values {
                request = request.header("authorization", value);
            }
            let response = app
                .clone()
                .oneshot(request.body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(response.status(), expected, "{path}");
        }
    }
}
