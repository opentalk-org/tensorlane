use super::AppError;
use crate::runtime::{Batch, Runtime};
use axum::{
    Json, Router,
    body::Body,
    extract::{Multipart, Path, State, multipart::MultipartError},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post, put},
};
use prost::Message;
use tensorlane_protocol::{
    AssetDownload, DataResponse, EndRequest, InitResponse, MetricBatch, UploadStatus,
};
use uuid::Uuid;

impl<E: Into<anyhow::Error>> From<E> for AppError {
    fn from(error: E) -> Self {
        let error = error.into();
        let status = if error.downcast_ref::<MultipartError>().is_some() {
            StatusCode::BAD_REQUEST
        } else if let Some(failure) = error.downcast_ref::<crate::job::Failure>() {
            if failure.retryable {
                StatusCode::SERVICE_UNAVAILABLE
            } else {
                StatusCode::UNPROCESSABLE_ENTITY
            }
        } else {
            match error.to_string().as_str() {
                "run not found"
                | "unknown stream"
                | "unknown input asset"
                | "input asset not found" => StatusCode::NOT_FOUND,
                "server is shutting down" | "shared cache has insufficient free space" => {
                    StatusCode::SERVICE_UNAVAILABLE
                }
                "run is cancelled"
                | "conflicting retry of upload ID"
                | "conflicting retry of metric request ID"
                | "conflicting retry of asset ID" => StatusCode::CONFLICT,
                "upload ID must not be nil"
                | "invalid upload SHA256"
                | "asset ID must match upload ID"
                | "asset name must not be empty"
                | "artifact name must not be empty"
                | "artifact size does not match upload size"
                | "asset kind must be checkpoint or file"
                | "upload must start with its spec"
                | "upload is missing its file"
                | "upload has unexpected size"
                | "upload SHA256 does not match"
                | "upload contains unexpected fields"
                | "metric request ID must not be nil"
                | "invalid scalar metric"
                | "invalid array metric" => StatusCode::BAD_REQUEST,
                _ => StatusCode::INTERNAL_SERVER_ERROR,
            }
        };
        Self::new(status, error)
    }
}

pub(super) fn router(runtime: Runtime, auth: crate::auth::Auth) -> Router {
    Router::new()
        .route("/runs/{run_id}/init", post(initialize))
        .route("/runs/{run_id}/heartbeat", post(heartbeat))
        .route("/runs/{run_id}/end", post(end))
        .route(
            "/runs/{run_id}/streams/{stream}/batches/{sequence}",
            get(batch),
        )
        .route("/runs/{run_id}/inputs/{name}", get(input_asset))
        .route("/runs/{run_id}/inputs/{name}/bytes", get(input_bytes))
        .route("/runs/{run_id}/metrics/{request_id}", put(save_metrics))
        .route("/uploads/{upload_id}", put(save_upload))
        .layer(axum::extract::DefaultBodyLimit::disable())
        .layer(axum::middleware::from_fn_with_state(
            auth,
            crate::auth::http,
        ))
        .with_state(runtime)
}

async fn initialize(
    State(engine): State<Runtime>,
    Path(id): Path<Uuid>,
) -> Result<Json<InitResponse>, AppError> {
    Ok(Json(engine.initialize(id).await?))
}

async fn heartbeat(
    State(engine): State<Runtime>,
    Path(id): Path<Uuid>,
) -> Result<StatusCode, AppError> {
    engine.active(id).await?;
    engine.repo.touch(id).await?;
    Ok(StatusCode::NO_CONTENT)
}

async fn end(
    State(engine): State<Runtime>,
    Path(id): Path<Uuid>,
    Json(request): Json<EndRequest>,
) -> Result<StatusCode, AppError> {
    engine.end(id, request.failed).await?;
    Ok(StatusCode::NO_CONTENT)
}

async fn save_metrics(
    State(engine): State<Runtime>,
    Path((run, request)): Path<(Uuid, Uuid)>,
    Json(batch): Json<MetricBatch>,
) -> Result<StatusCode, AppError> {
    crate::metric_http::save(&engine, run, request, batch).await?;
    Ok(StatusCode::NO_CONTENT)
}

async fn save_upload(
    State(engine): State<Runtime>,
    Path(id): Path<Uuid>,
    multipart: Multipart,
) -> Result<Json<UploadStatus>, AppError> {
    Ok(Json(
        crate::upload_http::save(&engine, id, multipart).await?,
    ))
}

async fn batch(
    State(engine): State<Runtime>,
    Path((id, stream, sequence)): Path<(Uuid, String, u64)>,
) -> Result<Response, AppError> {
    let path = match engine.batch(id, &stream, sequence).await {
        Ok(Batch::Pending) => return Ok(StatusCode::ACCEPTED.into_response()),
        Ok(Batch::End) => return Ok(StatusCode::NO_CONTENT.into_response()),
        Ok(Batch::Ready(path)) => path,
        Err(error)
            if error
                .downcast_ref::<std::io::Error>()
                .is_some_and(|error| error.kind() == std::io::ErrorKind::NotFound) =>
        {
            return Ok(StatusCode::ACCEPTED.into_response());
        }
        Err(error) => return Err(error.into()),
    };
    let tail = DataResponse {
        batch_id: sequence,
        ..Default::default()
    }
    .encode_to_vec();
    let data = match tokio::task::spawn_blocking(move || {
        let file = std::fs::File::open(path)?;
        let data = unsafe { memmap2::Mmap::map(&file)? };
        Ok::<_, std::io::Error>(bytes::Bytes::from_owner(data))
    })
    .await?
    {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(StatusCode::ACCEPTED.into_response());
        }
        data => data?,
    };
    let length = data.len() + tail.len();
    let stream =
        futures::stream::iter([Ok::<_, std::io::Error>(data), Ok(bytes::Bytes::from(tail))]);
    Ok(Response::builder()
        .header("content-type", "application/x-protobuf")
        .header("content-length", length)
        .body(Body::from_stream(stream))?)
}

async fn input_asset(
    State(engine): State<Runtime>,
    Path((id, name)): Path<(Uuid, String)>,
) -> Result<Json<AssetDownload>, AppError> {
    Ok(Json(engine.asset(id, &name).await?.download))
}

async fn input_bytes(
    State(engine): State<Runtime>,
    Path((id, name)): Path<(Uuid, String)>,
    headers: HeaderMap,
) -> Result<Response, AppError> {
    let source = engine.asset(id, &name).await?;
    let range = headers
        .get("range")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("");
    let range = crate::asset_http::parse_range(range, source.download.size)
        .map_err(|e| AppError::new(StatusCode::RANGE_NOT_SATISFIABLE, e))?;
    if let Some(tag) = headers.get("if-match")
        && tag.to_str()? != source.download.etag
    {
        return Err(AppError::new(
            StatusCode::PRECONDITION_FAILED,
            anyhow::anyhow!("input asset changed during download"),
        ));
    }
    Ok(crate::asset_http::download(engine, source, range).await?)
}
