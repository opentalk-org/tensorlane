use super::AppError;
use axum::{
    Json, Router,
    extract::{Multipart, Path, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post, put},
};
use uuid::Uuid;

pub(super) fn router(runtime: crate::runtime::Runtime, auth: crate::auth::Auth) -> Router {
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
        .route(
            "/uploads/{upload_id}",
            put(save_upload).layer(axum::extract::DefaultBodyLimit::disable()),
        )
        .layer(axum::extract::DefaultBodyLimit::max(
            tensorlane_protocol::TRANSFER_CHUNK_BYTES,
        ))
        .layer(axum::middleware::from_fn_with_state(
            auth,
            crate::auth::http,
        ))
        .with_state(runtime)
}

async fn save_metrics(
    State(engine): State<crate::runtime::Runtime>,
    Path((run, request)): Path<(Uuid, Uuid)>,
    Json(batch): Json<tensorlane_protocol::MetricBatch>,
) -> Result<StatusCode, AppError> {
    crate::metric_http::save(&engine, run, request, batch)
        .await
        .map_err(runtime_error)?;
    Ok(StatusCode::NO_CONTENT)
}

async fn save_upload(
    State(engine): State<crate::runtime::Runtime>,
    Path(id): Path<Uuid>,
    multipart: Multipart,
) -> Result<Json<tensorlane_protocol::UploadStatus>, AppError> {
    Ok(Json(
        crate::upload_http::save(&engine, id, multipart)
            .await
            .map_err(runtime_error)?,
    ))
}

fn runtime_error(error: anyhow::Error) -> AppError {
    if error
        .downcast_ref::<axum::extract::multipart::MultipartError>()
        .is_some()
    {
        return AppError::new(StatusCode::BAD_REQUEST, error);
    }
    if let Some(failure) = error.downcast_ref::<crate::job::Failure>() {
        let status = if failure.retryable {
            StatusCode::SERVICE_UNAVAILABLE
        } else {
            StatusCode::UNPROCESSABLE_ENTITY
        };
        return AppError::new(status, error);
    }
    let status = match error.to_string().as_str() {
        "run not found" | "unknown stream" | "unknown input asset" | "input asset not found" => {
            StatusCode::NOT_FOUND
        }
        "run is not running" => StatusCode::CONFLICT,
        "server is shutting down"
        | "asset download capacity reached"
        | "shared cache has insufficient free space" => StatusCode::SERVICE_UNAVAILABLE,
        "conflicting retry of upload ID"
        | "conflicting retry of upload chunk"
        | "conflicting retry of metric request ID"
        | "conflicting retry of asset ID"
        | "upload is already committed" => StatusCode::CONFLICT,
        "upload ID must not be nil"
        | "upload exceeds 16 GiB"
        | "invalid upload SHA256"
        | "asset ID must match upload ID"
        | "asset name must not be empty"
        | "artifact name must not be empty"
        | "artifact size does not match upload size"
        | "asset kind must be checkpoint or file"
        | "upload must start with its spec"
        | "upload spec exceeds 4 MiB"
        | "upload is missing its file"
        | "upload has unexpected size"
        | "upload SHA256 does not match"
        | "upload contains unexpected fields"
        | "metric request ID must not be nil"
        | "metric requests must not exceed 1000 metrics"
        | "invalid scalar metric"
        | "invalid array metric" => StatusCode::BAD_REQUEST,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    AppError::new(status, error)
}

async fn initialize(
    State(engine): State<crate::runtime::Runtime>,
    Path(id): Path<Uuid>,
) -> Result<Json<tensorlane_protocol::InitResponse>, AppError> {
    Ok(Json(engine.initialize(id).await.map_err(runtime_error)?))
}

async fn heartbeat(
    State(engine): State<crate::runtime::Runtime>,
    Path(id): Path<Uuid>,
) -> Result<StatusCode, AppError> {
    engine.active(id).await.map_err(runtime_error)?;
    Ok(StatusCode::NO_CONTENT)
}

async fn end(
    State(engine): State<crate::runtime::Runtime>,
    Path(id): Path<Uuid>,
    Json(request): Json<tensorlane_protocol::EndRequest>,
) -> Result<StatusCode, AppError> {
    engine
        .end(id, request.failed)
        .await
        .map_err(runtime_error)?;
    Ok(StatusCode::NO_CONTENT)
}

async fn batch(
    State(engine): State<crate::runtime::Runtime>,
    Path((id, stream, sequence)): Path<(Uuid, String, u64)>,
) -> Result<Response, AppError> {
    use prost::Message;
    match engine
        .batch(id, &stream, sequence)
        .await
        .map_err(runtime_error)?
    {
        crate::runtime::Batch::Pending => Ok(StatusCode::ACCEPTED.into_response()),
        crate::runtime::Batch::End => Ok(StatusCode::NO_CONTENT.into_response()),
        crate::runtime::Batch::Ready(path) => {
            let tail = tensorlane_protocol::DataResponse {
                batch_id: sequence,
                ..Default::default()
            }
            .encode_to_vec();
            let data = tokio::task::spawn_blocking(move || {
                let file = std::fs::File::open(path)?;
                let data = unsafe { memmap2::Mmap::map(&file)? };
                Ok::<_, std::io::Error>(bytes::Bytes::from_owner(data))
            })
            .await??;
            let length = data.len() + tail.len();
            let stream = futures::stream::iter([
                Ok::<_, std::io::Error>(data),
                Ok(bytes::Bytes::from(tail)),
            ]);
            Ok(Response::builder()
                .header("content-type", "application/x-protobuf")
                .header("content-length", length)
                .body(axum::body::Body::from_stream(stream))
                .map_err(anyhow::Error::from)?)
        }
    }
}

async fn input_asset(
    State(engine): State<crate::runtime::Runtime>,
    Path((id, name)): Path<(Uuid, String)>,
) -> Result<Json<tensorlane_protocol::AssetDownload>, AppError> {
    Ok(Json(
        engine
            .asset(id, &name)
            .await
            .map_err(runtime_error)?
            .download,
    ))
}

async fn input_bytes(
    State(engine): State<crate::runtime::Runtime>,
    Path((id, name)): Path<(Uuid, String)>,
    headers: HeaderMap,
) -> Result<Response, AppError> {
    let source = engine.asset(id, &name).await.map_err(runtime_error)?;
    let range = headers
        .get("range")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("");
    let range = crate::asset_http::parse_range(range, source.download.size)
        .map_err(|e| AppError::new(StatusCode::RANGE_NOT_SATISFIABLE, e))?;
    if headers
        .get("if-match")
        .map(|tag| tag.to_str())
        .transpose()?
        .is_some_and(|tag| tag != source.download.etag)
    {
        return Err(AppError::new(
            StatusCode::PRECONDITION_FAILED,
            anyhow::anyhow!("input asset changed during download"),
        ));
    }
    crate::asset_http::download(engine, source, range)
        .await
        .map_err(Into::into)
}
