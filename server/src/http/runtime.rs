use super::AppError;
use axum::{
    Json, Router,
    extract::{Path, State},
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
        .route("/uploads/{upload_id}", put(create_upload))
        .route("/uploads/{upload_id}/chunks/{index}", put(upload_chunk))
        .route("/uploads/{upload_id}/commit", post(commit_upload))
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
    headers: HeaderMap,
    Json(batch): Json<tensorlane_protocol::MetricBatch>,
) -> Result<StatusCode, AppError> {
    crate::metric_http::save(&engine, run, session(&headers)?, request, batch)
        .await
        .map_err(runtime_error)?;
    Ok(StatusCode::NO_CONTENT)
}

async fn create_upload(
    State(engine): State<crate::runtime::Runtime>,
    Path(id): Path<Uuid>,
    headers: HeaderMap,
    Json(spec): Json<tensorlane_protocol::UploadSpec>,
) -> Result<Json<tensorlane_protocol::UploadStatus>, AppError> {
    Ok(Json(
        crate::upload_http::create(&engine, id, session(&headers)?, spec)
            .await
            .map_err(runtime_error)?,
    ))
}

async fn upload_chunk(
    State(engine): State<crate::runtime::Runtime>,
    Path((id, index)): Path<(Uuid, u64)>,
    headers: HeaderMap,
    bytes: axum::body::Bytes,
) -> Result<StatusCode, AppError> {
    crate::upload_http::chunk(&engine, id, session(&headers)?, index, &bytes)
        .await
        .map_err(runtime_error)?;
    Ok(StatusCode::NO_CONTENT)
}

async fn commit_upload(
    State(engine): State<crate::runtime::Runtime>,
    Path(id): Path<Uuid>,
    headers: HeaderMap,
) -> Result<Response, AppError> {
    let status = crate::upload_http::commit(&engine, id, session(&headers)?)
        .await
        .map_err(runtime_error)?;
    if status.committed {
        Ok(Json(status).into_response())
    } else {
        Ok((StatusCode::ACCEPTED, [("retry-after", "1")], Json(status)).into_response())
    }
}

fn session(headers: &HeaderMap) -> Result<Uuid, AppError> {
    let session = headers
        .get(tensorlane_protocol::SESSION_HEADER)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<Uuid>().ok())
        .filter(|value| !value.is_nil());
    session.ok_or_else(|| {
        AppError::new(
            StatusCode::BAD_REQUEST,
            anyhow::anyhow!("a nonnil x-tensorlane-session UUID header is required"),
        )
    })
}

fn runtime_error(error: anyhow::Error) -> AppError {
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
        "run is terminal"
        | "run is not running"
        | "run belongs to another client session"
        | "run is not initialized" => StatusCode::CONFLICT,
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
        | "upload chunk offset overflows"
        | "upload chunk is outside the file"
        | "upload chunk has unexpected size"
        | "upload is missing chunks"
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
    headers: HeaderMap,
) -> Result<Json<tensorlane_protocol::InitResponse>, AppError> {
    Ok(Json(
        engine
            .initialize(id, session(&headers)?)
            .await
            .map_err(runtime_error)?,
    ))
}

async fn heartbeat(
    State(engine): State<crate::runtime::Runtime>,
    Path(id): Path<Uuid>,
    headers: HeaderMap,
) -> Result<StatusCode, AppError> {
    engine
        .heartbeat(id, session(&headers)?)
        .await
        .map_err(runtime_error)?;
    Ok(StatusCode::NO_CONTENT)
}

async fn end(
    State(engine): State<crate::runtime::Runtime>,
    Path(id): Path<Uuid>,
    headers: HeaderMap,
    Json(request): Json<tensorlane_protocol::EndRequest>,
) -> Result<StatusCode, AppError> {
    engine
        .end(id, session(&headers)?, request.failed)
        .await
        .map_err(runtime_error)?;
    Ok(StatusCode::NO_CONTENT)
}

async fn batch(
    State(engine): State<crate::runtime::Runtime>,
    Path((id, stream, sequence)): Path<(Uuid, String, u64)>,
    headers: HeaderMap,
) -> Result<Response, AppError> {
    use prost::Message;
    match engine
        .batch(id, session(&headers)?, &stream, sequence)
        .await
        .map_err(runtime_error)?
    {
        crate::runtime::Batch::Pending => {
            Ok((StatusCode::ACCEPTED, [("retry-after", "1")]).into_response())
        }
        crate::runtime::Batch::End => Ok(StatusCode::NO_CONTENT.into_response()),
        crate::runtime::Batch::Ready(path) => {
            use futures::StreamExt;
            let tail = tensorlane_protocol::DataResponse {
                batch_id: sequence,
                ..Default::default()
            }
            .encode_to_vec();
            let file = tokio::fs::File::open(path).await?;
            let length = file.metadata().await?.len() + tail.len() as u64;
            let stream =
                tokio_util::io::ReaderStream::new(file).chain(futures::stream::once(async move {
                    Ok(bytes::Bytes::from(tail))
                }));
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
    headers: HeaderMap,
) -> Result<Json<tensorlane_protocol::AssetDownload>, AppError> {
    Ok(Json(
        engine
            .asset(id, session(&headers)?, &name)
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
    let source = engine
        .asset(id, session(&headers)?, &name)
        .await
        .map_err(runtime_error)?;
    let range = headers
        .get("range")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("");
    crate::asset_http::parse_range(range, source.download.size)
        .map_err(|e| AppError::new(StatusCode::RANGE_NOT_SATISFIABLE, e))?;
    if headers
        .get("if-match")
        .and_then(|h| h.to_str().ok())
        .is_some_and(|tag| tag != source.download.etag)
    {
        return Err(AppError::new(
            StatusCode::PRECONDITION_FAILED,
            anyhow::anyhow!("input asset changed during download"),
        ));
    }
    crate::asset_http::download(engine, source, headers)
        .await
        .map_err(Into::into)
}
