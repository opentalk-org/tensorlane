use super::AppError;
use crate::runtime::{Batch, Runtime};
use axum::{
    Json, Router,
    body::Body,
    extract::{Multipart, Path, State, multipart::MultipartError},
    http::{HeaderMap, HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post, put},
};
use base64::{Engine, engine::general_purpose::STANDARD};
use prost::Message;
use tensorlane_protocol::{
    AssetDownload, DataResponse, EndRequest, InitResponse, MetricBatch, TUS_VERSION, UploadSpec,
    UploadStatus,
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
                "upload not found" => StatusCode::NOT_FOUND,
                "unsupported tus version" => StatusCode::PRECONDITION_FAILED,
                "tus upload data must be application/offset+octet-stream" => {
                    StatusCode::UNSUPPORTED_MEDIA_TYPE
                }
                "run is cancelled"
                | "upload offset does not match"
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
                | "invalid tus upload headers"
                | "upload exceeds its length"
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
        .route("/uploads", post(create_upload).options(upload_options))
        .route(
            "/uploads/{upload_id}",
            put(save_upload).patch(append_upload).head(upload_offset),
        )
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

fn tus_response(status: StatusCode, offset: Option<u64>) -> Response {
    let mut response = status.into_response();
    let headers = response.headers_mut();
    headers.insert("Tus-Resumable", HeaderValue::from_static(TUS_VERSION));
    if let Some(offset) = offset {
        headers.insert("Upload-Offset", offset.into());
    }
    response
}

fn tus_header<'a>(headers: &'a HeaderMap, name: &str) -> anyhow::Result<&'a str> {
    let value = headers.get(name).and_then(|v| v.to_str().ok());
    value.ok_or_else(|| anyhow::anyhow!("invalid tus upload headers"))
}

/// The request's tus version, and whether its body carries upload data.
fn tus_request(headers: &HeaderMap) -> anyhow::Result<bool> {
    anyhow::ensure!(
        headers.get("Tus-Resumable").map(HeaderValue::as_bytes) == Some(TUS_VERSION.as_bytes()),
        "unsupported tus version"
    );
    Ok(headers.get(header::CONTENT_TYPE).map(HeaderValue::as_bytes)
        == Some(b"application/offset+octet-stream"))
}

async fn upload_options() -> Response {
    let mut response = tus_response(StatusCode::NO_CONTENT, None);
    let headers = response.headers_mut();
    headers.insert("Tus-Version", HeaderValue::from_static(TUS_VERSION));
    headers.insert(
        "Tus-Extension",
        HeaderValue::from_static("creation,creation-with-upload"),
    );
    response
}

/// tus creation (with upload): `Upload-Metadata` holds the client's upload `id`
/// and its JSON `spec`, both base64.
async fn create_upload(
    State(engine): State<Runtime>,
    headers: HeaderMap,
    body: Body,
) -> Result<Response, AppError> {
    let data = tus_request(&headers)?;
    let length: u64 = tus_header(&headers, "Upload-Length")?
        .parse()
        .map_err(|_| anyhow::anyhow!("invalid tus upload headers"))?;
    let (mut id, mut spec) = (None, None);
    for pair in tus_header(&headers, "Upload-Metadata")?.split(',') {
        let (key, value) = pair.trim().split_once(' ').unwrap_or((pair.trim(), ""));
        let value = STANDARD
            .decode(value)
            .map_err(|_| anyhow::anyhow!("invalid tus upload headers"))?;
        match key {
            "id" => id = Some(String::from_utf8(value)?.parse::<Uuid>()?),
            "spec" => spec = Some(serde_json::from_slice::<UploadSpec>(&value)?),
            _ => {}
        }
    }
    let (Some(id), Some(spec)) = (id, spec) else {
        return Err(anyhow::anyhow!("invalid tus upload headers").into());
    };
    if spec.size != length {
        return Err(anyhow::anyhow!("invalid tus upload headers").into());
    }
    let session = crate::tus_upload::create(&engine, id, spec).await?;
    let mut offset = crate::tus_upload::offset(&engine, id, &session).await?;
    if data && offset == 0 && length > 0 {
        let _slot = engine.upload_slots.acquire().await?;
        offset = crate::tus_upload::receive(&engine, id, &session, offset, body).await?;
    }
    if offset == length {
        crate::tus_upload::finish(&engine, id, &session).await?;
    }
    let mut response = tus_response(StatusCode::CREATED, Some(offset));
    response.headers_mut().insert(
        header::LOCATION,
        HeaderValue::from_str(&format!("/uploads/{id}"))?,
    );
    Ok(response)
}

async fn append_upload(
    State(engine): State<Runtime>,
    Path(id): Path<Uuid>,
    headers: HeaderMap,
    body: Body,
) -> Result<Response, AppError> {
    if !tus_request(&headers)? {
        return Err(
            anyhow::anyhow!("tus upload data must be application/offset+octet-stream").into(),
        );
    }
    let requested: u64 = tus_header(&headers, "Upload-Offset")?
        .parse()
        .map_err(|_| anyhow::anyhow!("invalid tus upload headers"))?;
    let session = crate::tus_upload::session(&engine, id).await?;
    let mut offset = crate::tus_upload::offset(&engine, id, &session).await?;
    if requested != offset {
        return Err(anyhow::anyhow!("upload offset does not match").into());
    }
    if offset < session.spec.size {
        let _slot = engine.upload_slots.acquire().await?;
        offset = crate::tus_upload::receive(&engine, id, &session, offset, body).await?;
    }
    if offset == session.spec.size {
        crate::tus_upload::finish(&engine, id, &session).await?;
    }
    Ok(tus_response(StatusCode::NO_CONTENT, Some(offset)))
}

async fn upload_offset(
    State(engine): State<Runtime>,
    Path(id): Path<Uuid>,
) -> Result<Response, AppError> {
    let session = crate::tus_upload::session(&engine, id).await?;
    let offset = crate::tus_upload::offset(&engine, id, &session).await?;
    if offset == session.spec.size {
        crate::tus_upload::finish(&engine, id, &session).await?;
    }
    let mut response = tus_response(StatusCode::OK, Some(offset));
    let headers = response.headers_mut();
    headers.insert("Upload-Length", session.spec.size.into());
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    Ok(response)
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
