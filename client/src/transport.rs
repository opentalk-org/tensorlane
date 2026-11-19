use crate::semaphore::{MemoryBudget, MemoryLease};
use anyhow::{Context, Result, bail, ensure};
use base64::{Engine, engine::general_purpose::STANDARD};
use reqwest::{
    Method, StatusCode, Url,
    header::{AUTHORIZATION, CONTENT_LENGTH, CONTENT_TYPE, HeaderMap, HeaderValue},
};
use std::{
    fs::File,
    io::{Seek, SeekFrom},
    sync::Arc,
    time::Duration,
};
use tensorlane_protocol::{EndRequest, InitResponse, TUS_VERSION, UploadSpec, upload_chunk_bytes};
use tokio::{
    io::AsyncReadExt,
    time::{Instant, sleep},
};
use tokio_util::io::ReaderStream;

#[derive(Clone)]
pub struct HttpClient {
    client: reqwest::Client,
    base: Url,
}

enum Body {
    Bytes(Vec<u8>),
}

/// A failed resumable-upload request: retried after reading the stored offset,
/// or fatal.
enum Failure {
    Retry(String),
    Fatal(anyhow::Error),
}

pub async fn connect(addr: &str, key: Option<&str>) -> Result<HttpClient> {
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
    let base = if addr.contains("://") {
        Url::parse(addr)?
    } else {
        Url::parse(&format!("http://{addr}"))?
    };
    ensure!(
        matches!(base.scheme(), "http" | "https"),
        "use an http:// or https:// server address"
    );
    ensure!(
        base.username().is_empty()
            && base.password().is_none()
            && base.query().is_none()
            && base.fragment().is_none(),
        "server address must not contain credentials, query parameters, or a fragment"
    );
    let mut headers = HeaderMap::new();
    if let Some(key) = key {
        ensure!(
            key.len() >= 32 && key.bytes().all(|b| b.is_ascii_graphic()),
            "API key must contain at least 32 printable ASCII characters without spaces"
        );
        let mut value = HeaderValue::from_str(&format!("Bearer {key}"))?;
        value.set_sensitive(true);
        headers.insert(AUTHORIZATION, value);
    }
    let client = reqwest::Client::builder()
        .default_headers(headers)
        .connect_timeout(Duration::from_secs(10))
        .redirect(reqwest::redirect::Policy::none())
        .build()?;
    Ok(HttpClient { client, base })
}

impl HttpClient {
    pub async fn request(
        &self,
        method: Method,
        parts: &[&str],
        body: Option<Vec<u8>>,
        headers: &[(&str, String)],
    ) -> Result<(StatusCode, HeaderMap, Vec<u8>)> {
        let (status, headers, bytes, _) = self
            .send(method, parts, body.map(Body::Bytes), headers, None)
            .await?;
        Ok((status, headers, bytes))
    }

    /// Uploads `file` with the tus protocol: `POST /uploads` creates the upload with
    /// its first chunk (a small file is one request), `PATCH /uploads/{id}` sends
    /// each further chunk. After any failure the client reads the stored offset with
    /// `HEAD` and continues from there, so a failure resends at most one chunk.
    pub async fn upload(&self, id: &str, spec: &UploadSpec, file: &File) -> Result<()> {
        let chunk = upload_chunk_bytes(spec.size);
        let metadata = format!(
            "id {},spec {}",
            STANDARD.encode(id),
            STANDARD.encode(serde_json::to_vec(spec)?)
        );
        let path = format!("/uploads/{id}");
        // None until the upload is known to exist.
        let mut offset: Option<u64> = None;
        let mut resync = false;
        let mut attempt = 0;
        let mut failing: Option<(Instant, Instant)> = None;
        loop {
            let result = if resync {
                match self.tus(Method::HEAD, &["uploads", id], &[], None).await {
                    Ok(stored) => {
                        offset = Some(stored);
                        resync = false;
                        continue;
                    }
                    Err(Failure::Retry(error)) if error.starts_with("TensorLane HTTP 404") => {
                        offset = None;
                        resync = false;
                        continue;
                    }
                    Err(failure) => Err(failure),
                }
            } else if let Some(sent) = offset {
                let length = chunk.min(spec.size - sent);
                let headers = [("Upload-Offset", sent.to_string())];
                let body = Some((file, sent, length));
                let result = self
                    .tus(Method::PATCH, &["uploads", id], &headers, body)
                    .await;
                match result {
                    // A conflict at an offset the server still reports is not a lost
                    // reply (for example a cancelled run).
                    Err(Failure::Retry(error)) if error.starts_with("TensorLane HTTP 409") => {
                        match self.tus(Method::HEAD, &["uploads", id], &[], None).await {
                            Ok(stored) if stored == sent => {
                                Err(Failure::Fatal(anyhow::anyhow!(error)))
                            }
                            Ok(stored) => {
                                offset = Some(stored);
                                continue;
                            }
                            Err(failure) => Err(failure),
                        }
                    }
                    result => result,
                }
            } else {
                let headers = [
                    ("Upload-Length", spec.size.to_string()),
                    ("Upload-Metadata", metadata.clone()),
                ];
                let body = Some((file, 0, chunk.min(spec.size)));
                self.tus(Method::POST, &["uploads"], &headers, body).await
            };
            match result {
                Ok(stored) => {
                    if let Some((since, _)) = failing.take() {
                        eprintln!("TensorLane {path} recovered after {:.0?}", since.elapsed());
                    }
                    attempt = 0;
                    if stored == spec.size {
                        return Ok(());
                    }
                    offset = Some(stored);
                }
                Err(Failure::Fatal(error)) => return Err(error),
                Err(Failure::Retry(error)) => {
                    let now = Instant::now();
                    if failing.is_none_or(|(_, logged)| now - logged >= Duration::from_secs(60)) {
                        let since = failing.map_or(now, |(since, _)| since);
                        eprintln!(
                            "TensorLane {path} failing for {:.0?}, retrying: {error}",
                            since.elapsed()
                        );
                        failing = Some((since, now));
                    }
                    let backoff = Duration::from_millis((100u64 << attempt.min(7)).min(10_000));
                    let jitter =
                        Duration::from_millis((uuid::Uuid::new_v4().as_u128() % 20) as u64);
                    sleep(backoff + jitter).await;
                    attempt += 1;
                    resync = true;
                }
            }
        }
    }

    /// One resumable-upload request, with `length` bytes of `file` from `start` as
    /// its body; returns the stored offset (`Upload-Offset`).
    async fn tus(
        &self,
        method: Method,
        parts: &[&str],
        headers: &[(&str, String)],
        body: Option<(&File, u64, u64)>,
    ) -> std::result::Result<u64, Failure> {
        let mut url = self.base.clone();
        url.path_segments_mut()
            .map_err(|_| Failure::Fatal(anyhow::anyhow!("invalid HTTP base address")))?
            .pop_if_empty()
            .extend(parts);
        let mut request = self
            .client
            .request(method.clone(), url)
            .header("Tus-Resumable", TUS_VERSION);
        for (name, value) in headers {
            request = request.header(*name, value);
        }
        if let Some((file, start, length)) = body {
            let mut input = file.try_clone().map_err(|e| Failure::Fatal(e.into()))?;
            input
                .seek(SeekFrom::Start(start))
                .map_err(|e| Failure::Fatal(e.into()))?;
            let stream = ReaderStream::new(tokio::fs::File::from_std(input).take(length));
            request = request
                .header(CONTENT_TYPE, "application/offset+octet-stream")
                .header(CONTENT_LENGTH, length)
                .body(reqwest::Body::wrap_stream(stream));
        }
        let response = request
            .send()
            .await
            .map_err(|error| Failure::Retry(format!("{error:#}")))?;
        let status = response.status();
        if status.is_success() {
            return response
                .headers()
                .get("Upload-Offset")
                .and_then(|v| v.to_str().ok()?.parse().ok())
                .ok_or_else(|| {
                    Failure::Fatal(anyhow::anyhow!("TensorLane upload response has no offset"))
                });
        }
        let bytes = response.bytes().await.unwrap_or_default();
        let message = serde_json::from_slice::<serde_json::Value>(&bytes)
            .ok()
            .and_then(|v| v["message"].as_str().map(str::to_owned))
            .unwrap_or_else(|| String::from_utf8_lossy(&bytes).into_owned());
        let error = format!("TensorLane HTTP {status}: {message}");
        // A PATCH conflict or missing upload is resolved by reading the stored offset;
        // a HEAD 404 means the upload must be created.
        let resolvable = match status {
            StatusCode::CONFLICT | StatusCode::NOT_FOUND => method == Method::PATCH,
            _ => false,
        } || (method == Method::HEAD && status == StatusCode::NOT_FOUND);
        if retryable(status) || resolvable {
            Err(Failure::Retry(error))
        } else {
            Err(Failure::Fatal(anyhow::anyhow!(error)))
        }
    }

    pub async fn batch(
        &self,
        parts: &[&str],
        memory: Arc<MemoryBudget>,
        sequence: u64,
    ) -> Result<(StatusCode, Vec<u8>, MemoryLease)> {
        let (status, _, bytes, lease) = self
            .send(Method::GET, parts, None, &[], Some((memory, sequence)))
            .await?;
        Ok((
            status,
            bytes,
            lease.context("batch response has no memory reservation")?,
        ))
    }

    pub async fn initialize(&self, run: &str) -> Result<InitResponse> {
        let json = [("content-type", "application/json".into())];
        let body = Some(b"{}".to_vec());
        let (_, _, bytes) = self
            .request(Method::POST, &["runs", run, "init"], body, &json)
            .await?;
        Ok(serde_json::from_slice(&bytes)?)
    }

    pub async fn end(&self, run: &str, failed: bool) -> Result<()> {
        let json = [("content-type", "application/json".into())];
        let body = Some(serde_json::to_vec(&EndRequest { failed })?);
        self.request(Method::POST, &["runs", run, "end"], body, &json)
            .await?;
        Ok(())
    }

    async fn send(
        &self,
        method: Method,
        parts: &[&str],
        body: Option<Body>,
        headers: &[(&str, String)],
        memory: Option<(Arc<MemoryBudget>, u64)>,
    ) -> Result<(StatusCode, HeaderMap, Vec<u8>, Option<MemoryLease>)> {
        ensure!(
            !parts.iter().any(|part| matches!(*part, "." | "..")),
            "HTTP path segment must not be . or .."
        );
        let mut url = self.base.clone();
        url.path_segments_mut()
            .map_err(|_| anyhow::anyhow!("invalid HTTP base address"))?
            .pop_if_empty()
            .extend(parts);
        let mut lease = None;
        let mut attempt = 0;
        let mut failing: Option<(Instant, Instant)> = None;
        loop {
            let mut request = self.client.request(method.clone(), url.clone());
            for (name, value) in headers {
                request = request.header(*name, value);
            }
            match &body {
                Some(Body::Bytes(bytes)) => request = request.body(bytes.clone()),
                None => {}
            }
            let (error, retry_after) = match request.send().await {
                Err(error) => (format!("{error:#}"), None),
                Ok(response) if response.status() == StatusCode::ACCEPTED => {
                    sleep(Duration::from_millis((25u64 << attempt.min(7)).min(2_000))).await;
                    attempt += 1;
                    continue;
                }
                Ok(response) if response.status().is_success() => {
                    let status = response.status();
                    let headers = response.headers().clone();
                    if lease.is_none()
                        && let Some((memory, sequence)) = &memory
                    {
                        let bytes = match status {
                            StatusCode::NO_CONTENT => 0,
                            _ => response.content_length().unwrap_or(u64::MAX),
                        };
                        let bytes = usize::try_from(bytes).unwrap_or(usize::MAX);
                        lease = Some(memory.acquire(*sequence, bytes).await?);
                    }
                    match response.bytes().await {
                        Ok(bytes) => {
                            if let Some((since, _)) = failing {
                                eprintln!(
                                    "TensorLane {} recovered after {:.0?}",
                                    url.path(),
                                    since.elapsed()
                                );
                            }
                            return Ok((status, headers, bytes.into(), lease));
                        }
                        Err(error) => (format!("{error:#}"), None),
                    }
                }
                Ok(response) => {
                    let status = response.status();
                    let retry_after = response
                        .headers()
                        .get("retry-after")
                        .and_then(|v| v.to_str().ok()?.parse::<u64>().ok())
                        .map(|v| Duration::from_secs(v.min(30)));
                    let bytes = response.bytes().await.unwrap_or_default();
                    let message = serde_json::from_slice::<serde_json::Value>(&bytes)
                        .ok()
                        .and_then(|v| v["message"].as_str().map(str::to_owned))
                        .unwrap_or_else(|| String::from_utf8_lossy(&bytes).into_owned());
                    if !retryable(status) {
                        bail!("TensorLane HTTP {status}: {message}");
                    }
                    (format!("TensorLane HTTP {status}: {message}"), retry_after)
                }
            };
            let now = Instant::now();
            if failing.is_none_or(|(_, logged)| now - logged >= Duration::from_secs(60)) {
                let since = failing.map_or(now, |(since, _)| since);
                eprintln!(
                    "TensorLane {} failing for {:.0?}, retrying: {error}",
                    url.path(),
                    since.elapsed()
                );
                failing = Some((since, now));
            }
            let backoff = Duration::from_millis((100u64 << attempt.min(7)).min(10_000));
            let jitter = Duration::from_millis((uuid::Uuid::new_v4().as_u128() % 20) as u64);
            sleep(retry_after.unwrap_or(backoff) + jitter).await;
            attempt += 1;
        }
    }
}

/// Statuses worth retrying: timeouts, rate limits, server errors and 499 (a proxy
/// that dropped the request).
fn retryable(status: StatusCode) -> bool {
    status == StatusCode::REQUEST_TIMEOUT
        || status == StatusCode::TOO_MANY_REQUESTS
        || status.as_u16() == 499
        || status.is_server_error()
}
