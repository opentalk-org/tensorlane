use crate::semaphore::{MemoryBudget, MemoryLease};
use anyhow::{Context, Result, bail, ensure};
use futures_util::StreamExt;
use reqwest::{
    Method, StatusCode, Url,
    header::{AUTHORIZATION, HeaderMap, HeaderValue},
};
use serde::{Serialize, de::DeserializeOwned};
use std::{sync::Arc, time::Duration};
use tensorlane_protocol::{EndRequest, InitResponse, UploadSpec, UploadStatus};
use tokio::time::Instant;
use tokio::{fs::File, io::AsyncSeekExt};
use tokio_util::io::ReaderStream;

#[derive(Clone)]
pub struct HttpClient {
    client: reqwest::Client,
    base: Url,
    retry_timeout: Duration,
}

enum RequestBody<'a> {
    Bytes(Vec<u8>),
    Upload(&'a UploadSpec, &'a File),
}

pub async fn connect(addr: &str, key: Option<&str>) -> Result<HttpClient> {
    static PROVIDER: std::sync::Once = std::sync::Once::new();
    PROVIDER.call_once(|| {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
    });
    let url = if addr.contains("://") {
        addr.to_owned()
    } else {
        format!("http://{addr}")
    };
    let base = Url::parse(&url)?;
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
    let retry_seconds = std::env::var("TENSORLANE_RETRY_TIMEOUT_SECONDS")
        .unwrap_or_else(|_| "600".into())
        .parse()
        .context("invalid TENSORLANE_RETRY_TIMEOUT_SECONDS")?;
    Ok(HttpClient {
        client: reqwest::Client::builder()
            .default_headers(headers)
            .connect_timeout(Duration::from_secs(10))
            .redirect(reqwest::redirect::Policy::none())
            .build()?,
        base,
        retry_timeout: Duration::from_secs(retry_seconds),
    })
}

impl HttpClient {
    fn url(&self, parts: &[&str]) -> Result<Url> {
        ensure!(
            !parts.iter().any(|part| matches!(*part, "." | "..")),
            "HTTP path segment must not be . or .."
        );
        let mut url = self.base.clone();
        url.path_segments_mut()
            .map_err(|_| anyhow::anyhow!("invalid HTTP base address"))?
            .pop_if_empty()
            .extend(parts);
        Ok(url)
    }

    pub async fn request(
        &self,
        method: Method,
        parts: &[&str],
        body: Option<Vec<u8>>,
        headers: &[(&str, String)],
        limit: usize,
    ) -> Result<(StatusCode, HeaderMap, Vec<u8>)> {
        let (status, headers, bytes, _) = self
            .request_inner(
                method,
                parts,
                body.map(RequestBody::Bytes),
                headers,
                limit,
                None,
            )
            .await?;
        Ok((status, headers, bytes))
    }

    pub async fn upload(&self, id: &str, spec: &UploadSpec, file: &File) -> Result<UploadStatus> {
        let (_, _, bytes, _) = self
            .request_inner(
                Method::PUT,
                &["uploads", id],
                Some(RequestBody::Upload(spec, file)),
                &[],
                1024 * 1024,
                None,
            )
            .await?;
        Ok(serde_json::from_slice(&bytes)?)
    }

    pub async fn batch(
        &self,
        parts: &[&str],
        memory: Arc<MemoryBudget>,
        sequence: u64,
    ) -> Result<(StatusCode, Vec<u8>, MemoryLease)> {
        let (status, _, bytes, lease) = self
            .request_inner(
                Method::GET,
                parts,
                None,
                &[],
                crate::MAX_BATCH_BYTES,
                Some((memory, sequence)),
            )
            .await?;
        Ok((
            status,
            bytes,
            lease.context("batch response has no memory reservation")?,
        ))
    }

    async fn request_inner(
        &self,
        method: Method,
        parts: &[&str],
        body: Option<RequestBody<'_>>,
        headers: &[(&str, String)],
        limit: usize,
        memory: Option<(Arc<MemoryBudget>, u64)>,
    ) -> Result<(StatusCode, HeaderMap, Vec<u8>, Option<MemoryLease>)> {
        let mut lease = None;
        let url = self.url(parts)?;
        let mut started = Instant::now();
        let mut attempt = 0;
        loop {
            let remaining = if self.retry_timeout.is_zero() {
                Duration::from_secs(120)
            } else {
                self.retry_timeout
                    .saturating_sub(started.elapsed())
                    .max(Duration::from_millis(1))
            };
            let mut request = self.client.request(method.clone(), url.clone());
            if memory.is_none()
                && !(self.retry_timeout.is_zero() && matches!(&body, Some(RequestBody::Upload(..))))
            {
                let timeout = if matches!(&body, Some(RequestBody::Upload(..))) {
                    remaining
                } else {
                    remaining.min(Duration::from_secs(120))
                };
                request = request.timeout(timeout);
            }
            if let Some(body) = &body {
                request = match body {
                    RequestBody::Bytes(bytes) => request.body(bytes.clone()),
                    RequestBody::Upload(spec, file) => {
                        let mut input = file.try_clone().await?;
                        input.seek(std::io::SeekFrom::Start(0)).await?;
                        let part = reqwest::multipart::Part::stream_with_length(
                            reqwest::Body::wrap_stream(ReaderStream::new(input)),
                            spec.size,
                        )
                        .file_name("file");
                        request.multipart(
                            reqwest::multipart::Form::new()
                                .text("spec", serde_json::to_string(spec)?)
                                .part("file", part),
                        )
                    }
                };
            }
            for (name, value) in headers {
                request = request.header(*name, value);
            }
            let result: Result<_> = async {
                // Reqwest's read timeout also covers sending the request body.
                let response = if matches!(&body, Some(RequestBody::Upload(..))) {
                    request.send().await?
                } else {
                    tokio::time::timeout(Duration::from_secs(30), request.send()).await??
                };
                let status = response.status();
                let headers = response.headers().clone();
                let capacity = response.content_length().unwrap_or(0).min(limit as u64) as usize;
                if status.is_success()
                    && status != StatusCode::ACCEPTED
                    && lease.is_none()
                    && let Some((memory, sequence)) = &memory
                {
                    let bytes = if status == StatusCode::NO_CONTENT {
                        0
                    } else {
                        response.content_length().unwrap_or(limit as u64)
                    };
                    ensure!(
                        bytes <= limit as u64,
                        "HTTP response exceeds its size limit"
                    );
                    let waiting = Instant::now();
                    lease = Some(memory.acquire(*sequence, bytes as usize).await?);
                    started += waiting.elapsed();
                }
                let receive = async {
                    let mut stream = response.bytes_stream();
                    let mut bytes = Vec::with_capacity(capacity);
                    while let Some(chunk) =
                        tokio::time::timeout(Duration::from_secs(30), stream.next()).await?
                    {
                        let chunk = chunk?;
                        ensure!(
                            bytes
                                .len()
                                .checked_add(chunk.len())
                                .is_some_and(|size| size <= limit),
                            "HTTP response exceeds its size limit"
                        );
                        bytes.extend_from_slice(&chunk);
                    }
                    Ok((status, headers, bytes))
                };
                tokio::time::timeout(remaining.min(Duration::from_secs(120)), receive).await?
            }
            .await;
            let (error, retry_after) = match result {
                Ok((status, headers, bytes))
                    if status != StatusCode::ACCEPTED && status.is_success() =>
                {
                    return Ok((status, headers, bytes, lease));
                }
                Ok((status, headers, bytes)) => {
                    let message = serde_json::from_slice::<serde_json::Value>(&bytes)
                        .ok()
                        .and_then(|v| v["message"].as_str().map(str::to_owned))
                        .unwrap_or_else(|| String::from_utf8_lossy(&bytes).into_owned());
                    let retryable = status == StatusCode::ACCEPTED
                        || status == StatusCode::REQUEST_TIMEOUT
                        || status == StatusCode::TOO_MANY_REQUESTS
                        || status.is_server_error();
                    if !retryable {
                        bail!("TensorLane HTTP {status}: {message}");
                    }
                    let delay = headers
                        .get("retry-after")
                        .and_then(|v| v.to_str().ok())
                        .and_then(|v| v.parse::<u64>().ok())
                        .map(|v| Duration::from_secs(v.min(30)))
                        .or_else(|| {
                            (status == StatusCode::ACCEPTED).then_some(Duration::from_millis(25))
                        });
                    (format!("TensorLane HTTP {status}: {message}"), delay)
                }
                Err(error) => (format!("{error:#}"), None),
            };
            ensure!(
                self.retry_timeout.is_zero() || started.elapsed() < self.retry_timeout,
                "HTTP recovery deadline exceeded: {error}"
            );
            let delay = retry_after
                .unwrap_or_else(|| Duration::from_millis((100u64 << attempt.min(7)).min(10_000)));
            let jitter = Duration::from_millis((uuid::Uuid::new_v4().as_u128() % 20) as u64);
            let wait = delay + jitter;
            let wait = if self.retry_timeout.is_zero() {
                wait
            } else {
                wait.min(self.retry_timeout.saturating_sub(started.elapsed()))
            };
            tokio::time::sleep(wait).await;
            attempt += 1;
        }
    }

    pub async fn json<T: DeserializeOwned, B: Serialize>(
        &self,
        method: Method,
        parts: &[&str],
        body: &B,
    ) -> Result<T> {
        let (_, _, bytes) = self
            .request(
                method,
                parts,
                Some(serde_json::to_vec(body)?),
                &[("content-type", "application/json".into())],
                8 * 1024 * 1024,
            )
            .await?;
        Ok(serde_json::from_slice(&bytes)?)
    }

    pub async fn initialize(&self, run: &str) -> Result<InitResponse> {
        self.json(Method::POST, &["runs", run, "init"], &serde_json::json!({}))
            .await
    }

    pub async fn end(&self, run: &str, failed: bool) -> Result<()> {
        let mut ending = self.clone();
        ending.retry_timeout = Duration::from_secs(5);
        ending
            .request(
                Method::POST,
                &["runs", run, "end"],
                Some(serde_json::to_vec(&EndRequest { failed })?),
                &[("content-type", "application/json".into())],
                1024 * 1024,
            )
            .await?;
        Ok(())
    }
}
