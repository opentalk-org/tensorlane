use crate::semaphore::{MemoryBudget, MemoryLease};
use anyhow::{Context, Result, bail, ensure};
use reqwest::{
    Method, StatusCode, Url,
    header::{AUTHORIZATION, HeaderMap, HeaderValue},
    multipart::{Form, Part},
};
use std::{fs::File, io::Seek, sync::Arc, time::Duration};
use tensorlane_protocol::{EndRequest, InitResponse, UploadSpec};
use tokio::time::{Instant, sleep};
use tokio_util::io::ReaderStream;

#[derive(Clone)]
pub struct HttpClient {
    client: reqwest::Client,
    base: Url,
}

enum Body<'a> {
    Bytes(Vec<u8>),
    Upload(&'a UploadSpec, &'a File),
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

    pub async fn upload(&self, id: &str, spec: &UploadSpec, file: &File) -> Result<()> {
        let body = Some(Body::Upload(spec, file));
        self.send(Method::PUT, &["uploads", id], body, &[], None)
            .await?;
        Ok(())
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
        body: Option<Body<'_>>,
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
                Some(Body::Upload(spec, file)) => {
                    let mut input = file.try_clone()?;
                    input.rewind()?;
                    let stream = ReaderStream::new(tokio::fs::File::from_std(input));
                    let part =
                        Part::stream_with_length(reqwest::Body::wrap_stream(stream), spec.size)
                            .file_name("file");
                    let form = Form::new().text("spec", serde_json::to_string(spec)?);
                    request = request.multipart(form.part("file", part));
                }
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
                    let retryable = status == StatusCode::REQUEST_TIMEOUT
                        || status == StatusCode::TOO_MANY_REQUESTS
                        || status.is_server_error();
                    if !retryable {
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
