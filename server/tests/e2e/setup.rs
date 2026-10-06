use super::services::{Services, port};
use anyhow::{Context, Result, ensure};
use aws_sdk_s3::{
    config::{BehaviorVersion, Credentials, Region},
    primitives::ByteStream,
};
use bytes::Bytes;
use prost::Message;
use reqwest::{Method, StatusCode};
use serde_json::{Value, json};
use std::{
    path::PathBuf,
    process::{Child, Command, Stdio},
    time::Duration,
};
use tempfile::TempDir;
use tensorlane_protocol::{DataResponse, InitResponse};
use uuid::Uuid;

pub const KEY: &str = "tensorlane-test-key-01234567890123456789";

pub async fn eventually<T, F, Fut>(description: &str, mut check: F) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<Option<T>>>,
{
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    loop {
        if let Some(value) = check()
            .await
            .with_context(|| format!("waiting for {description}"))?
        {
            return Ok(value);
        }
        ensure!(
            tokio::time::Instant::now() < deadline,
            "timed out waiting for {description}"
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

pub fn config(dataset: Uuid, batches: u64) -> Value {
    let sql = "SELECT sample_id, position AS batch_idx, position AS sample_idx, metadata_json, blobs_json FROM example_samples WHERE dataset_id={dataset:UUID} AND position < {count:UInt64} ORDER BY batch_idx, sample_idx";
    json!({"queries": {"training": {"sql": sql, "params": {"dataset":dataset, "count":batches}}, "validation": {"sql":sql, "params":{"dataset":dataset,"count":1},"repeat":true}}, "tensorlane":{"assets":{}}, "app":{"optimizer":{"lr":0.1}}})
}

pub struct TestEnv {
    pub database: clickhouse::Client,
    pub s3: aws_sdk_s3::Client,
    pub http: reqwest::Client,
    pub url: String,
    pub cache: PathBuf,
    server: Child,
    command: Command,
    _services: Services,
    _temp: TempDir,
}
impl TestEnv {
    pub async fn start() -> Result<Self> {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        let temp = tempfile::tempdir()?;
        let (services, database_url, s3_url) = Services::start(temp.path()).await?;
        let database = clickhouse::Client::default()
            .with_url(&database_url)
            .with_user("default")
            .with_password("test");
        eventually("ClickHouse", || async {
            Ok(database.query("SELECT 1").fetch_one::<u8>().await.ok())
        })
        .await?;
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()?;
        eventually("S3", || async {
            Ok(http
                .get(format!("{s3_url}/minio/health/live"))
                .send()
                .await
                .ok()
                .filter(|r| r.status().is_success()))
        })
        .await?;
        for sql in include_str!("schema.sql")
            .split(';')
            .filter(|sql| !sql.trim().is_empty())
        {
            database.query(sql).execute().await?;
        }
        let s3 = aws_sdk_s3::Client::from_conf(
            aws_sdk_s3::config::Builder::new()
                .behavior_version(BehaviorVersion::latest())
                .endpoint_url(&s3_url)
                .region(Region::new("us-east-1"))
                .credentials_provider(Credentials::new(
                    "minioadmin",
                    "minioadmin",
                    None,
                    None,
                    "test",
                ))
                .force_path_style(true)
                .build(),
        );
        s3.create_bucket().bucket("tensorlane-test").send().await?;
        let cache = temp.path().join("cache");
        let url = format!("http://127.0.0.1:{}", port()?);
        let mut command = Command::new(env!("CARGO_BIN_EXE_tensorlane"));
        command
            .args([
                "--clickhouse-url",
                &database_url,
                "--clickhouse-user",
                "default",
                "--clickhouse-password",
                "test",
                "--s3-endpoint",
                &s3_url,
                "--s3-region",
                "us-east-1",
                "--s3-key",
                "minioadmin",
                "--s3-secret",
                "minioadmin",
                "--bucket",
                "tensorlane-test",
                "--http-port",
                url.rsplit(':').next().unwrap(),
                "--checkpoint-prefix",
                "checkpoints",
                "--metrics-prefix",
                "metrics",
                "--api-key",
                KEY,
            ])
            .arg("--cache-dir")
            .arg(&cache)
            .env("RUST_LOG", "tensorlane=warn")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit());
        let server = command.spawn()?;
        let env = Self {
            database,
            s3,
            http,
            url,
            cache,
            server,
            command,
            _services: services,
            _temp: temp,
        };
        env.ready().await?;
        Ok(env)
    }
    pub fn request(&self, method: Method, path: &str) -> reqwest::RequestBuilder {
        self.request_at(&self.url, method, path)
    }
    pub fn request_at(&self, url: &str, method: Method, path: &str) -> reqwest::RequestBuilder {
        self.http
            .request(method, format!("{url}{path}"))
            .bearer_auth(KEY)
    }
    pub async fn ready(&self) -> Result<()> {
        self.ready_at(&self.url).await
    }
    pub async fn ready_at(&self, url: &str) -> Result<()> {
        eventually("server", || async {
            Ok(self
                .http
                .get(format!("{url}/readyz"))
                .send()
                .await
                .ok()
                .filter(|r| r.status().is_success()))
        })
        .await?;
        Ok(())
    }
    pub async fn restart(&mut self) -> Result<()> {
        self.restart_after(Duration::ZERO).await
    }
    pub async fn restart_after(&mut self, downtime: Duration) -> Result<()> {
        self.server.kill()?;
        self.server.wait()?;
        tokio::time::sleep(downtime).await;
        self.server = self.command.spawn()?;
        self.ready().await
    }
    pub fn replica(&mut self) -> Result<(Child, String)> {
        let url = format!("http://127.0.0.1:{}", port()?);
        let mut command = Command::new(self.command.get_program());
        let mut args: Vec<_> = self.command.get_args().map(|a| a.to_os_string()).collect();
        let index = args.iter().position(|a| a == "--http-port").unwrap();
        args[index + 1] = url.rsplit(':').next().unwrap().into();
        let cache = self
            .cache
            .parent()
            .unwrap()
            .join(format!("replica-{}", url.rsplit(':').next().unwrap()));
        let index = args.iter().position(|a| a == "--cache-dir").unwrap();
        args[index + 1] = cache.into_os_string();
        command
            .args(args)
            .env("RUST_LOG", "tensorlane=warn")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit());
        let child = command.spawn()?;
        Ok((child, url))
    }
    pub async fn create_run(&self, config: Value) -> Result<String> {
        let project = Uuid::new_v4();
        self.database
            .query("INSERT INTO projects VALUES (?,'test','',now64(6),now64(6))")
            .bind(project)
            .execute()
            .await?;
        let response: Value = self
            .request(Method::POST, "/runs")
            .json(&json!({"project_id":project,"name":"test","config":config}))
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        Ok(response["run_id"].as_str().unwrap().into())
    }
    pub async fn init(&self, id: &str) -> Result<InitResponse> {
        Ok(self
            .request(Method::POST, &format!("/runs/{id}/init"))
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?)
    }
    pub async fn batch(
        &self,
        id: &str,
        stream: &str,
        sequence: u64,
    ) -> Result<Option<DataResponse>> {
        self.batch_at(&self.url, id, stream, sequence).await
    }
    pub async fn batch_at(
        &self,
        url: &str,
        id: &str,
        stream: &str,
        sequence: u64,
    ) -> Result<Option<DataResponse>> {
        eventually("batch", || async {
            let response = self
                .request_at(
                    url,
                    Method::GET,
                    &format!("/runs/{id}/streams/{stream}/batches/{sequence}"),
                )
                .send()
                .await?;
            match response.status() {
                StatusCode::ACCEPTED => Ok(None),
                StatusCode::NO_CONTENT => Ok(Some(None)),
                _ => Ok(Some(Some(DataResponse::decode(
                    response.error_for_status()?.bytes().await?,
                )?))),
            }
        })
        .await
    }
    pub async fn put_object(&self, key: &str, bytes: Bytes) -> Result<()> {
        self.s3
            .put_object()
            .bucket("tensorlane-test")
            .key(key)
            .body(ByteStream::from(bytes))
            .send()
            .await?;
        Ok(())
    }
    pub async fn seed(&self, count: usize) -> Result<Uuid> {
        let id = Uuid::new_v4();
        self.put_object(&format!("datasets/{id}"), Bytes::from_static(b"payload"))
            .await?;
        for position in 0..count {
            self.database
                .query("INSERT INTO example_samples VALUES (?,?,?,?,?)")
                .bind(id)
                .bind(position as u64)
                .bind(format!("sample-{position}"))
                .bind(json!({"position":position}).to_string())
                .bind(json!({"payload":{"object":format!("datasets/{id}")}}).to_string())
                .execute()
                .await?;
        }
        Ok(id)
    }
    pub async fn status(&self, id: &str) -> Result<String> {
        Ok(self
            .database
            .query(
                "SELECT toString(argMax(status,timestamp)) FROM run_status WHERE run_id=toUUID(?)",
            )
            .bind(id)
            .fetch_one()
            .await?)
    }
}
impl Drop for TestEnv {
    fn drop(&mut self) {
        let _ = self.server.kill();
        let _ = self.server.wait();
    }
}
