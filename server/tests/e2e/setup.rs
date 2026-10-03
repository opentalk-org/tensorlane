use crate::proto::{
    self, metrics_request, save_asset_request, tensor_lane_client::TensorLaneClient,
};
use anyhow::{Context, Result, ensure};
use aws_sdk_s3::{
    config::{BehaviorVersion, Credentials, Region},
    primitives::ByteStream,
};
use bytes::{Bytes, BytesMut};
use serde_json::{Value, json};
use std::{
    collections::BTreeSet,
    net::TcpListener,
    path::{Path, PathBuf},
    process::{Child, Command, ExitStatus, Stdio},
    time::Duration,
};
use tempfile::TempDir;
use testcontainers_modules::{
    clickhouse::ClickHouse,
    minio::MinIO,
    testcontainers::{
        ContainerAsync, ImageExt,
        core::{IntoContainerPort, WaitFor, wait::HttpWaitStrategy},
        runners::AsyncRunner,
    },
};
use tokio::{sync::mpsc, task::JoinHandle, time::sleep};
use tokio_stream::wrappers::ReceiverStream;
use tonic::transport::{Channel, Endpoint};
use uuid::Uuid;

pub const TIMESTAMP_MS: i64 = 1_700_000_000_123;
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
        sleep(Duration::from_millis(25)).await;
    }
}
pub fn run_config(dataset_id: Uuid, batches: u64) -> Value {
    let sql = "WITH pool AS (SELECT * FROM example_samples WHERE dataset_id = {dataset_id:UUID})
        SELECT p.sample_id AS sample_id, toUInt64(intDiv(n.number, {batch_size:UInt64})) AS batch_idx,
        toUInt64(n.number) AS sample_idx, p.metadata_json AS metadata_json, p.blobs_json AS blobs_json
        FROM numbers({batches:UInt64} * {batch_size:UInt64}) AS n CROSS JOIN pool AS p
        WHERE p.position = modulo(n.number + {dataset_offset:UInt64}, (SELECT count() FROM pool))
        ORDER BY batch_idx, sample_idx";
    json!({"dataset_id":dataset_id,"asset_type":"e2e-model","seed":1,
        "queries":{"training":sql,"validation":sql},"params":{"dataset_offset":0,"batch_size":1},
        "training":{"batches":batches},"validation":{"batches":1},"assets":{},"optimizer":{"lr":0.1}})
}
pub fn tar_file(name: &str, content: &[u8]) -> Result<Bytes> {
    let mut builder = tar::Builder::new(Vec::new());
    let mut header = tar::Header::new_gnu();
    header.set_size(content.len() as u64);
    header.set_mode(0o644);
    header.set_cksum();
    builder.append_data(&mut header, name, content)?;
    Ok(Bytes::from(builder.into_inner()?))
}
pub fn scalar(step: u64) -> proto::MetricsRequest {
    proto::MetricsRequest {
        payload: Some(metrics_request::Payload::Metric(proto::ScalarMetric {
            step,
            timestamp_unix_ms: TIMESTAMP_MS + step as i64,
            name: format!("loss/{}", step % 2),
            value: step as f32 / 4.0,
        })),
    }
}
pub fn array(step: u64) -> proto::MetricsRequest {
    proto::MetricsRequest {
        payload: Some(metrics_request::Payload::ArrayMetric(proto::ArrayMetric {
            step,
            timestamp_unix_ms: TIMESTAMP_MS + step as i64,
            name: "activations".into(),
            value: vec![step as f32, -1.0],
        })),
    }
}
pub fn files(path: &Path) -> Result<BTreeSet<PathBuf>> {
    Ok(std::fs::read_dir(path)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<std::io::Result<_>>()?)
}
fn port() -> Result<u16> {
    Ok(TcpListener::bind("127.0.0.1:0")?.local_addr()?.port())
}
#[derive(Default)]
struct Services {
    children: Vec<Child>,
    clickhouse: Option<ContainerAsync<ClickHouse>>,
    s3: Option<ContainerAsync<MinIO>>,
}
impl Drop for Services {
    fn drop(&mut self) {
        for child in &mut self.children {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}
pub struct TestEnv {
    pub clickhouse: clickhouse::Client,
    pub s3: aws_sdk_s3::Client,
    pub http: reqwest::Client,
    pub grpc: TensorLaneClient<Channel>,
    pub http_url: String,
    pub grpc_url: String,
    pub cache_dir: PathBuf,
    pub bucket: &'static str,
    pub server: Child,
    pub server_command: Command,
    services: Services,
    _temp: TempDir,
}
impl TestEnv {
    pub async fn start() -> Result<Self> {
        Self::start_with_history(None).await
    }
    pub async fn start_with_history(history: Option<(Uuid, Value, Value)>) -> Result<Self> {
        let temp = tempfile::tempdir()?;
        let mut services = Services::default();
        let (clickhouse_url, s3_url) = if std::env::var_os("TENSORLANE_TEST_LOCAL").is_some() {
            let http_port = port()?;
            let tcp_port = port()?;
            let s3_port = port()?;
            let config_path = temp.path().join("clickhouse.xml");
            std::fs::write(
                &config_path,
                format!(
                    "<clickhouse><logger><level>error</level><log>{root}/clickhouse.log</log><errorlog>{root}/clickhouse-error.log</errorlog></logger><path>{root}/ch/</path><tmp_path>{root}/tmp/</tmp_path><listen_host>127.0.0.1</listen_host><http_port>{http_port}</http_port><tcp_port>{tcp_port}</tcp_port><background_pool_size>16</background_pool_size><background_schedule_pool_size>4</background_schedule_pool_size><background_message_broker_schedule_pool_size>2</background_message_broker_schedule_pool_size><background_distributed_schedule_pool_size>2</background_distributed_schedule_pool_size><profiles><default><max_threads>4</max_threads><max_memory_usage>1000000000</max_memory_usage></default></profiles><users><default><password>test</password><networks><ip>127.0.0.1</ip></networks><profile>default</profile><quota>default</quota><access_management>1</access_management></default></users><quotas><default/></quotas></clickhouse>",
                    root = temp.path().display()
                ),
            )?;
            let clickhouse =
                std::env::var("TENSORLANE_TEST_CLICKHOUSE").unwrap_or_else(|_| "clickhouse".into());
            services.children.push(
                Command::new(clickhouse)
                    .args(["server", "--config-file"])
                    .arg(&config_path)
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .spawn()?,
            );
            let minio = std::env::var("TENSORLANE_TEST_MINIO").unwrap_or_else(|_| "minio".into());
            services.children.push(
                Command::new(minio)
                    .arg("server")
                    .arg(temp.path().join("minio"))
                    .args([
                        "--address",
                        &format!("127.0.0.1:{s3_port}"),
                        "--console-address",
                        "127.0.0.1:0",
                    ])
                    .env("MINIO_ROOT_USER", "minioadmin")
                    .env("MINIO_ROOT_PASSWORD", "minioadmin")
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .spawn()?,
            );
            (
                format!("http://127.0.0.1:{http_port}"),
                format!("http://127.0.0.1:{s3_port}"),
            )
        } else {
            let ch = ClickHouse::default()
                .with_tag("26.7.4.58")
                .with_env_var("CLICKHOUSE_USER", "default")
                .with_env_var("CLICKHOUSE_PASSWORD", "test")
                .with_ready_conditions(vec![WaitFor::http(
                    HttpWaitStrategy::new("/ping")
                        .with_port(8123.tcp())
                        .with_expected_status_code(200_u16),
                )])
                .start()
                .await?;
            let s3 = MinIO::default()
                .with_name("quay.io/minio/minio")
                .with_env_var("MINIO_ROOT_USER", "minioadmin")
                .with_env_var("MINIO_ROOT_PASSWORD", "minioadmin")
                .start()
                .await?;
            let urls = (
                format!(
                    "http://{}:{}",
                    ch.get_host().await?,
                    ch.get_host_port_ipv4(8123).await?
                ),
                format!(
                    "http://{}:{}",
                    s3.get_host().await?,
                    s3.get_host_port_ipv4(9000).await?
                ),
            );
            services.clickhouse = Some(ch);
            services.s3 = Some(s3);
            urls
        };
        let clickhouse = clickhouse::Client::default()
            .with_url(&clickhouse_url)
            .with_user("default")
            .with_password("test");
        eventually("ClickHouse startup", || async {
            Ok(clickhouse.query("SELECT 1").fetch_one::<u8>().await.ok())
        })
        .await?;
        let http = reqwest::Client::new();
        eventually("MinIO startup", || async {
            Ok(http
                .get(format!("{s3_url}/minio/health/live"))
                .send()
                .await
                .ok()
                .filter(|response| response.status().is_success())
                .map(|_| ()))
        })
        .await?;
        let migrations = Path::new(env!("CARGO_MANIFEST_DIR")).join("../db/migrations");
        let mut paths = std::fs::read_dir(migrations)?
            .map(|entry| entry.map(|entry| entry.path()))
            .collect::<std::io::Result<Vec<_>>>()?;
        paths.retain(|path| path.extension().is_some_and(|extension| extension == "sql"));
        paths.sort();
        for path in paths {
            for statement in std::fs::read_to_string(&path)?
                .lines()
                .filter(|line| !line.trim_start().starts_with("--"))
                .collect::<Vec<_>>()
                .join("\n")
                .split(';')
            {
                if !statement.trim().is_empty() {
                    clickhouse
                        .query(statement)
                        .execute()
                        .await
                        .with_context(|| format!("applying {}", path.display()))?;
                }
            }
            if path.file_name().unwrap() == "20260907153221.sql"
                && let Some((id, data, training)) = &history
            {
                clickhouse.query("INSERT INTO runs (id,project_id,name,data_config,train_config) VALUES (?,?,?,?,?)")
                        .bind(*id).bind(Uuid::new_v4()).bind("legacy").bind(data.to_string()).bind(training.to_string()).execute().await?;
            }
        }
        clickhouse.query("CREATE TABLE example_samples (dataset_id UUID, position UInt64, sample_id String, metadata_json String, blobs_json String) ENGINE=MergeTree ORDER BY (dataset_id,position)").execute().await?;
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
                    "e2e",
                ))
                .force_path_style(true)
                .build(),
        );
        let bucket = "tensorlane-test";
        s3.create_bucket().bucket(bucket).send().await?;
        let cache_dir = temp.path().join("cache");
        let http_url = format!("http://127.0.0.1:{}", port()?);
        let grpc_url = format!("http://127.0.0.1:{}", port()?);
        let mut server_command = Command::new(env!("CARGO_BIN_EXE_tensorlane"));
        server_command
            .args([
                "--clickhouse-url",
                &clickhouse_url,
                "--clickhouse-user",
                "default",
                "--clickhouse-password",
                "test",
                "--s3-endpoint",
                &s3_url,
                "--s3-key",
                "minioadmin",
                "--s3-secret",
                "minioadmin",
                "--bucket",
                bucket,
                "--http-port",
                http_url.rsplit(':').next().unwrap(),
                "--grpc-port",
                grpc_url.rsplit(':').next().unwrap(),
                "--checkpoint-prefix",
                "checkpoints",
                "--metrics-prefix",
                "metrics",
            ])
            .arg("--cache-dir")
            .arg(&cache_dir)
            .env("RUST_LOG", "tensorlane=warn")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit());
        let server = server_command.spawn()?;
        let endpoint = Endpoint::from_shared(grpc_url.clone())?;
        let mut env = Self {
            clickhouse,
            s3,
            http,
            grpc: TensorLaneClient::new(endpoint.connect_lazy())
                .max_decoding_message_size(64 * 1024 * 1024),
            http_url,
            grpc_url,
            cache_dir,
            bucket,
            server,
            server_command,
            services,
            _temp: temp,
        };
        eventually("server startup", || async {
            Ok(env
                .http
                .get(format!("{}/runs", env.http_url))
                .send()
                .await
                .ok()
                .filter(|response| response.status().is_success())
                .map(|_| ()))
        })
        .await?;
        env.grpc = TensorLaneClient::new(endpoint.connect().await?)
            .max_decoding_message_size(64 * 1024 * 1024);
        Ok(env)
    }
    pub async fn seed_dataset(&self, count: usize) -> Result<Uuid> {
        let id = Uuid::new_v4();
        let key = format!("datasets/{id}/pack");
        let packed: Vec<u8> = (0..count * 16).map(|index| (index % 251) as u8).collect();
        self.put_object(&key, Bytes::from(packed)).await?;
        self.put_object(
            &format!("datasets/{id}/context"),
            Bytes::from_static(b"arbitrary context bytes"),
        )
        .await?;
        for position in 0..count {
            let metadata=json!({"position":position,"label":format!("sample-{position}"),"nested":{"items":[1,2,3]}}).to_string();
            let blobs=json!({"payload":{"object":key,"byte_offset":position*16,"byte_length":16},"context":{"object":format!("datasets/{id}/context")}}).to_string();
            self.clickhouse.query("INSERT INTO example_samples (dataset_id,position,sample_id,metadata_json,blobs_json) VALUES (?,?,?,?,?)")
                .bind(id).bind(position as u64).bind(format!("sample-{position}")).bind(metadata).bind(blobs).execute().await?;
        }
        Ok(id)
    }
    pub async fn create_project(&self) -> Result<Uuid> {
        let id = Uuid::new_v4();
        self.clickhouse
            .query("INSERT INTO projects (id,name,description,created_at,updated_at) VALUES (?,'e2e','',now64(6),now64(6))")
            .bind(id)
            .execute()
            .await?;
        Ok(id)
    }
    pub async fn create_run(&self, config: Value) -> Result<String> {
        let project_id = self.create_project().await?;
        let response = self
            .http
            .post(format!("{}/runs", self.http_url))
            .json(&json!({"project_id":project_id,"name":"e2e","config":config}))
            .send()
            .await?
            .error_for_status()?;
        let value: Value = response.json().await?;
        Ok(value["run_id"].as_str().unwrap().to_owned())
    }
    pub async fn init_run(&self, id: &str) -> Result<proto::InitResponse> {
        Ok(self
            .grpc
            .clone()
            .init(proto::InitRequest { run_id: id.into() })
            .await?
            .into_inner())
    }
    pub async fn end_run(&self, id: &str) -> Result<()> {
        self.grpc
            .clone()
            .end(proto::EndRequest {
                run_id: id.into(),
                failed: false,
            })
            .await?;
        Ok(())
    }
    pub async fn batches(
        &self,
        id: &str,
        validation: bool,
        count: usize,
    ) -> Result<Vec<proto::DataResponse>> {
        self.stream_batches(
            id,
            if validation { "validation" } else { "training" },
            count,
        )
        .await
    }
    pub async fn stream_batches(
        &self,
        id: &str,
        name: &str,
        count: usize,
    ) -> Result<Vec<proto::DataResponse>> {
        let requests = (0..count)
            .map(|_| proto::DataRequest {
                run_id: id.into(),
                stream: name.into(),
            })
            .collect::<Vec<_>>();
        let mut stream = self
            .grpc
            .clone()
            .data(tokio_stream::iter(requests))
            .await?
            .into_inner();
        let mut batches = Vec::new();
        while let Some(batch) = stream.message().await? {
            batches.push(batch);
        }
        Ok(batches)
    }
    pub async fn put_object(&self, key: &str, body: Bytes) -> Result<()> {
        self.s3
            .put_object()
            .bucket(self.bucket)
            .key(key)
            .body(ByteStream::from(body))
            .send()
            .await?;
        Ok(())
    }
    pub async fn object(&self, key: &str) -> Result<Bytes> {
        Ok(self
            .s3
            .get_object()
            .bucket(self.bucket)
            .key(key)
            .send()
            .await?
            .body
            .collect()
            .await?
            .into_bytes())
    }
    pub async fn asset(&self, id: &str, name: &str) -> Result<(proto::AssetMetadata, Bytes)> {
        let mut stream = self
            .grpc
            .clone()
            .asset(proto::AssetRequest {
                run_id: id.into(),
                name: name.into(),
            })
            .await?
            .into_inner();
        let Some(proto::asset_response::Payload::Metadata(metadata)) =
            stream.message().await?.context("missing metadata")?.payload
        else {
            anyhow::bail!("expected metadata")
        };
        let mut bytes = BytesMut::new();
        while let Some(message) = stream.message().await? {
            let Some(proto::asset_response::Payload::Chunk(chunk)) = message.payload else {
                anyhow::bail!("expected chunk")
            };
            bytes.extend_from_slice(&chunk);
        }
        Ok((metadata, bytes.freeze()))
    }
    pub async fn checkpoint(&self, id: &str, step: u64, body: Bytes) -> Result<String> {
        self.save_asset(id, "model", step, "checkpoint", Uuid::new_v4(), body)
            .await
    }
    pub async fn save_asset(
        &self,
        id: &str,
        name: &str,
        step: u64,
        kind: &str,
        asset_id: Uuid,
        body: Bytes,
    ) -> Result<String> {
        let mut requests = vec![proto::SaveAssetRequest {
            payload: Some(save_asset_request::Payload::Metadata(
                proto::SaveAssetMetadata {
                    content_type: "application/x-tar".into(),
                    run_id: id.into(),
                    asset_id: asset_id.to_string(),
                    name: name.into(),
                    step,
                    kind: kind.into(),
                    asset_type: None,
                    metadata_json: "{}".into(),
                },
            )),
        }];
        for chunk in body.chunks(256 * 1024) {
            requests.push(proto::SaveAssetRequest {
                payload: Some(save_asset_request::Payload::Chunk(Bytes::copy_from_slice(
                    chunk,
                ))),
            });
        }
        Ok(self
            .grpc
            .clone()
            .save_asset(tokio_stream::iter(requests))
            .await?
            .into_inner()
            .asset_id)
    }
    pub async fn metrics_stream(&self, id: &str) -> Result<MetricsStream> {
        let (sender, receiver) = mpsc::channel(32);
        sender
            .send(proto::MetricsRequest {
                payload: Some(metrics_request::Payload::Metadata(
                    proto::MetricsStreamMetadata {
                        run_id: id.into(),
                        automatic: false,
                    },
                )),
            })
            .await?;
        let mut client = self.grpc.clone();
        let response = tokio::spawn(async move {
            Ok(client
                .metrics(ReceiverStream::new(receiver))
                .await?
                .into_inner())
        });
        Ok(MetricsStream { sender, response })
    }
    pub async fn count(&self, table: &str, id: &str) -> Result<u64> {
        ensure!(["metrics", "array_metrics", "artifacts", "assets"].contains(&table));
        Ok(self
            .clickhouse
            .query(&format!(
                "SELECT count() FROM {table} WHERE run_id=toUUID(?)"
            ))
            .bind(id)
            .fetch_one()
            .await?)
    }
    pub async fn wait_count(&self, table: &str, id: &str, expected: u64) -> Result<()> {
        eventually("row count", || async {
            Ok((self.count(table, id).await? == expected).then_some(()))
        })
        .await
    }
    pub async fn status(&self, id: &str) -> Result<String> {
        Ok(self
            .clickhouse
            .query(
                "SELECT toString(argMax(status,timestamp)) FROM run_status WHERE run_id=toUUID(?)",
            )
            .bind(id)
            .fetch_one()
            .await?)
    }
    pub fn run_cache(&self, id: &str) -> PathBuf {
        self.cache_dir.join("runs").join(id)
    }
    pub fn signal(&mut self, signal: &str) -> Result<()> {
        ensure!(
            Command::new("kill")
                .args([signal, &self.server.id().to_string()])
                .status()?
                .success()
        );
        Ok(())
    }
    pub async fn wait_exit(&mut self) -> Result<ExitStatus> {
        eventually("server exit", || {
            let result = self.server.try_wait();
            async move { Ok(result?) }
        })
        .await
    }
    pub async fn wait_shutdown(&self) -> Result<()> {
        eventually("shutdown admission", || async {
            let result = self
                .grpc
                .clone()
                .init(proto::InitRequest {
                    run_id: Uuid::new_v4().to_string(),
                })
                .await;
            Ok(result
                .err()
                .filter(|status| status.code() == tonic::Code::Unavailable)
                .map(|_| ()))
        })
        .await
    }
    pub async fn pause_s3(&self) -> Result<()> {
        if let Some(container) = &self.services.s3 {
            container.pause().await?;
        } else {
            ensure!(
                Command::new("kill")
                    .args(["-STOP", &self.services.children[1].id().to_string()])
                    .status()?
                    .success()
            );
        }
        Ok(())
    }
    pub async fn resume_s3(&self) -> Result<()> {
        if let Some(container) = &self.services.s3 {
            container.unpause().await?;
        } else {
            ensure!(
                Command::new("kill")
                    .args(["-CONT", &self.services.children[1].id().to_string()])
                    .status()?
                    .success()
            );
        }
        Ok(())
    }
}
impl Drop for TestEnv {
    fn drop(&mut self) {
        let _ = self.server.kill();
        let _ = self.server.wait();
    }
}
pub struct MetricsStream {
    sender: mpsc::Sender<proto::MetricsRequest>,
    response: JoinHandle<Result<proto::MetricsResponse>>,
}
impl MetricsStream {
    pub async fn send(&self, request: proto::MetricsRequest) -> Result<()> {
        self.sender.send(request).await?;
        Ok(())
    }
    pub async fn artifact(
        &self,
        step: u64,
        name: &str,
        content_type: &str,
        body: Bytes,
    ) -> Result<()> {
        self.send(proto::MetricsRequest {
            payload: Some(metrics_request::Payload::Artifact(proto::ArtifactMetric {
                step,
                timestamp_unix_ms: TIMESTAMP_MS,
                name: name.into(),
                content_type: content_type.into(),
                size_bytes: body.len() as u64,
            })),
        })
        .await?;
        for chunk in body.chunks(256 * 1024) {
            self.send(proto::MetricsRequest {
                payload: Some(metrics_request::Payload::ArtifactChunk(
                    proto::ArtifactChunk {
                        data: Bytes::copy_from_slice(chunk),
                    },
                )),
            })
            .await?;
        }
        Ok(())
    }
    pub async fn finish(self) -> Result<proto::MetricsResponse> {
        drop(self.sender);
        self.response.await?
    }
}
