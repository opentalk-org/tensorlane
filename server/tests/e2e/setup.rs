#![allow(dead_code)]

use std::{
    collections::BTreeSet,
    io::Cursor,
    net::TcpListener,
    path::{Path, PathBuf},
    process::{Child, Command, ExitStatus, Stdio},
    time::Duration,
};

use anyhow::{Context, Result, bail, ensure};
use aws_sdk_s3::config::{BehaviorVersion, Credentials, Region};
use aws_sdk_s3::primitives::ByteStream;
use bytes::{BufMut, Bytes, BytesMut};
use serde_json::{Value, json};
use tempfile::TempDir;
use testcontainers_modules::{
    clickhouse::ClickHouse,
    minio::MinIO,
    testcontainers::{
        ContainerAsync, ImageExt,
        core::{ExecCommand, IntoContainerPort, WaitFor, wait::HttpWaitStrategy},
        runners::AsyncRunner,
    },
};
use tokio::{sync::mpsc, task::JoinHandle, time::sleep};
use tokio_stream::wrappers::ReceiverStream;
use tonic::transport::{Channel, Endpoint};
use uuid::Uuid;

use crate::proto::{
    self, checkpoint_request, metrics_request, tensor_lane_client::TensorLaneClient,
};

pub const TIMESTAMP_MS: i64 = 1_700_000_000_123;

pub async fn eventually<T, F, Fut>(description: &str, mut check: F) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<Option<T>>>,
{
    loop {
        if let Some(value) = check()
            .await
            .with_context(|| format!("waiting for {description}"))?
        {
            return Ok(value);
        }
        sleep(Duration::from_millis(25)).await;
    }
}

pub fn data_config(dataset_id: Uuid, batches: u64) -> Value {
    json!({
        "dataset_id": dataset_id, "asset_type": "e2e-model", "seed": 1,
        "max_text_tokens": 128, "plbert_languages": ["en", "de", "fr"],
        "validation": {"samples": 1, "max_seconds": 33.0},
        "training": [{"batches": batches, "max_seconds": 33.0}], "assets": {}
    })
}

pub fn tar_file(name: &str, content: &[u8]) -> Result<Bytes> {
    let mut archive = tar::Builder::new(Vec::new());
    let mut header = tar::Header::new_gnu();
    header.set_size(content.len() as u64);
    header.set_mode(0o644);
    header.set_mtime(0);
    header.set_cksum();
    archive.append_data(&mut header, name, content)?;
    Ok(archive.into_inner()?.into())
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
    if !path.exists() {
        return Ok(BTreeSet::new());
    }
    Ok(std::fs::read_dir(path)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<std::io::Result<_>>()?)
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
    _cache: TempDir,
    _clickhouse: ContainerAsync<ClickHouse>,
    _s3: ContainerAsync<MinIO>,
}

impl TestEnv {
    pub async fn pause_s3(&self) -> Result<()> {
        self._s3.pause().await?;
        Ok(())
    }

    pub async fn resume_s3(&self) -> Result<()> {
        self._s3.unpause().await?;
        Ok(())
    }

    pub async fn seed_dataset(&self, audio_files: usize) -> Result<Uuid> {
        ensure!(
            audio_files > 0,
            "dataset must contain at least one audio file"
        );
        let dataset_id = Uuid::new_v4();
        let durations = [
            0.25_f32, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0,
        ];
        let languages = ["en", "de", "fr"];
        let phonemes = ["həloʊ", "wɜːld", "test", "ɑ"];
        self.clickhouse
            .query(
                "INSERT INTO datasets (id, updated_at, name) VALUES (?, now64(6), 'e2e dataset')",
            )
            .bind(dataset_id.to_string())
            .execute()
            .await?;

        for start in (0..audio_files).step_by(32) {
            let pack_id = Uuid::new_v4();
            let key = format!("datasets/{dataset_id}/{pack_id}.pack");
            let mut packed = Vec::new();
            let mut audio_rows = Vec::new();
            let mut segment_rows = Vec::new();
            let mut membership_rows = Vec::new();
            for index in start..start.saturating_add(32).min(audio_files) {
                let audio_id = Uuid::new_v4();
                let duration = durations[index % durations.len()];
                let speaker = format!("speaker-{}", index % 12);
                let mut cursor = Cursor::new(Vec::new());
                let mut writer = hound::WavWriter::new(
                    &mut cursor,
                    hound::WavSpec {
                        channels: 1,
                        sample_rate: 24_000,
                        bits_per_sample: 16,
                        sample_format: hound::SampleFormat::Int,
                    },
                )?;
                let frequency = 110.0 + (index % 12) as f32 * 30.0;
                for frame in 0..(duration * 24_000.0) as usize {
                    let phase = std::f32::consts::TAU * frequency * frame as f32 / 24_000.0;
                    writer.write_sample((phase.sin() * 8_000.0) as i16)?;
                }
                writer.finalize()?;
                let wave = cursor.into_inner();
                audio_rows.push(json!({
                    "id": audio_id,
                    "name": format!("sample-{index}.wav"),
                    "bucket_file_id": pack_id,
                    "byte_offset": packed.len(),
                    "byte_length": wave.len(),
                    "duration": duration,
                    "language": languages[index % languages.len()],
                    "virtual": false,
                    "storage_kind": "packed"
                }));
                segment_rows.push(json!({
                    "id": Uuid::new_v4(),
                    "audio_file_id": audio_id,
                    "start_seconds": 0,
                    "end_seconds": duration,
                    "phon": phonemes[index % phonemes.len()],
                    "text": format!("Test sample {index}"),
                    "speaker_id": speaker,
                    "metadata": json!({"_source": {"annotations": {"speaker_id": speaker}}}).to_string()
                }));
                membership_rows.push(json!({"dataset_id": dataset_id, "audio_file_id": audio_id}));
                packed.extend_from_slice(&wave);
            }
            let size = packed.len() as u64;
            self.s3
                .put_object()
                .bucket(self.bucket)
                .key(&key)
                .body(ByteStream::from(packed))
                .send()
                .await?;
            self.clickhouse
                .query(
                    "INSERT INTO bucket_files (id, kind, path, size, used_bytes)
                VALUES (?, 'audio', ?, ?, ?)",
                )
                .bind(pack_id.to_string())
                .bind(&key)
                .bind(size)
                .bind(size)
                .execute()
                .await?;
            for (table, rows) in [
                ("audio_files", audio_rows),
                ("audio_segments", segment_rows),
                ("dataset_audio_files", membership_rows),
            ] {
                let payload = rows
                    .iter()
                    .map(serde_json::to_string)
                    .collect::<Result<Vec<_>, _>>()?
                    .join("\n");
                self.clickhouse
                    .query(&format!(
                        "INSERT INTO {table} FORMAT JSONEachRow\n{payload}"
                    ))
                    .execute()
                    .await
                    .with_context(|| format!("seeding {table}"))?;
            }
        }
        Ok(dataset_id)
    }

    pub async fn start() -> Result<Self> {
        let clickhouse_container = ClickHouse::default()
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
        let s3_container = MinIO::default()
            .with_name("quay.io/minio/minio")
            .with_env_var("MINIO_ROOT_USER", "minioadmin")
            .with_env_var("MINIO_ROOT_PASSWORD", "minioadmin")
            .start()
            .await?;

        let clickhouse_url = format!(
            "http://{}:{}",
            clickhouse_container.get_host().await?,
            clickhouse_container.get_host_port_ipv4(8123).await?,
        );
        let clickhouse = clickhouse::Client::default()
            .with_url(&clickhouse_url)
            .with_user("default")
            .with_password("test")
            .with_setting("allow_experimental_json_type", "1")
            .with_setting("input_format_binary_read_json_as_string", "1")
            .with_setting("output_format_binary_write_json_as_string", "1");
        let migrations_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../db/migrations");
        let mut migrations = std::fs::read_dir(migrations_dir)?
            .map(|entry| entry.map(|entry| entry.path()))
            .collect::<std::io::Result<Vec<_>>>()?;
        migrations.retain(|path| path.extension().is_some_and(|extension| extension == "sql"));
        migrations.sort();
        for path in migrations {
            let sql = std::fs::read_to_string(&path)?;
            let mut command = clickhouse_container
                .exec(ExecCommand::new([
                    "clickhouse-client",
                    "--password",
                    "test",
                    "--multiquery",
                    "--query",
                    &sql,
                ]))
                .await?;
            let errors = command.stderr_to_vec().await?;
            ensure!(
                command.exit_code().await? == Some(0),
                "applying {}: {}",
                path.display(),
                String::from_utf8_lossy(&errors),
            );
        }

        let s3_url = format!(
            "http://{}:{}",
            s3_container.get_host().await?,
            s3_container.get_host_port_ipv4(9000).await?,
        );
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

        let cache = tempfile::tempdir()?;
        let http_port = TcpListener::bind("127.0.0.1:0")?;
        let grpc_port = TcpListener::bind("127.0.0.1:0")?;
        let http_address = http_port.local_addr()?;
        let grpc_address = grpc_port.local_addr()?;
        let http_url = format!("http://{http_address}");
        let grpc_url = format!("http://{grpc_address}");
        let http = reqwest::Client::new();
        let endpoint = Endpoint::from_shared(grpc_url.clone())?;
        drop((http_port, grpc_port));
        let server = Command::new(env!("CARGO_BIN_EXE_tensorlane"))
            .args([
                "--clickhouse-url",
                &clickhouse_url,
                "--clickhouse-user",
                "default",
                "--clickhouse-password",
                "test",
            ])
            .args([
                "--s3-endpoint",
                &s3_url,
                "--s3-key",
                "minioadmin",
                "--s3-secret",
                "minioadmin",
                "--bucket",
                bucket,
            ])
            .args([
                "--http-port",
                &http_address.port().to_string(),
                "--grpc-port",
                &grpc_address.port().to_string(),
            ])
            .args([
                "--checkpoint-prefix",
                "checkpoints",
                "--metrics-prefix",
                "metrics",
            ])
            .arg("--cache-dir")
            .arg(cache.path())
            .env(
                "RUST_LOG",
                std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()),
            )
            .stdin(Stdio::null())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .spawn()
            .context("starting tensorlane server")?;
        let mut environment = Self {
            clickhouse,
            s3,
            http,
            grpc: TensorLaneClient::new(endpoint.connect_lazy()),
            http_url,
            grpc_url,
            cache_dir: cache.path().to_path_buf(),
            bucket,
            server,
            _cache: cache,
            _clickhouse: clickhouse_container,
            _s3: s3_container,
        };
        loop {
            if let Some(status) = environment.server.try_wait()? {
                bail!("tensorlane exited during startup: {status}");
            }
            if let Ok(response) = environment
                .http
                .get(format!("{}/runs", environment.http_url))
                .send()
                .await
                && response.status().is_success()
                && let Ok(channel) = endpoint.connect().await
            {
                environment.grpc =
                    TensorLaneClient::new(channel).max_decoding_message_size(64 * 1024 * 1024);
                break;
            }
            sleep(Duration::from_millis(100)).await;
        }
        Ok(environment)
    }

    pub async fn create_run(&self, config: Value, train_config: Value) -> Result<String> {
        let response = self.http.post(format!("{}/runs", self.http_url)).json(&json!({
            "project_id": Uuid::new_v4(), "name": "e2e", "data_config": config, "train_config": train_config,
        })).send().await?;
        let status = response.status();
        let body = response.text().await?;
        ensure!(
            status == reqwest::StatusCode::CREATED,
            "create run: {status}: {body}"
        );
        let result: Value = serde_json::from_str(&body)?;
        ensure!(result["status"] == "queued", "run was not queued: {result}");
        Ok(result["run_id"]
            .as_str()
            .context("missing run ID")?
            .to_owned())
    }

    pub async fn init_run(&self, run_id: &str) -> Result<proto::InitResponse> {
        Ok(self
            .grpc
            .clone()
            .init(proto::InitRequest {
                run_id: run_id.into(),
            })
            .await?
            .into_inner())
    }

    pub async fn end_run(&self, run_id: &str) -> Result<()> {
        self.grpc
            .clone()
            .end(proto::EndRequest {
                run_id: run_id.into(),
            })
            .await?;
        Ok(())
    }

    pub async fn batches(
        &self,
        run_id: &str,
        validation: bool,
        requests: usize,
    ) -> Result<Vec<proto::DataResponse>> {
        let requests = (0..requests)
            .map(|_| proto::DataRequest {
                run_id: run_id.into(),
                split: if validation {
                    proto::Split::Validation
                } else {
                    proto::Split::Training
                } as i32,
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

    pub async fn asset(&self, run_id: &str, name: &str) -> Result<(proto::AssetMetadata, Bytes)> {
        let mut stream = self
            .grpc
            .clone()
            .asset(proto::AssetRequest {
                run_id: run_id.into(),
                name: name.into(),
            })
            .await?
            .into_inner();
        let first = stream.message().await?.context("missing asset metadata")?;
        let Some(proto::asset_response::Payload::Metadata(metadata)) = first.payload else {
            anyhow::bail!("asset did not start with metadata");
        };
        let mut body = BytesMut::new();
        while let Some(message) = stream.message().await? {
            let Some(proto::asset_response::Payload::Chunk(chunk)) = message.payload else {
                anyhow::bail!("expected asset chunk");
            };
            body.put(chunk);
        }
        Ok((metadata, body.freeze()))
    }

    pub async fn checkpoint(&self, run_id: &str, step: u64, body: Bytes) -> Result<()> {
        let mut requests = vec![proto::CheckpointRequest {
            payload: Some(checkpoint_request::Payload::Metadata(
                proto::CheckpointMetadata {
                    run_id: run_id.into(),
                    step,
                },
            )),
        }];
        for offset in (0..body.len()).step_by(256 * 1024) {
            requests.push(proto::CheckpointRequest {
                payload: Some(checkpoint_request::Payload::Chunk(
                    body.slice(offset..(offset + 256 * 1024).min(body.len())),
                )),
            });
        }
        self.grpc
            .clone()
            .checkpoint(tokio_stream::iter(requests))
            .await?;
        Ok(())
    }

    pub async fn metrics_stream(&self, run_id: &str) -> Result<MetricsStream> {
        let (sender, receiver) = mpsc::channel(32);
        sender
            .send(proto::MetricsRequest {
                payload: Some(metrics_request::Payload::Metadata(
                    proto::MetricsStreamMetadata {
                        run_id: run_id.into(),
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

    pub async fn count(&self, table: &str, run_id: &str) -> Result<u64> {
        ensure!(
            ["metrics", "array_metrics", "artifacts", "assets"].contains(&table),
            "unsupported count table"
        );
        Ok(self
            .clickhouse
            .query(&format!(
                "SELECT count() FROM {table} WHERE run_id = toUUID(?)"
            ))
            .bind(run_id)
            .fetch_one::<u64>()
            .await?)
    }

    pub async fn wait_count(&self, table: &str, run_id: &str, expected: u64) -> Result<()> {
        eventually("ClickHouse row count", || async {
            Ok((self.count(table, run_id).await? == expected).then_some(()))
        })
        .await
    }

    pub async fn status(&self, run_id: &str) -> Result<String> {
        Ok(self.clickhouse.query("SELECT toString(argMax(status, timestamp)) FROM run_status WHERE run_id = toUUID(?)")
            .bind(run_id).fetch_one::<String>().await?)
    }

    pub fn run_cache(&self, run_id: &str) -> PathBuf {
        self.cache_dir.join("runs").join(run_id)
    }

    pub fn signal(&mut self, signal: &str) -> Result<()> {
        ensure!(self.server.try_wait()?.is_none(), "server already exited");
        ensure!(
            std::process::Command::new("/bin/kill")
                .args([signal, &self.server.id().to_string()])
                .status()?
                .success(),
            "sending {signal} failed"
        );
        Ok(())
    }

    pub async fn wait_exit(&mut self) -> Result<ExitStatus> {
        loop {
            if let Some(status) = self.server.try_wait()? {
                return Ok(status);
            }
            sleep(Duration::from_millis(25)).await;
        }
    }

    pub async fn wait_shutdown(&self) -> Result<()> {
        eventually("shutdown admission rejection", || async {
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
        for offset in (0..body.len()).step_by(256 * 1024) {
            self.send(proto::MetricsRequest {
                payload: Some(metrics_request::Payload::ArtifactChunk(
                    proto::ArtifactChunk {
                        data: body.slice(offset..(offset + 256 * 1024).min(body.len())),
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
