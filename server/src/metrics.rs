use std::path::{Component, Path, PathBuf};

use clickhouse::Client;
use serde::Serialize;
use time::OffsetDateTime;
use tokio::fs::{self, File};
use tokio::io::AsyncWriteExt;
use tonic::{Status, Streaming};
use tracing::{debug, trace, warn};
use uuid::Uuid;

use crate::proto::{
    ArrayMetric, ArtifactMetric, MetricsRequest, MetricsResponse, ScalarMetric, metrics_request,
};

#[derive(clickhouse::Row, Serialize)]
struct ScalarRecord {
    #[serde(with = "clickhouse::serde::time::datetime64::nanos")]
    timestamp: OffsetDateTime,
    #[serde(with = "clickhouse::serde::uuid")]
    run_id: Uuid,
    step: u64,
    name: String,
    value: f32,
}

#[derive(clickhouse::Row, Serialize)]
struct ArrayRecord {
    #[serde(with = "clickhouse::serde::time::datetime64::nanos")]
    timestamp: OffsetDateTime,
    #[serde(with = "clickhouse::serde::uuid")]
    run_id: Uuid,
    step: u64,
    name: String,
    value: Vec<f32>,
}

#[derive(clickhouse::Row, Serialize)]
struct ArtifactRecord {
    #[serde(with = "clickhouse::serde::uuid")]
    id: Uuid,
    #[serde(with = "clickhouse::serde::uuid")]
    run_id: Uuid,
    step: u64,
    #[serde(with = "clickhouse::serde::time::datetime64::nanos")]
    timestamp: OffsetDateTime,
    name: String,
    path: String,
    content_type: String,
    size_bytes: u64,
}

struct PendingArtifact {
    metadata: ArtifactMetric,
    target: PathBuf,
    part: PathBuf,
    file: File,
    received: u64,
}

pub async fn receive(
    client: &Client,
    root: &Path,
    run_id: Uuid,
    mut stream: Streaming<MetricsRequest>,
) -> Result<MetricsResponse, Status> {
    let training_dir = root.join(run_id.to_string());
    fs::create_dir_all(&training_dir).await.map_err(internal)?;
    let artifacts_dir = training_dir.join("artifacts");
    fs::create_dir_all(&artifacts_dir).await.map_err(internal)?;

    let mut pending: Option<PendingArtifact> = None;
    let mut response = MetricsResponse::default();
    let result = async {
        while let Some(request) = stream.message().await? {
            match request.payload {
                Some(metrics_request::Payload::Metadata(_)) => {
                    return Err(Status::invalid_argument(
                        "metrics stream metadata may only appear first",
                    ));
                }
                Some(metrics_request::Payload::Metric(metric)) => {
                    trace!(
                        run = %run_id,
                        step = metric.step,
                        timestamp_unix_ms = metric.timestamp_unix_ms,
                        metric = %metric.name,
                        value = metric.value,
                        "metric received"
                    );
                    store_metric(client, run_id, metric).await?;
                    response.metrics_received += 1;
                }
                Some(metrics_request::Payload::ArrayMetric(metric)) => {
                    trace!(
                        run = %run_id,
                        step = metric.step,
                        timestamp_unix_ms = metric.timestamp_unix_ms,
                        metric = %metric.name,
                        values = metric.value.len(),
                        "array metric received"
                    );
                    store_array_metric(client, run_id, metric).await?;
                    response.array_metrics_received += 1;
                }
                Some(metrics_request::Payload::Artifact(artifact)) => {
                    if let Some(previous) = pending.take() {
                        warn!(
                            artifact = %previous.metadata.name,
                            received = previous.received,
                            expected = previous.metadata.size_bytes,
                            "artifact upload cancelled by a new artifact"
                        );
                        discard_artifact(previous).await?;
                    }
                    debug!(
                        run = %run_id,
                        step = artifact.step,
                        timestamp_unix_ms = artifact.timestamp_unix_ms,
                        artifact = %artifact.name,
                        content_type = %artifact.content_type,
                        bytes = artifact.size_bytes,
                        "receiving artifact"
                    );
                    let artifact = begin_artifact(&artifacts_dir, artifact).await?;
                    if artifact.metadata.size_bytes == 0 {
                        let name = artifact.metadata.name.clone();
                        let step = artifact.metadata.step;
                        finish_artifact(client, run_id, artifact).await?;
                        debug!(
                            run = %run_id,
                            step,
                            artifact = %name,
                            bytes = 0,
                            "artifact received"
                        );
                        response.artifacts_received += 1;
                    } else {
                        pending = Some(artifact);
                    }
                }
                Some(metrics_request::Payload::ArtifactChunk(chunk)) => {
                    let Some(mut artifact) = pending.take() else {
                        return Err(Status::invalid_argument(
                            "artifact chunk has no artifact metadata",
                        ));
                    };
                    let chunk_size = chunk.data.len() as u64;
                    if artifact.received + chunk_size > artifact.metadata.size_bytes {
                        pending = Some(artifact);
                        return Err(Status::invalid_argument(
                            "artifact contains more bytes than declared",
                        ));
                    }
                    if let Err(error) = artifact.file.write_all(&chunk.data).await {
                        pending = Some(artifact);
                        return Err(internal(error));
                    }
                    artifact.received += chunk_size;
                    response.artifact_bytes_received += chunk_size;
                    if artifact.received == artifact.metadata.size_bytes {
                        let name = artifact.metadata.name.clone();
                        let step = artifact.metadata.step;
                        let bytes = artifact.metadata.size_bytes;
                        finish_artifact(client, run_id, artifact).await?;
                        debug!(
                            run = %run_id,
                            step,
                            artifact = %name,
                            bytes,
                            "artifact received"
                        );
                        response.artifacts_received += 1;
                    } else {
                        pending = Some(artifact);
                    }
                }
                None => return Err(Status::invalid_argument("metrics message has no payload")),
            }
        }

        if pending.is_some() {
            return Err(Status::invalid_argument(
                "metrics stream ended before the artifact was complete",
            ));
        }
        Ok(())
    }
    .await;

    if result.is_err() {
        if let Some(artifact) = pending {
            let _ = discard_artifact(artifact).await;
        }
    }
    result?;
    Ok(response)
}

async fn store_metric(client: &Client, run_id: Uuid, metric: ScalarMetric) -> Result<(), Status> {
    if metric.name.is_empty() {
        return Err(Status::invalid_argument("metric name cannot be empty"));
    }
    let row = ScalarRecord {
        timestamp: metric_timestamp(metric.timestamp_unix_ms)?,
        run_id,
        step: metric.step,
        name: metric.name,
        value: metric.value,
    };
    let mut insert = client
        .insert::<ScalarRecord>("metrics")
        .await
        .map_err(internal)?;
    insert.write(&row).await.map_err(internal)?;
    insert.end().await.map_err(internal)
}

async fn store_array_metric(
    client: &Client,
    run_id: Uuid,
    metric: ArrayMetric,
) -> Result<(), Status> {
    if metric.name.is_empty() {
        return Err(Status::invalid_argument("metric name cannot be empty"));
    }
    let row = ArrayRecord {
        timestamp: metric_timestamp(metric.timestamp_unix_ms)?,
        run_id,
        step: metric.step,
        name: metric.name,
        value: metric.value,
    };
    let mut insert = client
        .insert::<ArrayRecord>("array_metrics")
        .await
        .map_err(internal)?;
    insert.write(&row).await.map_err(internal)?;
    insert.end().await.map_err(internal)
}

async fn begin_artifact(
    artifacts_dir: &Path,
    metadata: ArtifactMetric,
) -> Result<PendingArtifact, Status> {
    let relative = artifact_path(&metadata.name)?;
    let target = artifacts_dir.join(relative);
    let parent = target
        .parent()
        .ok_or_else(|| Status::invalid_argument("artifact name has no parent directory"))?;
    fs::create_dir_all(parent).await.map_err(internal)?;
    let filename = target
        .file_name()
        .ok_or_else(|| Status::invalid_argument("artifact name has no filename"))?
        .to_string_lossy();
    let part = target.with_file_name(format!("{filename}.part"));
    let file = File::create(&part).await.map_err(internal)?;
    Ok(PendingArtifact {
        metadata,
        target,
        part,
        file,
        received: 0,
    })
}

async fn finish_artifact(
    client: &Client,
    run_id: Uuid,
    artifact: PendingArtifact,
) -> Result<(), Status> {
    artifact.file.sync_all().await.map_err(internal)?;
    drop(artifact.file);
    if fs::try_exists(&artifact.target).await.map_err(internal)? {
        fs::remove_file(&artifact.target).await.map_err(internal)?;
    }
    fs::rename(&artifact.part, &artifact.target)
        .await
        .map_err(internal)?;
    let row = ArtifactRecord {
        id: Uuid::new_v4(),
        run_id,
        step: artifact.metadata.step,
        timestamp: metric_timestamp(artifact.metadata.timestamp_unix_ms)?,
        name: artifact.metadata.name,
        path: artifact.target.to_string_lossy().into_owned(),
        content_type: artifact.metadata.content_type,
        size_bytes: artifact.metadata.size_bytes,
    };
    let mut insert = client
        .insert::<ArtifactRecord>("artifacts")
        .await
        .map_err(internal)?;
    insert.write(&row).await.map_err(internal)?;
    insert.end().await.map_err(internal)
}

async fn discard_artifact(artifact: PendingArtifact) -> Result<(), Status> {
    drop(artifact.file);
    fs::remove_file(artifact.part).await.map_err(internal)
}

fn metric_timestamp(timestamp_unix_ms: i64) -> Result<OffsetDateTime, Status> {
    OffsetDateTime::from_unix_timestamp_nanos(i128::from(timestamp_unix_ms) * 1_000_000)
        .map_err(internal)
}

fn artifact_path(name: &str) -> Result<&Path, Status> {
    if name.is_empty() {
        return Err(Status::invalid_argument("artifact name cannot be empty"));
    }
    let path = Path::new(name);
    if path
        .components()
        .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(Status::invalid_argument(
            "artifact name must be a relative path without parent components",
        ));
    }
    Ok(path)
}

fn internal(error: impl std::fmt::Display) -> Status {
    Status::internal(format!("{error:#}"))
}
