use std::path::{Path, PathBuf};

use anyhow::Context;
use aws_sdk_s3::{
    primitives::{ByteStream, Length},
    types::{CompletedMultipartUpload, CompletedPart},
};
use clickhouse::Client;
use serde::Serialize;
use time::OffsetDateTime;
use tokio::fs;
use tokio_util::task::TaskTracker;
use tracing::{error, info};
use uuid::Uuid;

#[derive(clickhouse::Row, Serialize)]
struct CheckpointRecord {
    #[serde(with = "clickhouse::serde::uuid")]
    id: Uuid,
    #[serde(with = "clickhouse::serde::time::datetime64::micros")]
    updated_at: OffsetDateTime,
    kind: i8,
    name: String,
    step: u64,
    path: String,
    size: u64,
    #[serde(with = "serde_big_array::BigArray")]
    content_hash: [u8; 64],
    #[serde(rename = "type")]
    asset_type: String,
    metadata: String,
    #[serde(with = "clickhouse::serde::uuid")]
    run_id: Uuid,
    #[serde(with = "clickhouse::serde::uuid")]
    ancestor_asset_id: Uuid,
    deleted: bool,
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

#[derive(Clone)]
pub struct UploadStore {
    s3: aws_sdk_s3::Client,
    database: Client,
    bucket: &'static str,
    checkpoint_prefix: &'static str,
    metrics_prefix: &'static str,
    staging_dir: &'static Path,
    tasks: TaskTracker,
}

impl UploadStore {
    pub fn new(
        s3: aws_sdk_s3::Client,
        database: Client,
        bucket: &'static str,
        checkpoint_prefix: &'static str,
        metrics_prefix: &'static str,
        staging_dir: &'static Path,
    ) -> anyhow::Result<Self> {
        let checkpoint_prefix = checkpoint_prefix.trim_matches('/');
        let metrics_prefix = metrics_prefix.trim_matches('/');
        anyhow::ensure!(!checkpoint_prefix.is_empty(), "checkpoint prefix is empty");
        anyhow::ensure!(!metrics_prefix.is_empty(), "metrics prefix is empty");
        Ok(Self {
            s3,
            database,
            bucket,
            checkpoint_prefix,
            metrics_prefix,
            staging_dir,
            tasks: TaskTracker::new(),
        })
    }

    pub fn staging_path(&self, id: Uuid) -> PathBuf {
        self.staging_dir.join(id.to_string())
    }

    pub fn checkpoint(
        &self,
        id: Uuid,
        run_id: Uuid,
        step: u64,
        size: u64,
        content_hash: [u8; 64],
        asset_type: String,
    ) {
        let store = self.clone();
        self.tasks.spawn(async move {
            if let Err(err) = store
                .upload_checkpoint(id, run_id, step, size, content_hash, asset_type)
                .await
            {
                error!(checkpoint = %id, run = %run_id, error = format!("{err:#}"), "checkpoint upload failed");
            }
        });
    }

    pub fn artifact(
        &self,
        id: Uuid,
        run_id: Uuid,
        step: u64,
        timestamp: OffsetDateTime,
        name: String,
        content_type: String,
        size: u64,
    ) {
        let store = self.clone();
        self.tasks.spawn(async move {
            if let Err(err) = store
                .upload_artifact(
                    id,
                    run_id,
                    step,
                    timestamp,
                    name,
                    content_type,
                    size,
                )
                .await
            {
                error!(artifact = %id, run = %run_id, error = format!("{err:#}"), "metric artifact upload failed");
            }
        });
    }

    pub async fn finish(&self) {
        self.tasks.close();
        self.tasks.wait().await;
    }

    async fn upload_checkpoint(
        &self,
        id: Uuid,
        run_id: Uuid,
        step: u64,
        size: u64,
        content_hash: [u8; 64],
        asset_type: String,
    ) -> anyhow::Result<()> {
        let local_path = self.staging_path(id);
        let key = format!("{}/{}", self.checkpoint_prefix, id);
        self.upload(&local_path, &key, "application/x-tar").await?;
        let row = CheckpointRecord {
            id,
            updated_at: OffsetDateTime::now_utc(),
            kind: 1,
            name: format!("{run_id}_{step:09}"),
            step,
            path: key.clone(),
            size,
            content_hash,
            asset_type,
            metadata: "{}".to_owned(),
            run_id,
            ancestor_asset_id: Uuid::nil(),
            deleted: false,
        };
        let mut insert = self.database.insert::<CheckpointRecord>("assets").await?;
        insert.write(&row).await?;
        insert.end().await?;
        fs::remove_file(&local_path).await?;
        info!(checkpoint = %id, run = %run_id, key, size, "checkpoint uploaded");
        Ok(())
    }

    async fn upload_artifact(
        &self,
        id: Uuid,
        run_id: Uuid,
        step: u64,
        timestamp: OffsetDateTime,
        name: String,
        content_type: String,
        size_bytes: u64,
    ) -> anyhow::Result<()> {
        let local_path = self.staging_path(id);
        let key = format!("{}/{}", self.metrics_prefix, id);
        self.upload(&local_path, &key, "application/x-tar").await?;
        let row = ArtifactRecord {
            id,
            run_id,
            step,
            timestamp,
            name,
            path: key.clone(),
            content_type,
            size_bytes,
        };
        let mut insert = self.database.insert::<ArtifactRecord>("artifacts").await?;
        insert.write(&row).await?;
        insert.end().await?;
        fs::remove_file(&local_path).await?;
        info!(artifact = %id, run = %run_id, key, size_bytes, "metric artifact uploaded");
        Ok(())
    }

    async fn upload(&self, local_path: &Path, key: &str, content_type: &str) -> anyhow::Result<()> {
        let size = fs::metadata(local_path)
            .await
            .with_context(|| format!("reading staged upload {}", local_path.display()))?
            .len();
        if size == 0 {
            self.s3
                .put_object()
                .bucket(self.bucket)
                .key(key)
                .content_type(content_type)
                .body(ByteStream::from_static(b""))
                .send()
                .await
                .with_context(|| format!("uploading empty object to s3://{}/{key}", self.bucket))?;
            return Ok(());
        }
        let part_size = (64 * 1024 * 1024_u64).max(size.div_ceil(10_000));
        anyhow::ensure!(
            part_size <= 5 * 1024 * 1024 * 1024,
            "upload exceeds S3 multipart size limits"
        );
        let upload = self
            .s3
            .create_multipart_upload()
            .bucket(self.bucket)
            .key(key)
            .content_type(content_type)
            .send()
            .await
            .with_context(|| format!("starting multipart upload to s3://{}/{key}", self.bucket))?;
        let upload_id = upload
            .upload_id()
            .context("multipart upload response missing upload ID")?;
        let result: anyhow::Result<()> = async {
            let mut parts = Vec::new();
            for index in 0..size.div_ceil(part_size) {
                let offset = index * part_size;
                let length = part_size.min(size - offset);
                let part_number = i32::try_from(index + 1)?;
                let body = ByteStream::read_from()
                    .path(local_path)
                    .offset(offset)
                    .length(Length::Exact(length))
                    .build()
                    .await
                    .with_context(|| format!("reading upload part {part_number}"))?;
                let part = self
                    .s3
                    .upload_part()
                    .bucket(self.bucket)
                    .key(key)
                    .upload_id(upload_id)
                    .part_number(part_number)
                    .body(body)
                    .send()
                    .await
                    .with_context(|| format!("uploading part {part_number}"))?;
                parts.push(
                    CompletedPart::builder()
                        .part_number(part_number)
                        .e_tag(part.e_tag().context("upload part response missing ETag")?)
                        .build(),
                );
            }
            self.s3
                .complete_multipart_upload()
                .bucket(self.bucket)
                .key(key)
                .upload_id(upload_id)
                .multipart_upload(
                    CompletedMultipartUpload::builder()
                        .set_parts(Some(parts))
                        .build(),
                )
                .send()
                .await
                .context("completing multipart upload")?;
            Ok(())
        }
        .await;
        if result.is_err() {
            if let Err(err) = self
                .s3
                .abort_multipart_upload()
                .bucket(self.bucket)
                .key(key)
                .upload_id(upload_id)
                .send()
                .await
            {
                error!(
                    key,
                    upload_id,
                    error = format!("{err:#}"),
                    "aborting multipart upload failed"
                );
            }
        }
        result.with_context(|| {
            format!(
                "uploading {} to s3://{}/{key}",
                local_path.display(),
                self.bucket
            )
        })
    }
}
