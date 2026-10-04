use std::path::Path;

use clickhouse::Client;
use serde::Serialize;
use time::OffsetDateTime;
use uuid::Uuid;

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
}

impl UploadStore {
    pub fn new(
        s3: aws_sdk_s3::Client,
        database: Client,
        bucket: &'static str,
        checkpoint_prefix: &'static str,
        metrics_prefix: &'static str,
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
        })
    }

    pub fn asset_key(&self, id: Uuid) -> String {
        format!("{}/{}", self.checkpoint_prefix, id)
    }

    pub async fn save_asset(
        &self,
        record: &crate::asset_repo::AssetRecord,
        local_path: &Path,
        content_type: &str,
    ) -> anyhow::Result<()> {
        self.upload(local_path, &record.path, content_type).await?;
        let mut insert = self
            .database
            .insert::<crate::asset_repo::AssetRecord>("assets")
            .await?;
        insert.write(record).await?;
        insert.end().await?;
        Ok(())
    }

    pub async fn save_artifact(
        &self,
        id: Uuid,
        run_id: Uuid,
        metadata: &tensorlane_protocol::ArtifactMetric,
        local_path: &Path,
    ) -> anyhow::Result<()> {
        let key = format!("{}/{}", self.metrics_prefix, id);
        self.upload(local_path, &key, &metadata.content_type)
            .await?;
        let row = ArtifactRecord {
            id,
            run_id,
            step: metadata.step,
            timestamp: OffsetDateTime::from_unix_timestamp_nanos(
                i128::from(metadata.timestamp_unix_ms) * 1_000_000,
            )?,
            name: metadata.name.clone(),
            path: key,
            content_type: metadata.content_type.clone(),
            size_bytes: metadata.size_bytes,
        };
        let client = self
            .database
            .clone()
            .with_setting("insert_deduplication_token", id.to_string());
        let mut insert = client.insert::<ArtifactRecord>("artifacts").await?;
        insert.write(&row).await?;
        insert.end().await?;
        Ok(())
    }

    async fn upload(&self, path: &Path, key: &str, content_type: &str) -> anyhow::Result<()> {
        crate::s3_upload::upload(&self.s3, self.bucket, path, key, content_type).await
    }
}
