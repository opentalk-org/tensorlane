use std::path::Path;

use crate::{db, runtime::Runtime};
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

impl Runtime {
    pub fn asset_key(&self, id: Uuid) -> String {
        format!("{}/{}", self.checkpoint_prefix, id)
    }

    pub async fn save_asset(
        &self,
        record: &crate::asset_repo::AssetRecord,
        local_path: &Path,
        content_type: &str,
    ) -> anyhow::Result<()> {
        crate::s3_upload::upload(
            &self.s3,
            self.bucket,
            local_path,
            &record.path,
            content_type,
        )
        .await?;
        let mut insert = db::request(
            self.database
                .insert::<crate::asset_repo::AssetRecord>("assets"),
        )
        .await?
        .with_timeouts(Some(db::TIMEOUT), Some(db::TIMEOUT));
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
        crate::s3_upload::upload(
            &self.s3,
            self.bucket,
            local_path,
            &key,
            &metadata.content_type,
        )
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
        let mut insert = db::request(client.insert::<ArtifactRecord>("artifacts"))
            .await?
            .with_timeouts(Some(db::TIMEOUT), Some(db::TIMEOUT));
        insert.write(&row).await?;
        insert.end().await?;
        Ok(())
    }
}
