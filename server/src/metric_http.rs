use anyhow::{Result, ensure};
use serde::Serialize;
use sha2::{Digest, Sha256};
use tensorlane_protocol::MetricBatch;
use time::OffsetDateTime;
use uuid::Uuid;

use crate::{db, runtime::Runtime};

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

pub async fn save(
    engine: &Runtime,
    run: Uuid,
    session: Uuid,
    request: Uuid,
    batch: MetricBatch,
) -> Result<()> {
    ensure!(!request.is_nil(), "metric request ID must not be nil");
    ensure!(
        batch.scalars.len() + batch.arrays.len() <= 1000,
        "metric requests must not exceed 1000 metrics"
    );
    for metric in &batch.scalars {
        ensure!(
            !metric.name.is_empty() && metric.value.is_finite(),
            "invalid scalar metric"
        );
    }
    for metric in &batch.arrays {
        ensure!(
            !metric.name.is_empty() && metric.value.iter().all(|v| v.is_finite()),
            "invalid array metric"
        );
    }
    engine.active(run, session).await?;
    let intent = format!("metrics/{run}/{request}");
    let hash = hex::encode(Sha256::digest(serde_json::to_vec(&batch)?));
    let existing = engine.create_state(&intent, &hash).await?;
    ensure!(existing == hash, "conflicting retry of metric request ID");
    let receipt = format!("{intent}/committed");
    if engine.read_state::<bool>(&receipt).await? == Some(true) {
        return Ok(());
    }
    let client = engine
        .database
        .clone()
        .with_setting("insert_deduplication_token", format!("{run}/{request}"));
    if !batch.scalars.is_empty() {
        let mut insert = db::request(client.insert::<ScalarRecord>("metrics"))
            .await?
            .with_timeouts(Some(db::TIMEOUT), Some(db::TIMEOUT));
        for metric in batch.scalars {
            insert
                .write(&ScalarRecord {
                    timestamp: timestamp(metric.timestamp_unix_ms)?,
                    run_id: run,
                    step: metric.step,
                    name: metric.name,
                    value: metric.value,
                })
                .await?;
        }
        insert.end().await?;
    }
    if !batch.arrays.is_empty() {
        let mut insert = db::request(client.insert::<ArrayRecord>("array_metrics"))
            .await?
            .with_timeouts(Some(db::TIMEOUT), Some(db::TIMEOUT));
        for metric in batch.arrays {
            insert
                .write(&ArrayRecord {
                    timestamp: timestamp(metric.timestamp_unix_ms)?,
                    run_id: run,
                    step: metric.step,
                    name: metric.name,
                    value: metric.value,
                })
                .await?;
        }
        insert.end().await?;
    }
    engine.create_state(&receipt, &true).await?;
    Ok(())
}

fn timestamp(ms: i64) -> Result<OffsetDateTime> {
    Ok(OffsetDateTime::from_unix_timestamp_nanos(
        i128::from(ms) * 1_000_000,
    )?)
}
