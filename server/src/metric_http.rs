use anyhow::{Result, ensure};
use serde::Serialize;
use sha2::{Digest, Sha256};
use tensorlane_protocol::MetricBatch;
use time::OffsetDateTime;
use tokio::fs;
use uuid::Uuid;

use crate::{
    db,
    runtime::Runtime,
    shared_cache::{Lock, write_atomic},
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
    let dir = engine.run_dir(run).join("metrics");
    fs::create_dir_all(&dir).await?;
    let _lock = Lock::acquire(&dir.join(format!("{request}.lock"))).await?;
    let hash = hex::encode(Sha256::digest(serde_json::to_vec(&batch)?));
    let intent = dir.join(format!("{request}.intent"));
    if let Ok(existing) = fs::read_to_string(&intent).await {
        ensure!(existing == hash, "conflicting retry of metric request ID");
    } else {
        write_atomic(&intent, hash.as_bytes()).await?;
    }
    let receipt = dir.join(format!("{request}.committed"));
    if fs::try_exists(&receipt).await? {
        return Ok(());
    }
    let client = engine
        .database
        .clone()
        .with_setting("insert_deduplication_token", request.to_string());
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
    write_atomic(&receipt, hash.as_bytes()).await
}

fn timestamp(ms: i64) -> Result<OffsetDateTime> {
    Ok(OffsetDateTime::from_unix_timestamp_nanos(
        i128::from(ms) * 1_000_000,
    )?)
}
