use std::{env, time::Instant};

use serde_json::json;

use super::QuerySampler;
use crate::db::fetch_samples;

/// Measures metadata transfer and preparation without including audio downloads.
#[tokio::test]
#[ignore = "requires live ClickHouse credentials and TENSORLANE_BENCHMARK_BATCHES"]
async fn live_plan_memory() -> anyhow::Result<()> {
    let database = clickhouse::Client::default()
        .with_url(env::var("CLICKHOUSE_URL")?)
        .with_user(env::var("CLICKHOUSE_USER")?)
        .with_password(env::var("CLICKHOUSE_PASSWORD")?);
    let count: u64 = env::var("TENSORLANE_BENCHMARK_BATCHES")?.parse()?;
    assert!(count > 0 && count % 5 == 0);
    let config: serde_json::Value =
        serde_json::from_str(include_str!("../../../sample-configs.json"))?;
    let data = &config["data_config"];
    let validation = fetch_samples(
        &database,
        include_str!("../../../queries/validation.sql"),
        &[
            ("dataset_id", data["dataset_id"].clone()),
            ("sample_size", data["validation"]["samples"].clone()),
            ("max_duration", data["validation"]["max_seconds"].clone()),
            ("max_text", data["max_text_tokens"].clone()),
        ],
    )
    .await?;
    let validation_ids: Vec<_> = validation.iter().map(|row| row.audio_id).collect();
    let started = Instant::now();
    let rows = fetch_samples(
        &database,
        include_str!("../../../queries/training.sql"),
        &[
            ("dataset_id", data["dataset_id"].clone()),
            ("validation_ids", json!(validation_ids)),
            ("seed", data["seed"].clone()),
            (
                "stage_batches",
                json!([count * 2 / 5, count * 2 / 5, count / 5]),
            ),
            ("stage_seconds", json!([150.0, 150.0, 90.0])),
            ("max_duration", json!(150.0)),
            ("max_text", data["max_text_tokens"].clone()),
        ],
    )
    .await?;
    let fetched_seconds = started.elapsed().as_secs_f64();
    let sample_count = rows.len();
    println!(
        "BENCHMARK {}",
        json!({"phase":"fetched", "batches":count, "samples":sample_count, "seconds":fetched_seconds})
    );
    let prepare_started = Instant::now();
    let languages: Vec<String> = serde_json::from_value(data["plbert_languages"].clone())?;
    let sampler = QuerySampler::new(rows, &languages)?;
    assert_eq!(sampler.len() as u64, count);
    let prepared_seconds = prepare_started.elapsed().as_secs_f64();
    let text_bytes: usize = sampler
        .batches
        .as_slice()
        .iter()
        .flatten()
        .map(|sample| sample.text.len())
        .sum();
    println!(
        "BENCHMARK {}",
        json!({"phase":"prepared", "batches":count, "samples":sample_count,
        "fetch_seconds":fetched_seconds, "prepare_seconds":prepared_seconds,
        "total_seconds":started.elapsed().as_secs_f64(), "text_bytes":text_bytes})
    );
    std::hint::black_box(&sampler);
    Ok(())
}
