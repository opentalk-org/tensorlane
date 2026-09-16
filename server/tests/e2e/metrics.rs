use std::time::Duration;

use anyhow::Result;
use serde::Deserialize;
use serde_json::json;

use crate::setup::{TIMESTAMP_MS, TestEnv, array, data_config, scalar};

#[derive(clickhouse::Row, Deserialize)]
struct ArrayMetric {
    step: u64,
    name: String,
    value: Vec<f32>,
    timestamp_ms: i64,
}

#[rstest::rstest]
#[tokio::test]
#[timeout(Duration::from_secs(30))]
async fn metrics_buffer_independently_flush_at_1000_and_flush_tails_on_close() -> Result<()> {
    let env = TestEnv::start().await?;
    let dataset = env.seed_dataset(2).await?;
    let run = env.create_run(data_config(dataset, 1), json!({})).await?;
    env.init_run(&run).await?;
    let stream = env.metrics_stream(&run).await?;
    for step in 0..999 {
        stream.send(scalar(step)).await?;
        stream.send(array(step)).await?;
    }
    assert_eq!(env.count("metrics", &run).await?, 0);
    assert_eq!(env.count("array_metrics", &run).await?, 0);

    stream.send(scalar(999)).await?;
    env.wait_count("metrics", &run, 1000).await?;
    assert_eq!(env.count("array_metrics", &run).await?, 0);

    stream.send(array(999)).await?;
    env.wait_count("array_metrics", &run, 1000).await?;

    for step in 1000..1007 {
        stream.send(scalar(step)).await?;
        stream.send(array(step)).await?;
    }
    assert_eq!(env.count("metrics", &run).await?, 1000);
    assert_eq!(env.count("array_metrics", &run).await?, 1000);
    let accepted = stream.finish().await?;
    assert_eq!(accepted.metrics_received, 1007);
    assert_eq!(accepted.array_metrics_received, 1007);
    assert_eq!(accepted.artifacts_received, 0);

    let second = env.metrics_stream(&run).await?;
    for step in 1007..1010 {
        second.send(scalar(step)).await?;
        second.send(array(step)).await?;
    }
    let accepted = second.finish().await?;
    assert_eq!(accepted.metrics_received, 3);
    assert_eq!(accepted.array_metrics_received, 3);
    env.end_run(&run).await?;

    let scalars = env.clickhouse.query("SELECT step, name, value, toUnixTimestamp64Milli(timestamp) FROM metrics WHERE run_id = toUUID(?) ORDER BY step")
        .bind(&run).fetch_all::<(u64, String, f32, i64)>().await?;
    let arrays = env.clickhouse.query("SELECT step, name, value, toUnixTimestamp64Milli(timestamp) AS timestamp_ms FROM array_metrics WHERE run_id = toUUID(?) ORDER BY step")
        .bind(&run).fetch_all::<ArrayMetric>().await?;
    assert_eq!(scalars.len(), 1010);
    assert_eq!(arrays.len(), 1010);
    for (step, (scalar, array)) in scalars.iter().zip(&arrays).enumerate() {
        let step = step as u64;
        assert_eq!(
            scalar,
            &(
                step,
                format!("loss/{}", step % 2),
                step as f32 / 4.0,
                TIMESTAMP_MS + step as i64
            )
        );
        assert_eq!(array.step, step);
        assert_eq!(array.name, "activations");
        assert_eq!(array.value, [step as f32, -1.0]);
        assert_eq!(array.timestamp_ms, TIMESTAMP_MS + step as i64);
    }
    Ok(())
}
