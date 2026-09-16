use std::time::Duration;

use anyhow::Result;
use bytes::Bytes;
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};

use crate::setup::{TIMESTAMP_MS, TestEnv, data_config, eventually, tar_file};

#[derive(clickhouse::Row, Deserialize)]
struct Checkpoint {
    id: String,
    run: String,
    step: u64,
    path: String,
    size: u64,
    hash: String,
    asset_type: String,
    kind: String,
    name: String,
}

#[derive(clickhouse::Row, Deserialize)]
struct Artifact {
    id: String,
    run: String,
    step: u64,
    timestamp_ms: i64,
    name: String,
    path: String,
    content_type: String,
    size_bytes: u64,
}

async fn assert_object(env: &TestEnv, key: &str, expected: &Bytes) -> Result<()> {
    let head = env
        .s3
        .head_object()
        .bucket(env.bucket)
        .key(key)
        .send()
        .await?;
    assert_eq!(head.content_length(), Some(expected.len() as i64));
    assert_eq!(head.content_type(), Some("application/x-tar"));
    let downloaded = env.object(key).await?;
    assert_eq!(Sha256::digest(&downloaded), Sha256::digest(expected));
    assert_eq!(&downloaded, expected);
    Ok(())
}

#[rstest::rstest]
#[tokio::test]
#[timeout(Duration::from_secs(60))]
async fn multipart_checkpoint_matches_s3_and_clickhouse_metadata() -> Result<()> {
    let env = TestEnv::start().await?;
    let dataset = env.seed_dataset(2).await?;
    let mut config = data_config(dataset, 1);
    config["asset_type"] = json!("custom-model-type");
    let run = env.create_run(config, json!({})).await?;
    env.init_run(&run).await?;
    let contents: Vec<u8> = (0..64 * 1024 * 1024 + 123)
        .map(|index| (index % 251) as u8)
        .collect();
    let archive = tar_file("weights/model.pth", &contents)?;
    env.checkpoint(&run, 42, archive.clone()).await?;
    env.wait_count("assets", &run, 1).await?;
    let row = env.clickhouse.query("SELECT toString(id) AS id, toString(run_id) AS run, step, path, size, toString(content_hash) AS hash, type AS asset_type, toString(kind) AS kind, name FROM assets FINAL WHERE run_id = toUUID(?)")
        .bind(&run).fetch_one::<Checkpoint>().await?;
    assert_eq!(row.run, run);
    assert_eq!(row.step, 42);
    assert_eq!(row.path, format!("checkpoints/{}", row.id));
    assert_eq!(row.size, archive.len() as u64);
    assert_eq!(row.hash, hex::encode(Sha256::digest(&archive)));
    assert_eq!(row.asset_type, "custom-model-type");
    assert_eq!(row.kind, "checkpoint");
    assert_eq!(row.name, format!("{run}_000000042"));
    assert_object(&env, &row.path, &archive).await?;
    let head = env
        .s3
        .head_object()
        .bucket(env.bucket)
        .key(&row.path)
        .send()
        .await?;
    assert!(
        head.e_tag().is_some_and(|etag| etag.ends_with("-2\"")),
        "expected two uploaded parts: {:?}",
        head.e_tag()
    );
    eventually("checkpoint staging cleanup", || async {
        Ok((!env.cache_dir.join("uploads").join(&row.id).exists()).then_some(()))
    })
    .await?;
    env.end_run(&run).await?;
    assert_eq!(env.count("assets", &run).await?, 1);
    Ok(())
}

#[rstest::rstest]
#[tokio::test]
#[timeout(Duration::from_secs(60))]
async fn metric_artifact_matches_s3_and_clickhouse_metadata() -> Result<()> {
    let env = TestEnv::start().await?;
    let dataset = env.seed_dataset(2).await?;
    let run = env.create_run(data_config(dataset, 1), json!({})).await?;
    env.init_run(&run).await?;
    let archive = tar_file("validation/audio.wav", &[37; 1024 * 1024 + 19])?;
    let stream = env.metrics_stream(&run).await?;
    stream
        .artifact(17, "validation/audio", "audio/wav", archive.clone())
        .await?;
    let accepted = stream.finish().await?;
    assert_eq!(accepted.artifacts_received, 1);
    assert_eq!(accepted.artifact_bytes_received, archive.len() as u64);
    env.wait_count("artifacts", &run, 1).await?;
    let row = env.clickhouse.query("SELECT toString(id) AS id, toString(run_id) AS run, step, toUnixTimestamp64Milli(timestamp) AS timestamp_ms, name, path, content_type, size_bytes FROM artifacts WHERE run_id = toUUID(?)")
        .bind(&run).fetch_one::<Artifact>().await?;
    assert_eq!(row.run, run);
    assert_eq!(row.step, 17);
    assert_eq!(row.timestamp_ms, TIMESTAMP_MS);
    assert_eq!(row.name, "validation/audio");
    assert_eq!(row.path, format!("metrics/{}", row.id));
    assert_eq!(row.content_type, "audio/wav");
    assert_eq!(row.size_bytes, archive.len() as u64);
    assert_object(&env, &row.path, &archive).await?;
    eventually("artifact staging cleanup", || async {
        Ok((!env.cache_dir.join("uploads").join(&row.id).exists()).then_some(()))
    })
    .await?;
    env.end_run(&run).await?;
    assert_eq!(env.count("artifacts", &run).await?, 1);
    Ok(())
}
