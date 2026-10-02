use crate::setup::{TestEnv, eventually, files, run_config};
use anyhow::Result;
use bytes::Bytes;
use serde_json::json;
use std::time::Duration;
#[rstest::rstest]
#[tokio::test]
#[timeout(Duration::from_secs(60))]
async fn data_cache_is_bounded_and_removed_per_run() -> Result<()> {
    let env = TestEnv::start().await?;
    let dataset = env.seed_dataset(2).await?;
    let a = env.create_run(run_config(dataset, 40)).await?;
    let b = env.create_run(run_config(dataset, 1)).await?;
    env.init_run(&a).await?;
    env.init_run(&b).await?;
    let cache = env.run_cache(&a).join("data/0");
    eventually("twenty cached batches", || async {
        Ok((files(&cache)?.len() == 20).then_some(()))
    })
    .await?;
    env.pause_s3().await?;
    assert_eq!(env.batches(&a, false, 1).await?.len(), 1);
    env.end_run(&a).await?;
    assert!(!env.run_cache(&a).exists());
    assert!(env.run_cache(&b).exists());
    env.resume_s3().await?;
    env.end_run(&b).await?;
    Ok(())
}
#[tokio::test]
async fn assets_are_cached_unchanged_and_initialization_failure_cleans_up() -> Result<()> {
    let env = TestEnv::start().await?;
    let dataset = env.seed_dataset(2).await?;
    let body = Bytes::from(vec![29; 3 * 1024 * 1024 + 13]);
    env.put_object("inputs/model", body.clone()).await?;
    let mut config = run_config(dataset, 1);
    config["assets"] = json!({"model":{"object":"inputs/model"}});
    let id = env.create_run(config).await?;
    env.init_run(&id).await?;
    env.s3
        .delete_object()
        .bucket(env.bucket)
        .key("inputs/model")
        .send()
        .await?;
    assert_eq!(env.asset(&id, "model").await?.1, body);
    env.end_run(&id).await?;
    let mut invalid = run_config(dataset, 1);
    invalid["assets"] = json!({"missing":{"object":"does-not-exist"}});
    let failed = env.create_run(invalid).await?;
    assert!(env.init_run(&failed).await.is_err());
    assert!(!env.run_cache(&failed).exists());
    Ok(())
}
#[tokio::test]
async fn missing_blob_fails_instead_of_skipping_a_batch() -> Result<()> {
    let env = TestEnv::start().await?;
    let dataset = env.seed_dataset(2).await?;
    let mut config = run_config(dataset, 1);
    config["queries"]["training"] = json!(
        "SELECT 'missing' AS sample_id,toUInt64(0) AS batch_idx,toUInt64(0) AS sample_idx,'{}' AS metadata_json,'{\"payload\":{\"object\":\"missing\"}}' AS blobs_json"
    );
    let id = env.create_run(config).await?;
    env.init_run(&id).await?;
    assert!(env.batches(&id, false, 1).await.is_err());
    env.end_run(&id).await?;
    Ok(())
}

#[tokio::test]
async fn oversized_batches_and_short_ranges_propagate_loading_errors() -> Result<()> {
    let env = TestEnv::start().await?;
    let dataset = env.seed_dataset(1).await?;
    env.put_object("oversized", Bytes::from(vec![1; 64 * 1024 * 1024]))
        .await?;
    let mut config = run_config(dataset, 1);
    config["queries"]["training"] = json!(
        "SELECT 'big' AS sample_id,toUInt64(0) AS batch_idx,toUInt64(0) AS sample_idx,'{}' AS metadata_json,'{\"payload\":{\"object\":\"oversized\"}}' AS blobs_json"
    );
    let run = env.create_run(config.clone()).await?;
    env.init_run(&run).await?;
    let error = env.batches(&run, false, 1).await.unwrap_err().to_string();
    assert!(error.contains("64 MiB"), "{error}");
    env.end_run(&run).await?;
    env.put_object("short", Bytes::from_static(b"abc")).await?;
    config["queries"]["training"] = json!(
        "SELECT 'range' AS sample_id,toUInt64(0) AS batch_idx,toUInt64(0) AS sample_idx,'{}' AS metadata_json,'{\"payload\":{\"object\":\"short\",\"byte_offset\":1,\"byte_length\":8}}' AS blobs_json"
    );
    let run = env.create_run(config).await?;
    env.init_run(&run).await?;
    assert!(
        env.batches(&run, false, 1)
            .await
            .unwrap_err()
            .to_string()
            .contains("unexpected byte length")
    );
    env.end_run(&run).await?;
    Ok(())
}
