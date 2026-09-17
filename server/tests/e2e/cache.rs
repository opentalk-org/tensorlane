use std::{collections::BTreeMap, time::Duration};

use anyhow::Result;
use bytes::Bytes;
use serde_json::json;

use crate::setup::{TestEnv, data_config, eventually, files};

#[rstest::rstest]
#[tokio::test]
#[timeout(Duration::from_secs(60))]
async fn data_cache_populates_removes_consumed_files_and_cleans_up_per_run() -> Result<()> {
    let env = TestEnv::start().await?;
    let dataset = env.seed_dataset(2).await?;
    let mut config = data_config(dataset, 2);
    config["training"][0]["max_seconds"] = json!(0.375);
    let run = env.create_run(config.clone(), json!({})).await?;
    let other = env.create_run(config, json!({})).await?;
    env.init_run(&run).await?;
    env.init_run(&other).await?;
    let training = env.run_cache(&run).join("data/training");
    let validation = env.run_cache(&run).join("data/validation");
    let before = eventually("two prefetched training batches (three samples)", || async {
        let paths = files(&training)?;
        let complete = paths.iter().try_fold(true, |complete, path| {
            Ok::<_, std::io::Error>(complete && path.metadata()?.len() == 12_000)
        })?;
        Ok((paths.len() == 3 && complete).then_some(paths))
    })
    .await?;
    let validation_before = eventually("full validation cache", || async {
        let paths = files(&validation)?;
        Ok((paths.len() == 20).then_some(paths))
    })
    .await?;
    let original = before
        .iter()
        .map(|path| Ok((path.clone(), std::fs::read(path)?)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    assert!(original.values().all(|wave| wave.len() == 12_000));
    let response = env.batches(&run, false, 1).await?;
    assert_eq!(response.len(), 1);
    assert_eq!(response[0].batch.len(), 2);
    let after = files(&training)?;
    assert_eq!(after.len(), 1);
    assert!(after.is_subset(&before));
    let consumed: Vec<_> = before.difference(&after).collect();
    assert_eq!(consumed.len(), 2);
    for sample in &response[0].batch {
        assert!(consumed.iter().any(|path| sample.wave.as_ref() == original[*path]));
    }
    assert_eq!(env.batches(&run, false, 2).await?.len(), 1);
    assert!(files(&training)?.is_empty());
    assert_eq!(env.batches(&run, true, 1).await?.len(), 1);
    assert_eq!(
        validation_before
            .iter()
            .filter(|path| !path.exists())
            .count(),
        1
    );

    env.end_run(&run).await?;
    assert!(!env.run_cache(&run).exists());
    assert!(env.run_cache(&other).is_dir());
    assert_eq!(env.batches(&other, false, 1).await?.len(), 1);
    env.end_run(&other).await?;
    assert!(!env.run_cache(&other).exists());
    Ok(())
}

#[rstest::rstest]
#[tokio::test]
#[timeout(Duration::from_secs(60))]
async fn assets_are_prefetched_served_from_cache_and_removed_on_finish() -> Result<()> {
    let env = TestEnv::start().await?;
    let dataset = env.seed_dataset(2).await?;
    let body = Bytes::from(vec![29; 3 * 1024 * 1024 + 13]);
    env.put_object("inputs/model", body.clone()).await?;
    let mut config = data_config(dataset, 1);
    config["assets"] = json!({"model": {"object": "inputs/model"}});
    let run = env.create_run(config, json!({})).await?;
    let initialized = env.init_run(&run).await?;
    assert_eq!(initialized.assets, ["model"]);
    let path = env.run_cache(&run).join("assets/model");
    assert_eq!(std::fs::read(&path)?, body);
    env.s3
        .delete_object()
        .bucket(env.bucket)
        .key("inputs/model")
        .send()
        .await?;
    let (metadata, downloaded) = env.asset(&run, "model").await?;
    assert!(metadata.entrypoint.is_none());
    assert_eq!(downloaded, body);
    assert!(path.exists());
    env.end_run(&run).await?;
    assert!(!env.run_cache(&run).exists());
    Ok(())
}

#[rstest::rstest]
#[tokio::test]
#[timeout(Duration::from_secs(60))]
async fn failed_asset_initialization_removes_entire_run_cache() -> Result<()> {
    let env = TestEnv::start().await?;
    let dataset = env.seed_dataset(2).await?;
    env.put_object("inputs/available", Bytes::from_static(b"cached asset"))
        .await?;
    let mut config = data_config(dataset, 1);
    config["assets"] = json!({
        "available": {"object": "inputs/available"},
        "missing": {"object": "inputs/does-not-exist"}
    });
    let failed = env.create_run(config, json!({})).await?;
    assert!(env.init_run(&failed).await.is_err());
    assert!(!env.run_cache(&failed).exists());
    let healthy = env.create_run(data_config(dataset, 1), json!({})).await?;
    env.init_run(&healthy).await?;
    assert_eq!(env.batches(&healthy, false, 1).await?.len(), 1);
    env.end_run(&healthy).await?;
    Ok(())
}

#[rstest::rstest]
#[tokio::test]
#[timeout(Duration::from_secs(60))]
async fn finish_cancels_blocked_prefetch_and_removes_cache() -> Result<()> {
    let env = TestEnv::start().await?;
    let dataset = env.seed_dataset(2).await?;
    let mut config = data_config(dataset, 40);
    config["training"][0]["max_seconds"] = json!(0.375);
    let run = env.create_run(config, json!({})).await?;
    env.init_run(&run).await?;
    eventually("training cache filled", || async {
        Ok((files(&env.run_cache(&run).join("data/training"))?.len() == 30).then_some(()))
    })
    .await?;
    env.pause_s3().await?;
    assert_eq!(env.batches(&run, false, 1).await?.len(), 1);
    env.end_run(&run).await?;
    assert!(!env.run_cache(&run).exists());
    env.resume_s3().await?;
    assert_eq!(env.status(&run).await?, "succeeded");
    Ok(())
}
