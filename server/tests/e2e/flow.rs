use std::time::Duration;

use anyhow::Result;
use bytes::Bytes;
use serde_json::{Value, json};

use crate::setup::{TestEnv, array, data_config, scalar, tar_file};

#[rstest::rstest]
#[tokio::test]
#[timeout(Duration::from_secs(60))]
async fn full_run_exercises_every_http_and_grpc_endpoint() -> Result<()> {
    let env = TestEnv::start().await?;
    let dataset = env.seed_dataset(2).await?;
    let asset = Bytes::from_static(b"opaque model weights, not an archive");
    env.put_object("inputs/model", asset.clone()).await?;
    let mut config = data_config(dataset, 2);
    config["assets"] = json!({"model": {"object": "inputs/model", "entrypoint": "weights"}});
    let train_config = json!({"steps": 2, "nested": {"anything": true}});
    let run = env.create_run(config.clone(), train_config.clone()).await?;
    let fetched: Value = env
        .http
        .get(format!("{}/runs/{run}", env.http_url))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert_eq!(fetched["run_id"], run);
    assert_eq!(fetched["data_config"], config);
    assert_eq!(fetched["train_config"], train_config);
    assert_eq!(fetched["status"], "queued");
    let listed: Vec<Value> = env
        .http
        .get(format!("{}/runs", env.http_url))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0]["run_id"], run);

    let initialized = env.init_run(&run).await?;
    assert_eq!(initialized.run_id, run);
    assert_eq!(
        serde_json::from_str::<Value>(&initialized.train_config)?,
        train_config
    );
    assert_eq!(initialized.assets, ["model"]);
    let (metadata, downloaded) = env.asset(&run, "model").await?;
    assert_eq!(metadata.entrypoint.as_deref(), Some("weights"));
    assert_eq!(downloaded, asset);
    let training = env.batches(&run, false, 3).await?;
    assert_eq!(training.len(), 2);
    assert!(training.iter().all(|batch| !batch.batch.is_empty()));
    assert_eq!(env.batches(&run, true, 1).await?.len(), 1);

    env.checkpoint(&run, 2, tar_file("model", b"trained weights")?)
        .await?;
    let metrics = env.metrics_stream(&run).await?;
    metrics.send(scalar(2)).await?;
    metrics.send(array(2)).await?;
    let artifact = tar_file("audio.wav", b"test audio")?;
    metrics
        .artifact(2, "validation/audio", "audio/wav", artifact.clone())
        .await?;
    let accepted = metrics.finish().await?;
    assert_eq!(accepted.metrics_received, 1);
    assert_eq!(accepted.array_metrics_received, 1);
    assert_eq!(accepted.artifacts_received, 1);
    assert_eq!(accepted.artifact_bytes_received, artifact.len() as u64);
    env.wait_count("assets", &run, 1).await?;
    env.wait_count("artifacts", &run, 1).await?;
    assert_eq!(env.count("metrics", &run).await?, 1);
    assert_eq!(env.count("array_metrics", &run).await?, 1);
    env.end_run(&run).await?;
    assert_eq!(env.status(&run).await?, "succeeded");
    assert!(!env.run_cache(&run).exists());
    Ok(())
}
