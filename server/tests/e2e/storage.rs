use std::time::Duration;

use anyhow::Result;
use bytes::Bytes;
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};

use crate::setup::{TIMESTAMP_MS, TestEnv, eventually, run_config, tar_file};

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

async fn assert_object(
    env: &TestEnv,
    key: &str,
    expected: &Bytes,
    content_type: &str,
) -> Result<()> {
    let head = env
        .s3
        .head_object()
        .bucket(env.bucket)
        .key(key)
        .send()
        .await?;
    assert_eq!(head.content_length(), Some(expected.len() as i64));
    assert_eq!(head.content_type(), Some(content_type));
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
    let mut config = run_config(dataset, 1);
    config["tensorlane"]["asset_type"] = json!("custom-model-type");
    let run = env.create_run(config).await?;
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
    assert_eq!(row.name, "model");
    assert_object(&env, &row.path, &archive, "application/x-tar").await?;
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
        Ok(crate::setup::files(&env.cache_dir.join("uploads"))?
            .is_empty()
            .then_some(()))
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
    let run = env.create_run(run_config(dataset, 1)).await?;
    env.init_run(&run).await?;
    let archive = Bytes::from(vec![37; 1024 * 1024 + 19]);
    let stream = env.metrics_stream(&run).await?;
    stream
        .artifact(
            17,
            "validation/results",
            "application/octet-stream",
            archive.clone(),
        )
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
    assert_eq!(row.name, "validation/results");
    assert_eq!(row.path, format!("metrics/{}", row.id));
    assert_eq!(row.content_type, "application/octet-stream");
    assert_eq!(row.size_bytes, archive.len() as u64);
    assert_object(&env, &row.path, &archive, "application/octet-stream").await?;
    eventually("artifact staging cleanup", || async {
        Ok(crate::setup::files(&env.cache_dir.join("uploads"))?
            .is_empty()
            .then_some(()))
    })
    .await?;
    env.end_run(&run).await?;
    assert_eq!(env.count("artifacts", &run).await?, 1);
    Ok(())
}

#[tokio::test]
async fn named_asset_lineage_survives_new_runs_and_failed_saves() -> Result<()> {
    let env = TestEnv::start().await?;
    let dataset = env.seed_dataset(2).await?;
    let run = env.create_run(run_config(dataset, 1)).await?;
    env.init_run(&run).await?;
    let body = tar_file("model", b"weights")?;
    let a = env
        .save_asset(
            &run,
            "model",
            1,
            "checkpoint",
            uuid::Uuid::new_v4(),
            body.clone(),
        )
        .await?;
    let b_id = uuid::Uuid::new_v4();
    let b = env
        .save_asset(&run, "model", 2, "checkpoint", b_id, body.clone())
        .await?;
    assert_eq!(
        env.save_asset(&run, "model", 2, "checkpoint", b_id, body.clone())
            .await?,
        b
    );
    assert!(
        env.save_asset(
            &run,
            "model",
            2,
            "checkpoint",
            b_id,
            tar_file("model", b"changed")?
        )
        .await
        .is_err()
    );
    let independent = env
        .save_asset(&run, "other", 0, "file", uuid::Uuid::new_v4(), body.clone())
        .await?;
    let parent = |id: String| {
        let client = env.http.clone();
        let url = format!("{}/assets/{id}", env.http_url);
        async move {
            Ok::<_, anyhow::Error>(
                client
                    .get(url)
                    .send()
                    .await?
                    .error_for_status()?
                    .json::<serde_json::Value>()
                    .await?,
            )
        }
    };
    assert_eq!(parent(b.clone()).await?["ancestor_asset_id"], a);
    assert_eq!(
        parent(independent).await?["ancestor_asset_id"],
        uuid::Uuid::nil().to_string()
    );
    env.end_run(&run).await?;
    let mut config = run_config(dataset, 1);
    config["tensorlane"]["assets"] = json!({"model":{"asset_id":b}});
    config["tensorlane"]
        .as_object_mut()
        .unwrap()
        .remove("asset_type");
    let next = env.create_run(config).await?;
    env.init_run(&next).await?;
    assert_eq!(env.asset(&next, "model").await?.1, body);
    let c = env
        .save_asset(
            &next,
            "model",
            3,
            "checkpoint",
            uuid::Uuid::new_v4(),
            body.clone(),
        )
        .await?;
    let row = parent(c.clone()).await?;
    assert_eq!(row["ancestor_asset_id"], b);
    assert_eq!(row["asset_type"], "e2e-model");
    env.end_run(&next).await?;
    env.init_run(&next).await?;
    let d = env
        .save_asset(&next, "model", 4, "checkpoint", uuid::Uuid::new_v4(), body)
        .await?;
    assert_eq!(parent(d).await?["ancestor_asset_id"], c);
    env.end_run(&next).await?;
    Ok(())
}

#[tokio::test]
async fn simultaneous_saves_form_one_chain_and_committed_ids_are_idempotent() -> Result<()> {
    let env = TestEnv::start().await?;
    let dataset = env.seed_dataset(1).await?;
    let run = env.create_run(run_config(dataset, 1)).await?;
    env.init_run(&run).await?;
    let body = tar_file("weights", b"weights")?;
    let first = uuid::Uuid::new_v4();
    let second = uuid::Uuid::new_v4();
    let (a, b) = tokio::join!(
        env.save_asset(&run, "model", 1, "checkpoint", first, body.clone()),
        env.save_asset(&run, "model", 2, "checkpoint", second, body.clone())
    );
    a?;
    b?;
    let history: serde_json::Value = env
        .http
        .get(format!("{}/runs/{run}/assets?name=model", env.http_url))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert_eq!(history.as_array().unwrap().len(), 2);
    assert_eq!(
        history[0]["ancestor_asset_id"],
        uuid::Uuid::nil().to_string()
    );
    assert_eq!(history[1]["ancestor_asset_id"], history[0]["id"]);
    let retry = uuid::Uuid::new_v4();
    let (a, b) = tokio::join!(
        env.save_asset(&run, "model", 3, "file", retry, body.clone()),
        env.save_asset(&run, "model", 3, "file", retry, body.clone())
    );
    assert_eq!(a?, b?);
    assert_eq!(env.count("assets", &run).await?, 3);
    let conflict = uuid::Uuid::new_v4();
    let (a, b) = tokio::join!(
        env.save_asset(&run, "model", 4, "file", conflict, body.clone()),
        env.save_asset(&run, "other", 4, "file", conflict, body)
    );
    assert!(a.is_ok() != b.is_ok());
    assert_eq!(env.count("assets", &run).await?, 4);
    env.end_run(&run).await?;
    Ok(())
}

#[tokio::test]
async fn insert_failure_does_not_advance_lineage() -> Result<()> {
    let env = TestEnv::start().await?;
    let dataset = env.seed_dataset(1).await?;
    let run = env.create_run(run_config(dataset, 1)).await?;
    env.init_run(&run).await?;
    let body = tar_file("weights", b"weights")?;
    let first = env
        .save_asset(
            &run,
            "model",
            1,
            "checkpoint",
            uuid::Uuid::new_v4(),
            body.clone(),
        )
        .await?;
    env.clickhouse
        .query("ALTER TABLE assets ADD CONSTRAINT reject_step CHECK step != 2")
        .execute()
        .await?;
    let failed = uuid::Uuid::new_v4();
    assert!(
        env.save_asset(&run, "model", 2, "checkpoint", failed, body.clone())
            .await
            .is_err()
    );
    assert_eq!(env.count("assets", &run).await?, 1);
    env.clickhouse
        .query("ALTER TABLE assets DROP CONSTRAINT reject_step")
        .execute()
        .await?;
    let second = env
        .save_asset(&run, "model", 3, "checkpoint", uuid::Uuid::new_v4(), body)
        .await?;
    let row: serde_json::Value = env
        .http
        .get(format!("{}/assets/{second}", env.http_url))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert_eq!(row["ancestor_asset_id"], first);
    env.end_run(&run).await?;
    Ok(())
}
