use crate::setup::{TestEnv, array, run_config, scalar, tar_file};
use anyhow::Result;
use bytes::Bytes;
use serde_json::{Value, json};
use std::{path::Path, time::Duration};
#[rstest::rstest]
#[tokio::test]
#[timeout(Duration::from_secs(60))]
async fn full_run_exercises_http_grpc_and_generic_data() -> Result<()> {
    let env = TestEnv::start().await?;
    let dataset = env.seed_dataset(3).await?;
    let asset = Bytes::from_static(b"opaque input weights");
    env.put_object("inputs/model", asset.clone()).await?;
    let mut config = run_config(dataset, 2);
    config["assets"] = json!({"model":{"object":"inputs/model","entrypoint":"weights"}});
    config["optimizer"] = json!({"nested":{"anything":true}});
    let run = env.create_run(config.clone()).await?;
    let fetched: Value = env
        .http
        .get(format!("{}/runs/{run}", env.http_url))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert_eq!(fetched["config"], config);
    assert_eq!(fetched["status"], "queued");
    let init = env.init_run(&run).await?;
    assert_eq!(serde_json::from_str::<Value>(&init.config)?, config);
    assert_eq!(init.streams, ["training", "validation"]);
    assert_eq!(init.assets, ["model"]);
    let (meta, body) = env.asset(&run, "model").await?;
    assert_eq!(meta.entrypoint.as_deref(), Some("weights"));
    assert_eq!(body, asset);
    let batches = env.batches(&run, false, 3).await?;
    assert_eq!(batches.len(), 2);
    assert_eq!(
        batches
            .iter()
            .map(|batch| batch.batch_id)
            .collect::<Vec<_>>(),
        [0, 1]
    );
    assert_eq!(
        batches[0].batch[0].blobs["payload"].as_ref(),
        (0..16).collect::<Vec<u8>>()
    );
    assert_eq!(
        batches[0].batch[0].blobs["context"].as_ref(),
        b"arbitrary context bytes"
    );
    let saved = env
        .checkpoint(&run, 2, tar_file("model", b"trained weights")?)
        .await?;
    let history: Value = env
        .http
        .get(format!("{}/runs/{run}/assets?name=model", env.http_url))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert_eq!(history[0]["id"], saved);
    assert_eq!(history[0]["name"], "model");
    let metrics = env.metrics_stream(&run).await?;
    metrics.send(scalar(2)).await?;
    metrics.send(array(2)).await?;
    metrics
        .artifact(
            2,
            "validation/report",
            "application/json",
            tar_file("report.json", b"{}")?,
        )
        .await?;
    let accepted = metrics.finish().await?;
    assert_eq!(accepted.metrics_received, 1);
    assert_eq!(accepted.array_metrics_received, 1);
    env.wait_count("artifacts", &run, 1).await?;
    env.end_run(&run).await?;
    assert_eq!(env.status(&run).await?, "succeeded");
    Ok(())
}
#[tokio::test]
async fn offsets_and_batch_sizes_continue_across_separate_runs() -> Result<()> {
    let env = TestEnv::start().await?;
    let dataset = env.seed_dataset(12).await?;
    let mut first = run_config(dataset, 2);
    first["training"]["batch_size"] = json!(3);
    let mut second = run_config(dataset, 3);
    second["start_params"]["dataset_offset"] = json!(6);
    second["training"]["batch_size"] = json!(2);
    let a = env.create_run(first).await?;
    let b = env.create_run(second).await?;
    env.init_run(&a).await?;
    env.init_run(&b).await?;
    let mut positions = Vec::new();
    for batch in env
        .batches(&a, false, 3)
        .await?
        .into_iter()
        .chain(env.batches(&b, false, 4).await?)
    {
        for sample in batch.batch {
            positions.push(
                serde_json::from_str::<Value>(&sample.metadata_json)?["position"]
                    .as_u64()
                    .unwrap(),
            );
        }
    }
    assert_eq!(positions, (0..12).collect::<Vec<u64>>());
    env.end_run(&a).await?;
    env.end_run(&b).await?;
    Ok(())
}
#[tokio::test]
async fn third_query_and_metadata_only_samples_work() -> Result<()> {
    let env = TestEnv::start().await?;
    let dataset = env.seed_dataset(2).await?;
    let mut config = run_config(dataset, 1);
    config["queries"]["evaluation"] = json!(
        "SELECT 'text' AS sample_id, toUInt64(7) AS batch_idx, toUInt64(0) AS sample_idx, '{\"text\":\"hello\"}' AS metadata_json, '{}' AS blobs_json"
    );
    let id = env.create_run(config).await?;
    env.init_run(&id).await?;
    let batches = env.stream_batches(&id, "evaluation", 2).await?;
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].query_batch_idx, 7);
    assert!(batches[0].batch[0].blobs.is_empty());
    env.end_run(&id).await?;
    Ok(())
}
#[tokio::test]
async fn invalid_config_is_rejected_and_historical_configs_remain_readable() -> Result<()> {
    let env = TestEnv::start().await?;
    let response=env.http.post(format!("{}/runs",env.http_url)).json(&json!({"project_id":uuid::Uuid::new_v4(),"name":"invalid","config":{"queries":{"training":"SELECT 1"},"training":[{}]}})).send().await?;
    assert_eq!(response.status(), 400);
    let id = uuid::Uuid::new_v4();
    let historical = json!({"data_config":{"training":[{},{}]},"train_config":{"model":"legacy"}});
    env.clickhouse
        .query("INSERT INTO runs (id,project_id,name,config) VALUES (?,?,?,?)")
        .bind(id)
        .bind(uuid::Uuid::new_v4())
        .bind("history")
        .bind(historical.to_string())
        .execute()
        .await?;
    let value: Value = env
        .http
        .get(format!("{}/runs/{id}", env.http_url))
        .send()
        .await?
        .json()
        .await?;
    assert_eq!(value["config"], historical);
    assert!(env.init_run(&id.to_string()).await.is_err());
    Ok(())
}

#[tokio::test]
async fn migrations_preserve_legacy_sections_and_verify_before_retirement() -> Result<()> {
    let id = uuid::Uuid::new_v4();
    let data = json!({"queries":{"training":"old SQL"},"collision":"data","training":[{},{}],"application":[1,2]});
    let training = json!({"collision":"train","nested":{"anything":true}});
    let env = TestEnv::start_with_history(Some((id, data.clone(), training.clone()))).await?;
    let fetched: Value = env
        .http
        .get(format!("{}/runs/{id}", env.http_url))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert_eq!(
        fetched["config"],
        json!({"data_config":data,"train_config":training})
    );
    let columns: Vec<String> = env
        .clickhouse
        .query("SELECT name FROM system.columns WHERE database=currentDatabase() AND table='runs'")
        .fetch_all()
        .await?;
    assert!(columns.contains(&"config".into()));
    assert!(!columns.contains(&"data_config".into()) && !columns.contains(&"train_config".into()));
    assert!(env.init_run(&id.to_string()).await.is_err());
    env.clickhouse.query("CREATE TABLE runs_guard (config String,data_config String,train_config String) ENGINE=Memory").execute().await?;
    env.clickhouse
        .query("INSERT INTO runs_guard VALUES ('','{}','{}')")
        .execute()
        .await?;
    let migration = include_str!("../../../db/migrations/20261002120200.sql");
    for statement in migration
        .split(';')
        .filter(|statement| !statement.trim().is_empty())
    {
        let result = env
            .clickhouse
            .query(&statement.replace("runs", "runs_guard"))
            .execute()
            .await;
        if result.is_err() {
            break;
        }
    }
    assert_eq!(env.clickhouse.query("SELECT count() FROM system.columns WHERE database=currentDatabase() AND table='runs_guard' AND name IN ('data_config','train_config')").fetch_one::<u64>().await?,2);
    assert_eq!(env.clickhouse.query("SELECT count() FROM system.tables WHERE database=currentDatabase() AND name='audio_files'").fetch_one::<u64>().await?,1);
    Ok(())
}

#[tokio::test]
async fn repeat_replays_the_original_query_and_finite_validation_can_override_default() -> Result<()>
{
    let env = TestEnv::start().await?;
    let dataset = env.seed_dataset(2).await?;
    let mut config = run_config(dataset, 2);
    config["training"]["repeat"] = json!(true);
    config["validation"]["repeat"] = json!(false);
    let id = env.create_run(config).await?;
    env.init_run(&id).await?;
    env.clickhouse.query("ALTER TABLE example_samples UPDATE metadata_json='{\"position\":99}' WHERE dataset_id=? SETTINGS mutations_sync=2").bind(dataset).execute().await?;
    let batches = env.batches(&id, false, 5).await?;
    assert_eq!(
        batches
            .iter()
            .map(|batch| batch.batch_id)
            .collect::<Vec<_>>(),
        [0, 1, 2, 3, 4]
    );
    assert_eq!(
        batches
            .iter()
            .map(|batch| batch.query_batch_idx)
            .collect::<Vec<_>>(),
        [0, 1, 0, 1, 0]
    );
    for batch in batches {
        assert!(
            serde_json::from_str::<Value>(&batch.batch[0].metadata_json)?["position"]
                .as_u64()
                .unwrap()
                < 2
        );
    }
    assert_eq!(env.batches(&id, true, 2).await?.len(), 1);
    env.end_run(&id).await?;
    Ok(())
}

#[tokio::test]
async fn mismatched_batch_count_and_unsorted_query_fail_initialization() -> Result<()> {
    let env = TestEnv::start().await?;
    let dataset = env.seed_dataset(2).await?;
    let mut config = run_config(dataset, 2);
    config["queries"]["training"] = json!(
        "SELECT 'x' AS sample_id,toUInt64(0) AS batch_idx,toUInt64(0) AS sample_idx,'{}' AS metadata_json,'{}' AS blobs_json"
    );
    let id = env.create_run(config.clone()).await?;
    assert!(
        env.init_run(&id)
            .await
            .unwrap_err()
            .to_string()
            .contains("expected 2")
    );
    config["queries"]["training"] = json!(
        "SELECT 'x' AS sample_id,number AS batch_idx,number AS sample_idx,'{}' AS metadata_json,'{}' AS blobs_json FROM numbers(2) ORDER BY number DESC"
    );
    let id = env.create_run(config).await?;
    assert!(env.init_run(&id).await.is_err());
    assert!(!env.run_cache(&id).exists());
    Ok(())
}

#[tokio::test]
async fn complete_examples_continue_through_the_native_python_pipeline() -> Result<()> {
    let Some(python) = std::env::var_os("TENSORLANE_TEST_PYTHON") else {
        return Ok(());
    };
    let env = TestEnv::start().await?;
    let dataset = env.seed_dataset(18).await?;
    let mut first: Value = serde_json::from_str(include_str!("../../../sample-configs.json"))?;
    first["config"]["dataset_id"] = json!(dataset);
    first["config"]["training"] = json!({"batches":3,"batch_size":3});
    first["config"]["num_workers"] = json!(2);
    first["config"]["validation"] = json!({"samples":4,"batch_size":2});
    let mut second: Value =
        serde_json::from_str(include_str!("../../../sample-configs-stage2.json"))?;
    second["config"]["training"] = json!({"batches":2,"batch_size":2});
    second["config"]["validation"] = json!({"samples":4,"batch_size":2});
    let temp = tempfile::tempdir()?;
    let output = temp.path().join("training");
    first["config"]["output_dir"] = json!(output);
    let run = env.create_run(first["config"].clone()).await?;
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let result = tokio::process::Command::new(&python)
        .current_dir(root)
        .args([
            "-m",
            "accelerate.commands.launch",
            "--multi_gpu",
            "--num_processes=2",
            "--num_machines=1",
            "--mixed_precision=no",
            "--dynamo_backend=no",
            "--num_cpu_threads_per_process=1",
            "--main_process_port=0",
            "client/examples/train.py",
        ])
        .env("ACCELERATE_USE_CPU", "true")
        .env("TENSORLANE_RUN_ID", &run)
        .env("TENSORLANE_ADDR", &env.grpc_url)
        .output()
        .await?;
    anyhow::ensure!(
        result.status.success(),
        "first training failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let progress: Value = serde_json::from_slice(&std::fs::read(output.join("progress.json"))?)?;
    assert_eq!(progress["dataset_offset"], 9);
    assert_eq!(env.status(&run).await?, "succeeded");
    let first_asset = progress["asset_id"].as_str().unwrap();
    second["config"]["dataset_id"] = json!(dataset);
    second["config"]["num_workers"] = json!(2);
    second["config"]["output_dir"] = json!(output);
    second["config"]["start_params"]["dataset_offset"] = progress["dataset_offset"].clone();
    second["config"]["assets"]["model"]["asset_id"] = progress["asset_id"].clone();
    let run = env.create_run(second["config"].clone()).await?;
    let result = tokio::process::Command::new(&python)
        .current_dir(root)
        .args([
            "-m",
            "accelerate.commands.launch",
            "--multi_gpu",
            "--num_processes=2",
            "--num_machines=1",
            "--mixed_precision=no",
            "--dynamo_backend=no",
            "--num_cpu_threads_per_process=1",
            "--main_process_port=0",
            "client/examples/train2.py",
        ])
        .env("ACCELERATE_USE_CPU", "true")
        .env("TENSORLANE_RUN_ID", &run)
        .env("TENSORLANE_ADDR", &env.grpc_url)
        .output()
        .await?;
    anyhow::ensure!(
        result.status.success(),
        "second training failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let progress: Value = serde_json::from_slice(&std::fs::read(output.join("progress.json"))?)?;
    assert_eq!(progress["dataset_offset"], 13);
    let row: Value = env
        .http
        .get(format!(
            "{}/assets/{}",
            env.http_url,
            progress["asset_id"].as_str().unwrap()
        ))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert_eq!(row["ancestor_asset_id"], first_asset);
    assert_eq!(row["metadata"]["dataset_offset"], 13);
    assert_eq!(env.status(&run).await?, "succeeded");
    Ok(())
}
