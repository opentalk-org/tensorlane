use std::time::Duration;

use anyhow::Result;
use bytes::Bytes;
use serde_json::{Value, json};
use tonic::Code;

use crate::{
    proto,
    setup::{TestEnv, data_config, eventually, files, tar_file},
};

async fn shutdown_with_runs(running: usize, initializing: usize) -> Result<()> {
    let mut env = TestEnv::start().await?;
    let dataset = env.seed_dataset(2).await?;
    env.put_object("inputs/slow-asset", Bytes::from_static(b"model"))
        .await?;
    let mut config = data_config(dataset, 2);
    config["assets"] = json!({"model": {"object": "inputs/slow-asset"}});
    let mut runs = Vec::new();
    for _ in 0..running {
        let run = env.create_run(config.clone(), json!({})).await?;
        env.init_run(&run).await?;
        runs.push(run);
    }

    let mut pending = Vec::new();
    if initializing > 0 {
        env.pause_s3().await?;
        for _ in 0..initializing {
            let run = env.create_run(config.clone(), json!({})).await?;
            let mut client = env.grpc.clone();
            let request = proto::InitRequest {
                run_id: run.clone(),
            };
            pending.push(tokio::spawn(async move { client.init(request).await }));
            runs.push(run);
        }
        for run in &runs {
            eventually("admitted initializing run", || async {
                Ok((env.status(run).await? == "running").then_some(()))
            })
            .await?;
        }
        assert!(pending.iter().all(|task| !task.is_finished()));
    }

    env.signal("-TERM")?;
    env.wait_shutdown().await?;
    assert!(env.server.try_wait()?.is_none());
    let listed: Vec<Value> = env
        .http
        .get(format!("{}/runs", env.http_url))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert_eq!(listed.len(), runs.len());
    let rejected = env.create_run(config, json!({})).await?;
    let result = env
        .grpc
        .clone()
        .init(proto::InitRequest {
            run_id: rejected.clone(),
        })
        .await;
    assert_eq!(result.unwrap_err().code(), Code::Unavailable);
    assert_eq!(env.status(&rejected).await?, "queued");
    assert!(!env.run_cache(&rejected).exists());

    if initializing > 0 {
        assert!(pending.iter().all(|task| !task.is_finished()));
        env.resume_s3().await?;
        for task in pending {
            task.await??;
        }
    }
    for (index, run) in runs.iter().enumerate() {
        assert_eq!(env.batches(run, false, 1).await?.len(), 1);
        assert_eq!(env.batches(run, true, 1).await?.len(), 1);
        env.end_run(run).await?;
        assert!(!env.run_cache(run).exists());
        assert_eq!(env.status(run).await?, "succeeded");
        if index + 1 < runs.len() {
            assert!(env.server.try_wait()?.is_none());
        }
    }
    assert!(env.wait_exit().await?.success());
    Ok(())
}

#[rstest::rstest]
#[tokio::test]
#[timeout(Duration::from_secs(60))]
async fn shutdown_waits_for_running_run_and_rejects_new_init() -> Result<()> {
    shutdown_with_runs(1, 0).await
}

#[rstest::rstest]
#[tokio::test]
#[timeout(Duration::from_secs(60))]
async fn shutdown_waits_for_multiple_running_runs() -> Result<()> {
    shutdown_with_runs(3, 0).await
}

#[rstest::rstest]
#[tokio::test]
#[timeout(Duration::from_secs(60))]
async fn shutdown_waits_for_initializing_run() -> Result<()> {
    shutdown_with_runs(0, 1).await
}

#[rstest::rstest]
#[tokio::test]
#[timeout(Duration::from_secs(60))]
async fn shutdown_waits_for_multiple_initializing_runs() -> Result<()> {
    shutdown_with_runs(0, 3).await
}

#[rstest::rstest]
#[tokio::test]
#[timeout(Duration::from_secs(60))]
async fn shutdown_waits_for_running_and_initializing_runs() -> Result<()> {
    shutdown_with_runs(1, 2).await
}

#[rstest::rstest]
#[tokio::test]
#[timeout(Duration::from_secs(60))]
async fn failed_initialization_releases_shutdown_and_cleans_cache() -> Result<()> {
    let mut env = TestEnv::start().await?;
    let dataset = env.seed_dataset(2).await?;
    let mut config = data_config(dataset, 1);
    config["assets"] = json!({"missing": {"object": "inputs/missing"}});
    let run = env.create_run(config, json!({})).await?;
    env.pause_s3().await?;
    let mut client = env.grpc.clone();
    let request = proto::InitRequest {
        run_id: run.clone(),
    };
    let initialization = tokio::spawn(async move { client.init(request).await });
    eventually("admitted initialization", || async {
        Ok((env.status(&run).await? == "running").then_some(()))
    })
    .await?;
    env.signal("-TERM")?;
    env.wait_shutdown().await?;
    assert!(!initialization.is_finished());
    assert!(env.server.try_wait()?.is_none());
    env.resume_s3().await?;
    let result = initialization.await?;
    assert_eq!(result.unwrap_err().code(), Code::Internal);
    assert!(env.wait_exit().await?.success());
    assert!(!env.run_cache(&run).exists());
    Ok(())
}

#[rstest::rstest]
#[tokio::test]
#[timeout(Duration::from_secs(60))]
async fn sigterm_immediately_after_startup_exits_cleanly() -> Result<()> {
    let mut env = TestEnv::start().await?;
    env.signal("-TERM")?;
    assert!(env.wait_exit().await?.success());
    Ok(())
}

#[rstest::rstest]
#[tokio::test]
#[timeout(Duration::from_secs(60))]
async fn sigint_with_no_runs_exits_cleanly() -> Result<()> {
    let mut env = TestEnv::start().await?;
    env.signal("-INT")?;
    assert!(env.wait_exit().await?.success());
    Ok(())
}

#[rstest::rstest]
#[tokio::test]
#[timeout(Duration::from_secs(60))]
async fn queued_but_not_initialized_runs_do_not_block_shutdown() -> Result<()> {
    let mut env = TestEnv::start().await?;
    let run = env
        .create_run(data_config(uuid::Uuid::new_v4(), 1), json!({}))
        .await?;
    env.signal("-TERM")?;
    assert!(env.wait_exit().await?.success());
    assert_eq!(env.status(&run).await?, "queued");
    Ok(())
}

#[rstest::rstest]
#[tokio::test]
#[timeout(Duration::from_secs(60))]
async fn shutdown_waits_for_pending_checkpoint_and_artifact_uploads() -> Result<()> {
    let mut env = TestEnv::start().await?;
    let dataset = env.seed_dataset(2).await?;
    let run = env.create_run(data_config(dataset, 1), json!({})).await?;
    env.init_run(&run).await?;
    env.pause_s3().await?;
    let archive = tar_file("weights", b"upload must finish before server exits")?;
    env.checkpoint(&run, 1, archive.clone()).await?;
    let stream = env.metrics_stream(&run).await?;
    stream
        .artifact(1, "artifact", "application/octet-stream", archive.clone())
        .await?;
    stream.finish().await?;
    assert_eq!(files(&env.cache_dir.join("uploads"))?.len(), 2);
    assert_eq!(env.count("assets", &run).await?, 0);
    assert_eq!(env.count("artifacts", &run).await?, 0);
    env.signal("-TERM")?;
    env.wait_shutdown().await?;
    env.end_run(&run).await?;
    assert!(env.server.try_wait()?.is_none());
    env.resume_s3().await?;
    assert!(env.wait_exit().await?.success());
    assert_eq!(env.count("assets", &run).await?, 1);
    assert_eq!(env.count("artifacts", &run).await?, 1);
    for table in ["assets", "artifacts"] {
        let path = env
            .clickhouse
            .query(&format!(
                "SELECT path FROM {table} WHERE run_id = toUUID(?)"
            ))
            .bind(&run)
            .fetch_one::<String>()
            .await?;
        assert_eq!(env.object(&path).await?, archive);
    }
    assert!(files(&env.cache_dir.join("uploads"))?.is_empty());
    Ok(())
}
