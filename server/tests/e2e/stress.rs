use super::setup::{KEY, TestEnv};
use anyhow::{Context, Result, ensure};
use bytes::Bytes;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{path::Path, time::Duration};
use uuid::Uuid;

pub(super) struct TrainingClient {
    pub process: tokio::process::Child,
    pub group: i32,
}

impl Drop for TrainingClient {
    fn drop(&mut self) {
        unsafe {
            libc::kill(-self.group, libc::SIGKILL);
        }
    }
}

async fn barrier(
    root: &Path,
    runs: &[String],
    clients: &mut [TrainingClient],
    suffix: &str,
) -> Result<()> {
    let started = tokio::time::Instant::now();
    while !runs
        .iter()
        .all(|id| root.join(format!("{id}.{suffix}")).exists())
    {
        for client in &mut *clients {
            ensure!(
                client.process.try_wait()?.is_none(),
                "stress client {} exited before {suffix}",
                client.group
            );
        }
        ensure!(
            started.elapsed() < Duration::from_secs(180),
            "stress {suffix} timed out"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "four simultaneous Python trainings with a server restart"]
async fn four_concurrent_trainings_survive_restart() -> Result<()> {
    let python = std::env::var_os("TENSORLANE_TEST_PYTHON")
        .context("set TENSORLANE_TEST_PYTHON to a TensorLane Python environment")?;
    let mut env = TestEnv::start().await?;
    let root = tempfile::Builder::new()
        .prefix("tl-stress-")
        .tempdir_in("/tmp")?;
    let dataset = Uuid::new_v4();
    env.put_object("stress/data", Bytes::from(vec![23; 512 * 32768]))
        .await?;
    env.put_object("stress/model", Bytes::from(vec![17; 16 * 1024 * 1024]))
        .await?;
    env.database.query("INSERT INTO example_samples SELECT ?,number,concat('sample-',toString(number)),concat('{\"position\":',toString(number),'}'),'{}' FROM numbers(512)")
        .bind(dataset).execute().await?;
    let sql = "SELECT sample_id, intDiv(position,{batch_size:UInt64}) AS batch_idx, modulo(position,{batch_size:UInt64}) AS sample_idx, metadata_json, concat('{\"payload\":{\"object\":\"stress/data\",\"byte_offset\":',toString(position*{blob_bytes:UInt64}),',\"byte_length\":',toString({blob_bytes:UInt64}),'}}') AS blobs_json FROM example_samples WHERE dataset_id={dataset:UUID} ORDER BY batch_idx,sample_idx";
    let config = json!({"queries":{"training":{"sql":sql,"params":{"dataset":dataset,"batch_size":8,"blob_bytes":32768}}},"tensorlane":{"assets":{"model":{"object":"stress/model"}}}});
    let mut runs = Vec::new();
    for _ in 0..4 {
        runs.push(env.create_run(config.clone()).await?);
    }
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("../client/tests/server_stress.py");
    let mut clients = Vec::new();
    for id in &runs {
        let process = tokio::process::Command::new(&python)
            .arg(&script)
            .args([&env.url, id])
            .arg(root.path())
            .env("TENSORLANE_API_KEY", KEY)
            .env("OMP_NUM_THREADS", "1")
            .env("MKL_NUM_THREADS", "1")
            .env("OPENBLAS_NUM_THREADS", "1")
            .process_group(0)
            .kill_on_drop(true)
            .spawn()?;
        clients.push(TrainingClient {
            group: process.id().unwrap() as i32,
            process,
        });
    }
    barrier(root.path(), &runs, &mut clients, "ready").await?;
    for id in &runs {
        assert_eq!(env.status(id).await?, "running");
    }
    tokio::fs::write(root.path().join("go"), b"").await?;
    barrier(root.path(), &runs, &mut clients, "restart").await?;
    let resume = root.path().join("resume");
    let release = async {
        tokio::time::sleep(Duration::from_millis(250)).await;
        tokio::fs::write(resume, b"").await
    };
    let (restart, released) = tokio::join!(env.restart_after(Duration::from_secs(3)), release);
    restart?;
    released?;
    for mut client in clients {
        let status =
            tokio::time::timeout(Duration::from_secs(180), client.process.wait()).await??;
        ensure!(status.success(), "stress client failed: {status}");
    }
    let mut reports = Vec::new();
    for id in &runs {
        assert_eq!(env.status(id).await?, "succeeded");
        let metrics = env.database.query("SELECT count(),uniqExact(step) FROM metrics WHERE run_id=toUUID(?) AND name='loss'")
            .bind(id).fetch_one::<(u64,u64)>().await?;
        assert_eq!(metrics, (64, 64));
        assert_eq!(
            env.database
                .query("SELECT count() FROM assets FINAL WHERE run_id=toUUID(?)")
                .bind(id)
                .fetch_one::<u64>()
                .await?,
            1
        );
        let report: Value = serde_json::from_slice(
            &tokio::fs::read(root.path().join(format!("{id}.json"))).await?,
        )?;
        let asset = report["asset_id"].as_str().unwrap();
        let record = env
            .database
            .query("SELECT size,toString(content_hash) FROM assets FINAL WHERE id=toUUID(?)")
            .bind(asset)
            .fetch_one::<(u64, String)>()
            .await?;
        let object = env
            .s3
            .get_object()
            .bucket("tensorlane-test")
            .key(format!("checkpoints/{asset}"))
            .send()
            .await?
            .body
            .collect()
            .await?
            .into_bytes();
        assert_eq!(record.0, report["checkpoint_bytes"].as_u64().unwrap());
        assert_eq!(record.0, object.len() as u64);
        assert_eq!(record.1, report["checkpoint_sha256"].as_str().unwrap());
        assert_eq!(hex::encode(Sha256::digest(object)), record.1);
        reports.push(report);
    }
    env.ready().await?;
    println!(
        "stress passed: 4 concurrent runs, 256 batches, 2048 samples, 256 unique metrics, 4 verified checkpoints; survived a 3-second server outage"
    );
    println!("{}", json!({"runs":reports}));
    Ok(())
}
