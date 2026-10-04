use super::setup::{KEY, TestEnv, config};
use anyhow::{Result, ensure};
use bytes::Bytes;
use serde_json::json;
use std::{path::Path, time::Duration};

pub async fn check(env: &TestEnv) -> Result<()> {
    let Some(python) = std::env::var_os("TENSORLANE_TEST_PYTHON") else {
        return Ok(());
    };
    let mut settings = config(env.seed(4).await?, 4);
    settings["tensorlane"]["assets"] = json!({"model":{"object":"training/initial"}});
    env.put_object("training/initial", Bytes::from_static(b"initial weights"))
        .await?;
    let id = env.create_run(settings).await?;
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("../client/tests/server_training.py");
    let mut command = tokio::process::Command::new(python);
    command
        .arg(script)
        .args([&env.url, &id])
        .env("TENSORLANE_API_KEY", KEY)
        .env("OMP_NUM_THREADS", "1")
        .env("MKL_NUM_THREADS", "1")
        .kill_on_drop(true);
    let output = tokio::time::timeout(Duration::from_secs(45), command.output()).await??;
    ensure!(
        output.status.success(),
        "Python training failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(env.status(&id).await?, "succeeded");
    assert_eq!(
        env.database
            .query("SELECT count() FROM assets FINAL WHERE run_id=toUUID(?)")
            .bind(&id)
            .fetch_one::<u64>()
            .await?,
        1
    );
    assert_eq!(
        env.database
            .query("SELECT count() FROM metrics WHERE run_id=toUUID(?) AND name='loss'")
            .bind(&id)
            .fetch_one::<u64>()
            .await?,
        4
    );
    Ok(())
}
