use crate::setup::{TestEnv, run_config};
use anyhow::Result;
use serde_json::{Value, json};
use uuid::Uuid;

#[tokio::test]
async fn create_run_requires_an_existing_project() -> Result<()> {
    let env = TestEnv::start().await?;
    let config = run_config(Uuid::new_v4(), 1);
    for project_id in [Uuid::new_v4(), Uuid::nil()] {
        let response = env
            .http
            .post(format!("{}/runs", env.http_url))
            .json(&json!({"project_id":project_id,"name":"missing project","config":config}))
            .send()
            .await?;
        assert_eq!(response.status(), 404);
        assert_eq!(
            response.json::<Value>().await?["message"],
            "Project not found"
        );
    }
    for table in ["runs", "run_status"] {
        let count = env
            .clickhouse
            .query(&format!("SELECT count() FROM {table}"))
            .fetch_one::<u64>()
            .await?;
        assert_eq!(count, 0, "missing projects must not create {table} rows");
    }

    let project_id = env.create_project().await?;
    let response = env
        .http
        .post(format!("{}/runs", env.http_url))
        .json(&json!({"project_id":project_id,"name":"existing project","config":config}))
        .send()
        .await?;
    assert_eq!(response.status(), 201);
    let created: Value = response.json().await?;
    assert_eq!(created["status"], "queued");
    let run_id = created["run_id"].as_str().unwrap();
    let run: Value = env
        .http
        .get(format!("{}/runs/{run_id}", env.http_url))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert_eq!(run["project_id"], project_id.to_string());
    assert_eq!(run["config"], config);
    assert_eq!(env.status(run_id).await?, "queued");
    Ok(())
}
