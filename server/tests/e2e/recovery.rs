use super::setup::{KEY, TestEnv, config, eventually};
use anyhow::Result;
use prost::Message;
use reqwest::Method;
use serde_json::json;
use tensorlane_protocol::{DataResponse, SESSION_HEADER};

pub async fn check(env: &mut TestEnv) -> Result<()> {
    let dataset = env.seed(5).await?;
    let id = env.create_run(config(dataset, 5)).await?;
    env.init(&id).await?;
    let first = env.batch(&id, "training", 0).await?.unwrap();
    env.database
        .query("TRUNCATE TABLE example_samples")
        .execute()
        .await?;
    env.restart().await?;
    assert_eq!(env.status(&id).await?, "running");
    assert_eq!(env.init(&id).await?.run_id, id);
    assert_eq!(env.batch(&id, "training", 0).await?.unwrap(), first);
    assert_eq!(
        env.batch(&id, "training", 4).await?.unwrap().batch[0].sample_id,
        "sample-4"
    );
    let (mut replica, url) = env.replica()?;
    let result = async {
        eventually("replica", || async {
            Ok(env
                .http
                .get(format!("{url}/runs"))
                .bearer_auth(KEY)
                .send()
                .await
                .ok()
                .filter(|r| r.status().is_success()))
        })
        .await?;
        let path = format!("{url}/runs/{id}/streams/training/batches/2");
        let batch = eventually("replica batch", || async {
            let response = env
                .http
                .get(&path)
                .bearer_auth(KEY)
                .header(SESSION_HEADER, env.session.to_string())
                .send()
                .await?
                .error_for_status()?;
            if response.status() == reqwest::StatusCode::ACCEPTED {
                return Ok(None);
            }
            Ok(Some(DataResponse::decode(response.bytes().await?)?))
        })
        .await?;
        assert_eq!(batch.batch_id, 2);
        assert_eq!(batch.batch[0].sample_id, "sample-2");
        for _ in 0..2 {
            env.request(Method::POST, &format!("/runs/{id}/end"))
                .json(&json!({"failed":false}))
                .send()
                .await?
                .error_for_status()?;
        }
        assert_eq!(env.status(&id).await?, "succeeded");
        anyhow::Ok(())
    }
    .await;
    let _ = replica.kill();
    let _ = replica.wait();
    result?;
    let dataset = env.seed(2).await?;
    let mut runs = Vec::new();
    for _ in 0..8 {
        let id = env.create_run(config(dataset, 2)).await?;
        env.init(&id).await?;
        runs.push(id);
    }
    let batches =
        futures::future::join_all(runs.iter().map(|id| env.batch(id, "training", 1))).await;
    for batch in batches {
        assert_eq!(batch?.unwrap().batch[0].sample_id, "sample-1");
    }
    Ok(())
}
