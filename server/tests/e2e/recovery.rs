use super::setup::{KEY, TestEnv, config};
use anyhow::Result;
use prost::Message;
use reqwest::{Method, StatusCode};
use serde_json::json;
use tensorlane_protocol::DataResponse;
use uuid::Uuid;

pub async fn check(env: &mut TestEnv) -> Result<()> {
    let dataset = env.seed(5).await?;
    let mut settings = config(dataset, 5);
    let queries = settings["queries"].as_object_mut().unwrap();
    let query = queries.remove("training").unwrap();
    queries.insert("packed_audio".into(), query);
    let id = env.create_run(settings).await?;
    env.init(&id).await?;
    let first = env.batch(&id, "packed_audio", 0).await?.unwrap();
    env.restart().await?;
    assert_eq!(env.status(&id).await?, "running");
    assert_eq!(env.init(&id).await?.run_id, id);
    let replay = env.batch(&id, "packed_audio", 0).await?.unwrap();
    assert_eq!(replay.batch, first.batch);
    assert_eq!(replay.query_batch_idx, first.query_batch_idx);
    assert_eq!(
        env.batch(&id, "packed_audio", 4).await?.unwrap().batch[0].sample_id,
        "sample-4"
    );
    let (mut replica, url) = env.replica()?;
    let result = async {
        env.ready_at(&url).await?;
        let contended = env.create_run(config(dataset, 1)).await?;
        env.put_object(
            &format!("checkpoints/.tensorlane/runs/{contended}/session"),
            bytes::Bytes::from(serde_json::to_vec(&(Uuid::new_v4(), 0i64))?),
        )
        .await?;
        let candidates = [Uuid::new_v4(), Uuid::new_v4()];
        let addresses = [&env.url, &url];
        let responses = futures::future::join_all(addresses.iter().zip(candidates).map(
            |(address, session)| {
                env.http
                    .post(format!("{address}/runs/{contended}/init"))
                    .bearer_auth(KEY)
                    .header("x-tensorlane-session", session.to_string())
                    .send()
            },
        ))
        .await;
        for response in responses {
            response?.error_for_status()?;
        }
        for address in addresses {
            env.http
                .post(format!("{address}/runs/{contended}/init"))
                .bearer_auth(KEY)
                .header("x-tensorlane-session", candidates[0].to_string())
                .send()
                .await?
                .error_for_status()?;
        }
        env.http
            .post(format!("{url}/runs/{contended}/end"))
            .bearer_auth(KEY)
            .header("x-tensorlane-session", candidates[0].to_string())
            .json(&json!({"failed":false}))
            .send()
            .await?
            .error_for_status()?;
        assert_eq!(env.status(&contended).await?, "succeeded");
        env.http
            .post(format!("{url}/runs/{contended}/end"))
            .bearer_auth(KEY)
            .json(&json!({"failed":false}))
            .send()
            .await?
            .error_for_status()?;
        let initialized: tensorlane_protocol::InitResponse = env
            .request_at(&url, Method::POST, &format!("/runs/{id}/init"))
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        assert_eq!(initialized.run_id, id);
        assert_eq!(
            env.http
                .post(format!("{url}/runs/{id}/init"))
                .bearer_auth(KEY)
                .header("x-tensorlane-session", Uuid::new_v4().to_string())
                .send()
                .await?
                .status(),
            StatusCode::OK
        );
        for sequence in 0..5 {
            let address = if sequence % 2 == 0 { &url } else { &env.url };
            let batch = env
                .batch_at(address, &id, "packed_audio", sequence)
                .await?
                .unwrap();
            assert_eq!(batch.stream, "packed_audio");
            assert_eq!(batch.batch_id, sequence);
            assert_eq!(batch.query_batch_idx, sequence);
            assert_eq!(batch.batch[0].sample_id, format!("sample-{sequence}"));
            assert_eq!(batch.batch[0].blobs["payload"], b"payload".as_slice());
        }
        tokio::fs::remove_dir_all(&env.cache).await?;
        let continuation = env
            .request_at(
                &url,
                Method::GET,
                &format!("/runs/{id}/streams/packed_audio/batches/0"),
            )
            .send();
        let (restarted, response) = tokio::join!(
            env.restart_after(std::time::Duration::from_millis(100)),
            continuation,
        );
        restarted?;
        let surviving = DataResponse::decode(response?.error_for_status()?.bytes().await?)?;
        assert_eq!(surviving.batch, first.batch);
        let regenerated = env.batch(&id, "packed_audio", 0).await?.unwrap();
        assert_eq!(regenerated.batch, first.batch);
        assert_eq!(regenerated.query_batch_idx, first.query_batch_idx);
        assert!(env.batch_at(&url, &id, "packed_audio", 5).await?.is_none());
        for address in [&url, &env.url] {
            env.request_at(address, Method::POST, &format!("/runs/{id}/end"))
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
