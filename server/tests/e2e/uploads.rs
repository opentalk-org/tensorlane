use super::setup::{TestEnv, config};
use anyhow::Result;
use bytes::Bytes;
use reqwest::{
    Method, StatusCode,
    multipart::{Form, Part},
};
use serde_json::json;
use sha2::{Digest, Sha256};
use tensorlane_protocol::{TRANSFER_CHUNK_BYTES, UploadStatus};
use uuid::Uuid;

pub async fn check(env: &mut TestEnv) -> Result<()> {
    let (mut replica, url) = env.replica()?;
    let result = check_replicas(env, &url).await;
    let _ = replica.kill();
    let _ = replica.wait();
    result
}

async fn check_replicas(env: &mut TestEnv, replica: &str) -> Result<()> {
    env.ready_at(replica).await?;
    let id = env.create_run(config(env.seed(1).await?, 1)).await?;
    env.init(&id).await?;
    let request = Uuid::new_v4();
    let metrics = json!({"scalars":[{"step":1,"timestamp_unix_ms":1700000000123i64,"name":"loss","value":0.5}],"arrays":[{"step":1,"timestamp_unix_ms":1700000000123i64,"name":"array","value":[1.0,-2.0]}]});
    let path = format!("/runs/{id}/metrics/{request}");
    let responses = futures::future::join_all([&env.url, replica].into_iter().map(|address| {
        env.request_at(address, Method::PUT, &path)
            .json(&metrics)
            .send()
    }))
    .await;
    for response in responses {
        assert_eq!(response?.status(), StatusCode::NO_CONTENT);
    }
    tokio::fs::remove_dir_all(&env.cache).await?;
    env.restart().await?;
    env.request(Method::PUT, &path)
        .json(&metrics)
        .send()
        .await?
        .error_for_status()?;
    for table in ["metrics", "array_metrics"] {
        assert_eq!(
            env.database
                .query(&format!(
                    "SELECT count() FROM {table} WHERE run_id=toUUID(?)"
                ))
                .bind(&id)
                .fetch_one::<u64>()
                .await?,
            1
        );
    }
    let other = env.create_run(config(Uuid::new_v4(), 0)).await?;
    env.init(&other).await?;
    env.request_at(
        replica,
        Method::PUT,
        &format!("/runs/{other}/metrics/{request}"),
    )
    .json(&metrics)
    .send()
    .await?
    .error_for_status()?;
    for table in ["metrics", "array_metrics"] {
        assert_eq!(
            env.database
                .query(&format!(
                    "SELECT count() FROM {table} WHERE run_id=toUUID(?)"
                ))
                .bind(&other)
                .fetch_one::<u64>()
                .await?,
            1
        );
    }
    let mut changed = metrics.clone();
    changed["scalars"][0]["value"] = json!(0.75);
    assert_eq!(
        env.request_at(replica, Method::PUT, &path)
            .json(&changed)
            .send()
            .await?
            .status(),
        StatusCode::CONFLICT
    );
    let body = Bytes::from(vec![19; 4 * TRANSFER_CHUNK_BYTES + 123]);
    let first = checkpoint(env, replica, &id, 1, body.clone(), true).await?;
    let second = checkpoint(env, replica, &id, 2, body.clone(), false).await?;
    let assets: serde_json::Value = env
        .request(Method::GET, &format!("/runs/{id}/assets"))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    let rows = assets.as_array().unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[1]["id"], second.to_string());
    assert_eq!(rows[1]["ancestor_asset_id"], first.to_string());
    let incomplete = Uuid::new_v4();
    let spec = json!({"size":1,"sha256":hex::encode(Sha256::digest(b"u")),"metadata":{"kind":"asset","metadata":{"run_id":id,"asset_id":incomplete,"name":"model","step":99,"kind":"checkpoint","asset_type":"test-model","metadata_json":"{}","content_type":"application/octet-stream"}}});
    assert_eq!(
        env.request_at(replica, Method::PUT, &format!("/uploads/{incomplete}"))
            .multipart(
                Form::new()
                    .text("spec", spec.to_string())
                    .part("file", Part::bytes(Vec::new()).file_name("model"))
            )
            .send()
            .await?
            .status(),
        StatusCode::BAD_REQUEST
    );
    env.request(Method::POST, &format!("/runs/{id}/end"))
        .json(&json!({"failed": true}))
        .send()
        .await?
        .error_for_status()?;
    assert_eq!(env.status(&id).await?, "failed");
    let initialized = env.init(&id).await?;
    assert_eq!(initialized.run_id, id);
    let pinned = initialized.checkpoint.unwrap();
    assert_eq!(pinned.name, "model");
    assert_eq!(pinned.asset_id, second.to_string());
    assert_eq!(env.status(&id).await?, "running");
    checkpoint(env, replica, &id, 3, Bytes::from_static(b"newer"), false).await?;
    let metadata: tensorlane_protocol::AssetDownload = env
        .request(Method::GET, &format!("/runs/{id}/inputs/{second}"))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert_eq!(metadata.metadata.asset_id, Some(second.to_string()));
    let bad_hash = Uuid::new_v4();
    let mut invalid = spec.clone();
    invalid["sha256"] = json!(hex::encode(Sha256::digest(b"y")));
    invalid["metadata"]["metadata"]["asset_id"] = json!(bad_hash);
    assert_eq!(
        env.request_at(replica, Method::PUT, &format!("/uploads/{bad_hash}"))
            .multipart(
                Form::new()
                    .text("spec", invalid.to_string())
                    .part("file", Part::bytes(b"x".to_vec()).file_name("model"))
            )
            .send()
            .await?
            .status(),
        StatusCode::BAD_REQUEST
    );
    for failed in [incomplete, bad_hash] {
        assert_eq!(
            env.database
                .query("SELECT count() FROM assets WHERE id=?")
                .bind(failed)
                .fetch_one::<u64>()
                .await?,
            0
        );
        let key = format!("checkpoints/{failed}");
        let error = env
            .s3
            .head_object()
            .bucket("tensorlane-test")
            .key(&key)
            .send()
            .await
            .unwrap_err();
        assert_eq!(error.raw_response().unwrap().status().as_u16(), 404);
        assert!(
            env.s3
                .list_multipart_uploads()
                .bucket("tensorlane-test")
                .prefix(&key)
                .send()
                .await?
                .uploads()
                .is_empty()
        );
    }
    let empty = Uuid::new_v4();
    let mut empty_spec = spec;
    empty_spec["size"] = json!(0);
    empty_spec["sha256"] = json!(hex::encode(Sha256::digest(b"")));
    empty_spec["metadata"]["metadata"]["asset_id"] = json!(empty);
    empty_spec["metadata"]["metadata"]["kind"] = json!("file");
    empty_spec["metadata"]["metadata"]["name"] = json!("empty");
    let status: UploadStatus = env
        .request_at(replica, Method::PUT, &format!("/uploads/{empty}"))
        .multipart(
            Form::new()
                .text("spec", empty_spec.to_string())
                .part("file", Part::bytes(Vec::new()).file_name("empty")),
        )
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert!(status.committed);
    assert_eq!(
        env.s3
            .head_object()
            .bucket("tensorlane-test")
            .key(format!("checkpoints/{empty}"))
            .send()
            .await?
            .content_length(),
        Some(0)
    );
    let object = env
        .s3
        .get_object()
        .bucket("tensorlane-test")
        .key(format!("checkpoints/{first}"))
        .send()
        .await?
        .body
        .collect()
        .await?
        .into_bytes();
    assert_eq!(object, body);
    assert_eq!(
        hex::encode(Sha256::digest(&object)),
        hex::encode(Sha256::digest(&body))
    );
    Ok(())
}

async fn checkpoint(
    env: &mut TestEnv,
    replica: &str,
    run: &str,
    step: u64,
    body: Bytes,
    restart: bool,
) -> Result<Uuid> {
    let id = Uuid::new_v4();
    let path = format!("/uploads/{id}");
    let spec = json!({"size":body.len(),"sha256":hex::encode(Sha256::digest(&body)),"metadata":{"kind":"asset","metadata":{"run_id":run,"asset_id":id,"name":"model","step":step,"kind":"checkpoint","asset_type":"test-model","metadata_json":"{}","content_type":"application/octet-stream"}}});
    for (index, address) in [env.url.clone(), replica.to_owned(), env.url.clone()]
        .into_iter()
        .enumerate()
    {
        let status: UploadStatus = env
            .request_at(&address, Method::PUT, &path)
            .multipart(
                Form::new()
                    .text("spec", spec.to_string())
                    .part("file", Part::bytes(body.to_vec()).file_name("model")),
            )
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        assert!(status.committed);
        if restart && index == 0 {
            tokio::fs::remove_dir_all(&env.cache).await?;
            env.restart().await?;
        }
    }
    Ok(id)
}
