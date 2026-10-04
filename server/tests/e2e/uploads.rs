use super::setup::{TestEnv, config, eventually};
use anyhow::Result;
use bytes::Bytes;
use reqwest::{Method, StatusCode};
use serde_json::json;
use sha2::{Digest, Sha256};
use tensorlane_protocol::{TRANSFER_CHUNK_BYTES, UploadStatus};
use uuid::Uuid;

pub async fn check(env: &mut TestEnv) -> Result<()> {
    let id = env.create_run(config(env.seed(1).await?, 1)).await?;
    env.init(&id).await?;
    let request = Uuid::new_v4();
    let metrics = json!({"scalars":[{"step":1,"timestamp_unix_ms":1700000000123i64,"name":"loss","value":0.5}],"arrays":[{"step":1,"timestamp_unix_ms":1700000000123i64,"name":"array","value":[1.0,-2.0]}]});
    let path = format!("/runs/{id}/metrics/{request}");
    for _ in 0..2 {
        assert_eq!(
            env.request(Method::PUT, &path)
                .json(&metrics)
                .send()
                .await?
                .status(),
            StatusCode::NO_CONTENT
        );
    }
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
    let receipt = env
        .cache
        .join("runs")
        .join(&id)
        .join("metrics")
        .join(request.to_string())
        .with_extension("committed");
    if receipt.exists() {
        tokio::fs::remove_file(receipt).await?;
    }
    env.request(Method::PUT, &path)
        .json(&metrics)
        .send()
        .await?
        .error_for_status()?;
    assert_eq!(
        env.database
            .query("SELECT count() FROM metrics WHERE run_id=toUUID(?)")
            .bind(&id)
            .fetch_one::<u64>()
            .await?,
        1
    );
    let mut changed = metrics.clone();
    changed["scalars"][0]["value"] = json!(0.75);
    assert!(
        !env.request(Method::PUT, &path)
            .json(&changed)
            .send()
            .await?
            .status()
            .is_success()
    );
    let body = Bytes::from(vec![19; TRANSFER_CHUNK_BYTES + 123]);
    let first = checkpoint(env, &id, 1, body.clone(), true).await?;
    let second = checkpoint(env, &id, 2, body, false).await?;
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
    assert_eq!(object.len(), TRANSFER_CHUNK_BYTES + 123);
    Ok(())
}

async fn checkpoint(
    env: &mut TestEnv,
    run: &str,
    step: u64,
    body: Bytes,
    restart: bool,
) -> Result<Uuid> {
    let id = Uuid::new_v4();
    let path = format!("/uploads/{id}");
    let spec = json!({"size":body.len(),"sha256":hex::encode(Sha256::digest(&body)),"metadata":{"kind":"asset","metadata":{"run_id":run,"asset_id":id,"name":"model","step":step,"kind":"checkpoint","asset_type":"test-model","metadata_json":"{}","content_type":"application/octet-stream"}}});
    env.request(Method::PUT, &path)
        .json(&spec)
        .send()
        .await?
        .error_for_status()?;
    if restart {
        tokio::fs::remove_file(
            env.cache
                .join("http-uploads")
                .join(id.to_string())
                .join("data"),
        )
        .await?;
        env.restart().await?;
        env.request(Method::PUT, &path)
            .json(&spec)
            .send()
            .await?
            .error_for_status()?;
    }
    for (index, chunk) in body.chunks(TRANSFER_CHUNK_BYTES).enumerate() {
        for _ in 0..2 {
            env.request(Method::PUT, &format!("{path}/chunks/{index}"))
                .body(chunk.to_vec())
                .send()
                .await?
                .error_for_status()?;
        }
        if restart && index == 0 {
            env.restart().await?;
        }
    }
    eventually("upload commit", || async {
        let response = env
            .request(Method::POST, &format!("{path}/commit"))
            .send()
            .await?
            .error_for_status()?;
        let status: UploadStatus = response.json().await?;
        Ok(status.committed.then_some(()))
    })
    .await?;
    for _ in 0..2 {
        assert!(
            env.request(Method::POST, &format!("{path}/commit"))
                .send()
                .await?
                .error_for_status()?
                .json::<UploadStatus>()
                .await?
                .committed
        );
    }
    Ok(id)
}
