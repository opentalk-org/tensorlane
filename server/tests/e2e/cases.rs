use super::setup::{TestEnv, config};
use anyhow::{Result, ensure};
use bytes::Bytes;
use prost::Message;
use reqwest::{Method, StatusCode};
use serde_json::json;
use tensorlane_protocol::{AssetDownload, DataResponse};
use uuid::Uuid;

pub async fn check(env: &TestEnv) -> Result<()> {
    let paths = [
        "/runs",
        "/uploads/00000000-0000-0000-0000-000000000001",
        "/runs/00000000-0000-0000-0000-000000000001/init",
        "/runs/00000000-0000-0000-0000-000000000001/inputs/model/bytes",
        "/unknown",
    ];
    for method in [Method::GET, Method::POST, Method::PUT] {
        for path in paths {
            assert_eq!(
                env.http
                    .request(method.clone(), format!("{}{path}", env.url))
                    .send()
                    .await?
                    .status(),
                StatusCode::UNAUTHORIZED
            );
        }
    }
    let dataset = env.seed(4).await?;
    let id = env.create_run(config(dataset, 4)).await?;
    let initialized = env.init(&id).await?;
    assert_eq!(initialized.run_id, id);
    assert_eq!(env.init(&id).await?.streams, vec!["training", "validation"]);
    assert_eq!(env.status(&id).await?, "running");
    for sequence in [2, 0, 2, 3, 1] {
        let batch = env.batch(&id, "training", sequence).await?.unwrap();
        assert_eq!(batch.batch_id, sequence);
        assert_eq!(batch.query_batch_idx, sequence);
        assert_eq!(batch.batch[0].sample_id, format!("sample-{sequence}"));
        assert_eq!(batch.batch[0].blobs["payload"].as_ref(), b"payload");
    }
    assert!(env.batch(&id, "training", 4).await?.is_none());
    assert_eq!(
        env.batch(&id, "validation", 500)
            .await?
            .unwrap()
            .query_batch_idx,
        0
    );
    let large_dataset = env.seed(1).await?;
    let object = format!("datasets/{large_dataset}");
    env.put_object(&object, Bytes::from(vec![31; 16 * 1024 * 1024]))
        .await?;
    let large_run = env.create_run(config(large_dataset, 1)).await?;
    env.init(&large_run).await?;
    let expected = env.batch(&large_run, "training", 0).await?.unwrap();
    let response = env
        .request(
            Method::GET,
            &format!("/runs/{large_run}/streams/training/batches/0"),
        )
        .send()
        .await?
        .error_for_status()?;
    let length = response.content_length().unwrap();
    tokio::fs::remove_file(
        env.cache
            .join("runs")
            .join(&large_run)
            .join("data/0/0.batch"),
    )
    .await?;
    let replacement = Bytes::from(vec![47; 16 * 1024 * 1024]);
    env.put_object(&object, replacement.clone()).await?;
    let refreshed = env.batch(&large_run, "training", 0).await?.unwrap();
    assert_eq!(refreshed.batch[0].blobs["payload"], replacement);
    let original = response.bytes().await?;
    assert_eq!(original.len() as u64, length);
    assert_eq!(DataResponse::decode(original)?, expected);
    assert_eq!(
        env.http
            .post(format!("{}/runs/{id}/init", env.url))
            .bearer_auth(super::setup::KEY)
            .header(
                tensorlane_protocol::SESSION_HEADER,
                Uuid::new_v4().to_string()
            )
            .send()
            .await?
            .status(),
        StatusCode::CONFLICT
    );
    let mut input = config(dataset, 1);
    input["tensorlane"]["assets"] = json!({"model":{"object":"inputs/model"}});
    env.put_object("inputs/model", Bytes::from(vec![7; 9 * 1024 * 1024]))
        .await?;
    let asset_run = env.create_run(input).await?;
    env.init(&asset_run).await?;
    ensure!(
        !env.cache.join("assets").exists(),
        "init fetched assets before an asset request"
    );
    let path = format!("/runs/{asset_run}/inputs/model");
    let metadata: AssetDownload = env
        .request(Method::GET, &path)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert_eq!(metadata.size, 9 * 1024 * 1024);
    assert_eq!(
        env.request(Method::GET, &format!("{path}/bytes"))
            .send()
            .await?
            .status(),
        StatusCode::RANGE_NOT_SATISFIABLE
    );
    assert_eq!(
        env.request(Method::GET, &format!("{path}/bytes"))
            .header("Range", "bytes=0-5")
            .header("If-Match", "wrong")
            .send()
            .await?
            .status(),
        StatusCode::PRECONDITION_FAILED
    );
    for _ in 0..2 {
        let response = env
            .request(Method::GET, &format!("{path}/bytes"))
            .header("Range", "bytes=0-4194303")
            .header("If-Match", &metadata.etag)
            .send()
            .await?
            .error_for_status()?;
        assert_eq!(response.status(), StatusCode::PARTIAL_CONTENT);
        assert_eq!(response.bytes().await?.as_ref(), vec![7; 4 * 1024 * 1024]);
    }
    let mut bad = config(dataset, 1);
    bad["queries"]["training"]["sql"] = json!("SELECT broken syntax");
    let bad_id = env.create_run(bad).await?;
    env.init(&bad_id).await?;
    let response = env
        .request(
            Method::GET,
            &format!("/runs/{bad_id}/streams/training/batches/0"),
        )
        .send()
        .await?;
    assert_eq!(
        response.status(),
        StatusCode::UNPROCESSABLE_ENTITY,
        "{}",
        response.text().await?
    );
    let mut typed = config(dataset, 1);
    typed["queries"]["training"]["sql"] = json!(
        "SELECT 'typed' AS sample_id, toUInt64(0) AS batch_idx, toUInt64(0) AS sample_idx, '{}' AS metadata_json, '{}' AS blobs_json WHERE {signed:Int64}=-9223372036854775808 AND {wide:UInt64}=18446744073709551615 AND isNull({nil:Nullable(Int64)}) AND {array:Array(Int64)}=[-1,2] AND {tuple:Tuple(String, Int64)}=('x',-4) AND {map:Map(String, Int64)}=map('x',-5) AND {decimal:Decimal(38, 18)}=toDecimal128('1.000000000000000001',18)"
    );
    typed["queries"]["training"]["params"] = json!({"signed":i64::MIN,"wide":u64::MAX,"nil":null,"array":[-1,2],"tuple":["x",-4],"map":{"x":-5},"decimal":{"$clickhouse":"1.000000000000000001"}});
    let typed_id = env.create_run(typed).await?;
    env.init(&typed_id).await?;
    assert_eq!(
        env.batch(&typed_id, "training", 0).await?.unwrap().batch[0].sample_id,
        "typed"
    );
    Ok(())
}
