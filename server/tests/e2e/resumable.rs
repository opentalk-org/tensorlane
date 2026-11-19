use super::setup::{KEY, TestEnv, config};
use anyhow::Result;
use base64::{Engine, engine::general_purpose::STANDARD};
use reqwest::{Method, Response, StatusCode};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tensorlane_protocol::upload_part_bytes;
use uuid::Uuid;

pub async fn check(env: &mut TestEnv) -> Result<()> {
    let (mut replica, url) = env.replica()?;
    let result = check_replicas(env, &url).await;
    let _ = replica.kill();
    let _ = replica.wait();
    result
}

fn spec(run: &str, id: Uuid, name: &str, body: &[u8]) -> Value {
    json!({"size":body.len(),"sha256":hex::encode(Sha256::digest(body)),"metadata":{"kind":"asset","metadata":{"run_id":run,"asset_id":id,"name":name,"step":1,"kind":"checkpoint","asset_type":"test-model","metadata_json":"{}","content_type":"application/octet-stream"}}})
}

async fn create(
    env: &TestEnv,
    address: &str,
    id: Uuid,
    spec: &Value,
    first: Option<&[u8]>,
) -> Result<Response> {
    let metadata = format!(
        "id {},spec {}",
        STANDARD.encode(id.to_string()),
        STANDARD.encode(spec.to_string())
    );
    let mut request = env
        .request_at(address, Method::POST, "/uploads")
        .header("Tus-Resumable", "1.0.0")
        .header("Upload-Length", spec["size"].to_string())
        .header("Upload-Metadata", metadata);
    if let Some(first) = first {
        request = request
            .header("Content-Type", "application/offset+octet-stream")
            .body(first.to_vec());
    }
    Ok(request.send().await?)
}

async fn append(
    env: &TestEnv,
    address: &str,
    id: Uuid,
    offset: usize,
    bytes: &[u8],
) -> Result<Response> {
    Ok(env
        .request_at(address, Method::PATCH, &format!("/uploads/{id}"))
        .header("Tus-Resumable", "1.0.0")
        .header("Upload-Offset", offset.to_string())
        .header("Content-Type", "application/offset+octet-stream")
        .body(bytes.to_vec())
        .send()
        .await?)
}

fn offset(response: &Response) -> u64 {
    response.headers()["Upload-Offset"]
        .to_str()
        .unwrap()
        .parse()
        .unwrap()
}

async fn stored_offset(env: &TestEnv, address: &str, id: Uuid) -> Result<u64> {
    let response = env
        .request_at(address, Method::HEAD, &format!("/uploads/{id}"))
        .header("Tus-Resumable", "1.0.0")
        .send()
        .await?
        .error_for_status()?;
    assert_eq!(response.headers()["Cache-Control"], "no-store");
    Ok(offset(&response))
}

async fn asset_rows(env: &TestEnv, id: Uuid) -> Result<u64> {
    Ok(env
        .database
        .query("SELECT count() FROM assets WHERE id=?")
        .bind(id)
        .fetch_one::<u64>()
        .await?)
}

async fn object(env: &TestEnv, id: Uuid) -> Result<Vec<u8>> {
    Ok(env
        .s3
        .get_object()
        .bucket("tensorlane-test")
        .key(format!("checkpoints/{id}"))
        .send()
        .await?
        .body
        .collect()
        .await?
        .into_bytes()
        .to_vec())
}

async fn check_replicas(env: &mut TestEnv, replica: &str) -> Result<()> {
    env.ready_at(replica).await?;
    let run = env.create_run(config(env.seed(1).await?, 1)).await?;
    env.init(&run).await?;

    let options = env
        .request(Method::OPTIONS, "/uploads")
        .send()
        .await?
        .error_for_status()?;
    assert_eq!(options.headers()["Tus-Version"], "1.0.0");
    assert_eq!(
        options.headers()["Tus-Extension"],
        "creation,creation-with-upload"
    );

    // A small file is one request: created, stored, completed and published.
    let small = Uuid::new_v4();
    let body = b"small checkpoint".to_vec();
    let response = create(
        env,
        replica,
        small,
        &spec(&run, small, "small", &body),
        Some(&body),
    )
    .await?
    .error_for_status()?;
    assert_eq!(response.status(), StatusCode::CREATED);
    assert_eq!(response.headers()["Location"], format!("/uploads/{small}"));
    assert_eq!(response.headers()["Tus-Resumable"], "1.0.0");
    assert_eq!(offset(&response), body.len() as u64);
    assert_eq!(asset_rows(env, small).await?, 1);
    assert_eq!(object(env, small).await?, body);
    // Repeating the creation or asking for the offset changes nothing.
    let again = create(
        env,
        &env.url,
        small,
        &spec(&run, small, "small", &body),
        Some(&body),
    )
    .await?
    .error_for_status()?;
    assert_eq!(offset(&again), body.len() as u64);
    assert_eq!(stored_offset(env, replica, small).await?, body.len() as u64);
    assert_eq!(asset_rows(env, small).await?, 1);

    // An empty file needs no data.
    let empty = Uuid::new_v4();
    let response = create(env, replica, empty, &spec(&run, empty, "empty", b""), None)
        .await?
        .error_for_status()?;
    assert_eq!(offset(&response), 0);
    assert_eq!(asset_rows(env, empty).await?, 1);
    assert!(object(env, empty).await?.is_empty());

    // A large file in chunks across replicas and a restart; a chunk that ends
    // inside a part stores only its whole parts.
    let part = upload_part_bytes(3 * 16 * 1024 * 1024) as usize;
    let large: Vec<u8> = (0..2 * part + 123).map(|i| (i % 251) as u8).collect();
    let id = Uuid::new_v4();
    let large_spec = spec(&run, id, "model", &large);
    let response = create(
        env,
        &env.url,
        id,
        &large_spec,
        Some(&large[..part + part / 2]),
    )
    .await?
    .error_for_status()?;
    assert_eq!(offset(&response), part as u64);
    assert_eq!(asset_rows(env, id).await?, 0);
    assert_eq!(stored_offset(env, replica, id).await?, part as u64);
    let stale = append(env, replica, id, 0, &large[..part]).await?;
    assert_eq!(stale.status(), StatusCode::CONFLICT);
    tokio::fs::remove_dir_all(&env.cache).await?;
    env.restart().await?;
    let response = append(env, replica, id, part, &large[part..2 * part])
        .await?
        .error_for_status()?;
    assert_eq!(response.status(), StatusCode::NO_CONTENT);
    assert_eq!(offset(&response), 2 * part as u64);
    let response = append(env, &env.url, id, 2 * part, &large[2 * part..])
        .await?
        .error_for_status()?;
    assert_eq!(offset(&response), large.len() as u64);
    assert_eq!(asset_rows(env, id).await?, 1);
    assert_eq!(object(env, id).await?, large);
    // A resent last chunk after a lost reply is a conflict at the old offset; the
    // client then finds the upload complete.
    let resent = append(env, replica, id, 2 * part, &large[2 * part..]).await?;
    assert_eq!(resent.status(), StatusCode::CONFLICT);
    assert_eq!(stored_offset(env, replica, id).await?, large.len() as u64);
    assert_eq!(asset_rows(env, id).await?, 1);

    // The whole file's SHA-256 is checked before the object is completed.
    let corrupt = Uuid::new_v4();
    let mut corrupt_spec = spec(&run, corrupt, "model", &large);
    corrupt_spec["sha256"] = json!(hex::encode(Sha256::digest(b"other")));
    create(
        env,
        replica,
        corrupt,
        &corrupt_spec,
        Some(&large[..2 * part]),
    )
    .await?
    .error_for_status()?;
    let response = append(env, &env.url, corrupt, 2 * part, &large[2 * part..]).await?;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        response.json::<Value>().await?["message"],
        "upload SHA256 does not match"
    );
    assert_eq!(stored_offset(env, replica, corrupt).await?, 2 * part as u64);
    assert_eq!(asset_rows(env, corrupt).await?, 0);

    // Protocol errors.
    let conflicting = spec(&run, small, "small", b"different");
    assert_eq!(
        create(env, replica, small, &conflicting, None)
            .await?
            .status(),
        StatusCode::CONFLICT
    );
    let missing = Uuid::new_v4();
    assert_eq!(
        append(env, replica, missing, 0, b"x").await?.status(),
        StatusCode::NOT_FOUND
    );
    let response = env
        .request_at(replica, Method::PATCH, &format!("/uploads/{corrupt}"))
        .header("Tus-Resumable", "0.2.2")
        .header("Upload-Offset", (2 * part).to_string())
        .header("Content-Type", "application/offset+octet-stream")
        .body(b"x".to_vec())
        .send()
        .await?;
    assert_eq!(response.status(), StatusCode::PRECONDITION_FAILED);
    let response = env
        .request_at(replica, Method::PATCH, &format!("/uploads/{corrupt}"))
        .header("Tus-Resumable", "1.0.0")
        .header("Upload-Offset", (2 * part).to_string())
        .body(b"x".to_vec())
        .send()
        .await?;
    assert_eq!(response.status(), StatusCode::UNSUPPORTED_MEDIA_TYPE);
    let longer = append(env, replica, corrupt, 2 * part, &[0; 1000]).await?;
    assert_eq!(longer.status(), StatusCode::BAD_REQUEST);
    python_client(env, &run).await
}

/// The Python client's uploads of a small and a multi-chunk file.
async fn python_client(env: &TestEnv, run: &str) -> Result<()> {
    let Some(python) = std::env::var_os("TENSORLANE_TEST_PYTHON") else {
        return Ok(());
    };
    let size = 2 * tensorlane_protocol::upload_chunk_bytes(1 << 30) + 77;
    let script =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../client/tests/server_upload.py");
    let output = tokio::time::timeout(
        std::time::Duration::from_secs(120),
        tokio::process::Command::new(python)
            .arg(script)
            .args([&env.url, run, &size.to_string()])
            .env("TENSORLANE_API_KEY", KEY)
            .kill_on_drop(true)
            .output(),
    )
    .await??;
    anyhow::ensure!(
        output.status.success(),
        "Python upload failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout)?;
    let ids: Vec<Uuid> = stdout.lines().map(str::parse).collect::<Result<_, _>>()?;
    assert_eq!(ids.len(), 2);
    assert_eq!(object(env, ids[0]).await?, b"small checkpoint");
    let large = object(env, ids[1]).await?;
    assert_eq!(large.len() as u64, size);
    assert!(large.iter().enumerate().all(|(i, &b)| b == (i % 251) as u8));
    for id in ids {
        assert_eq!(asset_rows(env, id).await?, 1);
    }
    Ok(())
}
