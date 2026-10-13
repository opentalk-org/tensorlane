use anyhow::{Context, Result, ensure};
use axum::extract::Multipart;
use serde::Serialize;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use tensorlane_protocol::{UploadMetadata, UploadSpec, UploadStatus};
use time::OffsetDateTime;
use uuid::Uuid;

use crate::runtime::Runtime;

#[derive(clickhouse::Row, Serialize)]
struct ArtifactRecord {
    #[serde(with = "clickhouse::serde::uuid")]
    id: Uuid,
    #[serde(with = "clickhouse::serde::uuid")]
    run_id: Uuid,
    step: u64,
    #[serde(with = "clickhouse::serde::time::datetime64::nanos")]
    timestamp: OffsetDateTime,
    name: String,
    path: String,
    content_type: String,
    size_bytes: u64,
}

pub async fn save(engine: &Runtime, id: Uuid, mut multipart: Multipart) -> Result<UploadStatus> {
    let field = multipart
        .next_field()
        .await?
        .context("upload must start with its spec")?;
    ensure!(
        field.name() == Some("spec"),
        "upload must start with its spec"
    );
    let mut spec: UploadSpec = serde_json::from_slice(&field.bytes().await?)?;
    ensure!(!id.is_nil(), "upload ID must not be nil");
    ensure!(
        spec.sha256.len() == 64 && spec.sha256.bytes().all(|b| b.is_ascii_hexdigit()),
        "invalid upload SHA256"
    );
    spec.sha256.make_ascii_lowercase();
    let (run, key, content_type) = match &spec.metadata {
        UploadMetadata::Asset { metadata } => {
            ensure!(
                metadata.asset_id.parse::<Uuid>()? == id,
                "asset ID must match upload ID"
            );
            ensure!(!metadata.name.is_empty(), "asset name must not be empty");
            crate::asset_repo::kind_value(&metadata.kind)?;
            serde_json::from_str::<Map<String, Value>>(&metadata.metadata_json)?;
            (
                metadata.run_id.parse()?,
                engine.asset_key(id),
                &metadata.content_type,
            )
        }
        UploadMetadata::Artifact { run_id, metadata } => {
            ensure!(!metadata.name.is_empty(), "artifact name must not be empty");
            ensure!(
                metadata.size_bytes == spec.size,
                "artifact size does not match upload size"
            );
            OffsetDateTime::from_unix_timestamp_nanos(
                i128::from(metadata.timestamp_unix_ms) * 1_000_000,
            )?;
            (
                run_id.parse()?,
                format!("{}/{id}", engine.metrics_prefix),
                &metadata.content_type,
            )
        }
    };
    engine.active(run).await?;
    let fingerprint = hex::encode(Sha256::digest(serde_json::to_vec(&spec)?));
    let existing = engine
        .create_state(&format!("uploads/{id}"), &fingerprint)
        .await?;
    ensure!(existing == fingerprint, "conflicting retry of upload ID");
    let _slot = engine.upload_slots.acquire().await?;
    crate::s3_upload::upload(
        &engine.s3,
        engine.bucket,
        &key,
        content_type,
        &spec,
        &fingerprint,
        &mut multipart,
    )
    .await?;
    match &spec.metadata {
        UploadMetadata::Asset { metadata } => {
            crate::asset_commit::save(engine, id, metadata, &spec.sha256, spec.size).await?;
        }
        UploadMetadata::Artifact { metadata, .. } => {
            let client = engine
                .database
                .clone()
                .with_setting("insert_deduplication_token", id.to_string());
            let mut insert = client.insert::<ArtifactRecord>("artifacts").await?;
            insert
                .write(&ArtifactRecord {
                    id,
                    run_id: run,
                    step: metadata.step,
                    timestamp: OffsetDateTime::from_unix_timestamp_nanos(
                        i128::from(metadata.timestamp_unix_ms) * 1_000_000,
                    )?,
                    name: metadata.name.clone(),
                    path: key,
                    content_type: metadata.content_type.clone(),
                    size_bytes: metadata.size_bytes,
                })
                .await?;
            insert.end().await?;
        }
    }
    Ok(UploadStatus { committed: true })
}
