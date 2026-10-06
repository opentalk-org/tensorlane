use anyhow::{Result, ensure};
use axum::extract::Multipart;
use sha2::{Digest, Sha256};
use tensorlane_protocol::{TRANSFER_CHUNK_BYTES, UploadMetadata, UploadSpec, UploadStatus};
use uuid::Uuid;

use crate::runtime::Runtime;

pub async fn save(engine: &Runtime, id: Uuid, mut multipart: Multipart) -> Result<UploadStatus> {
    let mut field = multipart
        .next_field()
        .await?
        .ok_or_else(|| anyhow::anyhow!("upload must start with its spec"))?;
    ensure!(
        field.name() == Some("spec"),
        "upload must start with its spec"
    );
    let mut bytes = Vec::new();
    while let Some(chunk) = field.chunk().await? {
        ensure!(
            bytes.len() + chunk.len() <= TRANSFER_CHUNK_BYTES,
            "upload spec exceeds 4 MiB"
        );
        bytes.extend_from_slice(&chunk);
    }
    drop(field);
    let mut spec: UploadSpec = serde_json::from_slice(&bytes)?;
    ensure!(!id.is_nil(), "upload ID must not be nil");
    ensure!(
        spec.size <= 16 * 1024 * 1024 * 1024,
        "upload exceeds 16 GiB"
    );
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
            let _: serde_json::Map<String, serde_json::Value> =
                serde_json::from_str(&metadata.metadata_json)?;
            (
                metadata.run_id.parse()?,
                engine.asset_key(id),
                metadata.content_type.as_str(),
            )
        }
        UploadMetadata::Artifact { run_id, metadata } => {
            ensure!(!metadata.name.is_empty(), "artifact name must not be empty");
            ensure!(
                metadata.size_bytes == spec.size,
                "artifact size does not match upload size"
            );
            time::OffsetDateTime::from_unix_timestamp_nanos(
                i128::from(metadata.timestamp_unix_ms) * 1_000_000,
            )?;
            (
                run_id.parse()?,
                format!("{}/{id}", engine.metrics_prefix),
                metadata.content_type.as_str(),
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
    match spec.metadata {
        UploadMetadata::Asset { metadata } => {
            crate::asset_commit::save(engine, id, &metadata, &spec.sha256, spec.size).await?;
        }
        UploadMetadata::Artifact { metadata, .. } => {
            engine.save_artifact(id, run, &metadata).await?;
        }
    }
    Ok(UploadStatus { committed: true })
}
