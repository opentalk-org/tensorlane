use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};
use tensorlane_protocol::{TRANSFER_CHUNK_BYTES, UploadMetadata, UploadSpec, UploadStatus};
use tokio::{
    fs,
    io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt},
};
use uuid::Uuid;

use crate::{
    runtime::Runtime,
    shared_cache::{Lock, write_atomic},
};

fn directory(engine: &Runtime, id: Uuid) -> std::path::PathBuf {
    engine.cache.join("http-uploads").join(id.to_string())
}

fn run_id(spec: &UploadSpec) -> Result<Uuid> {
    Ok(match &spec.metadata {
        UploadMetadata::Asset { metadata } => metadata.run_id.parse()?,
        UploadMetadata::Artifact { run_id, .. } => run_id.parse()?,
    })
}

async fn spec(engine: &Runtime, id: Uuid, session: Uuid) -> Result<UploadSpec> {
    let spec: UploadSpec =
        serde_json::from_slice(&fs::read(directory(engine, id).join("spec.json")).await?)?;
    engine.active(run_id(&spec)?, session).await?;
    Ok(spec)
}

pub async fn create(
    engine: &Runtime,
    id: Uuid,
    session: Uuid,
    spec: UploadSpec,
) -> Result<UploadStatus> {
    ensure!(!id.is_nil(), "upload ID must not be nil");
    ensure!(
        spec.size <= 16 * 1024 * 1024 * 1024,
        "upload exceeds 16 GiB"
    );
    ensure!(
        spec.sha256.len() == 64 && spec.sha256.bytes().all(|b| b.is_ascii_hexdigit()),
        "invalid upload SHA256"
    );
    match &spec.metadata {
        UploadMetadata::Asset { metadata } => {
            ensure!(
                metadata.asset_id.parse::<Uuid>()? == id,
                "asset ID must match upload ID"
            );
            ensure!(!metadata.name.is_empty(), "asset name must not be empty");
            crate::asset_repo::kind_value(&metadata.kind)?;
            let _: serde_json::Map<String, serde_json::Value> =
                serde_json::from_str(&metadata.metadata_json)?;
        }
        UploadMetadata::Artifact { metadata, .. } => {
            ensure!(!metadata.name.is_empty(), "artifact name must not be empty");
            ensure!(
                metadata.size_bytes == spec.size,
                "artifact size does not match upload size"
            );
            time::OffsetDateTime::from_unix_timestamp_nanos(
                i128::from(metadata.timestamp_unix_ms) * 1_000_000,
            )?;
        }
    }
    engine.active(run_id(&spec)?, session).await?;
    let dir = directory(engine, id);
    fs::create_dir_all(&dir).await?;
    let _lock = Lock::acquire(&dir.join("upload.lock")).await?;
    let path = dir.join("spec.json");
    if let Ok(bytes) = fs::read(&path).await {
        let existing: UploadSpec = serde_json::from_slice(&bytes)?;
        ensure!(existing == spec, "conflicting retry of upload ID");
    } else {
        crate::cache_limits::space(&dir, spec.size).await?;
        write_atomic(&path, &serde_json::to_vec(&spec)?).await?;
    }
    let committed = fs::try_exists(dir.join("committed")).await?;
    if !committed {
        let file = fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(dir.join("data"))
            .await?;
        file.sync_all().await?;
    }
    Ok(UploadStatus { committed })
}

pub async fn chunk(
    engine: &Runtime,
    id: Uuid,
    session: Uuid,
    index: u64,
    bytes: &[u8],
) -> Result<()> {
    let spec = spec(engine, id, session).await?;
    let offset = index
        .checked_mul(TRANSFER_CHUNK_BYTES as u64)
        .context("upload chunk offset overflows")?;
    ensure!(offset < spec.size, "upload chunk is outside the file");
    let size = (spec.size - offset).min(TRANSFER_CHUNK_BYTES as u64);
    ensure!(
        bytes.len() as u64 == size,
        "upload chunk has unexpected size"
    );
    let dir = directory(engine, id);
    let _lock = Lock::acquire(&dir.join("upload.lock")).await?;
    let receipt = dir.join(format!("{index}.received"));
    let hash = hex::encode(Sha256::digest(bytes));
    if let Ok(existing) = fs::read_to_string(&receipt).await {
        ensure!(existing == hash, "conflicting retry of upload chunk");
        return Ok(());
    }
    ensure!(
        !fs::try_exists(dir.join("committed")).await?,
        "upload is already committed"
    );
    let mut file = fs::OpenOptions::new()
        .write(true)
        .open(dir.join("data"))
        .await?;
    file.seek(std::io::SeekFrom::Start(offset)).await?;
    file.write_all(bytes).await?;
    file.sync_all().await?;
    write_atomic(&receipt, hash.as_bytes()).await
}

pub async fn commit(engine: &Runtime, id: Uuid, session: Uuid) -> Result<UploadStatus> {
    let spec = spec(engine, id, session).await?;
    let dir = directory(engine, id);
    if fs::try_exists(dir.join("committed")).await? {
        return Ok(UploadStatus { committed: true });
    }
    crate::job::check(&dir.join("error")).await?;
    let Some(lock) = Lock::try_acquire(&dir.join("upload.lock")).await? else {
        return Ok(UploadStatus { committed: false });
    };
    for index in 0..spec.size.div_ceil(TRANSFER_CHUNK_BYTES as u64) {
        ensure!(
            fs::try_exists(dir.join(format!("{index}.received"))).await?,
            "upload is missing chunks"
        );
    }
    let Ok(slot) = engine.upload_slots.clone().try_acquire_owned() else {
        return Ok(UploadStatus { committed: false });
    };
    let engine = engine.clone();
    engine.tasks.clone().spawn(async move {
        let (_lock, _slot) = (lock, slot);
        let result = async {
            let path = dir.join("data");
            ensure!(fs::metadata(&path).await?.len() == spec.size, "staged upload has unexpected size");
            let mut file = fs::File::open(&path).await?;
            let mut digest = Sha256::new();
            let mut buffer = vec![0; 256 * 1024];
            loop {
                let count = file.read(&mut buffer).await?;
                if count == 0 { break; }
                digest.update(&buffer[..count]);
            }
            ensure!(hex::encode(digest.finalize()) == spec.sha256, "staged upload SHA256 does not match");
            match spec.metadata {
                UploadMetadata::Asset { metadata } => {
                    crate::asset_commit::save(&engine, id, session, &metadata, &spec.sha256, spec.size, &path).await?;
                }
                UploadMetadata::Artifact { run_id, metadata } => {
                    engine.save_artifact(id, run_id.parse()?, &metadata, &path).await?;
                }
            }
            write_atomic(&dir.join("committed"), b"committed").await?;
            fs::remove_file(path).await?;
            anyhow::Ok(())
        }.await;
        if let Err(error) = result {
            tracing::warn!(upload = %id, error = %error, "upload commit failed; staged chunks retained for retry");
            crate::job::failed(&dir.join("error"), &error).await;
        }
    });
    Ok(UploadStatus { committed: false })
}
