use crate::{
    asset_repo::AssetRecord,
    runtime::Runtime,
    shared_cache::{Lock, write_atomic},
};
use anyhow::{Result, ensure};
use std::path::Path;
use tensorlane_protocol::SaveAssetMetadata;
use tokio::fs;
use uuid::Uuid;

pub async fn save(
    engine: &Runtime,
    id: Uuid,
    session: Uuid,
    metadata: &SaveAssetMetadata,
    hash: &str,
    size: u64,
    path: &Path,
) -> Result<()> {
    let run = metadata.run_id.parse()?;
    let (_, config) = engine.active(run, session).await?;
    if let Some(existing) = engine.repo.assets().get(id).await? {
        ensure!(
            existing.run_id == run
                && existing.name == metadata.name
                && existing.content_hash.as_slice() == hash.as_bytes()
                && existing.step == metadata.step
                && existing.kind == crate::asset_repo::kind_value(&metadata.kind)?
                && existing.metadata == metadata.metadata_json,
            "conflicting retry of asset ID"
        );
        return Ok(());
    }
    let _lineage = Lock::acquire(&engine.run_dir(run).join("lineage.lock")).await?;
    let intent = path.with_extension("asset.json");
    let record: AssetRecord = if fs::try_exists(&intent).await? {
        serde_json::from_slice(&fs::read(&intent).await?)?
    } else {
        let mut previous = engine
            .repo
            .assets()
            .for_run(run, Some(&metadata.name))
            .await?;
        let parent = match previous.pop() {
            Some(parent) => Some(parent),
            None => match config.assets.get(&metadata.name).and_then(|a| a.asset_id) {
                Some(id) => engine.repo.assets().get(id).await?,
                None => None,
            },
        };
        let now = time::OffsetDateTime::now_utc();
        let timestamp = now.replace_nanosecond(now.nanosecond() / 1000 * 1000)?;
        let updated_at = parent
            .as_ref()
            .map(|parent| timestamp.max(parent.updated_at + time::Duration::microseconds(1)))
            .unwrap_or(timestamp);
        let record = AssetRecord {
            id,
            updated_at,
            kind: crate::asset_repo::kind_value(&metadata.kind)?,
            name: metadata.name.clone(),
            step: metadata.step,
            path: engine.uploads.asset_key(id),
            size,
            content_hash: hash.as_bytes().try_into()?,
            asset_type: metadata
                .asset_type
                .clone()
                .or(config.asset_type)
                .or_else(|| parent.as_ref().map(|p| p.asset_type.clone()))
                .unwrap_or_else(|| "generic".into()),
            metadata: metadata.metadata_json.clone(),
            run_id: run,
            ancestor_asset_id: parent.map(|p| p.id).unwrap_or(Uuid::nil()),
            deleted: false,
        };
        write_atomic(&intent, &serde_json::to_vec(&record)?).await?;
        record
    };
    engine
        .uploads
        .save_asset(&record, path, &metadata.content_type)
        .await
}
