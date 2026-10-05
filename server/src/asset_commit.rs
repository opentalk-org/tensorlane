use crate::{asset_repo::AssetRecord, runtime::Runtime};
use anyhow::{Result, ensure};
use tensorlane_protocol::SaveAssetMetadata;
use uuid::Uuid;

pub async fn save(
    engine: &Runtime,
    id: Uuid,
    session: Uuid,
    metadata: &SaveAssetMetadata,
    hash: &str,
    size: u64,
) -> Result<()> {
    let run = metadata.run_id.parse()?;
    let config = engine.active(run, session).await?;
    if let Some(existing) = engine.repo.get_asset(id).await? {
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
    let intent = format!("assets/{id}");
    let record: AssetRecord = if let Some(record) = engine.read_state(&intent).await? {
        record
    } else {
        let mut previous = engine.repo.run_assets(run, Some(&metadata.name)).await?;
        let parent = match previous.pop() {
            Some(parent) => Some(parent),
            None => match config.assets.get(&metadata.name).and_then(|a| a.asset_id) {
                Some(id) => engine.repo.get_asset(id).await?,
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
            path: engine.asset_key(id),
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
        engine.create_state(&intent, &record).await?
    };
    engine.save_asset(&record).await
}
