use crate::{
    asset_repo::{AssetRecord, kind_value},
    runtime::Runtime,
};
use anyhow::{Result, ensure};
use tensorlane_protocol::SaveAssetMetadata;
use time::{Duration, OffsetDateTime};
use uuid::Uuid;

pub async fn save(
    engine: &Runtime,
    id: Uuid,
    metadata: &SaveAssetMetadata,
    hash: &str,
    size: u64,
) -> Result<()> {
    let run = metadata.run_id.parse()?;
    let kind = kind_value(&metadata.kind)?;
    let config = engine.active(run).await?;
    if let Some(existing) = engine.repo.get_asset(id).await? {
        ensure!(
            existing.run_id == run
                && existing.name == metadata.name
                && existing.content_hash.as_slice() == hash.as_bytes()
                && existing.step == metadata.step
                && existing.kind == kind
                && existing.metadata == metadata.metadata_json,
            "conflicting retry of asset ID"
        );
        return Ok(());
    }
    let intent = format!("assets/{id}");
    let record = match engine.read_state::<AssetRecord>(&intent).await? {
        Some(record) => record,
        None => {
            let mut parent = engine
                .repo
                .run_assets(run, Some(&metadata.name))
                .await?
                .pop();
            if parent.is_none()
                && let Some(input) = config.assets.get(&metadata.name).and_then(|a| a.asset_id)
            {
                parent = engine.repo.get_asset(input).await?;
            }
            let now = OffsetDateTime::now_utc();
            let mut updated_at = now.replace_nanosecond(now.nanosecond() / 1000 * 1000)?;
            if let Some(parent) = &parent {
                updated_at = updated_at.max(parent.updated_at + Duration::microseconds(1));
            }
            let record = AssetRecord {
                id,
                updated_at,
                kind,
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
                ancestor_asset_id: parent.map_or(Uuid::nil(), |p| p.id),
                deleted: false,
            };
            engine.create_state(&intent, &record).await?
        }
    };
    let client = engine
        .database
        .clone()
        .with_setting("insert_deduplication_token", id.to_string());
    let mut insert = client.insert::<AssetRecord>("assets").await?;
    insert.write(&record).await?;
    insert.end().await?;
    Ok(())
}
