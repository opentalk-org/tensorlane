use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use time::OffsetDateTime;
use uuid::Uuid;

#[derive(Clone, clickhouse::Row, Deserialize, Serialize)]
pub struct AssetRecord {
    #[serde(with = "clickhouse::serde::uuid")]
    pub id: Uuid,
    #[serde(with = "clickhouse::serde::time::datetime64::micros")]
    pub updated_at: OffsetDateTime,
    pub kind: i8,
    pub name: String,
    pub step: u64,
    pub path: String,
    pub size: u64,
    #[serde(with = "serde_big_array::BigArray")]
    pub content_hash: [u8; 64],
    #[serde(rename = "type")]
    pub asset_type: String,
    pub metadata: String,
    #[serde(with = "clickhouse::serde::uuid")]
    pub run_id: Uuid,
    #[serde(with = "clickhouse::serde::uuid")]
    pub ancestor_asset_id: Uuid,
    pub deleted: bool,
}
#[derive(Serialize)]
pub struct AssetInfo {
    pub id: Uuid,
    pub run_id: Uuid,
    pub ancestor_asset_id: Uuid,
    pub name: String,
    pub kind: &'static str,
    pub step: u64,
    pub asset_type: String,
    pub metadata: Map<String, Value>,
    pub path: String,
    pub size: u64,
    pub content_hash: String,
    #[serde(with = "time::serde::rfc3339")]
    pub updated_at: OffsetDateTime,
}
impl AssetRecord {
    pub fn info(self) -> Result<AssetInfo> {
        Ok(AssetInfo {
            id: self.id,
            run_id: self.run_id,
            ancestor_asset_id: self.ancestor_asset_id,
            name: self.name,
            kind: kind_name(self.kind)?,
            step: self.step,
            asset_type: self.asset_type,
            metadata: serde_json::from_str(&self.metadata)?,
            path: self.path,
            size: self.size,
            content_hash: String::from_utf8(self.content_hash.to_vec())?,
            updated_at: self.updated_at,
        })
    }
}
pub fn kind_name(kind: i8) -> Result<&'static str> {
    match kind {
        1 => Ok("checkpoint"),
        2 => Ok("file"),
        _ => anyhow::bail!("unknown asset kind"),
    }
}
pub fn kind_value(kind: &str) -> Result<i8> {
    match kind {
        "checkpoint" => Ok(1),
        "file" => Ok(2),
        _ => anyhow::bail!("asset kind must be checkpoint or file"),
    }
}
#[derive(Clone)]
pub struct AssetRepo {
    client: clickhouse::Client,
}
impl AssetRepo {
    pub fn new(client: clickhouse::Client) -> Self {
        Self { client }
    }
    pub async fn get(&self, id: Uuid) -> Result<Option<AssetRecord>> {
        Ok(self
            .client
            .query("SELECT ?fields FROM assets FINAL WHERE id = ? AND NOT deleted")
            .bind(id)
            .fetch_optional()
            .await?)
    }
    pub async fn for_run(&self, id: Uuid, name: Option<&str>) -> Result<Vec<AssetRecord>> {
        let sql = if name.is_some() {
            "SELECT ?fields FROM assets FINAL WHERE run_id = ? AND NOT deleted AND name = ? ORDER BY updated_at, id"
        } else {
            "SELECT ?fields FROM assets FINAL WHERE run_id = ? AND NOT deleted ORDER BY updated_at, id"
        };
        let mut query = self.client.query(sql).bind(id);
        if let Some(name) = name {
            query = query.bind(name);
        }
        Ok(query.fetch_all().await?)
    }
}
