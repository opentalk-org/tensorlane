use serde::Deserialize;
use serde_json::Value;

#[derive(clickhouse::Row, Deserialize)]
pub struct SampleRow {
    pub sample_id: String,
    pub batch_idx: u64,
    pub sample_idx: u64,
    pub metadata_json: String,
    pub blobs_json: String,
}

pub async fn fetch_samples(
    client: &clickhouse::Client,
    sql: &str,
    params: &std::collections::BTreeMap<String, Value>,
) -> anyhow::Result<Vec<SampleRow>> {
    let mut query = client.query(sql);
    for (name, value) in params {
        query = query.param(name, value);
    }
    query.fetch_all::<SampleRow>().await.map_err(Into::into)
}
