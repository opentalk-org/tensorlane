use serde::Deserialize;
use serde_json::Value;
use uuid::Uuid;

#[derive(clickhouse::Row, Deserialize)]
pub struct SampleRow {
    #[serde(with = "clickhouse::serde::uuid")]
    pub audio_id: Uuid,
    pub duration: f64,
    pub language: Option<String>,
    pub speaker_id: Option<String>,
    pub text: Option<String>,

    pub batch_idx: u64,
    pub sample_idx: u64,

    pub object_path: String,
    pub byte_offset: i64,
    pub byte_length: i64,
}

pub async fn fetch_samples(
    client: &clickhouse::Client,
    sql: &str,
    params: &[(&str, Value)],
) -> anyhow::Result<Vec<SampleRow>> {
    let mut query = client.query(sql);
    for (name, value) in params {
        query = query.param(name, value);
    }
    query.fetch_all::<SampleRow>().await.map_err(Into::into)
}
