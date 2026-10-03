use futures::Stream;
use serde::Deserialize;
use serde_json::Value;

#[derive(clickhouse::Row, Deserialize, prost::Message)]
pub struct SampleRow {
    #[prost(string, tag = "1")]
    pub sample_id: String,
    #[prost(uint64, tag = "2")]
    pub batch_idx: u64,
    #[prost(uint64, tag = "3")]
    pub sample_idx: u64,
    #[prost(string, tag = "4")]
    pub metadata_json: String,
    #[prost(string, tag = "5")]
    pub blobs_json: String,
}

pub fn stream_samples(
    client: &clickhouse::Client,
    sql: &str,
    params: &std::collections::BTreeMap<String, Value>,
) -> impl Stream<Item = anyhow::Result<SampleRow>> + Send + use<> {
    let mut query = client.query(sql);
    for (name, value) in params {
        query = query.param(name, value);
    }
    async_stream::try_stream! {
        let mut cursor = query.fetch::<SampleRow>()?;
        while let Some(row) = cursor.next().await? {
            yield row;
        }
    }
}
