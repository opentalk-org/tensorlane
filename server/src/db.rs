use std::{future::Future, time::Duration};

use futures::Stream;
use serde::Deserialize;
use serde_json::Value;

pub const TIMEOUT: Duration = Duration::from_secs(120);

pub async fn request<T>(
    operation: impl Future<Output = clickhouse::error::Result<T>>,
) -> clickhouse::error::Result<T> {
    tokio::time::timeout(TIMEOUT, operation)
        .await
        .map_err(|_| clickhouse::error::Error::TimedOut)?
}

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
    let query = crate::query_params::bind(client.query(sql), sql, params);
    async_stream::try_stream! {
        let mut cursor = query?.fetch::<SampleRow>()?;
        while let Some(row) = request(cursor.next()).await? {
            yield row;
        }
    }
}

#[cfg(test)]
mod tests {
    #[tokio::test(start_paused = true)]
    async fn stalled_database_request_expires() {
        let started = tokio::time::Instant::now();
        let result = super::request::<()>(std::future::pending()).await;
        assert!(matches!(result, Err(clickhouse::error::Error::TimedOut)));
        assert_eq!(started.elapsed(), super::TIMEOUT);
    }
}
