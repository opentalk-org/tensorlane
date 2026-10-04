use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const MAX_BATCH_BYTES: usize = 64 * 1024 * 1024;
pub const TRANSFER_CHUNK_BYTES: usize = 4 * 1024 * 1024;
pub const SESSION_HEADER: &str = "x-tensorlane-session";

#[derive(Clone, Serialize, Deserialize)]
pub struct InitResponse {
    pub run_id: String,
    pub config: String,
    pub assets: Vec<String>,
    pub streams: Vec<String>,
}

#[derive(Clone, Serialize, Deserialize, Default)]
pub struct AssetMetadata {
    pub entrypoint: Option<String>,
    pub asset_id: Option<String>,
    pub metadata_json: String,
    pub kind: String,
    pub asset_type: String,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct AssetDownload {
    pub metadata: AssetMetadata,
    pub size: u64,
    pub etag: String,
    pub sha256: Option<String>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct EndRequest {
    pub failed: bool,
}

#[derive(Clone, Serialize, Deserialize, PartialEq)]
pub struct SaveAssetMetadata {
    pub run_id: String,
    pub asset_id: String,
    pub name: String,
    pub step: u64,
    pub kind: String,
    pub asset_type: Option<String>,
    pub metadata_json: String,
    pub content_type: String,
}

#[derive(Clone, Serialize, Deserialize, PartialEq)]
pub struct ArtifactMetric {
    pub step: u64,
    pub timestamp_unix_ms: i64,
    pub name: String,
    pub content_type: String,
    pub size_bytes: u64,
}

#[derive(Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum UploadMetadata {
    Asset {
        metadata: SaveAssetMetadata,
    },
    Artifact {
        run_id: String,
        metadata: ArtifactMetric,
    },
}

#[derive(Clone, Serialize, Deserialize, PartialEq)]
pub struct UploadSpec {
    pub metadata: UploadMetadata,
    pub size: u64,
    pub sha256: String,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct UploadStatus {
    pub committed: bool,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ScalarMetric {
    pub step: u64,
    pub timestamp_unix_ms: i64,
    pub name: String,
    pub value: f32,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ArrayMetric {
    pub step: u64,
    pub timestamp_unix_ms: i64,
    pub name: String,
    pub value: Vec<f32>,
}

#[derive(Clone, Serialize, Deserialize, Default)]
pub struct MetricBatch {
    #[serde(default)]
    pub scalars: Vec<ScalarMetric>,
    #[serde(default)]
    pub arrays: Vec<ArrayMetric>,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct Sample {
    #[prost(string, tag = "6")]
    pub sample_id: String,
    #[prost(string, tag = "7")]
    pub metadata_json: String,
    #[prost(map = "string, bytes", tag = "8")]
    pub blobs: HashMap<String, Vec<u8>>,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct DataResponse {
    #[prost(message, repeated, tag = "1")]
    pub batch: Vec<Sample>,
    #[prost(string, tag = "2")]
    pub stream: String,
    #[prost(uint64, tag = "3")]
    pub batch_id: u64,
    #[prost(uint64, tag = "4")]
    pub query_batch_idx: u64,
    #[prost(double, tag = "5")]
    pub load_seconds: f64,
    #[prost(double, tag = "6")]
    pub server_wait_seconds: f64,
}
