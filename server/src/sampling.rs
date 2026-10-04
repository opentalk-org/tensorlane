use crate::db::SampleRow;
use anyhow::{Context, Result, ensure};
#[cfg(test)]
use futures::future::BoxFuture;
use serde::Deserialize;
use serde::de::IgnoredAny;
use std::collections::BTreeMap;

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlobRef {
    pub object: String,
    pub byte_offset: Option<u64>,
    pub byte_length: Option<u64>,
}
impl BlobRef {
    pub fn validate(&self) -> Result<()> {
        ensure!(!self.object.is_empty(), "blob object must not be empty");
        match (self.byte_offset, self.byte_length) {
            (None, None) => Ok(()),
            (Some(offset), Some(length)) => {
                ensure!(length > 0, "blob range length must be positive");
                offset
                    .checked_add(length - 1)
                    .context("blob range overflows")?;
                Ok(())
            }
            _ => anyhow::bail!("blob range needs both byte_offset and byte_length"),
        }
    }
}
#[derive(Clone)]
pub struct Sample {
    pub sample_id: String,
    pub metadata_json: String,
    pub blobs: BTreeMap<String, BlobRef>,
}
#[derive(Clone)]
pub struct BatchPlan {
    pub query_batch_idx: u64,
    pub samples: Vec<Sample>,
}
#[cfg(test)]
pub trait Sampler: Send {
    fn next_batch(&mut self) -> BoxFuture<'_, Result<Option<BatchPlan>>>;
}
mod plan;
pub use plan::QuerySampler;

impl SampleRow {
    fn sample(&self) -> Result<Sample> {
        let _: std::collections::HashMap<String, IgnoredAny> =
            serde_json::from_str(&self.metadata_json)
                .context("sample metadata must be a JSON object")?;
        let blobs: BTreeMap<String, BlobRef> = serde_json::from_str(&self.blobs_json)
            .context("blobs must be a JSON object of storage references")?;
        for (name, blob) in &blobs {
            ensure!(!name.is_empty(), "blob name must not be empty");
            blob.validate()?;
        }
        Ok(Sample {
            sample_id: self.sample_id.clone(),
            metadata_json: self.metadata_json.clone(),
            blobs,
        })
    }
}
#[cfg(test)]
#[path = "sampling/benchmark.rs"]
mod benchmark;
#[cfg(test)]
#[path = "sampling_tests.rs"]
mod tests;
