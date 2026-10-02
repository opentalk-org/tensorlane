use crate::db::SampleRow;
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use serde_json::{Map, Value};
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
pub trait Sampler: Send {
    fn next_batch(&mut self) -> Result<Option<BatchPlan>>;
}
pub struct QuerySampler {
    batches: Vec<BatchPlan>,
    next: usize,
    repeat: bool,
}
impl QuerySampler {
    pub fn new(rows: Vec<SampleRow>) -> Result<Self> {
        let mut batches: Vec<BatchPlan> = Vec::new();
        let mut previous = None;
        for row in rows {
            let key = (row.batch_idx, row.sample_idx);
            ensure!(
                previous.is_none_or(|last| last < key),
                "query rows must be strictly ordered by batch_idx, sample_idx"
            );
            previous = Some(key);
            let _: Map<String, Value> = serde_json::from_str(&row.metadata_json)
                .context("sample metadata must be a JSON object")?;
            let blobs: BTreeMap<String, BlobRef> = serde_json::from_str(&row.blobs_json)
                .context("blobs must be a JSON object of storage references")?;
            for (name, blob) in &blobs {
                ensure!(!name.is_empty(), "blob name must not be empty");
                blob.validate()?;
            }
            if batches
                .last()
                .is_none_or(|batch| batch.query_batch_idx != row.batch_idx)
            {
                batches.push(BatchPlan {
                    query_batch_idx: row.batch_idx,
                    samples: Vec::new(),
                });
            }
            batches.last_mut().unwrap().samples.push(Sample {
                sample_id: row.sample_id,
                metadata_json: row.metadata_json,
                blobs,
            });
        }
        Ok(Self {
            batches,
            next: 0,
            repeat: false,
        })
    }
    pub fn len(&self) -> usize {
        self.batches.len()
    }
    pub fn repeat(mut self) -> Self {
        self.repeat = true;
        self
    }
}
impl Sampler for QuerySampler {
    fn next_batch(&mut self) -> Result<Option<BatchPlan>> {
        if self.next == self.batches.len() {
            if !self.repeat || self.batches.is_empty() {
                return Ok(None);
            }
            self.next = 0;
        }
        let batch = self.batches[self.next].clone();
        self.next += 1;
        Ok(Some(batch))
    }
}
#[cfg(test)]
#[path = "sampling/benchmark.rs"]
mod benchmark;
#[cfg(test)]
#[path = "sampling_tests.rs"]
mod tests;
