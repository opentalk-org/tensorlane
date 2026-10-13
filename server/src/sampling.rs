use crate::{
    db::SampleRow,
    shared_cache::{TemporaryFile, write_atomic},
};
use anyhow::{Context, Result, ensure};
use futures::{Stream, TryStreamExt};
use prost::Message;
use serde::{Deserialize, de::IgnoredAny};
use std::{
    collections::{BTreeMap, HashMap},
    io::SeekFrom,
    path::Path,
};
use tokio::{
    fs::{self, File},
    io::{AsyncBufReadExt, AsyncReadExt, AsyncSeekExt, AsyncWriteExt, BufReader, BufWriter},
};

const IO_BUFFER_BYTES: usize = 256 * 1024;

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlobRef {
    pub object: String,
    pub byte_offset: Option<u64>,
    pub byte_length: Option<u64>,
}
#[derive(Clone)]
pub struct Sample {
    pub sample_id: String,
    pub metadata_json: String,
    pub blobs: BTreeMap<String, BlobRef>,
}
pub struct BatchPlan {
    pub query_batch_idx: u64,
    pub samples: Vec<Sample>,
}

impl SampleRow {
    fn sample(&self) -> Result<Sample> {
        let _: HashMap<String, IgnoredAny> = serde_json::from_str(&self.metadata_json)
            .context("sample metadata must be a JSON object")?;
        let blobs: BTreeMap<String, BlobRef> = serde_json::from_str(&self.blobs_json)
            .context("blobs must be a JSON object of storage references")?;
        for (name, blob) in &blobs {
            ensure!(!name.is_empty(), "blob name must not be empty");
            ensure!(!blob.object.is_empty(), "blob object must not be empty");
            match (blob.byte_offset, blob.byte_length) {
                (None, None) => {}
                (Some(offset), Some(length)) => ensure!(
                    length > 0 && offset.checked_add(length).is_some(),
                    "blob range is invalid"
                ),
                _ => anyhow::bail!("blob range needs both byte_offset and byte_length"),
            }
        }
        Ok(Sample {
            sample_id: self.sample_id.clone(),
            metadata_json: self.metadata_json.clone(),
            blobs,
        })
    }
}

pub struct QuerySampler {
    reader: BufReader<File>,
    index: File,
    repeat: bool,
    batches: u64,
}
impl QuerySampler {
    pub async fn create(
        name: &str,
        rows: impl Stream<Item = Result<SampleRow>> + Send,
        path: &Path,
    ) -> Result<()> {
        let part = TemporaryFile(path.with_extension(format!("{}.part", uuid::Uuid::new_v4())));
        let mut writer = BufWriter::with_capacity(IO_BUFFER_BYTES, File::create(&part.0).await?);
        let mut previous = None;
        let mut offsets = Vec::new();
        let mut bytes = 0u64;
        futures::pin_mut!(rows);
        while let Some(row) = rows.try_next().await? {
            let key = (row.batch_idx, row.sample_idx);
            ensure!(
                previous.is_none_or(|last| last < key),
                "query rows must be strictly ordered by batch_idx, sample_idx"
            );
            if let Err(error) = row.sample() {
                tracing::warn!(stream = name, sample = row.sample_id, error = %error, "skipping invalid sample");
                continue;
            }
            if previous.is_none_or(|(batch, _)| batch != row.batch_idx) {
                offsets.extend_from_slice(&bytes.to_le_bytes());
            }
            previous = Some(key);
            let encoded = row.encode_to_vec();
            writer.write_u64_le(encoded.len() as u64).await?;
            writer.write_all(&encoded).await?;
            bytes += encoded.len() as u64 + 8;
        }
        writer.flush().await?;
        writer.get_ref().sync_all().await?;
        fs::rename(&part.0, path).await?;
        write_atomic(&path.with_extension("index"), &offsets).await?;
        write_atomic(
            &path.with_extension("ready"),
            uuid::Uuid::new_v4().as_bytes(),
        )
        .await?;
        tracing::info!(
            stream = name,
            batches = offsets.len() / 8,
            bytes,
            "query plan spooled to disk"
        );
        Ok(())
    }

    pub async fn open(path: &Path, repeat: bool) -> Result<Self> {
        let index = File::open(path.with_extension("index")).await?;
        Ok(Self {
            reader: BufReader::with_capacity(IO_BUFFER_BYTES, File::open(path).await?),
            repeat,
            batches: index.metadata().await?.len() / 8,
            index,
        })
    }

    pub async fn batch_at(&mut self, sequence: u64) -> Result<Option<BatchPlan>> {
        if self.batches == 0 || (!self.repeat && sequence >= self.batches) {
            return Ok(None);
        }
        self.index
            .seek(SeekFrom::Start((sequence % self.batches) * 8))
            .await?;
        let offset = self.index.read_u64_le().await?;
        self.reader.seek(SeekFrom::Start(offset)).await?;
        let mut batch: Option<BatchPlan> = None;
        while !self.reader.fill_buf().await?.is_empty() {
            let mut record = vec![0; self.reader.read_u64_le().await? as usize];
            self.reader.read_exact(&mut record).await?;
            let row = SampleRow::decode(record.as_slice())?;
            match &mut batch {
                Some(batch) if batch.query_batch_idx != row.batch_idx => break,
                Some(batch) => batch.samples.push(row.sample()?),
                None => {
                    batch = Some(BatchPlan {
                        query_batch_idx: row.batch_idx,
                        samples: vec![row.sample()?],
                    })
                }
            }
        }
        Ok(batch)
    }
}

#[cfg(test)]
#[path = "sampling_tests.rs"]
mod tests;
