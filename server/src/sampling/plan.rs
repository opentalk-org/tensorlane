use super::BatchPlan;
use crate::{MAX_BATCH_BYTES, db::SampleRow};
use anyhow::{Context, Result, ensure};
use futures::{Stream, TryStreamExt};
use prost::Message;
use std::path::{Path, PathBuf};
use tokio::{
    fs::{self, File},
    io::{AsyncBufReadExt, AsyncReadExt, AsyncSeekExt, AsyncWriteExt, BufReader, BufWriter},
};

const IO_BUFFER_BYTES: usize = 256 * 1024;
const MAX_BATCH_SAMPLES: usize = 65_536;

struct PlanFile {
    path: PathBuf,
    keep: bool,
}
impl Drop for PlanFile {
    fn drop(&mut self) {
        if !self.keep {
            let _ = std::fs::remove_file(&self.path);
            let _ = std::fs::remove_file(self.path.with_extension("index"));
        }
    }
}

pub struct QuerySampler {
    reader: BufReader<File>,
    pending: Option<SampleRow>,
    repeat: bool,
    batches: u64,
    _file: PlanFile,
}
impl QuerySampler {
    pub async fn create(
        name: &str,
        rows: impl Stream<Item = Result<SampleRow>> + Send,
        path: &Path,
        repeat: bool,
    ) -> Result<Self> {
        let part = path.with_extension("part");
        let output = File::create_new(&part)
            .await
            .context("creating query plan")?;
        let mut file = PlanFile {
            path: part,
            keep: false,
        };
        let mut writer = BufWriter::with_capacity(IO_BUFFER_BYTES, output);
        let mut previous = None;
        let (mut batches, mut samples, mut bytes) = (0u64, 0u64, 0u64);
        let (mut batch_bytes, mut batch_samples) = (0usize, 0usize);
        let mut encoded = Vec::new();
        let mut offsets = Vec::new();
        futures::pin_mut!(rows);
        while let Some(row) = rows.try_next().await? {
            let size = row.encoded_len();
            let key = (row.batch_idx, row.sample_idx);
            ensure!(
                previous.is_none_or(|last| last < key),
                "query rows must be strictly ordered by batch_idx, sample_idx"
            );
            if previous.is_none_or(|(batch, _)| batch != row.batch_idx) {
                offsets.extend_from_slice(&bytes.to_le_bytes());
                batches += 1;
                batch_bytes = 0;
                batch_samples = 0;
            }
            previous = Some(key);
            batch_bytes += size + 4;
            batch_samples += 1;
            ensure!(
                batch_bytes <= MAX_BATCH_BYTES,
                "batch descriptors exceed 64 MiB"
            );
            ensure!(
                batch_samples <= MAX_BATCH_SAMPLES,
                "batch exceeds 65536 samples"
            );
            row.sample()?;
            bytes += size as u64 + 4;
            ensure!(bytes <= 512 * 1024 * 1024, "query plan exceeds 512 MiB");
            encoded.clear();
            row.encode(&mut encoded)?;
            writer.write_u32_le(size as u32).await?;
            writer.write_all(&encoded).await?;
            samples += 1;
        }
        writer.flush().await?;
        writer.get_ref().sync_all().await?;
        drop(writer);
        fs::rename(&file.path, path).await?;
        file.path = path.to_path_buf();
        crate::shared_cache::write_atomic(&path.with_extension("index"), &offsets).await?;
        let reader = BufReader::with_capacity(IO_BUFFER_BYTES, File::open(path).await?);
        tracing::info!(
            stream = name,
            samples,
            batches,
            bytes,
            "query plan spooled to disk"
        );
        Ok(Self {
            reader,
            pending: None,
            repeat,
            batches,
            _file: file,
        })
    }

    pub fn persist(&mut self) {
        self._file.keep = true;
    }

    pub async fn open(path: &Path, repeat: bool) -> Result<Self> {
        let length = fs::metadata(path.with_extension("index")).await?.len();
        ensure!(length % 8 == 0, "invalid query plan index");
        Ok(Self {
            reader: BufReader::with_capacity(IO_BUFFER_BYTES, File::open(path).await?),
            pending: None,
            repeat,
            batches: length / 8,
            _file: PlanFile {
                path: path.to_owned(),
                keep: true,
            },
        })
    }

    pub async fn batch_at(&mut self, sequence: u64) -> Result<Option<BatchPlan>> {
        if self.batches == 0 || (!self.repeat && sequence >= self.batches) {
            return Ok(None);
        }
        let mut index = File::open(self._file.path.with_extension("index")).await?;
        index
            .seek(std::io::SeekFrom::Start((sequence % self.batches) * 8))
            .await?;
        let offset = index.read_u64_le().await?;
        self.reader.seek(std::io::SeekFrom::Start(offset)).await?;
        self.pending = None;
        self.next_batch().await
    }

    async fn read_row(&mut self) -> Result<Option<SampleRow>> {
        if self.reader.fill_buf().await?.is_empty() {
            return Ok(None);
        }
        let size = self
            .reader
            .read_u32_le()
            .await
            .context("truncated plan record length")? as usize;
        ensure!(size <= MAX_BATCH_BYTES, "plan descriptor exceeds 64 MiB");
        let mut bytes = vec![0; size];
        self.reader
            .read_exact(&mut bytes)
            .await
            .context("truncated plan record")?;
        Ok(Some(
            SampleRow::decode(bytes.as_slice()).context("invalid plan record")?,
        ))
    }

    pub(super) async fn next_batch(&mut self) -> Result<Option<BatchPlan>> {
        let mut first = match self.pending.take() {
            Some(row) => Some(row),
            None => self.read_row().await?,
        };
        if first.is_none() && self.repeat && self.batches > 0 {
            self.reader.rewind().await?;
            first = self.read_row().await?;
            ensure!(first.is_some(), "repeating plan is unexpectedly empty");
        }
        let Some(first) = first else {
            return Ok(None);
        };
        let mut size = first.encoded_len() + 4;
        let mut batch = BatchPlan {
            query_batch_idx: first.batch_idx,
            samples: vec![first.sample()?],
        };
        while let Some(row) = self.read_row().await? {
            if row.batch_idx != batch.query_batch_idx {
                self.pending = Some(row);
                break;
            }
            size += row.encoded_len() + 4;
            ensure!(size <= MAX_BATCH_BYTES, "batch descriptors exceed 64 MiB");
            ensure!(
                batch.samples.len() < MAX_BATCH_SAMPLES,
                "batch exceeds 65536 samples"
            );
            batch.samples.push(row.sample()?);
        }
        Ok(Some(batch))
    }
}
