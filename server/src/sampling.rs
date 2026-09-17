use anyhow::{anyhow, ensure};
use blake2::{
    Blake2bVar,
    digest::{Update, VariableOutput},
};
use bytes::Bytes;
use uuid::Uuid;

use crate::{
    db::SampleRow,
    symbols::{TextCleaner, boundary_token_id, text_to_tensor_bytes},
};

#[derive(Clone)]
pub struct SampleObject {
    pub path: String,
    pub offset: i64,
    pub length: i64,
}

#[derive(Clone)]
pub struct Sample {
    pub duration: f64,
    pub audio_id: Uuid,
    pub language_id: i32,
    pub speaker_id: u64,
    pub text: Bytes,

    pub object: SampleObject,
}

impl Sample {
    fn new(
        audio_id: Uuid,
        duration: f64,
        text: String,
        text_cleaner: &mut TextCleaner,
        language: &String,
        plbert_langs: &[String],
        speaker_id: Option<String>,
        object: SampleObject,
    ) -> anyhow::Result<Self> {
        let boundary_token_id = boundary_token_id(text_cleaner)?;
        let text_tensor = text_to_tensor_bytes(text_cleaner, boundary_token_id, &text);
        let lang_norm = language.trim().to_lowercase().replace("_", "-");
        let language_id: i32 = if plbert_langs.is_empty() {
            0
        } else {
            plbert_langs
                .iter()
                .position(|l| {
                    l == &lang_norm
                        || l == lang_norm
                            .split_once("-")
                            .unwrap_or((lang_norm.as_str(), ""))
                            .0
                })
                .ok_or_else(|| anyhow!("training audio is missing its language"))?
                as i32
        };
        let mut hasher = Blake2bVar::new(8)?;
        let speaker = speaker_id.unwrap_or("0".to_string());
        // TODO: check the ylacombe/expresso thing too
        hasher.update(speaker.as_bytes());
        let mut digest = [0u8; 8];
        hasher.finalize_variable(&mut digest)?;

        let speaker_id = u64::from_be_bytes(digest) % ((1u64 << 63) - 1);

        Ok(Sample {
            audio_id,
            duration,
            language_id,
            speaker_id,
            text: text_tensor,
            object,
        })
    }
}

pub trait Sampler: Send {
    /// `Ok(None)` means the sampler is exhausted and the stream should end.
    fn next_batch(&mut self) -> anyhow::Result<Option<Vec<Sample>>>;
}

pub struct QuerySampler {
    batches: std::vec::IntoIter<Vec<Sample>>,
}

impl QuerySampler {
    pub fn len(&self) -> usize {
        self.batches.len()
    }

    pub fn repeat(self) -> impl Sampler {
        RepeatingQuerySampler {
            batches: self.batches.cycle(),
        }
    }

    pub fn audio_ids(&self) -> Vec<Uuid> {
        self.batches
            .as_slice()
            .iter()
            .flatten()
            .map(|sample| sample.audio_id)
            .collect()
    }

    pub fn new(rows: Vec<SampleRow>, languages: &[String]) -> anyhow::Result<Self> {
        let mut cleaner = TextCleaner::default();
        let mut batches = Vec::new();
        let mut batch = Vec::new();
        let mut previous = None;
        for row in rows {
            let key = (row.batch_idx, row.sample_idx);
            ensure!(
                previous.is_none_or(|last| last < key),
                "query rows must be strictly ordered by batch_idx, sample_idx"
            );
            if previous.is_some_and(|last: (u64, u64)| last.0 != row.batch_idx) {
                batches.push(std::mem::take(&mut batch));
            }
            previous = Some(key);
            batch.push(Sample::new(
                row.audio_id,
                row.duration,
                row.text.ok_or_else(|| anyhow!("sample is missing text"))?,
                &mut cleaner,
                &row.language
                    .ok_or_else(|| anyhow!("sample is missing language"))?,
                languages,
                row.speaker_id,
                SampleObject {
                    path: row.object_path,
                    offset: row.byte_offset,
                    length: row.byte_length,
                },
            )?);
        }
        if !batch.is_empty() {
            batches.push(batch);
        }
        Ok(Self {
            batches: batches.into_iter(),
        })
    }
}

impl Sampler for QuerySampler {
    fn next_batch(&mut self) -> anyhow::Result<Option<Vec<Sample>>> {
        Ok(self.batches.next())
    }
}

struct RepeatingQuerySampler {
    batches: std::iter::Cycle<std::vec::IntoIter<Vec<Sample>>>,
}

impl Sampler for RepeatingQuerySampler {
    fn next_batch(&mut self) -> anyhow::Result<Option<Vec<Sample>>> {
        Ok(self.batches.next())
    }
}

#[cfg(test)]
#[path = "sampling_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "sampling/benchmark.rs"]
mod benchmark;
