use anyhow::{anyhow, bail};
use blake2::{
    Blake2bVar,
    digest::{Update, VariableOutput},
};
use bytes::Bytes;
use rand::{
    SeedableRng,
    rngs::SmallRng,
    seq::{IndexedMutRandom, IndexedRandom},
};
use uuid::Uuid;

use crate::{
    db::SampleRow,
    run,
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

pub trait InfiniteSampler: Send {
    fn sample_batch(&mut self) -> anyhow::Result<Vec<Sample>>;
}

impl<T: InfiniteSampler> Sampler for T {
    fn next_batch(&mut self) -> anyhow::Result<Option<Vec<Sample>>> {
        self.sample_batch().map(Some)
    }
}

#[derive(Clone)]
pub struct DurationBin {
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub samples: Vec<Sample>,
    pub total_seconds: f64,
}

pub struct HistogramSampler {
    template: Vec<DurationBin>,
    bins: Vec<DurationBin>,
    max_seconds: f64,
    rng: SmallRng,
    seed: u64,
    // like a epoch number
    loops: u64,
}

/// `sample_rows` should be sorted by duration
pub fn bins_from_rows(
    sample_rows: Vec<SampleRow>,
    plbert_languages: &[String],
) -> anyhow::Result<Vec<DurationBin>> {
    let mut bins: Vec<DurationBin> = vec![];

    let mut text_cleaner = TextCleaner::default();

    for row in sample_rows {
        match row {
            SampleRow {
                audio_id,
                duration,
                language: Some(lang),
                text: Some(text),
                speaker_id,
                lower_bound: Some(lower_bound),
                upper_bound: Some(upper_bound),
                object_path,
                byte_offset,
                byte_length,
                ..
            } => {
                let sample = Sample::new(
                    audio_id,
                    duration,
                    text,
                    &mut text_cleaner,
                    &lang,
                    plbert_languages,
                    speaker_id,
                    SampleObject {
                        path: object_path,
                        offset: byte_offset,
                        length: byte_length,
                    },
                )?;

                if let Some(bin) = bins
                    .iter_mut()
                    .find(|b| b.lower_bound == lower_bound && b.upper_bound == upper_bound)
                {
                    bin.samples.push(sample);
                    bin.total_seconds += duration;
                } else {
                    let bin = DurationBin {
                        lower_bound,
                        upper_bound,
                        total_seconds: duration,
                        samples: vec![sample],
                    };
                    bins.push(bin);
                }
            }
            SampleRow { language: None, .. } => {
                bail!("training audio is missing its language");
            }
            _ => {}
        }
    }

    Ok(bins)
}

impl HistogramSampler {
    pub fn new(template: Vec<DurationBin>, max_seconds: f64, seed: u64) -> Self {
        // a sample longer than the packing budget can never join a batch;
        // keep it out so it doesn't distort bin weights (mirrors the SQL
        // `duration < max_seconds` filter, applied here per sequence)
        let bins: Vec<DurationBin> = template
            .into_iter()
            .filter_map(|mut bin| {
                bin.samples.retain(|s| s.duration < max_seconds);
                if bin.samples.is_empty() {
                    return None;
                }
                bin.total_seconds = bin.samples.iter().map(|s| s.duration).sum();
                Some(bin)
            })
            .collect();

        Self {
            template: bins.clone(),
            bins,
            max_seconds,
            rng: SmallRng::seed_from_u64(seed),
            seed,
            loops: 0,
        }
    }
}

/// Loops its sample set forever; never exhausts.
impl InfiniteSampler for HistogramSampler {
    fn sample_batch(&mut self) -> anyhow::Result<Vec<Sample>> {
        if self.bins.iter().all(|b| b.samples.is_empty()) {
            self.loops += 1;
            tracing::info!(loops = self.loops, "bins drained, looping with a new seed");
            self.bins = self.template.clone();
            self.rng = SmallRng::seed_from_u64(self.seed + self.loops);
        }

        let mut remaining = self.max_seconds;
        let mut batch: Vec<Sample> = vec![];
        let bin = self
            .bins
            .choose_weighted_mut(&mut self.rng, |b| b.total_seconds)?;

        loop {
            let eligable: Vec<(usize, f64)> = bin
                .samples
                .iter()
                .enumerate()
                .filter_map(|(i, s)| {
                    if s.duration <= remaining {
                        Some((i, s.duration))
                    } else {
                        None
                    }
                })
                .collect();
            if eligable.is_empty() {
                break;
            }
            let random_sample = eligable.choose_weighted(&mut self.rng, |s| s.1)?;
            batch.push(bin.samples.remove(random_sample.0));
            bin.total_seconds = (bin.total_seconds - random_sample.1).max(0.0);
            remaining -= random_sample.1;
        }

        Ok(batch)
    }
}

struct Sequence {
    batches: u64,
    max_seconds: f64,
}

/// Serves the whole schedule as one stream: exactly `batches` batches with each
/// sequence's shape, then the next sequence, then `None` when the schedule is done.
pub struct ScheduledSampler {
    template: Vec<DurationBin>,
    schedule: std::vec::IntoIter<Sequence>,
    seed: u64,
    current: Option<(HistogramSampler, u64)>,
}

impl ScheduledSampler {
    pub fn new(
        template: Vec<DurationBin>,
        sequences: &[run::SequenceConfig],
        seed: u64,
    ) -> Self {
        let schedule: Vec<Sequence> = sequences
            .iter()
            .map(|s| Sequence {
                batches: s.batches,
                max_seconds: s.max_seconds as f64,
            })
            .collect();
        Self {
            template,
            schedule: schedule.into_iter(),
            seed,
            current: None,
        }
    }
}

impl Sampler for ScheduledSampler {
    fn next_batch(&mut self) -> anyhow::Result<Option<Vec<Sample>>> {
        loop {
            if let Some((sampler, remaining)) = self.current.as_mut()
                && *remaining > 0
            {
                let batch = sampler.sample_batch()?;
                *remaining -= 1;
                return Ok(Some(batch));
            }

            let Some(sequence) = self.schedule.next() else {
                return Ok(None);
            };
            tracing::info!(
                batches = sequence.batches,
                max_seconds = sequence.max_seconds,
                "starting batch sequence"
            );
            self.current = Some((
                HistogramSampler::new(self.template.clone(), sequence.max_seconds, self.seed),
                sequence.batches,
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{db::SampleRow, run::SequenceConfig};
    use std::collections::HashMap;

    fn row(i: u128, duration: f64, lower: f64, upper: f64) -> SampleRow {
        SampleRow {
            audio_id: Uuid::from_u128(i),
            duration,
            language: Some("en".to_string()),
            speaker_id: Some(format!("spk-{i}")),
            text: Some("wˈʌn tˈuː".to_string()),
            lower_bound: Some(lower),
            upper_bound: Some(upper),
            object_path: format!("obj-{i}"),
            byte_offset: 0,
            byte_length: 0,
        }
    }

    /// 8 rows sorted by duration, exp2 bins for max 8: (1,2) (2,4) (4,8)
    fn eight_rows() -> Vec<SampleRow> {
        [
            (1.2, 1.0, 2.0),
            (1.7, 1.0, 2.0),
            (2.2, 2.0, 4.0),
            (3.0, 2.0, 4.0),
            (3.9, 2.0, 4.0),
            (4.5, 4.0, 8.0),
            (6.0, 4.0, 8.0),
            (7.5, 4.0, 8.0),
        ]
        .iter()
        .enumerate()
        .map(|(i, &(duration, lower, upper))| row(i as u128, duration, lower, upper))
        .collect()
    }

    fn template() -> Vec<DurationBin> {
        bins_from_rows(eight_rows(), &[]).unwrap()
    }

    fn batch_ids(sampler: &mut dyn Sampler, batches: usize) -> Vec<Vec<Uuid>> {
        (0..batches)
            .map(|_| {
                sampler
                    .next_batch()
                    .unwrap()
                    .expect("sampler exhausted early")
                    .iter()
                    .map(|s| s.audio_id)
                    .collect()
            })
            .collect()
    }

    #[test]
    fn bins_group_rows_by_bounds() {
        let bins = template();
        assert_eq!(bins.len(), 3);
        let by_bounds: HashMap<(u64, u64), &DurationBin> = bins
            .iter()
            .map(|b| ((b.lower_bound as u64, b.upper_bound as u64), b))
            .collect();
        assert_eq!(by_bounds[&(1, 2)].samples.len(), 2);
        assert_eq!(by_bounds[&(2, 4)].samples.len(), 3);
        assert_eq!(by_bounds[&(4, 8)].samples.len(), 3);
        assert!((by_bounds[&(1, 2)].total_seconds - 2.9).abs() < 1e-9);
    }

    #[test]
    fn bins_reject_missing_language() {
        let mut rows = eight_rows();
        rows[3].language = None;
        assert!(bins_from_rows(rows, &[]).is_err());
    }

    #[test]
    fn bins_map_plbert_language_ids() {
        let mut rows = eight_rows();
        rows[0].language = Some("pl".to_string());
        let langs = ["en".to_string(), "pl".to_string()];
        let bins = bins_from_rows(rows, &langs).unwrap();
        let sample = |id: u128| {
            bins.iter()
                .flat_map(|b| &b.samples)
                .find(|s| s.audio_id == Uuid::from_u128(id))
                .unwrap()
                .language_id
        };
        assert_eq!(sample(0), 1);
        assert_eq!(sample(1), 0);

        let mut rows = eight_rows();
        rows[0].language = Some("de".to_string());
        assert!(bins_from_rows(rows, &langs).is_err());
    }

    #[test]
    fn histogram_is_deterministic_per_seed() {
        let mut a = HistogramSampler::new(template(), 5.0, 7);
        let mut b = HistogramSampler::new(template(), 5.0, 7);
        assert_eq!(batch_ids(&mut a, 12), batch_ids(&mut b, 12));

        let mut c = HistogramSampler::new(template(), 5.0, 8);
        assert_ne!(batch_ids(&mut a, 12), batch_ids(&mut c, 12));
    }

    #[test]
    fn histogram_respects_packing_budget() {
        let mut sampler = HistogramSampler::new(template(), 5.0, 1);
        for _ in 0..20 {
            let batch = sampler.next_batch().unwrap().unwrap();
            assert!(!batch.is_empty());
            let total: f64 = batch.iter().map(|s| s.duration).sum();
            assert!(total <= 5.0, "batch of {total}s over the 5s budget");
        }
    }

    #[test]
    fn histogram_filters_unpackable_samples() {
        // 4.5, 6.0 and 7.5 can never fit a 4s batch and must not distort weights
        let mut sampler = HistogramSampler::new(template(), 4.0, 1);
        let mut seen: Vec<Uuid> = vec![];
        for _ in 0..30 {
            seen.extend(
                sampler
                    .next_batch()
                    .unwrap()
                    .unwrap()
                    .iter()
                    .map(|s| s.audio_id),
            );
        }
        for filtered in [5u128, 6, 7] {
            assert!(!seen.contains(&Uuid::from_u128(filtered)));
        }
        for kept in [0u128, 1, 2, 3, 4] {
            assert!(seen.contains(&Uuid::from_u128(kept)));
        }
    }

    #[test]
    fn histogram_loops_every_sample_once_per_epoch() {
        // budget swallows a whole bin per batch: 3 bins -> 3 batches per loop
        let mut sampler = HistogramSampler::new(template(), 100.0, 3);
        let mut counts: HashMap<Uuid, usize> = HashMap::new();
        for ids in batch_ids(&mut sampler, 9) {
            for id in ids {
                *counts.entry(id).or_default() += 1;
            }
        }
        assert_eq!(counts.len(), 8);
        assert!(
            counts.values().all(|&c| c == 3),
            "unfair looping: {counts:?}"
        );
    }

    #[test]
    fn histogram_errors_when_nothing_fits() {
        let mut sampler = HistogramSampler::new(template(), 1.0, 1);
        assert!(sampler.next_batch().is_err());
    }

    #[test]
    fn scheduled_serves_exact_schedule_then_ends() {
        let schedule = [
            SequenceConfig {
                batches: 2,
                max_seconds: 100.0,
            },
            SequenceConfig {
                batches: 3,
                max_seconds: 100.0,
            },
        ];
        let mut sampler = ScheduledSampler::new(template(), &schedule, 1);
        for _ in 0..5 {
            assert!(!sampler.next_batch().unwrap().unwrap().is_empty());
        }
        assert!(sampler.next_batch().unwrap().is_none());
        assert!(sampler.next_batch().unwrap().is_none());
    }

    #[test]
    fn scheduled_switches_shape_between_sequences() {
        let schedule = [
            SequenceConfig {
                batches: 2,
                max_seconds: 100.0,
            },
            SequenceConfig {
                batches: 2,
                max_seconds: 2.0,
            },
        ];
        let mut sampler = ScheduledSampler::new(template(), &schedule, 1);
        for _ in 0..2 {
            sampler.next_batch().unwrap().unwrap();
        }
        for _ in 0..2 {
            let batch = sampler.next_batch().unwrap().unwrap();
            let total: f64 = batch.iter().map(|s| s.duration).sum();
            assert!(total <= 2.0);
            assert!(batch.iter().all(|s| s.duration < 2.0));
        }
        assert!(sampler.next_batch().unwrap().is_none());
    }

    #[test]
    fn scheduled_reseeds_each_sequence() {
        // identical sequences replay the identical batch order (seed resets)
        let schedule = [
            SequenceConfig {
                batches: 3,
                max_seconds: 100.0,
            },
            SequenceConfig {
                batches: 3,
                max_seconds: 100.0,
            },
        ];
        let mut sampler = ScheduledSampler::new(template(), &schedule, 5);
        let ids = batch_ids(&mut sampler, 6);
        assert_eq!(ids[..3], ids[3..]);
    }

    #[test]
    fn scheduled_skips_zero_batch_sequences() {
        let schedule = [
            SequenceConfig {
                batches: 0,
                max_seconds: 100.0,
            },
            SequenceConfig {
                batches: 1,
                max_seconds: 100.0,
            },
        ];
        let mut sampler = ScheduledSampler::new(template(), &schedule, 1);
        assert!(sampler.next_batch().unwrap().is_some());
        assert!(sampler.next_batch().unwrap().is_none());
    }
}
