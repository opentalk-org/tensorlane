use rand::{RngExt, SeedableRng, rngs::SmallRng};
use serde::Deserialize;
use uuid::Uuid;

use crate::run;

pub const VALIDATION_SEED_SALT: u64 = 0x76616c;
pub const TRAINING_SEED_SALT: u64 = 0x747261;

const VALIDATION_SAMPLES_QUERY: &str = "
select ?fields
from some_cool_samples
where dataset_id = toUUID(?)
  and duration < ?
  and length(text) <= ?
order by duration, audio_id
limit ?
";

const TRAINING_SAMPLES_QUERY: &str = "
select ?fields
from some_cool_samples
where dataset_id = toUUID(?)
  and not has(?, toString(audio_id))
  and duration < ?
  and length(text) <= ?
order by duration, audio_id
";

#[derive(clickhouse::Row, Deserialize)]
pub struct SampleRow {
    #[serde(with = "clickhouse::serde::uuid")]
    pub audio_id: Uuid,
    pub duration: f64,
    pub language: Option<String>,
    pub speaker_id: Option<String>,
    pub text: Option<String>,

    pub lower_bound: Option<f64>,
    pub upper_bound: Option<f64>,

    pub object_path: String,
    pub byte_offset: i64,
    pub byte_length: i64,
}

pub async fn fetch_validation_samples(
    client: &clickhouse::Client,
    config: &run::DataConfig,
) -> anyhow::Result<Vec<SampleRow>> {
    client
        .query(VALIDATION_SAMPLES_QUERY)
        .bind(config.dataset_id.to_string())
        .bind(config.validation.max_seconds as f64)
        .bind(config.max_text_tokens)
        .bind(config.validation.samples)
        .fetch_all::<SampleRow>()
        .await
        .map_err(Into::into)
}

pub async fn fetch_training_samples(
    client: &clickhouse::Client,
    excluded_ids: &[Uuid],
    config: &run::DataConfig,
) -> anyhow::Result<Vec<SampleRow>> {
    let excluded_ids: Vec<String> = excluded_ids.iter().map(Uuid::to_string).collect();
    client
        .query(TRAINING_SAMPLES_QUERY)
        .bind(config.dataset_id.to_string())
        .bind(excluded_ids)
        .bind(config.training_max_seconds() as f64)
        .bind(config.max_text_tokens)
        .fetch_all::<SampleRow>()
        .await
        .map_err(Into::into)
}

pub fn synthetic_rows(
    max_seconds: f64,
    seed: u64,
    language: &str,
    count: usize,
    seed_salt: u64,
) -> Vec<SampleRow> {
    let mut rng = SmallRng::seed_from_u64(seed ^ seed_salt);

    let mut durations: Vec<f64> = (0..count)
        // log-uniform in [1, max_seconds) so the exp2 bins populate evenly
        .map(|_| rng.random_range(0f64..max_seconds.ln()).exp())
        .collect();
    durations.sort_by(f64::total_cmp);
    let max_duration = durations.last().copied().unwrap_or(1.0);

    durations
        .iter()
        .enumerate()
        .map(|(i, &duration)| {
            let (lower_bound, upper_bound) = exp2_bounds(duration, max_duration);
            SampleRow {
                audio_id: Uuid::from_u128(rng.random()),
                duration,
                language: Some(language.to_string()),
                speaker_id: Some(format!("spk-{}", i % 3)),
                text: Some("wˈʌn tˈuː θrˈiː".to_string()),
                lower_bound: Some(lower_bound),
                upper_bound: Some(upper_bound),
                object_path: "synthetic".to_string(),
                byte_offset: 0,
                byte_length: 0,
            }
        })
        .collect()
}

fn exp2_bounds(duration: f64, max_duration: f64) -> (f64, f64) {
    let top = max_duration.ceil().log2().floor() as i32;
    let bounds: Vec<f64> = (0..=top).map(|x| 2f64.powi(x)).collect();

    let bin_index = bounds
        .iter()
        .take_while(|&&bound| bound <= duration)
        .count();
    let lower = if bin_index == 0 {
        0.0
    } else {
        bounds[bin_index - 1]
    };
    let upper = 2f64.powi(bin_index as i32).min(max_duration);
    (lower, upper)
}
