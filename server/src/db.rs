use rand::{RngExt, SeedableRng, rngs::SmallRng};
use serde::Deserialize;
use uuid::Uuid;

use crate::run;

pub const VALIDATION_SEED_SALT: u64 = 0x76616c;
pub const TRAINING_SEED_SALT: u64 = 0x747261;

const VALIDATION_SAMPLES_QUERY: &str = "
with latest_segments as (
    select id, audio_file_id, start_seconds, end_seconds, phon, speaker_id
    from audio_segments
    qualify row_number() over (
        partition by audio_file_id, id order by updated_at desc
    ) = 1
),
segments as (
    select audio_file_id,
           if(count() = 1, nullIf(min(speaker_id), ''),
              cast(null as Nullable(String))) as speaker_id,
           arrayStringConcat(
               arrayMap(segment -> segment.4,
                   arraySort(segment -> (segment.1, segment.2, segment.3),
                       groupArray((start_seconds, end_seconds, id, phon)))),
               ' '
           ) as text
    from latest_segments
    where notEmpty(trimBoth(phon))
      and start_seconds < end_seconds
    group by audio_file_id
),
base as (
    select audio.id as audio_id,
           toFloat64(audio.latest.3) as duration,
           nullIf(audio.latest.6, '') as language,
           segments.speaker_id,
           segments.text,
           bucket.path as object_path,
           toInt64(audio.latest.2) as byte_offset,
           toInt64(audio.latest.4) as byte_length
    from (
        select id,
               argMax(tuple(bucket_file_id, byte_offset, duration, byte_length,
                            virtual, language), updated_at) as latest
        from audio_files
        group by id
    ) as audio
    inner join dataset_audio_files as membership final
        on membership.audio_file_id = audio.id
    inner join segments on segments.audio_file_id = audio.id
    inner join bucket_files as bucket on bucket.id = audio.latest.1
    where membership.dataset_id = toUUID(?)
      and not audio.latest.5
      and audio.latest.3 > 0
),
eligible as (
    select *
    from base
    where duration >= (select quantileExact(0.9)(duration) from base)
    order by audio_id
    limit ?
),
binned as (
    select *, max(duration) over () as max_duration
    from eligible
)
select audio_id,
       duration,
       language,
       speaker_id,
       toNullable(text) as text,
       toNullable(if(duration < 1, 0,
                     pow(2, floor(log2(duration))))) as lower_bound,
       toNullable(least(
           pow(2, if(duration < 1, 0, floor(log2(duration)) + 1)),
           max_duration
       )) as upper_bound,
       object_path,
       byte_offset,
       byte_length
from binned
where duration < ?
  and lengthUTF8(text) <= ?
order by duration, audio_id
";

const TRAINING_SAMPLES_QUERY: &str = "
with latest_segments as (
    select id, audio_file_id, start_seconds, end_seconds, phon, speaker_id
    from audio_segments
    qualify row_number() over (
        partition by audio_file_id, id order by updated_at desc
    ) = 1
),
segments as (
    select audio_file_id,
           if(count() = 1, nullIf(min(speaker_id), ''),
              cast(null as Nullable(String))) as speaker_id,
           arrayStringConcat(
               arrayMap(segment -> segment.4,
                   arraySort(segment -> (segment.1, segment.2, segment.3),
                       groupArray((start_seconds, end_seconds, id, phon)))),
               ' '
           ) as text
    from latest_segments
    where notEmpty(trimBoth(phon))
      and start_seconds < end_seconds
    group by audio_file_id
),
base as (
    select audio.id as audio_id,
           toFloat64(audio.latest.3) as duration,
           nullIf(audio.latest.6, '') as language,
           segments.speaker_id,
           segments.text,
           bucket.path as object_path,
           toInt64(audio.latest.2) as byte_offset,
           toInt64(audio.latest.4) as byte_length
    from (
        select id,
               argMax(tuple(bucket_file_id, byte_offset, duration, byte_length,
                            virtual, language), updated_at) as latest
        from audio_files
        group by id
    ) as audio
    inner join dataset_audio_files as membership final
        on membership.audio_file_id = audio.id
    inner join segments on segments.audio_file_id = audio.id
    inner join bucket_files as bucket on bucket.id = audio.latest.1
    where membership.dataset_id = toUUID(?)
      and not audio.latest.5
      and audio.latest.3 > 0
),
eligible as (
    select *, max(duration) over () as max_duration
    from base
    where not has(?, toString(audio_id))
)
select audio_id,
       duration,
       language,
       speaker_id,
       toNullable(text) as text,
       toNullable(if(duration < 1, 0,
                     pow(2, floor(log2(duration))))) as lower_bound,
       toNullable(least(
           pow(2, if(duration < 1, 0, floor(log2(duration)) + 1)),
           max_duration
       )) as upper_bound,
       object_path,
       byte_offset,
       byte_length
from eligible
where duration < ?
  and lengthUTF8(text) <= ?
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
        .bind(config.validation.samples)
        .bind(config.validation.max_seconds as f64)
        .bind(config.max_text_tokens)
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
