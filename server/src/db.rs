use rand::{RngExt, SeedableRng, rngs::SmallRng};
use serde::Deserialize;
use uuid::Uuid;

use crate::run;

pub const VALIDATION_SEED_SALT: u64 = 0x76616c;
pub const TRAINING_SEED_SALT: u64 = 0x747261;

const VALIDATION_SAMPLES_QUERY: &str = "
WITH
dataset_ids AS (
    SELECT audio_file_id FROM dataset_audio_files
    WHERE dataset_id = {dataset_id:UUID}
),
segments AS (
    SELECT * FROM (
        SELECT id, audio_file_id, start_seconds, end_seconds, phon, metadata
        FROM audio_segments
        WHERE audio_file_id IN (SELECT audio_file_id FROM dataset_ids)
        ORDER BY audio_file_id, id, updated_at DESC
        LIMIT 1 BY audio_file_id, id
    )
    WHERE trim(BOTH ' ' FROM phon) != '' AND start_seconds < end_seconds
),
eligible AS (
    SELECT * FROM (
        SELECT id, duration, language, bucket_file_id, byte_offset, byte_length, virtual
        FROM audio_files
        WHERE id IN (SELECT audio_file_id FROM dataset_ids)
        ORDER BY id, updated_at DESC
        LIMIT 1 BY id
    )
    WHERE NOT virtual AND duration > 0
      AND id IN (SELECT audio_file_id FROM segments)
    QUALIFY rank() OVER (ORDER BY duration DESC) <= intDiv(count() OVER (), 10) + 1
    ORDER BY toString(id)
    LIMIT {sample_size:UInt64}
),
selected AS (
    SELECT *, max(duration) OVER () AS max_duration FROM eligible
    QUALIFY duration < {max_duration:Float64}
),
agg AS (
    SELECT
        a.id AS audio_id, a.duration, a.byte_offset, a.byte_length,
        a.bucket_file_id, a.language, a.max_duration,
        if(count() = 1, min(if(
            JSONType(s.metadata, '_source', 'annotations', 'speaker_id') = 'Null',
            NULL, JSON_VALUE(s.metadata, '$._source.annotations.speaker_id')
        )), NULL) AS speaker_id,
        arrayStringConcat(arrayMap(x -> x.4, arraySort(groupArray((
            s.start_seconds, s.end_seconds, toString(s.id), s.phon
        )))), ' ') AS text
    FROM selected AS a
    ALL INNER JOIN segments AS s ON s.audio_file_id = a.id
    GROUP BY ALL
)
SELECT
    a.audio_id, toFloat64(a.duration) AS duration,
    toNullable(a.language) AS language, a.speaker_id, toNullable(a.text) AS text,
    toNullable(if(a.duration < 1, 0., pow(2, floor(log2(toFloat64(a.duration)))))) AS lower_bound,
    toNullable(least(if(a.duration < 1, 1., 2 * lower_bound), toFloat64(a.max_duration))) AS upper_bound,
    b.path AS object_path,
    toInt64(a.byte_offset) AS byte_offset, toInt64(a.byte_length) AS byte_length
FROM agg AS a
ALL INNER JOIN bucket_files AS b ON b.id = a.bucket_file_id
WHERE lengthUTF8(a.text) <= {max_text:UInt64}
ORDER BY a.duration, toString(a.audio_id)
SETTINGS function_json_value_return_type_allow_complex = 1
";

const TRAINING_SAMPLES_QUERY: &str = "
WITH
dataset_ids AS (
    SELECT audio_file_id FROM dataset_audio_files
    WHERE dataset_id = {dataset_id:UUID}
),
segments AS (
    SELECT * FROM (
        SELECT id, audio_file_id, start_seconds, end_seconds, phon, metadata
        FROM audio_segments
        WHERE audio_file_id IN (SELECT audio_file_id FROM dataset_ids)
        ORDER BY audio_file_id, id, updated_at DESC
        LIMIT 1 BY audio_file_id, id
    )
    WHERE trim(BOTH ' ' FROM phon) != '' AND start_seconds < end_seconds
),
eligible AS (
    SELECT * FROM (
        SELECT id, duration, language, bucket_file_id, byte_offset, byte_length, virtual
        FROM audio_files
        WHERE id IN (SELECT audio_file_id FROM dataset_ids)
        ORDER BY id, updated_at DESC
        LIMIT 1 BY id
    )
    WHERE NOT virtual AND duration > 0
      AND id IN (SELECT audio_file_id FROM segments)
      AND id NOT IN {validation_ids:Array(UUID)}
),
selected AS (
    SELECT *, max(duration) OVER () AS max_duration FROM eligible
    QUALIFY duration < {max_duration:Float64}
),
agg AS (
    SELECT
        a.id AS audio_id, a.duration, a.byte_offset, a.byte_length,
        a.bucket_file_id, a.language, a.max_duration,
        if(count() = 1, min(if(
            JSONType(s.metadata, '_source', 'annotations', 'speaker_id') = 'Null',
            NULL, JSON_VALUE(s.metadata, '$._source.annotations.speaker_id')
        )), NULL) AS speaker_id,
        arrayStringConcat(arrayMap(x -> x.4, arraySort(groupArray((
            s.start_seconds, s.end_seconds, toString(s.id), s.phon
        )))), ' ') AS text
    FROM selected AS a
    ALL INNER JOIN segments AS s ON s.audio_file_id = a.id
    GROUP BY ALL
)
SELECT
    a.audio_id, toFloat64(a.duration) AS duration,
    toNullable(a.language) AS language, a.speaker_id, toNullable(a.text) AS text,
    toNullable(if(a.duration < 1, 0., pow(2, floor(log2(toFloat64(a.duration)))))) AS lower_bound,
    toNullable(least(if(a.duration < 1, 1., 2 * lower_bound), toFloat64(a.max_duration))) AS upper_bound,
    b.path AS object_path,
    toInt64(a.byte_offset) AS byte_offset, toInt64(a.byte_length) AS byte_length
FROM agg AS a
ALL INNER JOIN bucket_files AS b ON b.id = a.bucket_file_id
WHERE lengthUTF8(a.text) <= {max_text:UInt64}
ORDER BY a.duration, toString(a.audio_id)
SETTINGS function_json_value_return_type_allow_complex = 1
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
        .param("dataset_id", config.dataset_id.to_string())
        .param("sample_size", u64::try_from(config.validation.samples)?)
        .param("max_duration", config.validation.max_seconds as f64)
        .param("max_text", u64::try_from(config.max_text_tokens)?)
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
        .param("dataset_id", config.dataset_id.to_string())
        .param("validation_ids", excluded_ids)
        .param("max_duration", config.training_max_seconds() as f64)
        .param("max_text", u64::try_from(config.max_text_tokens)?)
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
