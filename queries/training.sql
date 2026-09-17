WITH
{seed:UInt64} AS shuffle_seed,
{stage_batches:Array(UInt64)} AS counts,
{stage_seconds:Array(Float64)} AS seconds,
throwIf(empty(counts) OR length(counts) != length(seconds)
    OR arrayExists(x -> x = 0, counts)
    OR arrayExists(x -> NOT isFinite(x) OR x <= 0, seconds), 'invalid stages') AS invalid_stages,
arrayFold((acc, x) -> gcd(acc, x), counts, counts[1]) AS common_batches,
arrayMap(x -> toDecimal128(toString(x), 9), seconds) AS budgets,
arrayCumSum(arrayMap((n, d) -> n * d, counts, budgets)) AS stage_ends,
arrayPushFront(arrayPopBack(stage_ends), toDecimal128(0, 9)) AS stage_starts,
arrayPushFront(arrayPopBack(arrayCumSum(counts)), toUInt64(0)) AS batch_starts,
stage_ends[-1] / common_batches AS superbatch_seconds,
dataset_ids AS (
    SELECT audio_file_id FROM dataset_audio_files
    WHERE dataset_id = {dataset_id:UUID}
),
segments AS (
    SELECT * FROM (
        SELECT id, audio_file_id, start_seconds, end_seconds, phon, metadata
        FROM audio_segments
        WHERE audio_file_id IN (SELECT audio_file_id FROM dataset_ids)
        ORDER BY audio_file_id, id, updated_at DESC, start_seconds, end_seconds, phon, metadata
        LIMIT 1 BY audio_file_id, id
    )
    WHERE trim(BOTH ' ' FROM phon) != '' AND start_seconds < end_seconds
),
eligible AS (
    SELECT * FROM (
        SELECT id, duration, language, bucket_file_id, byte_offset, byte_length, virtual
        FROM audio_files
        WHERE id IN (SELECT audio_file_id FROM dataset_ids)
        ORDER BY id, updated_at DESC, duration, language, bucket_file_id, byte_offset, byte_length, virtual
        LIMIT 1 BY id
    )
    WHERE NOT virtual AND duration > 0
      AND id NOT IN {validation_ids:Array(UUID)}
),
selected AS (
    SELECT * FROM eligible
    WHERE duration < {max_duration:Float64}
),
agg AS (
    SELECT
        a.id AS audio_id, a.duration, a.byte_offset, a.byte_length,
        a.bucket_file_id, a.language,
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
),
-- Decimal sums keep boundary assignments stable across execution plans.
ordered AS (
    SELECT *, toDecimal128(toString(duration), 9) AS audio_seconds,
        sum(audio_seconds) OVER (ORDER BY duration, audio_id ROWS UNBOUNDED PRECEDING) AS cumulative_seconds,
        row_number() OVER (ORDER BY duration, audio_id) - 1 AS source_idx,
        max(duration) OVER () AS longest_audio
    FROM agg
    WHERE invalid_stages = 0 AND lengthUTF8(text) <= {max_text:UInt64}
),
-- Reuse the eligible rows without scanning the source again for each pass.
assigned AS MATERIALIZED (
    SELECT *, toUInt64(floor(cumulative_seconds / superbatch_seconds)) AS superbatch_idx
    FROM ordered
    WHERE throwIf(longest_audio >= arrayMin(seconds),
        'audio duration must be shorter than the smallest stage budget') = 0
),
superbatches AS (
    SELECT superbatch_idx, sum(audio_seconds) AS block_seconds, count() AS block_samples,
        min(cumulative_seconds - audio_seconds) AS source_start, min(source_idx) AS first_source_idx
    FROM assigned GROUP BY superbatch_idx
),
stream AS (
    SELECT *,
        sum(block_seconds) OVER stream_order - block_seconds AS start_seconds,
        sum(block_samples) OVER stream_order - block_samples AS start_idx
    FROM (SELECT *, sum(block_seconds) OVER () AS dataset_seconds FROM superbatches)
    ARRAY JOIN range(toUInt64(floor(stage_ends[-1] / dataset_seconds)) + 1) AS pass
    WINDOW stream_order AS (
        ORDER BY pass, cityHash64(shuffle_seed, pass, superbatch_idx), superbatch_idx
        ROWS UNBOUNDED PRECEDING
    )
    QUALIFY start_seconds < stage_ends[-1]
),
planned AS (
    SELECT a.*, start_seconds + cumulative_seconds - audio_seconds - source_start AS position,
        arrayCount(bound -> position >= bound, stage_starts) AS stage,
        batch_starts[stage] + toUInt64(floor((position - stage_starts[stage]) / budgets[stage])) AS batch_idx,
        toUInt64(start_idx + source_idx - first_source_idx) AS sample_idx
    FROM assigned AS a ALL INNER JOIN stream USING (superbatch_idx)
    WHERE position < stage_ends[-1]
)
SELECT
    a.audio_id, toFloat64(a.duration) AS duration,
    toNullable(a.language) AS language, a.speaker_id, toNullable(a.text) AS text,
    batch_idx, sample_idx,
    b.path AS object_path, toInt64(a.byte_offset) AS byte_offset, toInt64(a.byte_length) AS byte_length
FROM planned AS a
ALL INNER JOIN bucket_files AS b ON b.id = a.bucket_file_id
ORDER BY batch_idx, sample_idx
SETTINGS function_json_value_return_type_allow_complex = 1, enable_materialized_cte = 1
