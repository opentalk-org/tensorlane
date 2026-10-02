WITH
{seed:UInt64} AS shuffle_seed,
{batches:UInt64} AS batch_count,
toDecimal128(toString({max_seconds:Float64}), 9) AS batch_seconds,
toDecimal128(toString({superbatch_seconds:Float64}), 9) AS superbatch_seconds,
throwIf(batch_count = 0
    OR NOT isFinite({max_seconds:Float64}) OR {max_seconds:Float64} <= 0
    OR NOT isFinite({superbatch_seconds:Float64}) OR {superbatch_seconds:Float64} <= 0,
    'invalid batch or superbatch budget') AS invalid_parameters,
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
ordered AS (
    SELECT *, toDecimal128(toString(duration), 9) AS audio_seconds,
        sum(audio_seconds) OVER (ORDER BY duration, audio_id ROWS UNBOUNDED PRECEDING) AS cumulative_seconds,
        row_number() OVER (ORDER BY duration, audio_id) - 1 AS source_idx,
        max(duration) OVER () AS longest_audio
    FROM agg
    WHERE invalid_parameters = 0 AND lengthUTF8(text) <= {max_text:UInt64}
),
assigned AS MATERIALIZED (
    SELECT *, toUInt64(floor(cumulative_seconds / superbatch_seconds)) AS superbatch_idx
    FROM ordered
    WHERE throwIf(longest_audio >= {max_seconds:Float64},
        'audio duration must be shorter than the batch budget') = 0
),
superbatches AS (
    SELECT superbatch_idx, sum(audio_seconds) AS block_seconds, count() AS block_samples,
        min(cumulative_seconds - audio_seconds) AS source_start, min(source_idx) AS first_source_idx
    FROM assigned GROUP BY superbatch_idx
),
stream AS (
    SELECT *,
        sum(block_seconds) OVER stream_order - block_seconds AS start_seconds,
        sum(block_samples) OVER stream_order - block_samples AS start_idx,
        intDiv({dataset_offset:UInt64}, dataset_samples) * dataset_samples AS pass_start_idx
    FROM (
        SELECT *, sum(block_seconds) OVER () AS dataset_seconds,
            sum(block_samples) OVER () AS dataset_samples
        FROM superbatches
    )
    ARRAY JOIN range(
        toUInt64(intDiv({dataset_offset:UInt64}, dataset_samples)),
        toUInt64(intDiv({dataset_offset:UInt64}, dataset_samples))
            + toUInt64(floor(batch_count * batch_seconds / dataset_seconds)) + 2
    ) AS pass
    WINDOW stream_order AS (
        ORDER BY pass, cityHash64(shuffle_seed, pass, superbatch_idx), superbatch_idx
        ROWS UNBOUNDED PRECEDING
    )
),
planned AS (
    SELECT a.*, start_seconds + cumulative_seconds - audio_seconds - source_start AS position,
        toUInt64(pass_start_idx + start_idx + source_idx - first_source_idx) AS sample_idx
    FROM assigned AS a ALL INNER JOIN stream USING (superbatch_idx)
),
remaining AS (
    SELECT * FROM planned WHERE sample_idx >= {dataset_offset:UInt64}
),
batched AS (
    SELECT *, toUInt64(floor((position - min(position) OVER ()) / batch_seconds)) AS batch_idx
    FROM remaining
)
SELECT toString(a.audio_id) AS sample_id, batch_idx, sample_idx,
    toJSONString(CAST((toFloat64(a.duration), toNullable(a.language), a.speaker_id, a.text, sample_idx),
        'Tuple(duration Float64, language Nullable(String), speaker_id Nullable(String), text String, position UInt64)')) AS metadata_json,
    concat('{"audio":{"object":', toJSONString(b.path),
        ',"byte_offset":', toString(a.byte_offset),
        ',"byte_length":', toString(a.byte_length), '}}') AS blobs_json
FROM batched AS a
ALL INNER JOIN bucket_files AS b ON b.id = a.bucket_file_id
WHERE batch_idx < batch_count
ORDER BY batch_idx, sample_idx
SETTINGS function_json_value_return_type_allow_complex = 1,
    enable_materialized_cte = 1, output_format_json_named_tuples_as_objects = 1
