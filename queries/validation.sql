
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
    toUInt64(floor(sum(toDecimal128(toString(a.duration), 9)) OVER (
        ORDER BY a.duration, a.audio_id ROWS UNBOUNDED PRECEDING
    ) / toDecimal128(toString({max_duration:Float64}), 9))) AS batch_idx,
    toUInt64(row_number() OVER (ORDER BY a.duration, a.audio_id) - 1) AS sample_idx,
    b.path AS object_path,
    toInt64(a.byte_offset) AS byte_offset, toInt64(a.byte_length) AS byte_length
FROM agg AS a
ALL INNER JOIN bucket_files AS b ON b.id = a.bucket_file_id
WHERE lengthUTF8(a.text) <= {max_text:UInt64}
ORDER BY batch_idx, sample_idx
SETTINGS function_json_value_return_type_allow_complex = 1
