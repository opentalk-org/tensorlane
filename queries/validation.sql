SELECT sample_id,
    toUInt64(intDiv(position, {batch_size:UInt64})) AS batch_idx,
    position AS sample_idx,
    metadata_json, blobs_json
FROM (
    SELECT sample_id, metadata_json, blobs_json,
        toUInt64(row_number() OVER (ORDER BY sample_id) - 1) AS position
    FROM example_samples
    WHERE dataset_id = {dataset_id:UUID}
    ORDER BY sample_id
    LIMIT {samples:UInt64}
)
ORDER BY batch_idx, sample_idx
