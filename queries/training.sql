WITH ordered AS (
    SELECT sample_id, metadata_json, blobs_json,
        toUInt64(row_number() OVER (ORDER BY cityHash64({seed:UInt64}, sample_id), sample_id) - 1) AS position
    FROM example_samples
    WHERE dataset_id = {dataset_id:UUID}
)
SELECT sample_id,
    toUInt64(intDiv(position - {dataset_offset:UInt64}, {batch_size:UInt64})) AS batch_idx,
    position AS sample_idx,
    metadata_json, blobs_json
FROM ordered
WHERE position >= {dataset_offset:UInt64}
  AND position < {dataset_offset:UInt64} + {batches:UInt64} * {batch_size:UInt64}
ORDER BY batch_idx, sample_idx
