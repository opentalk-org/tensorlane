CREATE TABLE IF NOT EXISTS example_samples (
    dataset_id UUID,
    position UInt64,
    sample_id String,
    metadata_json String,
    blobs_json String
) ENGINE = MergeTree ORDER BY (dataset_id, position);

INSERT INTO example_samples
SELECT toUUID('e25b39ac-3400-4f9f-9fac-3e9c94e1a92b'),
    number, concat('sample-', toString(number)),
    toJSONString(map('position', number)), '{}'
FROM numbers(2000);
