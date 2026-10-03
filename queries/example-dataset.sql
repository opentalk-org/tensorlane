-- Project used by the bundled run configurations.
INSERT INTO projects (id, name, description, created_at, updated_at)
SELECT toUUID('f3b83939-af56-473a-8ad8-b77b12bfef37'),
    'TensorLane examples', 'Bundled example runs', now64(6), now64(6)
WHERE NOT EXISTS (
    SELECT 1 FROM projects WHERE id = toUUID('f3b83939-af56-473a-8ad8-b77b12bfef37')
);

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
