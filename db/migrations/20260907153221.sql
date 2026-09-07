-- Create "array_metrics" table
CREATE TABLE `array_metrics` (
  `timestamp` DateTime64(9),
  `run_id` UUID,
  `step` UInt64,
  `name` LowCardinality(String),
  `value` Array(Float32)
) ENGINE = MergeTree
PRIMARY KEY (`run_id`, `name`, `step`, `timestamp`) ORDER BY (`run_id`, `name`, `step`, `timestamp`) PARTITION BY (toYYYYMM(timestamp)) SETTINGS index_granularity = 8192;
-- Create "artifacts" table
CREATE TABLE `artifacts` (
  `id` UUID,
  `run_id` UUID,
  `step` UInt64,
  `timestamp` DateTime64(9),
  `name` String,
  `path` String,
  `content_type` LowCardinality(String),
  `size_bytes` UInt64
) ENGINE = MergeTree
PRIMARY KEY (`id`) ORDER BY (`id`) SETTINGS index_granularity = 8192;
-- Create "assets" table
CREATE TABLE `assets` (
  `id` UUID,
  `updated_at` DateTime64(6),
  `kind` Enum8('checkpoint' = 1, 'file' = 2),
  `name` String,
  `step` UInt64 DEFAULT 0,
  `path` String,
  `size` UInt64,
  `content_hash` FixedString(64),
  `type` LowCardinality(String),
  `metadata` String,
  `run_id` UUID DEFAULT '00000000-0000-0000-0000-000000000000',
  `ancestor_asset_id` UUID DEFAULT '00000000-0000-0000-0000-000000000000',
  `deleted` Bool
) ENGINE = ReplacingMergeTree(updated_at)
PRIMARY KEY (`id`) ORDER BY (`id`) SETTINGS index_granularity = 8192;
-- Create "audio_files" table
CREATE TABLE `audio_files` (
  `id` UUID,
  `updated_at` DateTime64(9),
  `name` String,
  `bucket_file_id` UUID DEFAULT '00000000-0000-0000-0000-000000000000',
  `byte_offset` UInt64,
  `duration` Float32,
  `byte_length` UInt64,
  `score` Float32,
  `language` LowCardinality(String),
  `style_prompt` String,
  `voice_prompt` String,
  `virtual` Bool,
  `storage_kind` Enum8('packed' = 1, 'external' = 2),
  `storage_ref` String,
  `metadata` String
) ENGINE = MergeTree
PRIMARY KEY (`id`, `updated_at`) ORDER BY (`id`, `updated_at`) SETTINGS index_granularity = 8192;
-- Create "audio_segments" table
CREATE TABLE `audio_segments` (
  `id` UUID,
  `audio_file_id` UUID,
  `updated_at` DateTime64(9),
  `position` UInt32,
  `start_seconds` Float32,
  `end_seconds` Float32,
  `text` String,
  `phon` String,
  `kind` LowCardinality(String),
  `accuracy` Float32 DEFAULT -1,
  `speaker_id` String,
  `metadata` String,
  `alignment` Array(Tuple(`word` String, `start` Float32, `end` Float32))
) ENGINE = MergeTree
PRIMARY KEY (`audio_file_id`, `id`, `updated_at`) ORDER BY (`audio_file_id`, `id`, `updated_at`) SETTINGS index_granularity = 8192;
-- Create "audio_waveforms" table
CREATE TABLE `audio_waveforms` (
  `audio_file_id` UUID,
  `updated_at` DateTime64(6),
  `pack_id` UUID,
  `byte_offset` UInt64,
  `byte_length` UInt64,
  `duration` Float64,
  `sample_rate` UInt32,
  `points_per_second` UInt32,
  `point_count` UInt64
) ENGINE = ReplacingMergeTree(updated_at)
PRIMARY KEY (`audio_file_id`) ORDER BY (`audio_file_id`) SETTINGS index_granularity = 8192;
-- Create "bucket_files" table
CREATE TABLE `bucket_files` (
  `id` UUID,
  `kind` Enum8('audio' = 1, 'waveform' = 2),
  `path` String,
  `size` UInt64,
  `used_bytes` UInt64
) ENGINE = MergeTree
PRIMARY KEY (`id`) ORDER BY (`id`) SETTINGS index_granularity = 8192;
-- Create "configs" table
CREATE TABLE `configs` (
  `id` UUID,
  `updated_at` DateTime64(6),
  `name` String,
  `type` LowCardinality(String),
  `metadata` JSON
) ENGINE = ReplacingMergeTree(updated_at)
PRIMARY KEY (`id`) ORDER BY (`id`) SETTINGS index_granularity = 8192;
-- Create "dataset_audio_files" table
CREATE TABLE `dataset_audio_files` (
  `dataset_id` UUID,
  `audio_file_id` UUID,
  `updated_at` DateTime64(6)
) ENGINE = ReplacingMergeTree(updated_at)
PRIMARY KEY (`dataset_id`, `audio_file_id`) ORDER BY (`dataset_id`, `audio_file_id`) SETTINGS index_granularity = 8192;
-- Create "datasets" table
CREATE TABLE `datasets` (
  `id` UUID,
  `updated_at` DateTime64(6),
  `name` String
) ENGINE = ReplacingMergeTree(updated_at)
PRIMARY KEY (`id`) ORDER BY (`id`) SETTINGS index_granularity = 8192;
-- Create "logs" table
CREATE TABLE `logs` (
  `run_id` UUID,
  `timestamp` DateTime64(9),
  `message` String
) ENGINE = MergeTree
PRIMARY KEY (`run_id`, `timestamp`) ORDER BY (`run_id`, `timestamp`) PARTITION BY (toYYYYMM(timestamp)) SETTINGS index_granularity = 8192;
-- Create "metrics" table
CREATE TABLE `metrics` (
  `timestamp` DateTime64(9),
  `run_id` UUID,
  `step` UInt64,
  `name` LowCardinality(String),
  `value` Float32
) ENGINE = MergeTree
PRIMARY KEY (`run_id`, `name`, `step`, `timestamp`) ORDER BY (`run_id`, `name`, `step`, `timestamp`) PARTITION BY (toYYYYMM(timestamp)) SETTINGS index_granularity = 8192;
-- Create "mos_comparisons" table
CREATE TABLE `mos_comparisons` (
  `id` UUID,
  `updated_at` DateTime64(6),
  `audio_a_id` UUID,
  `audio_b_id` UUID,
  `preferred_audio_id` UUID,
  `score_a` Float32,
  `score_b` Float32,
  `created_at` DateTime64(6)
) ENGINE = ReplacingMergeTree(updated_at)
PRIMARY KEY (`id`) ORDER BY (`id`) SETTINGS index_granularity = 8192;
-- Create "projects" table
CREATE TABLE `projects` (
  `id` UUID,
  `name` String,
  `description` String,
  `created_at` DateTime64(6),
  `updated_at` DateTime64(6)
) ENGINE = ReplacingMergeTree(updated_at)
PRIMARY KEY (`id`) ORDER BY (`id`) SETTINGS index_granularity = 8192;
-- Create "run_status" table
CREATE TABLE `run_status` (
  `timestamp` DateTime64(9),
  `run_id` UUID,
  `status` Enum8('running' = 1, 'succeeded' = 2, 'failed' = 3, 'cancelled' = 4, 'queued' = 5)
) ENGINE = MergeTree
PRIMARY KEY (`run_id`, `timestamp`) ORDER BY (`run_id`, `timestamp`) SETTINGS index_granularity = 8192;
-- Create "runs" table
CREATE TABLE `runs` (
  `id` UUID,
  `project_id` UUID,
  `name` String,
  `data_config` String,
  `train_config` String
) ENGINE = MergeTree
PRIMARY KEY (`project_id`, `id`) ORDER BY (`project_id`, `id`) SETTINGS index_granularity = 8192;
-- Create "statistics_entries" table
CREATE TABLE `statistics_entries` (
  `id` UUID,
  `updated_at` DateTime64(6),
  `name` String,
  `dataset_id` UUID DEFAULT '00000000-0000-0000-0000-000000000000',
  `payload` JSON,
  `metadata` JSON,
  `created_at` DateTime64(6)
) ENGINE = ReplacingMergeTree(updated_at)
PRIMARY KEY (`id`) ORDER BY (`id`) SETTINGS index_granularity = 8192;
