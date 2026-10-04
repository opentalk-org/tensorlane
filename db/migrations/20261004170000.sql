ALTER TABLE `array_metrics` MODIFY SETTING index_granularity = 8192, non_replicated_deduplication_window = 1000000;
ALTER TABLE `artifacts` MODIFY SETTING index_granularity = 8192, non_replicated_deduplication_window = 1000000;
ALTER TABLE `metrics` MODIFY SETTING index_granularity = 8192, non_replicated_deduplication_window = 1000000;
CREATE TABLE `run_sessions` (
 `run_id` UUID,
 `session_id` UUID,
 `updated_at` DateTime64(9)
) ENGINE = ReplacingMergeTree(updated_at)
PRIMARY KEY (`run_id`) ORDER BY (`run_id`) SETTINGS index_granularity = 8192;
