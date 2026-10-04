CREATE TABLE `default`.`run_sessions` (
  `run_id` UUID,
  `session_id` UUID,
  `updated_at` DateTime64(9)
) ENGINE = ReplacingMergeTree(updated_at)
ORDER BY (`run_id`)
PRIMARY KEY (`run_id`);
ALTER TABLE `default`.`metrics` MODIFY SETTING non_replicated_deduplication_window = 1000000;
ALTER TABLE `default`.`array_metrics` MODIFY SETTING non_replicated_deduplication_window = 1000000;
ALTER TABLE `default`.`artifacts` MODIFY SETTING non_replicated_deduplication_window = 1000000;
