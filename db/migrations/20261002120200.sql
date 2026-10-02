SELECT throwIf(countIf(config = '') != 0, 'run config backfill is incomplete') FROM runs;
ALTER TABLE runs DROP COLUMN data_config;
ALTER TABLE runs DROP COLUMN train_config;
