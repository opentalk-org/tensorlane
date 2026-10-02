ALTER TABLE runs UPDATE config = concat('{"data_config":', data_config, ',"train_config":', train_config, '}') WHERE config = '' SETTINGS mutations_sync = 2;
