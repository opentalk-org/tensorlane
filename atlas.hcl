locals {
  schema_dir    = "file://db/schema"
  migration_dir = "file://db/migrations"
}

env "local" {
  src     = local.schema_dir
  exclude = ["default.audio_files", "default.audio_segments", "default.audio_waveforms", "default.dataset_audio_files", "default.datasets", "default.bucket_files", "default.configs", "default.statistics_entries", "default.mos_comparisons", "default.example_samples"]
  diff {
    skip {
      drop_column = true
      drop_table  = true
    }
  }
  url = getenv("CLICKHOUSE_URL")
  dev = getenv("CLICKHOUSE_DEV_URL")

  migration {
    dir = local.migration_dir
  }
}

env "migration" {
  src     = local.schema_dir
  exclude = ["default.audio_files", "default.audio_segments", "default.audio_waveforms", "default.dataset_audio_files", "default.datasets", "default.bucket_files", "default.configs", "default.statistics_entries", "default.mos_comparisons", "default.example_samples"]
  diff {
    skip {
      drop_column = true
      drop_table  = true
    }
  }
  dev = getenv("CLICKHOUSE_DEV_URL")

  migration {
    dir = local.migration_dir
  }
}

env "prod" {
  url = getenv("CLICKHOUSE_URL")

  migration {
    dir = local.migration_dir
  }
}
