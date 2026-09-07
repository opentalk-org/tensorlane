locals {
  schema_dir    = "file://db/schema"
  migration_dir = "file://db/migrations"
}

env "local" {
  src = local.schema_dir
  url = getenv("CLICKHOUSE_URL")
  dev = getenv("CLICKHOUSE_DEV_URL")

  migration {
    dir = local.migration_dir
  }
}

env "migration" {
  src = local.schema_dir
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
