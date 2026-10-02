schema "default" {
  engine = sql("Atomic")
}

table "projects" {
  schema = schema.default
  engine = sql("ReplacingMergeTree(updated_at)")

  column "id" {
    type = UUID
  }
  column "name" {
    type = String
  }
  column "description" {
    type = String
  }
  column "created_at" {
    type = DateTime64(6)
  }
  column "updated_at" {
    type = DateTime64(6)
  }

  primary_key {
    columns = [column.id]
  }
  sort {
    columns = [column.id]
  }
}

table "runs" {
  schema = schema.default
  engine = MergeTree

  column "id" {
    type = UUID
  }
  column "project_id" {
    type = UUID
  }
  column "name" {
    type = String
  }
  column "config" {
    type = String
  }

  primary_key {
    columns = [column.project_id, column.id]
  }
  sort {
    columns = [column.project_id, column.id]
  }
}

table "run_status" {
  schema = schema.default
  engine = MergeTree

  column "timestamp" {
    type = DateTime64(9)
  }
  column "run_id" {
    type = UUID
  }
  column "status" {
    type = sql("Enum8('running' = 1, 'succeeded' = 2, 'failed' = 3, 'cancelled' = 4, 'queued' = 5)")
  }

  sort {
    columns = [column.run_id, column.timestamp]
  }
  primary_key {
    columns = [column.run_id, column.timestamp]
  }
}

table "metrics" {
  schema = schema.default
  engine = MergeTree

  column "timestamp" {
    type = DateTime64(9)
  }
  column "run_id" {
    type = UUID
  }
  column "step" {
    type = UInt64
  }
  column "name" {
    type = sql("LowCardinality(String)")
  }
  column "value" {
    type = Float32
  }

  partition {
    on {
      expr = "toYYYYMM(timestamp)"
    }
  }
  primary_key {
    columns = [column.run_id, column.name, column.step, column.timestamp]
  }
  sort {
    columns = [column.run_id, column.name, column.step, column.timestamp]
  }

}

table "array_metrics" {
  schema = schema.default
  engine = MergeTree

  column "timestamp" {
    type = DateTime64(9)
  }
  column "run_id" {
    type = UUID
  }
  column "step" {
    type = UInt64
  }
  column "name" {
    type = sql("LowCardinality(String)")
  }
  column "value" {
    type = sql("Array(Float32)")
  }

  partition {
    on {
      expr = "toYYYYMM(timestamp)"
    }
  }
  primary_key {
    columns = [column.run_id, column.name, column.step, column.timestamp]
  }
  sort {
    columns = [column.run_id, column.name, column.step, column.timestamp]
  }
}

table "logs" {
  schema = schema.default
  engine = MergeTree

  column "run_id" {
    type = UUID
  }
  column "timestamp" {
    type = DateTime64(9)
  }
  column "message" {
    type = String
  }

  partition {
    on {
      expr = "toYYYYMM(timestamp)"
    }
  }
  primary_key {
    columns = [column.run_id, column.timestamp]
  }
  sort {
    columns = [column.run_id, column.timestamp]
  }
}

table "artifacts" {
  schema = schema.default
  engine = MergeTree

  column "id" {
    type = UUID
  }
  column "run_id" {
    type = UUID
  }
  column "step" {
    type = UInt64
  }
  column "timestamp" {
    type = DateTime64(9)
  }
  column "name" {
    type = String
  }
  column "path" {
    type = String
  }
  column "content_type" {
    type = sql("LowCardinality(String)")
  }
  column "size_bytes" {
    type = UInt64
  }

  primary_key {
    columns = [column.id]
  }
  sort {
    columns = [column.id]
  }
}

table "assets" {
  schema = schema.default
  engine = sql("ReplacingMergeTree(updated_at)")

  column "id" {
    type = UUID
  }
  column "updated_at" {
    type = DateTime64(6)
  }
  column "kind" {
    type = sql("Enum8('checkpoint' = 1, 'file' = 2)")
  }
  column "name" {
    type = String
  }
  column "step" {
    type    = UInt64
    default = 0
  }
  column "path" {
    type = String
  }
  column "size" {
    type = UInt64
  }
  column "content_hash" {
    type = FixedString(64)
  }
  column "type" {
    type = sql("LowCardinality(String)")
  }
  column "metadata" {
    type = String
  }

  column "run_id" {
    type    = UUID
    default = "00000000-0000-0000-0000-000000000000"
  }

  column "ancestor_asset_id" {
    type    = UUID
    default = "00000000-0000-0000-0000-000000000000"
  }

  column "deleted" {
    type = Bool
  }

  primary_key {
    columns = [column.id]
  }
  sort {
    columns = [column.id]
  }
}
