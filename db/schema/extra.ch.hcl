table "audio_waveforms" {
  schema = schema.default
  engine = sql("ReplacingMergeTree(updated_at)")

  column "audio_file_id" {
    type = UUID
  }
  column "updated_at" {
    type = DateTime64(6)
  }
  column "pack_id" {
    type = UUID
  }
  column "byte_offset" {
    type = UInt64
  }
  column "byte_length" {
    type = UInt64
  }
  column "duration" {
    type = Float64
  }
  column "sample_rate" {
    type = UInt32
  }
  column "points_per_second" {
    type = UInt32
  }
  column "point_count" {
    type = UInt64
  }

  primary_key {
    columns = [column.audio_file_id]
  }
  sort {
    columns = [column.audio_file_id]
  }
}

table "statistics_entries" {
  schema = schema.default
  engine = sql("ReplacingMergeTree(updated_at)")

  column "id" {
    type = UUID
  }
  column "updated_at" {
    type = DateTime64(6)
  }
  column "name" {
    type = String
  }
  column "dataset_id" {
    type = UUID
    default = "00000000-0000-0000-0000-000000000000"
  }
  column "payload" {
    type = sql("JSON")
  }
  column "metadata" {
    type = sql("JSON")
  }
  column "created_at" {
    type = DateTime64(6)
  }

  primary_key {
    columns = [column.id]
  }
  sort {
    columns = [column.id]
  }
}

table "configs" {
  schema = schema.default
  engine = sql("ReplacingMergeTree(updated_at)")

  column "id" {
    type = UUID
  }
  column "updated_at" {
    type = DateTime64(6)
  }
  column "name" {
    type = String
  }
  column "type" {
    type = sql("LowCardinality(String)")
  }
  column "metadata" {
    type = sql("JSON")
  }

  primary_key {
    columns = [column.id]
  }
  sort {
    columns = [column.id]
  }
}

table "mos_comparisons" {
  schema = schema.default
  engine = sql("ReplacingMergeTree(updated_at)")

  column "id" {
    type = UUID
  }
  column "updated_at" {
    type = DateTime64(6)
  }
  column "audio_a_id" {
    type = UUID
  }
  column "audio_b_id" {
    type = UUID
  }
  column "preferred_audio_id" {
    type = UUID
  }
  column "score_a" {
    type = Float32
  }
  column "score_b" {
    type = Float32
  }
  column "created_at" {
    type = DateTime64(6)
  }

  primary_key {
    columns = [column.id]
  }
  sort {
    columns = [column.id]
  }
}
