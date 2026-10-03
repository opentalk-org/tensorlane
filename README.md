# TensorLane

Training data from ClickHouse and S3, delivered as Python batches.

SQL selects samples, orders them, and assigns batches. A Rust server fetches the referenced objects; Python workers transform the samples and collate them for your training loop. Dataset tables, decoding, and the model belong to your application.

## Setup

Requires Linux, Python 3.11+, ClickHouse, and an S3-compatible bucket. Put the connection settings in `.env` at the repository root:

```sh
CLICKHOUSE_URL=http://localhost:8123
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=''
AWS_ENDPOINT_URL=http://localhost:9001
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
S3_BUCKET=tensorlane
```

The Nix development shell loads `.env` and provides the build tools. Build the Python client:

```sh
nix develop
uv sync --group test
uv run --group test maturin develop
```

Apply the [database schema](db/README.md), then start the server in another terminal:

```sh
nix develop -c server
```

HTTP listens on `8180`; gRPC listens on `8181`. The development environment puts the server cache in `.tensorlane/cache`. Set `HTTP_PORT`, `GRPC_PORT`, or `CACHE_DIR` to override these defaults. Keep the cache on disk-backed storage.

## Read batches

For the bundled example, execute [example-dataset.sql](queries/example-dataset.sql) once in the configured ClickHouse database. It creates 2,000 samples with metadata and no S3 blobs. Create a run:

```sh
curl --fail-with-body http://localhost:8180/runs \
  -H 'Content-Type: application/json' \
  --data-binary @queries/examples/sample-configs.json
```

Set `TENSORLANE_RUN_ID` to the returned `run_id`. Save this as `read.py` and run it with `uv run python read.py` inside the development shell:

```python
import torch
import tensorlane


def transform(sample):
    return torch.tensor(sample.metadata["position"], dtype=torch.float32)


def collate(samples):
    return torch.stack(samples)


def main():
    with tensorlane.init(transform=transform, collate_fn=collate) as lane:
        with lane.batches("training") as batches:
            for batch in batches:
                print(batch.batch_id, batch.data)


if __name__ == "__main__":
    main()
```

Transforms receive `RawSample(sample_id, stream, metadata, blobs)`, where `blobs` maps names to bytes. Collators receive the ordered transformed samples. Define both callbacks at module scope for multiprocessing; worker tensors must be on CPU. Without callbacks, `batch.samples` contains the raw samples and `batch.data` is the same tuple.

`lane.config` holds the complete submitted configuration. Defaults are five workers, prefetch factor two, and one rank; set `num_workers`, `prefetch_factor`, and `ranks` in the run configuration or override them in `init`. Set `TENSORLANE_ADDR` for a server other than `localhost:8181`.

For multiple ranks, one local process starts the daemon and the others attach. Batches are assigned round robin within each stream. The [Accelerate example](client/examples/train.py) shows rank setup, training, and synchronization before close.

## Write a query

Each entry in `config.queries` defines a named stream. SQL must return:

| Column | Type | Contents |
| --- | --- | --- |
| `sample_id` | `String` | Sample identifier; duplicates are allowed |
| `batch_idx` | `UInt64` | Batch grouping; gaps are allowed |
| `sample_idx` | `UInt64` | Order within the batch |
| `metadata_json` | `String` | JSON object passed to the transform |
| `blobs_json` | `String` | JSON object mapping blob names to S3 references |

Rows must be strictly ordered by `(batch_idx, sample_idx)`. A blob reference is `{"object":"path/in/bucket"}` or `{"object":"packed/file","byte_offset":64,"byte_length":32}`. Both range fields are required together. Empty blob maps are valid.

SQL parameters use ClickHouse placeholders such as `{batch_size:UInt64}`. Values come from `dataset_id` and `seed`, then `config.params`, then the stream's own configuration object, with later values taking precedence. Application settings such as optimizer parameters are stored alongside them.

Each query runs once during initialization. TensorLane validates and writes the complete result to disk before returning from `init`. Training ends at EOF; validation repeats by default. Set `repeat` per stream to change this. Repetition reads the same plan file; blobs are fetched as needed, so use immutable objects when their contents must stay identical. If supplied, `batches` must match the query's distinct batch count.

## Storage and limits

Each stream has one plan file and at most 20 reusable batch-cache files. The same files are reused as training progresses. Plans are read one batch at a time; their disk usage grows with the query result.

| Limit | Value |
| --- | --- |
| Encoded batch | 64 MiB |
| Batch descriptors | 64 MiB / 65,536 samples |
| Concurrent S3 reads | 16 across the server |
| Client batch credits | `ranks × prefetch_factor` per stream |

These limits do not cap total RAM: active blob reads, decoded tensors, and ClickHouse query execution have separate memory costs. Run caches are removed on close. A client missing heartbeats for 60 seconds is marked failed and its cache is removed. Server restart marks previously running runs failed. Use one server per run database.

## Checkpoints and metrics

Declare input assets in `config.assets` by registered `asset_id` or S3 `object`. They are downloaded before initialization completes; `lane.asset("model")` returns the local path. TAR archives are extracted automatically.

```python
asset_id = lane.save_asset("model", "weights.pt", step=1000, kind="checkpoint")
lane.flush()
```

Saving returns an ID immediately and uploads in the background. Keep the source unchanged until `flush()` completes. Successful saves record their predecessor, preserving checkpoint history. Files upload as-is; directories become TAR archives.

A continuation is a new run: load the saved asset and pass the next sample position after completed training work. Prefetched batches do not count as completed work. See [train2.py](client/examples/train2.py) and its [run configuration](queries/examples/sample-configs-stage2.json).

Use `lane.metric(step, name, value)` for scalars and `lane.metric_artifact(...)` for files. Throughput and pipeline timings are reported every 10 seconds by default; disable them with `performance_metrics=False`. Run configuration, status, and asset history are available through [the HTTP API](server/src/http.rs); data transfer uses [gRPC](proto/tensorlane.proto).

## Development

```sh
nix develop -c cargo test --workspace --lib --bins
nix develop -c uv run --group test python -m unittest discover -s client/tests -p 'test_*.py'
nix develop -c cargo test -p tensorlane --test e2e -- --test-threads=1
```

Integration tests start ClickHouse and MinIO through Docker. Set `TENSORLANE_TEST_PYTHON` to the prepared environment's Python executable to include the training examples.

[Load benchmark](client/benchmarks/load_test.py) · [Audio sampling SQL](queries/training.sql) · [Audio run configuration](sample-configs.json) · [Schema and migrations](db/README.md)
