# TensorLane

TensorLane is a data loader for training on datasets stored in S3 and indexed in ClickHouse. You define the sampling and batching in SQL, then iterate over the batches in Python.

The [audio example](queries/training.sql) uses recording durations and transcripts to select samples and assign batches. Audio is read from byte ranges in S3 objects, so changing the sampling rules doesn't require repacking the dataset.

## Defining batches

A run has a query for each stream, such as `training` or `validation`. The queries can read any of your ClickHouse tables. Each result row describes one sample:

| Column | ClickHouse type | Meaning |
| --- | --- | --- |
| `sample_id` | `String` | Sample identifier |
| `batch_idx` | `UInt64` | Samples with the same value form a batch |
| `sample_idx` | `UInt64` | Sample order within that batch |
| `metadata_json` | `String` | JSON object passed to Python |
| `blobs_json` | `String` | Named references to files in S3 |

Return rows in strictly increasing `(batch_idx, sample_idx)` order. Batch sizes can vary; sample IDs can repeat. For example, this `blobs_json` requests part of an object:

```json
{"audio": {"object": "recordings/part-001", "byte_offset": 64, "byte_length": 32000}}
```

Omit both range fields to fetch the whole object. The Python sample will contain its bytes at `sample.blobs["audio"]`. Use `{}` for samples that only need metadata.

Put queries in `config.queries`. Parameters use ClickHouse syntax such as `{batch_size:UInt64}`. Values come from `config.dataset_id` and `config.seed`, then `config.params`, with per-stream settings taking precedence. See the [example run configuration](queries/examples/sample-configs.json).

## Reading in Python

The Rust server fetches the objects referenced by the query. Python workers call your `transform` on each sample, then `collate_fn` assembles the transformed samples into `batch.data`.

This reader works with the bundled dataset, which contains numeric metadata:

```python
import tensorlane
import torch


def transform(sample):
    return torch.tensor(sample.metadata["position"], dtype=torch.float32)


def main():
    with tensorlane.init(transform=transform, collate_fn=torch.stack) as lane:
        with lane.batches("training") as batches:
            for batch in batches:
                print(batch.batch_id, batch.data)


if __name__ == "__main__":
    main()
```

For files, decode `sample.blobs` inside `transform`. Define callbacks at module scope and return CPU tensors; workers use multiprocessing. Without callbacks, `batch.data` is a tuple of raw samples.

`init` reads `TENSORLANE_RUN_ID` and connects to `TENSORLANE_ADDR` (default `localhost:8181`). Its defaults are five workers and two prefetched batches per rank, per stream. Override them with `num_workers` and `prefetch_factor`.

The [training example](client/examples/train.py) shows a complete PyTorch loop with Accelerate, including batch distribution across local ranks and checkpoint uploads.

## Running locally

Requires Linux, Python 3.11+, ClickHouse and an S3-compatible bucket. In the repository root, create `.env` with your connection settings:

```sh
CLICKHOUSE_URL=http://localhost:8123
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=''
AWS_ENDPOINT_URL=http://localhost:9001
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
S3_BUCKET=tensorlane
```

Build the Python client in the Nix development shell, which loads `.env`:

```sh
nix develop
uv sync --group test
uv run --group test maturin develop
```

Apply the [database schema](db/README.md) and execute [example-dataset.sql](queries/example-dataset.sql) in ClickHouse. It creates 2,000 samples without S3 data. Start the server in another terminal:

```sh
nix develop -c server
```

Create a run over the example dataset:

```sh
curl --fail-with-body http://localhost:8180/runs \
  -H 'Content-Type: application/json' \
  --data-binary @queries/examples/sample-configs.json
```

Export the returned `run_id` as `TENSORLANE_RUN_ID`. Save the Python reader above as `read.py`, then run `uv run python read.py` in the development shell.

## Run behavior

Initialization executes each query once and saves the complete result to disk before returning. Training ends at the last batch; validation repeats by default. Set `repeat` per stream to change this. Repeating reuses the saved result, while S3 objects are fetched as needed.

The server reads each saved result one batch at a time. Each stream uses one query-result file and at most 20 reusable batch files under `CACHE_DIR` (`.tensorlane/cache` in the development shell). Keep this directory on disk with room for the full query results. Encoded batches are limited to 64 MiB; decoded data and worker processes also consume RAM. ClickHouse query memory is separate.

Closing a run removes its cache. A missing client heartbeat fails the run after 60 seconds. Server restarts fail previous active runs; run one server per database.

Save checkpoints with `lane.save_asset("model", path, kind="checkpoint")`, then call `lane.flush()` before modifying the source file. To load one in a new run, add it to `config.assets` and read its local path with `lane.asset("model")`. The [resume example](client/examples/train2.py) restores weights and continues from the number of samples actually trained on.

Use `lane.metric(step, name, value)` to record scalars and `lane.metric_artifact(...)` for files. Throughput and timing metrics are enabled by default. The full run configuration is available as `lane.config`.

## Development

```sh
nix develop -c cargo test --workspace --lib --bins
nix develop -c uv run --group test python -m unittest discover -s client/tests -p 'test_*.py'
nix develop -c cargo test -p tensorlane --test e2e -- --test-threads=1
```

Integration tests require Docker for ClickHouse and MinIO. Set `TENSORLANE_TEST_PYTHON` to the prepared Python executable to include the training examples.

[Load benchmark](client/benchmarks/load_test.py) · [HTTP routes](server/src/http.rs) · [gRPC protocol](proto/tensorlane.proto)
