# TensorLane

TensorLane is a data loader for training on datasets stored in S3 and indexed in ClickHouse. You define the sampling and batching in SQL, then iterate over the batches in Python.

The [example query](queries/training.sql) shuffles samples and assigns batches using generic metadata and blob references. TensorLane transfers blob bytes unchanged; decoding belongs in your Python transform.

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
{"payload": {"object": "packs/part-001", "byte_offset": 64, "byte_length": 32000}}
```

Omit both range fields to fetch the whole object. The Python sample will contain its bytes at `sample.blobs["payload"]`. Use `{}` for samples that only need metadata.

Run configuration has three separate sections:

- `tensorlane`: runtime settings (`ranks`, `num_workers`, `prefetch_factor`) and asset settings (`assets`, `asset_type`).
- `queries`: a list of query objects, each containing `key`, `sql`, `params`, and an optional `repeat` flag.
- `app`: custom application configuration. TensorLane stores it without interpreting its contents.

```json
{
  "tensorlane": {"num_workers": 5, "prefetch_factor": 2},
  "queries": [
    {
      "key": "training",
      "sql": "...",
      "params": {"batches": 20, "batch_size": 32},
      "repeat": false
    },
    {
      "key": "validation",
      "sql": "...",
      "params": {"samples": 1000, "batch_size": 32},
      "repeat": true
    }
  ],
  "app": {"optimizer": {"learning_rate": 0.0001}}
}
```

Parameters use ClickHouse syntax such as `{batch_size:UInt64}` and come only from that query's `params` object. Put inputs such as `dataset_id` and `seed` there too. SQL determines batch membership and count; `batches`, `samples`, and `batch_size` are ordinary SQL parameters with no built-in TensorLane meaning. See the executable [example run configuration](queries/examples/sample-configs.json).

Query keys must be unique and nonempty. Every stream ends when its query result ends unless `repeat` is explicitly `true`; repeating replays the saved query result. Keys such as `training` and `validation` have no special behavior.

This schema replaces the former query-name-to-SQL object and top-level stream settings. Existing stored configurations remain readable, but must be converted to this schema before initializing them with the updated server. Update server and client together; this change does not rewrite stored runs.

## Reading in Python

Pass a `transform` function to decode each sample and a `collate_fn` to combine the samples into `batch.data`.

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

For files, decode `sample.blobs` inside `transform`. Define callbacks at module scope and return CPU tensors. Without callbacks, `batch.data` is a tuple of raw samples.

Automatic performance metrics are reported by rank 0 under `tensorlane/<stream>/`: samples per second and mean server-load, transform, collation, data-wait, and application times. These describe rank 0's batches. Empty measurement windows are skipped; error counts are emitted when they increase. Set `performance_metrics=False` in `tensorlane.init` to disable automatic reporting.

Set `TENSORLANE_RUN_ID` to your run ID and `TENSORLANE_ADDR` to the server address (default `localhost:8181`).

The [training example](client/examples/train.py) shows a complete PyTorch loop with Accelerate and checkpoint uploads.

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

## Checkpoints and metrics

Save checkpoints with `lane.save_asset("model", path, kind="checkpoint")`, then call `lane.flush()` before modifying the source file. To load one in a new run, add it to `config.tensorlane.assets` and read its local path with `lane.asset("model")`. The [resume example](client/examples/train2.py) restores weights and continues from the number of samples actually trained on.

Use `lane.metric(step, name, value)` to record scalars and `lane.metric_artifact(...)` for files. Throughput and timing metrics are enabled by default. The full run configuration is available as `lane.config`; application settings are in `lane.config["app"]`. Runtime defaults are one rank, five workers, and two prefetched batches per rank and stream. Explicit `init` arguments override `config.tensorlane` settings. Always select a stream by name with `lane.batches("training")`.

## Development

```sh
nix develop -c cargo test --workspace --lib --bins
nix develop -c uv run --group test python -m unittest discover -s client/tests -p 'test_*.py'
nix develop -c cargo test -p tensorlane --test e2e -- --test-threads=1
```

Integration tests require Docker for ClickHouse and MinIO. Set `TENSORLANE_TEST_PYTHON` to the prepared Python executable to include the training examples.

[Example run configuration](sample-configs.json) · [Resume configuration](queries/examples/sample-configs-stage2.json) · [Load benchmark](client/benchmarks/load_test.py)
