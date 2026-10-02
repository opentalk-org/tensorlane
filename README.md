# TensorLane

TensorLane executes fixed ClickHouse query plans and streams generic samples from S3 through a Rust daemon, Python workers, a collator, and local ranks. Each run has one complete JSON configuration and one run ID. Dataset schemas and decoding belong to the application.

Create a run with `POST /runs` using [sample-configs.json](sample-configs.json), then initialize it with the returned `run_id`. The submitted `config` is stored and returned in full, including optimizer settings and arbitrary application fields. TensorLane parses its own settings separately. `GET /runs` and `GET /runs/{run_id}` expose the same configuration. Applications choose their training framework and launcher; the examples use Accelerate.

```python
from accelerate import Accelerator
import tensorlane


def main():
    accelerator = Accelerator()
    try:
        with tensorlane.init(
            ranks=accelerator.num_processes,
            rank=accelerator.local_process_index,
            start_daemon=accelerator.is_local_main_process,
        ) as lane:
            print(lane.config)
            with lane.batches(stream="training") as batches:
                for batch in batches:
                    for sample in batch.samples:
                        print(sample.sample_id, sample.metadata, sample.blobs)
            lane.flush()
            accelerator.wait_for_everyone()
    finally:
        accelerator.end_training()


if __name__ == "__main__":
    main()
```

Set `num_workers` and `prefetch_factor` directly inside the run JSON's `config`. They default to five workers and a prefetch factor of two. The optional `config.ranks` supplies the data delivery rank count when the application does not pass `ranks`; its default is one. Accelerate supplies the rank count in the examples. `lane.rank`, `lane.ranks`, `lane.num_workers`, and `lane.prefetch_factor` expose the resolved values. `lane.config` exposes the complete application configuration on every rank.

Pass `run_id` to `tensorlane.init` or set `TENSORLANE_RUN_ID`. The gRPC connection defaults to `localhost:8181` and can be changed with `addr` or `TENSORLANE_ADDR`. IPC directories and startup timeouts remain Python arguments. Transforms and collators must be importable for multiprocessing.

`tensorlane.init(run_id, transform=..., collate_fn=...)` reads worker and prefetch counts from the run JSON and infers the rank from `RANK` when omitted. Explicit `ranks`, `num_workers`, and `prefetch_factor` arguments override the JSON. Exactly one local caller starts the daemon; other ranks attach. `rank` and `start_daemon` remain optional overrides. Batches go to ranks round robin independently for each query. All ranks flush uploads and synchronize before closing so the daemon owner closes after rank uploads finish.

## Configuration and queries

The familiar `queries`, `assets`, `dataset_id`, `seed`, `asset_type`, `training`, and `validation` fields remain in a single `config` object. Training uses one parameter object. There are no stage arrays or runtime parameter updates.

Every nonempty name in `queries` defines an independent stream. Its optional same-named configuration object supplies query parameters. Query names cannot be `queries`, `assets`, `params`, `dataset_id`, `seed`, `asset_type`, `ranks`, `num_workers`, or `prefetch_factor`. Parameter binding proceeds through optional `dataset_id` and `seed`, shared `params`, then the query's object. Later values override earlier ones. SQL declares ClickHouse types with placeholders such as `{dataset_offset:UInt64}`; `repeat` is excluded from SQL parameters.

Each query executes once at initialization. Training and other streams are finite by default; validation repeats by default. A query object can override this with `repeat`. Repeating replays the original descriptor plan, even if the source table changes. Blob contents are read when prefetched, so applications that require identical bytes should use immutable objects. If `batches` is supplied, it must equal the number of distinct batches returned by that query. An empty plan terminates even when repeating.

Queries return these columns:

| Column | ClickHouse type | Meaning |
| --- | --- | --- |
| `sample_id` | `String` | Opaque identifier; duplicates are allowed |
| `batch_idx` | `UInt64` | Query-defined grouping; gaps are allowed |
| `sample_idx` | `UInt64` | Ordering within a batch |
| `metadata_json` | `String` | JSON object with application metadata |
| `blobs_json` | `String` | JSON object mapping names to S3 references |

Rows must be strictly ordered by `(batch_idx, sample_idx)`. A blob reference is `{"object":"objects/item"}` or `{"object":"objects/pack","byte_offset":64,"byte_length":32}`. Both range fields are required together; lengths must be positive and ranges must not overflow. Empty blob maps support text, tabular, and other metadata-only samples. Objects and ranges arrive as their original bytes, regardless of file format. Applications decode or unpack them.

Each stream keeps its descriptor plan in memory and up to 20 prefetched batches on disk. At most 16 blob reads run concurrently, with the existing AWS retry policy. Loading failures reach the reader. Encoded batches are limited to 64 MiB, with an explicit error for larger batches; asset uploads stay chunked. Generated local identifiers isolate arbitrary sample, stream, and asset names from filesystem paths. Consumed and partial files are removed, and closing a run removes its cache.

## Transforms and collation

```python
import torch
import tensorlane


def transform(sample):
    return {"features": torch.tensor(sample.metadata["features"]), "payload": sample.blobs}


def collate(samples):
    return torch.stack([sample["features"] for sample in samples])


with tensorlane.init(run_id, transform=transform, collate_fn=collate) as lane:
    with lane.batches("training") as batches:
        for batch in batches:
            print(batch.data)
```

A transform receives `RawSample(sample_id, stream, metadata, blobs)`. Without a transform, samples remain raw. Outputs can contain nested dictionaries, lists, tuples, ordinary values, bytes, and CPU tensors. Tensors are recursively detached and placed in shared memory; CUDA tensors fail in workers. A collator receives the ordered transformed samples. Either callback may be one callable or a mapping by query name; missing entries use the defaults, and unknown names fail initialization.

`Batch` carries `stream`, contiguous `batch_id`, original `query_batch_idx`, transformed `samples`, and collated `data`. Without a collator, `data` is the samples tuple. Worker supervision and independent per-stream batch credits bound buffering and propagate failures to readers.

[sample-configs.json](sample-configs.json) and [sample-configs-stage2.json](sample-configs-stage2.json) embed the duration-based sampling queries in `queries/`. Rebuild either with `queries/build-config.sh CONFIG_FILE`. Continuation uses the completed sample position; keep the seed, dataset, filters, validation exclusions, and `superbatch_seconds` fixed when changing `max_seconds`. Supply validation sample IDs explicitly in `params.validation_ids`.

## Separate runs and explicit continuation

[sample-configs.json](queries/examples/sample-configs.json) consumes 20 batches of 32 samples starting at position zero. [sample-configs-stage2.json](queries/examples/sample-configs-stage2.json) starts at position 640 with batches of 16 and loads the first run's registered model ID. Replace its example asset UUID with the ID returned by `save_asset`.

The example SQL assigns a deterministic sample position before batching. It assumes unique sample IDs within a dataset and stable dataset contents and seed. Changing batch size preserves the meaning of `dataset_offset`. TensorLane never derives continuation offsets from prefetching; the application counts completed training work and records the next position.

[queries/example-dataset.sql](queries/example-dataset.sql) creates and seeds an application-owned metadata-only dataset for the example scripts. Run it once against your dataset ClickHouse database. Submit the first JSON through `POST /runs`, then use its returned run ID:

```sh
TENSORLANE_RUN_ID=RUN_ID_1 nix develop -c uv run --group test accelerate launch client/examples/train.py
```

The script saves model weights, flushes them, and records the committed asset ID and completed dataset offset in `training-output/progress.json`. Put those two values into `config.assets.model.asset_id` and `config.params.dataset_offset` in [sample-configs-stage2.json](queries/examples/sample-configs-stage2.json). Submit that complete JSON through `POST /runs` to create a separate run, then execute:

```sh
TENSORLANE_RUN_ID=RUN_ID_2 nix develop -c uv run --group test accelerate launch client/examples/train2.py
```

Both scripts are self-contained. Stage one starts a new model; stage two loads its model directly from `lane.asset("model")`. Accelerate's launch configuration controls training ranks and devices. The scripts use `Accelerator.prepare`, `backward`, `reduce`, and `unwrap_model`; TensorLane batches already belong to their ranks, so only the model and optimizer go through `prepare`. For one training process, the same scripts work with `python` in place of `accelerate launch`. Each run executes one fixed configuration. Compatibility of SQL ordering and dataset contents is the application's responsibility.

The data-pipeline load benchmark is [client/benchmarks/load_test.py](client/benchmarks/load_test.py).

## Assets and history

Input assets accept exactly one source: `{"asset_id":"registered-uuid"}` or `{"object":"inputs/model","entrypoint":"weights"}`. Registered IDs resolve to the current nondeleted row. Inputs download once before workers start; followers receive the same paths. TAR inputs are automatically extracted. `lane.asset("model")` returns the original file or directory for archives saved by TensorLane, the extraction directory for archives with multiple top-level entries, or the selected relative `entrypoint` when supplied. Raw inputs remain unchanged. Archive links and paths outside the extraction directory are rejected. `lane.asset_metadata["model"]` exposes its registered ID, kind, type, metadata, and optional entrypoint.

```python
saved_id = lane.save_asset(
    "model", saved_model_path, step=1000, kind="checkpoint",
    metadata={"dataset_offset": next_offset},
)
lane.flush()
```

Files upload as their original bytes; only directories use TAR packaging. Both use multipart upload when needed. `kind` defaults to `"file"` and `step` to zero. The type resolves from an explicit `asset_type`, then `config.asset_type`, then the parent asset, then `"generic"`. Saving allocates and returns a UUID immediately. Uploads are queued in the background and do not hold up batch delivery or later saves, including while another thread is flushing. Keep the source unchanged until flush completes. `lane.flush()` and close wait for completion without a transfer deadline. An explicit `lane.flush(timeout=...)` limits only that caller's wait and leaves uploads running. Async applications can use `await lane.flush_async()`, `await lane.close_async()`, and `await tensorlane.init_async(...)` without blocking the event loop. Initialization still returns with all input assets ready; asset downloads run concurrently. Successful flush guarantees both the S3 object and database row exist; failures reach flush.

Commits serialize per `(run_id, name)` and attach the previous committed ID as `ancestor_asset_id`: loading A, saving B, and saving C produces A → B → C. A later run loading C and saving D produces C → D. The first save without a registered input uses the null UUID. Different names have independent chains. Failed saves preserve the head; reinitialization recovers it from persisted rows. Retries of committed IDs return the same ID, while conflicting retries fail. All earlier rows remain available.

`GET /assets/{asset_id}` returns the parent, run, name, step, kind, metadata, type, object path, size, hash, and timestamp. `GET /runs/{run_id}/assets` reads run history, optionally filtered by `?name=model`. Scalar metrics, array metrics, and metric artifacts retain their existing protobuf API and tables; Python's `metric` and `metric_artifact` APIs remain separate from asset saves.

Run completion records `succeeded` on a clean close and `failed` on initialization, worker, upload, or application-context errors. The daemon sends a heartbeat every 10 seconds; a run without a heartbeat for 60 seconds is failed and its server cache is removed. Server shutdown drains live runs and expires abandoned ones. On restart, the server marks previously running runs failed. One server owns the run database; rebuild both server and client for the heartbeat protocol.

## Automatic performance metrics

Performance metrics are enabled by default and written to the existing ClickHouse `metrics` table every 10 seconds, on `lane.flush()`, and on close. No training-loop changes are needed. Pass `performance_metrics=False` to `tensorlane.init` to disable collection for that rank. Automatic telemetry is best effort: upload failures are logged and do not fail the run.

Names use `tensorlane/rank/{rank}/{stream}/{metric}`, with stream names URL-escaped. Metrics include cumulative batches, samples and reader errors; batches and samples per second; seconds since the last batch; mean application and consumption-loop time; mean and maximum data-wait time; and the fraction of measured time spent waiting for data. Timings also cover server sample loading, server cache wait and read, client gRPC response wait, worker transform work summed across the batch, and collation. Each report uses the latest delivered batch ID as its step. Rates describe the reporting window and are per rank; sum matching rank rates for aggregate throughput. Initial windows include reader startup. gRPC wait includes client credit/backpressure delays. Transform work is summed across workers, not elapsed batch latency.

Application time is the interval from returning a batch until the next batch request or reader close. Consumption speed includes application work, validation, pauses, and data waits; it does not count optimizer updates or synchronize asynchronous GPU execution. `tensorlane/rank/{rank}/uploads/flush_seconds` measures how long application upload flushes wait. Automatic metrics use a separate connection so they do not flush application uploads.

Metric artifact files also upload unchanged, with inferred MIME types unless `content_type` is supplied. Directory artifacts are TAR archives and use `application/x-tar`. Existing stored TAR files remain readable through automatic extraction; new uploads do not rewrite old objects.

## Storage and verification

Apply the ordered migrations before running the updated server and rebuild the native Python extension with the updated protobuf. The migrations add `runs.config`, backfill each historical row as `{"data_config":...,"train_config":...}`, verify completion, then retire the old columns. This preserves both legacy sections without collisions. Historical snapshots remain readable and cannot be initialized as new single-stage runs. Asset IDs, history, metrics, and run IDs remain intact. Pause run creation during the backfill and column retirement so legacy writers cannot add unbackfilled rows between those steps. TensorLane's active schema excludes application dataset tables, while historical migrations remain unchanged and their data is retained.

Build and check in the Nix environment:

```sh
nix develop -c uv sync --group test
nix develop -c uv run --group test maturin develop
nix develop -c cargo test --workspace --lib --bins
nix develop -c uv run --group test python -m unittest discover -s client/tests -p 'test_*.py'
nix develop -c cargo test -p tensorlane --test e2e -- --test-threads=1
```

Integration tests use disposable ClickHouse and MinIO containers by default. They can also start isolated local binaries by setting `TENSORLANE_TEST_LOCAL=1`, `TENSORLANE_TEST_CLICKHOUSE=/path/to/clickhouse`, and `TENSORLANE_TEST_MINIO=/path/to/minio` before the Rust integration command. Set `TENSORLANE_TEST_PYTHON` to the absolute path of the prepared virtual environment's Python binary to include both training examples against the real services. Tests cover configuration round trips and migrations, explicit offsets and changed batch sizes, fixed plans, generic bytes and ranges, size limits, worker ordering and rank attachment, asset lineage and retries, durable flush, metrics, multipart uploads, and shutdown cleanup.
