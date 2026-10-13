# TensorLane architecture

Example setup: 2 training ranks, 2 transform workers, streams `training` and `validation`.
Repeated parts (ranks, workers, streams) are drawn twice as examples; real counts come from the run config.

```text
 TRAINING HOST                                  (2) upload messages (uploads.sock)           │  TENSORLANE SERVER                                           │  STORAGE
              ┌────────────────────────────┬──────────────────────┐                          │                                                              │
  ┌────────────────────────┐   ┌────────────────────────┐         │                          │                                                              │
  │ UPLOAD CLIENT          │   │ UPLOAD CLIENT          │         │                          │                                                              │
  │ rank 0 (Rust)          │   │ rank 1 (Rust)          │         │                          │                                                              │
  └────────────────────────┘   └────────────────────────┘         │                          │                                                              │
              ▲                            ▲                      │                          │                                                              │
              │(1) metric, save            │(1) metric, save      │                          │                                                              │
              │                            │                      │                          │                                                              │
  ┌────────────────────────┐   ┌────────────────────────┐         │                          │                                                              │
  │ YOUR TRAINING LOOP     │   │ YOUR TRAINING LOOP     │         │                          │                                                              │
  │ rank 0 process         │   │ rank 1 process         │◄────────┼─────────────┐            │                                                              │
  │ lane.asset(), batches  │   │ lane.asset(), batches  │         │             │            │                                                              │
  └────────────────────────┘   └────────────────────────┘         │             │            │                                                              │
        │           ▲                │           ▲                │             │            │                                                              │
        │(10) next  │                │(10) next  │                │             │            │                                                              │
        │           │(11) batch      │           │(11) batch      │             │            │                                                              │
        ▼           │                ▼           │                │             │            │                                                              │
  ┌────────────────────────┐   ┌────────────────────────┐         │             │            │                                                              │
  │ BATCH READER           │   │ BATCH READER           │         │             │            │                                                              │
  │ rank 0, stream training│   │ rank 1, stream training│─────┐   │             │            │                                                              │
  └────────────────────────┘   └────────────────────────┘     │   │             │            │                                                              │
        │           ▲                │           ▲            │   │             │            │                                                              │
        │(10) next  │(9) batch       │(10) next  │(9) batch   │   │             │            │                                                              │
        │           │                │           │            │   │             │            │                                                              │
        ▼           │                ▼           │            │   │             │            │                                                              │
  ┌─────────────────────────────────────────────────────┐     │   │             │            │                                                              │
  │ COLLATOR (1 process)                                │     │   │             │            │                                                              │
  │ joins samples into batches, runs collate_fn(samples)│     │   │             │            │                                                              │
  │ batch id % ranks → rank 0, rank 1, rank 0, ...      │     │   │             │            │                                                              │
  └─────────────────────────────────────────────────────┘     │(12) credits back│            │                                                              │
              ▲                            ▲                  │   (semaphores)  │            │                                                              │
              │(8) transformed samples     │(8)               │   │             │            │                                                              │
              │  (shared memory)           │                  │   │             │            │                                                              │
              │                            │                  │   │             │            │                                                              │
              │                            │                  │   │             │            │                                                              │
  ┌────────────────────────┐   ┌────────────────────────┐     │   │             │            │                                                              │
  │ TRANSFORM WORKER 1     │   │ TRANSFORM WORKER 2     │     │   │             │            │                                                              │
  │ process                │   │ process                │     │   │             │            │                                                              │
  │ runs transform(sample) │   │ runs transform(sample) │     │   │             │            │                                                              │
  └────────────────────────┘   └────────────────────────┘     │   │             │(18) cached │                                                              │
              ▲ (7) raw samples (work.sock)▲                  │   │             │files for   │                                                              │
    ┌─────────┴─────┬──────────────────────┘                  │   │             │lane.asset()│                                  ┌──────────────────────┐(5) │  ┌──────────────────────┐
    │               │                                         │   │             │            │                                  │ PLAN BUILDER         │────┼─►│ ClickHouse           │
 ┌─ DAEMON (Rust threads inside rank 0 process) ──────────────┼───┼─────────────┼──────────┐ │                               ┌─►│ runs stream SQL once │    │  │ your sample rows     │
 │  │               │                                         │   │             │          │ │  ┌────────────────────────┐(4)│  │ saves batch list     │ ◄──┼──└──────────────────────┘
 │  │ ┌────────────────────────┐                              │   │             │          │ │  │ BATCH SERVICE          │───┘  └──────────────────────┘    │
 │  │ │ PREFETCHER             │────(3) GET batch n───────────┼───┼─────────────┼──────────┼─┼─►│                        │                                  │
 │  │ │ stream training        │ ◄────────────────────────────┘   │             │          │ │  │ plan ready? if not,    │                                  │
 │  │ │ ≤32 batches, ≤2 GiB    │ ◄──(6) samples+blobs | 202 wait | 204 end──────┼──────────┼─┼──│ build it (4) and       │                                  │
 │  │ └────────────────────────┘                                  │             │          │ │  │ answer 202             │      ┌──────────────────────┐(5c)│  ┌──────────────────────┐
 │  │                                                             │             │          │ │  │                        │(5b)  │ BLOB LOADER          │────┼─►│ R2 / S3              │
 │  │ ┌────────────────────────┐                                  │             │          │ │  │ batch file cached?     │─────►│ ≤256 MiB per run     │    │  │ sample blobs         │
 │  │ │ PREFETCHER             │────(3) GET batch n───────────────┼─────────────┼──────────┼─┼─►│ if not, load (5b)      │      │ range GET per blob   │ ◄──┼──└──────────────────────┘
 │  └►│ stream validation      │                                  │             │          │ │  │                        │      └──────────────────────┘    │
 │    │ same credits/budget    │ ◄──(6) samples+blobs─────────────┼─────────────┼──────────┼─┼──│                        │                                  │
 │    └────────────────────────┘                                  │             │          │ │  └────────────────────────┘                                  │
 │                                                                │             │          │ │                                                              │
 │                 ┌──────────────────────────────────────────────┘             │          │ │                                                              │
 │                 ▼                                                            │          │ │                                                              │
 │    ┌────────────────────────┐                                                │          │ │  ┌────────────────────────┐                                  │  ┌──────────────────────┐
 │    │ UPLOADER               │────(13) PUT metrics────────────────────────────┼──────────┼─┼─►│ METRIC WRITER          │────(14) INSERT───────────────────┼─►│ ClickHouse           │
 │    │ metrics: 500 or 1 s    │                                                │          │ │  │ dedup by request id    │                                  │  │ metrics              │
 │    │ files: 16 in parallel  │────(15) PUT file + size + sha256───────────────┼──────────┼─┼┐ └────────────────────────┘                                  │  └──────────────────────┘
 │    │ checkpoints: in order  │────┐                                           │          │ ││                                                             │
 │    └────────────────────────┘    │                                           │          │ ││ ┌────────────────────────┐                                  │  ┌──────────────────────┐
 │                                  │                                           │          │ ││ │ UPLOAD HANDLER         │                                  │  │ R2 / S3 objects +    │
 │                                  │                                           │          │ │└►│ 2 at a time            │────(16) object + row─────────────┼─►│ ClickHouse assets,   │
 │                                  │                                           │          │ │  │ verifies size, sha256  │                                  │  │ artifacts            │
 │                                  │                                           │          │ │  └────────────────────────┘                                  │  └──────────────────────┘
 │                                  │(17) copy                                  │          │ │                                                              │
 │                                  │                                           │          │ │                                                              │
 │    ┌────────────────────────┐    │                                           │          │ │  ┌────────────────────────┐                                  │  ┌──────────────────────┐
 │    │ ASSET FETCHER          │────(19) GET asset ranges───────────────────────┼──────────┼─┼─►│ ASSET PROXY            │────(20) GET on cache miss────────┼─►│ R2 / S3              │
 │    │ 4 MiB ranges           │    │                                           │          │ │  │ caches each range      │                                  │  │ models, checkpoints  │
 │    │ ≤1024 in parallel      │ ◄──bytes───────────────────────────────────────┼──────────┼─┼──│ on server disk         │ ◄────────────────────────────────┼──└──────────────────────┘
 │    └────────────────────────┘    │                                           │          │ │  └────────────────────────┘                                  │
 │                 │                │                                           │          │ │                                                              │
 │                 │                │                                           │          │ │                                                              │
 │                 │ download       │   ┌────────────────────────────────┐      │          │ │                                                              │
 │                 │                │   │ ASSET CACHE (disk, per host)   │      │          │ │                                                              │
 │                 └────────────────┼──►│ ~/.cache/tensorlane/assets     │──────┘          │ │                                                              │
 │                                  └──►│ shared by all runs, LRU 40 GiB │                 │ │                                                              │
 │                                      └────────────────────────────────┘                 │ │                                                              │
 │                                                                                         │ │                                                              │
 │                                                                                         │ │                                                              │
 │                                                                                         │ │  ┌────────────────────────┐                                  │  ┌──────────────────────┐
 │    ┌────────────────────────┐                                                           │ │  │ RUN STATUS             │                                  │  │ ClickHouse           │
 │    │ HEARTBEAT              │────(21) POST init / heartbeat / end───────────────────────┼─┼─►│ + SWEEPER every 60 s   │────(22) INSERT status────────────┼─►│ runs, run_status     │
 │    │ init, every 30 s, end  │                                                           │ │  │ fails silent runs      │                                  │  └──────────────────────┘
 │    └────────────────────────┘                                                           │ │  └────────────────────────┘                                  │
 │                                                                                         │ │                                                              │
 │                                                                                         │ │                                                              │
 │                                                                                         │ │                                                              │
 └─────────────────────────────────────────────────────────────────────────────────────────┘ │                                                              │
                                                                                             │                                                              │

```

## Nodes

| Node | Runs where | What it does |
| --- | --- | --- |
| Your training loop (rank 0, rank 1) | Your script, one process per rank | Calls `tensorlane.init()`, iterates `lane.batches(stream)`, calls `lane.metric/metric_artifact/save_asset`, reads models via `lane.asset()`. |
| Upload client (rank 0, rank 1) | Rust object inside each rank process | Turns metric/artifact/checkpoint calls into messages for the daemon. Never waits for the network. |
| Batch reader (rank 0, rank 1) | Python object per rank per stream | Asks the collator for the next batch, hands it to your loop, then releases a prefetch credit. Can be closed early and reopened. |
| Collator | 1 process started by rank 0 | Collects transformed samples until a batch is complete, runs your `collate_fn(samples)`, gives batches to ranks in order (batch id % ranks). |
| Transform worker 1, 2 | `num_workers` processes started by rank 0 | Run your `transform(sample)` on each raw sample (metadata dict + blob bytes); tensors go to shared memory. They ignore job signals and exit when the training process dies. |
| Daemon | Rust threads inside the rank 0 process | Everything that talks to the server: prefetchers, uploader, asset fetcher, heartbeat. One per run per host. |
| Prefetcher (training, validation) | Daemon task, one per stream | Requests batches ahead of training: up to `ranks × prefetch_factor` (32) batches and 2 GiB of memory. |
| Uploader | Daemon task | Sends metrics in batches (500 or every 1 s), uploads up to 16 files at once, uploads checkpoints one at a time in order. Failures are logged and returned at `flush()`/`close()`. |
| Asset fetcher | Daemon startup | Downloads models and checkpoints in 4 MiB ranges, ≤1024 at once, verifies sha256, unpacks tar files. |
| Asset cache | Disk, `~/.cache/tensorlane/assets` | Shared by every run on the host. LRU, 40 GiB by default, keeps 5% of the disk free. In-use entries are never evicted. |
| Heartbeat | Daemon task | Sends `init` at start, a heartbeat every 30 s, and `end` (succeeded or failed) at close. |
| Batch service | Server | Answers "give me batch n of this stream" from its disk cache. Builds the plan and loads blobs on the first request. |
| Plan builder | Server background job | Runs your stream SQL once and saves which samples form which batch. |
| Blob loader | Server background job | Reads each sample's blob bytes from R2 (≤256 MiB in memory per run) and saves the finished batch file. |
| Metric writer | Server | Writes metric batches to ClickHouse; retried batches are deduplicated by request id. |
| Upload handler | Server | Streams uploaded files to R2 (2 at a time), checks size and sha256, records an asset or artifact row. |
| Asset proxy | Server | Serves model/checkpoint byte ranges; each 4 MiB range is fetched from R2 once and cached on server disk. |
| Run status + sweeper | Server | Records running/succeeded/failed. Every 60 s marks runs failed after 2 min without a heartbeat (only once the server has been up 2 min); a returning heartbeat sets them running again. |

## Flows

| # | From → to | What flows |
| --- | --- | --- |
| (1) | Training loop → upload client | `metric(step, name, value)`, `metric_artifact(step, path)`, `save_asset(name, path or state_dict)` |
| (2) | Upload clients → uploader | Upload messages over `uploads.sock` |
| (3) | Prefetcher → batch service | `GET /runs/{id}/streams/{stream}/batches/{n}` |
| (4) | Batch service → plan builder | First request for a stream starts the plan build; client gets 202 until it is ready |
| (5) | Plan builder ↔ ClickHouse | Your stream SQL; rows `(sample_id, batch_idx, sample_idx, metadata_json, blobs_json)` |
| (5b) | Batch service → blob loader | Batch file not cached yet |
| (5c) | Blob loader ↔ R2 | One range GET per sample blob |
| (6) | Batch service → prefetcher | Batch with sample metadata and blob bytes; 202 = not ready yet, 204 = stream ended |
| (7) | Prefetchers → transform workers | Raw samples over `work.sock`, spread across workers |
| (8) | Transform workers → collator | Transformed samples; tensors in shared memory |
| (9) | Collator → batch reader | One collated batch per request, in batch order |
| (10) | Training loop → reader → collator | `"next"` request |
| (11) | Batch reader → training loop | The batch your loop receives |
| (12) | Batch reader → prefetcher | Prefetch credit and memory released (POSIX semaphores), so the prefetcher can fetch another batch |
| (13) | Uploader → metric writer | `PUT /runs/{id}/metrics/{request_id}` |
| (14) | Metric writer → ClickHouse | Insert into `metrics` |
| (15) | Uploader → upload handler | `PUT /uploads/{id}`: file stream + size + sha256 + metadata |
| (16) | Upload handler → R2 + ClickHouse | Object in R2, then an `assets` or `artifacts` row |
| (17) | Uploader → asset cache | Uploaded checkpoint copied into the cache, so it is never downloaded again |
| (18) | Asset cache → training loops | `lane.asset("model")` returns a local path |
| (19) | Asset fetcher ↔ asset proxy | Asset info, then 4 MiB range requests and bytes |
| (20) | Asset proxy ↔ R2 | Range GET on server cache miss |
| (21) | Daemon → run status | `POST init`, `heartbeat` every 30 s, `end {failed}` |
| (22) | Run status → ClickHouse | Insert into `run_status` |
