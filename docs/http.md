# HTTP training protocol

TensorLane uses finite HTTP requests for initialization, batches, downloads,
uploads, metrics, heartbeats, and completion. Upgrade the server and Python client
together. Existing gRPC clients cannot use this server. Keep their old server
available until their training runs finish.

A client supplies `x-tensorlane-session: <UUID>` on training requests. Each Python
daemon starts with a fresh UUID. The first initialization claims ownership atomically in R2 and records it in
ClickHouse. HTTP retries from that daemon keep the same UUID, including across
server replacement. Another training execution cannot take over the run.

Resume by creating a new run with `POST /runs`, adding `resume_from: <old run UUID>`
to the existing project, name, and configuration fields. Creation selects the
source run's latest committed checkpoint and pins its asset ID under its original
name in the new run's input assets. Incomplete uploads are ignored. Python calls
remain `tensorlane.init(new_run_id)` and `lane.asset(name)`. The new run starts its
own batch sequences at zero; this does not restore prior stream positions.

| Request | Response |
| --- | --- |
| `POST /runs/{id}/init` | Run configuration, stream names, input names |
| `POST /runs/{id}/heartbeat` | 204; records the last heartbeat |
| `POST /runs/{id}/end` with `{"failed": false}` | 204; records completion |
| `GET /runs/{id}/streams/{name}/batches/{sequence}` | Protobuf batch, 202 while preparing, or 204 at EOF |
| `GET /runs/{id}/inputs/{name}` | Size, ETag, optional SHA-256, and asset metadata |
| `GET /runs/{id}/inputs/{name}/bytes` | 206 for an explicit byte range of at most 4 MiB |
| `PUT /runs/{id}/metrics/{request_id}` | 204; repeated IDs are deduplicated |
| `PUT /uploads/{id}` | Streams multipart `spec` JSON followed by `file` directly into the final R2 object; 200 after publication |

A batch sequence starts at zero. Retrying a sequence uses the cached query result.
For a repeating stream, the sequence keeps increasing while the saved query
result repeats. The protobuf messages are defined in `protocol/batch.proto`.
Run management routes remain JSON HTTP APIs.

Initialization does not execute data queries. The Python client starts a stream's
prefetch when its first `lane.batches(name)` reader connects. A batch request
creates a missing query plan; partially removed plans are rebuilt automatically.
Each rebuilt plan has a separate batch cache, so it cannot serve an older plan's
payloads. Requerying uses the run's original SQL and parameters. Deterministic SQL,
the same seed, and stable source data are required to reproduce the same samples
after a plan is evicted.

Each stream fetches up to `ranks * prefetch_factor` batches concurrently.
The same prefetch credits cover outstanding requests and completed batches, so
concurrency does not increase the configured buffer capacity. Responses reach
transform workers in batch sequence order, even when requests finish out of order.
Batch preparation loads samples and blobs concurrently and emits them in saved
query order, preserving seeded sampling and replay order.

Asset initialization does not fetch input contents. The first range request
streams bytes from S3 to the client while caching them. Interrupted ranges are
retried individually. Clients pin the ETag, validate each range, and verify the
final SHA-256 when the registered asset supplies one. Input object keys must be
immutable during a run. Asset IDs point to immutable saved objects.

Uploads stream through a bounded 16 MiB buffer into standard S3 multipart parts
on the final object key. The server validates length and SHA-256 before completing
the object, then publishes its asset or artifact row in ClickHouse. There are no
local upload files or intermediate payload objects. A failed request aborts its
multipart upload; an HTTP retry sends the whole file and can reach any replica.
Existing final objects are checked against the upload's content and metadata.

Small immutable R2 records hold run ownership, upload fingerprints, checkpoint
lineage, and metric retry receipts. They contain control metadata, never staged
payload bytes. ClickHouse insert tokens deduplicate metric and artifact retries
between insertion and receipt publication; the deduplication window must cover
the client's recovery period.

## Recovery and limits

Requests retry network failures, incomplete bodies, 202, 408, 429, and server
errors. Backoff includes jitter. The default recovery deadline is 600 seconds per
request; `TENSORLANE_RETRY_TIMEOUT_SECONDS=0` waits indefinitely. Each attempt has
a 10-second connect timeout and a 30-second read idle timeout. Ordinary requests
have a 120-second attempt limit; streaming uploads can use the remaining recovery
deadline. A Python `init(timeout=...)` deadline also covers asset downloads and local
worker startup. A `flush(timeout=...)` deadline includes queue space and automatic metrics.
A timeout leaves its upload running.

A missed heartbeat or a lost connection does not fail a run. Server startup and
shutdown leave run status intact. Cached query results can be regenerated. Only explicit run
completion changes a running run to succeeded or failed. Abandoned clients leave
running records; `run_sessions.updated_at` exposes their last heartbeat.

Server batch-loading admission uses `config.tensorlane.max_load_memory_bytes`
(default 256 MiB per run per server process), divided into independent stream
pools. A batch exceeding its stream's share may run alone within that stream;
its reservation lasts through cache publication. This is an estimated working-set
target. Query preparation uses independent stream locks. Each server process
allows two streaming uploads and eight asset range requests at once. Upload queues
have bounded capacity. Batches are limited to 64 MiB and query snapshots to
512 MiB. Uploads are limited to 16 GiB.

CPU tensor storage is shared between Python transform workers, the collate worker,
and readers. IPC still serializes Python metadata and tensor storage handles.
Collation may allocate a new batch tensor before its storage is shared.

## Replicas and cache

Replicas share ClickHouse and R2. Each replica has an independent disposable
`CACHE_DIR`; no shared filesystem or sticky routing is required. Local locks
coordinate preparation and eviction only within that replica. Cache loss causes
automatic query preparation and input downloads on the next request.

`CACHE_BYTES` defaults to 15 GiB and limits query plans, reusable batches, and asset
range files. Cleanup runs every minute and removes a query plan's index and ready
marker with it, under the same local lock used by batch readers and preparation.
Writes reserve at least 512 MiB of free space; insufficient space returns an error
so the client can retry. R2 control records are outside the disposable cache.
Configure the bucket to expire abandoned multipart uploads after hard process
termination.

The server accepts SIGTERM, stops admitting new preparations, and allows up to
60 seconds for HTTP requests and background jobs to finish. Provision a shutdown
grace period of at least 65 seconds. Another replica can serve retried requests
using the database and R2 records.

The migration `20261004170000` adds run sessions and enables insert deduplication
for local MergeTree tables. Review and apply it before starting this server.
This change does not apply migrations automatically.

## Query parameters

Use ClickHouse placeholders such as `{seed:Int64}` and `{items:Array(UInt64)}`.
Parameters are sent through ClickHouse's native `param_<name>` mechanism; they
are never interpolated into SQL. JSON numbers, strings, booleans, nulls, arrays,
tuples, and maps are supported. Use strings for integers or decimals that exceed
your JSON producer's precision.

For any type requiring native text, use `{"$clickhouse":"native value"}`. For
example, `{"price":{"$clickhouse":"1.000000000000000001"}}` preserves a decimal
exactly. ClickHouse validates the value against the placeholder's declared type.
The type must be supported by the ClickHouse server's query-parameter interface.
