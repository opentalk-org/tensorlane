# HTTP training protocol

TensorLane uses finite HTTP requests for initialization, batches, downloads,
uploads, metrics, and completion. Upgrade the server and Python client
together. Existing gRPC clients cannot use this server. Keep their old server
available until their training runs finish.

Training requests identify the run by its URL and use the configured API key.
Multiple clients may initialize and access the same running run, including after
server or client replacement. No client session header or ownership claim is
required. Clients coordinate their batch sequences and run completion.

Resume with `tensorlane.init()` using the same run ID. Initialization selects the latest committed checkpoint, returns its name and immutable asset ID in `checkpoint`, and exposes it through `lane.asset(name)`. Failed or completed runs return to running on initialization. Incomplete uploads are ignored. The client saves per-stream next batch numbers in the checkpoint's existing `_tensorlane.next_batches` metadata and restores them before fetching data. Prefetched batches do not advance this cursor. Save after training on the returned batches; synchronize ranks before saving a distributed checkpoint. Load model and optimizer state from the checkpoint before reading batches. There is no separate resume request or manually supplied resume step.

| Request | Response |
| --- | --- |
| `POST /runs/{id}/init` | Run configuration, stream names, input names, optional latest checkpoint |
| `POST /runs/{id}/heartbeat` | Legacy compatibility endpoint: 204 while the run is running; no state is recorded |
| `POST /runs/{id}/end` with `{"failed": false}` | 204; records completion |
| `GET /runs/{id}/streams/{name}/batches/{sequence}` | Protobuf batch, 202 while preparing, or 204 at EOF |
| `GET /runs/{id}/inputs/{name}` | Size, ETag, optional SHA-256, and asset metadata |
| `GET /runs/{id}/inputs/{name}/bytes` | 206 for an explicit byte range |
| `PUT /runs/{id}/metrics/{request_id}` | 204; repeated IDs are deduplicated |
| `PUT /uploads/{id}` | Streams multipart `spec` JSON followed by `file` directly into the final R2 object; 200 after publication |

A new stream starts at zero; a resumed stream starts at its checkpoint cursor.
Retrying a sequence uses the cached query result.
For a repeating stream, the sequence keeps increasing while the saved query
result repeats. The protobuf messages are defined in `protocol/batch.proto`.
Run management routes remain JSON HTTP APIs.

Initialization does not execute data queries. The Python client starts a stream's
prefetch when its first `lane.batches(name)` reader connects. A batch request
creates a missing query plan; partially removed plans are rebuilt automatically.
Each rebuilt plan has a separate batch cache, so it cannot serve an older plan's
payloads. Requerying uses the run's original SQL and parameters. Deterministic SQL,
the same seed, and stable source data are required to reproduce the same samples
after cache loss.

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

Uploads stream into standard S3 multipart parts on the final object key. Parts
start at 16 MiB and grow for large files to fit the storage provider's 10,000-part
maximum. The server buffers one part at a time. The server validates length and SHA-256 before completing
the object, then publishes its asset or artifact row in ClickHouse. There are no
local upload files or intermediate payload objects. A failed request aborts its
multipart upload; an HTTP retry sends the whole file and can reach any replica.
Existing final objects are checked against the upload's content and metadata.

Small immutable R2 records hold upload fingerprints, checkpoint
lineage, and metric retry receipts. They contain control metadata, never staged
payload bytes. ClickHouse insert tokens deduplicate metric and artifact retries
between insertion and receipt publication; the deduplication window must cover
the client's recovery period.

## Recovery and limits

Requests retry network failures, incomplete bodies, 202, 408, 429, and server
errors. Backoff includes jitter. The default recovery deadline is 600 seconds per
request; `TENSORLANE_RETRY_TIMEOUT_SECONDS=0` waits indefinitely. Each attempt has
a 10-second connect timeout and a 30-second response-body idle timeout. Ordinary
requests also allow 30 seconds for response headers and have a 120-second attempt
limit. Streaming uploads allow the remaining recovery deadline for sending the
file and waiting for response headers. A Python `init(timeout=...)` deadline also
covers asset downloads and local worker startup. A `flush(timeout=...)` deadline
includes queue space and automatic metrics.
A timeout leaves its upload running.

A lost connection does not fail a run. Server startup and shutdown leave run
status intact. Cached query results can be regenerated. Explicit run
completion changes a running run to succeeded or failed; initialization reopens it. Abandoned clients leave
running records. Clients do not send heartbeats or store server session IDs.

Server batch-loading admission uses `config.tensorlane.max_load_memory_bytes`
(default 256 MiB per run per server process), divided into independent stream
pools. A batch exceeding its stream's share may run alone within that stream;
its reservation lasts through cache publication. This is an estimated working-set
target. Query preparation uses independent stream locks. Each server process
allows two streaming uploads and eight asset range requests at once. Upload queues
have bounded capacity. There are no application byte or item-count caps on query
plans, batches, samples, blobs, uploads, metadata, JSON request/response bodies,
asset ranges, metrics, or IPC messages. Query plans stream to disk; the filesystem
must have enough space. Client asset downloads still use 4 MiB chunks, which do
not limit the total size. Storage-provider limits and available disk/RAM apply.

Remaining timeout and concurrency policies:

| Policy | Effect |
| --- | --- |
| 300 seconds per ClickHouse query | The server sends `max_execution_time=300`. |
| 120 seconds per database read/write wait | Includes waiting for the first query row or the next row. |
| 30 seconds until S3 response headers | The SDK read timeout includes sending an upload part. S3 operations also have a 120-second attempt timeout and a 600-second total timeout including retries. |
| 30 seconds for incoming upload chunks | A pause in the HTTP request body aborts that attempt. Time spent awaiting an S3 part write is outside this chunk timer. |
| 64 concurrent application HTTP requests, two uploads and eight asset ranges per server | Controls concurrent work. Waiting for an upload slot consumes the client's recovery deadline. |

The default 600-second client recovery deadline can be changed with
`TENSORLANE_RETRY_TIMEOUT_SECONDS`; zero disables it. That setting does not change
server, S3 or reverse-proxy timeouts. `lane.batches(timeout=120)` bounds only the
local collator connection and handshake; iteration has no per-step timeout.
`init()` and `flush()` have no Python deadline unless the caller supplies one.

`CACHE_BYTES` controls reusable batch and asset-range cache files; query plans
are excluded so large plans are not repeatedly evicted and rebuilt. Server
batch-loading and client prefetch memory targets control concurrent work and
allow a single oversized batch to proceed alone. None limits the total step count.

Native IPC messages and on-disk query-plan records use 64-bit length prefixes.
Update the Python client and restart its daemon when upgrading; old four-byte
IPC peers are incompatible. The server uses a new `plans-v2` cache directory,
so old disposable query plans are rebuilt rather than read with the new format.

CPU tensor storage is shared between Python transform workers, the collate worker,
and readers. IPC still serializes Python metadata and tensor storage handles.
Collation may allocate a new batch tensor before its storage is shared.

## Replicas and cache

`GET /healthz` returns HTTP 200 while the HTTP server is responsive. It does not
check external dependencies. `GET /readyz` returns HTTP 200 only when ClickHouse
answers `SELECT 1`, the configured S3 bucket accepts HEAD, and the cache directory
is writable. These checks run
concurrently with a two-second total deadline. Failures, timeouts, and shutdown
return HTTP 503. Both endpoints are unauthenticated and bypass the application's
request concurrency limit. Use `/healthz` for startup and liveness probes and
`/readyz` for readiness, with a readiness probe timeout longer than two seconds.

Replicas share ClickHouse and R2. Each replica has an independent disposable
`CACHE_DIR`; no shared filesystem or sticky routing is required. Local locks
coordinate preparation and eviction only within that replica. Cache loss causes
automatic query preparation and input downloads on the next request.

`CACHE_BYTES` defaults to 15 GiB and controls reusable batch and asset-range
files. Cleanup runs every minute under local locks; query plans and their index
and ready markers remain until cache loss. Writes check available filesystem
space without reserving an extra fixed margin; insufficient space returns an
error so the client can retry. R2 control records are outside the disposable cache.
Configure the bucket to expire abandoned multipart uploads after hard process
termination.

The server accepts SIGTERM, stops admitting new preparations, and allows up to
60 seconds for HTTP requests and background jobs to finish. Provision a shutdown
grace period of at least 65 seconds. Another replica can serve retried requests
using the database and R2 records.

The migration `20261004170000` enables insert deduplication for local MergeTree
tables. The server no longer uses the historical `run_sessions` table; its removal
has a separate forward migration. Existing migration files remain unchanged.
This server does not apply migrations automatically.

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
