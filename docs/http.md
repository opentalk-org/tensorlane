# HTTP training protocol

TensorLane uses finite HTTP requests for initialization, batches, downloads,
uploads, metrics, heartbeats, and completion. Upgrade the server and Python client
together. Existing gRPC clients cannot use this server. Keep their old server
available until their training runs finish.

A client supplies `x-tensorlane-session: <UUID>` on training requests. The Python
client persists this UUID in its IPC directory. The first initialization stores
ownership in ClickHouse. Another client session cannot take over the run.

| Request | Response |
| --- | --- |
| `POST /runs/{id}/init` | Run configuration, stream names, input names |
| `POST /runs/{id}/heartbeat` | 204; records the last heartbeat |
| `POST /runs/{id}/end` with `{"failed": false}` | 204; records completion |
| `GET /runs/{id}/streams/{name}/batches/{sequence}` | Protobuf batch, 202 while preparing, or 204 at EOF |
| `GET /runs/{id}/inputs/{name}` | Size, ETag, optional SHA-256, and asset metadata |
| `GET /runs/{id}/inputs/{name}/bytes` | 206 for an explicit byte range of at most 4 MiB |
| `PUT /runs/{id}/metrics/{request_id}` | 204; repeated IDs are deduplicated |
| `PUT /uploads/{id}` | Creates an upload with its size, SHA-256, and metadata |
| `PUT /uploads/{id}/chunks/{index}` | 204; writes one chunk of at most 4 MiB |
| `POST /uploads/{id}/commit` | 202 until S3 and ClickHouse writes finish, then 200 |

A batch sequence starts at zero. Retrying a sequence returns the same samples.
For a repeating stream, the sequence keeps increasing while the saved query
result repeats. The protobuf messages are defined in `protocol/batch.proto`.
Run management routes remain JSON HTTP APIs.

Asset initialization does not fetch input contents. The first range request
streams bytes from S3 to the client while caching them. Interrupted ranges are
retried individually. Clients pin the ETag, validate each range, and verify the
final SHA-256 when the registered asset supplies one. Input object keys must be
immutable during a run. Asset IDs point to immutable saved objects.

Uploads preserve received chunks, their hashes, and multipart S3 progress on the
shared volume. Commit requests are idempotent. Metrics use durable volume receipts
and ClickHouse insert tokens. ClickHouse deduplication covers the gap between
insertion and receipt publication; its deduplication window must cover the
client's recovery period. Completed receipts remain until the run is archived.

## Recovery and limits

Requests retry network failures, incomplete bodies, 202, 408, 429, and server
errors. Backoff includes jitter. The default recovery deadline is 600 seconds per
request; `TENSORLANE_RETRY_TIMEOUT_SECONDS=0` waits indefinitely. Each attempt has
a 10-second connect timeout, a 30-second read idle timeout, and a 120-second total
limit. A Python `init(timeout=...)` deadline also covers asset downloads and local
worker startup. A `flush(timeout=...)` timeout leaves its upload running.

A missed heartbeat or a lost connection does not fail a run. Server startup and
shutdown leave run status and saved query results intact. Only explicit run
completion changes a running run to succeeded or failed. Abandoned clients leave
running records; `run_sessions.updated_at` exposes their last heartbeat.

Per server process, preparation allows two queries, two batches, two upload
commits, four sample-blob reads, and eight asset range requests at once. Upload
queues have bounded capacity. Batches are limited to 64 MiB and query snapshots
to 512 MiB. Uploads are limited to 16 GiB.

## Shared storage

All server replicas need the same ClickHouse database and `CACHE_DIR` volume.
The volume must support POSIX advisory file locks, atomic rename, and fsync
across replicas. These locks serialize ownership changes, query preparation,
checkpoint lineage, and upload commits. A dead process releases its locks.
No sticky routing is required.

`CACHE_BYTES` defaults to 8 GiB and limits reusable batch and asset range files.
Cleanup runs every minute. Immutable query snapshots, upload staging, and receipts
are retained separately and need capacity planning and archival after runs finish.
Writes reserve at least 512 MiB of free space; insufficient space returns an error
so the client can retry. Cache cleanup does not remove query snapshots or receipts.
S3 providers should expire abandoned multipart uploads through a lifecycle rule.

The server accepts SIGTERM, stops admitting new preparations, and allows up to
60 seconds for HTTP requests and background jobs to finish. Provision a shutdown
grace period of at least 65 seconds. A hard stop also preserves completed files;
the next process resumes from them.

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
