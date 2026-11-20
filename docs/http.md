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
| `POST /runs/{id}/heartbeat` | 204 unless the run is cancelled; keeps the run alive (the client sends one every 30 s) |
| `POST /runs/{id}/end` with `{"failed": false}` | 204; records completion |
| `GET /runs/{id}/streams/{name}/batches/{sequence}` | Protobuf batch, 202 while preparing, or 204 at EOF |
| `GET /runs/{id}/inputs/{name}` | Size, ETag, optional SHA-256, and asset metadata |
| `GET /runs/{id}/inputs/{name}/bytes` | 206 for an explicit byte range |
| `PUT /runs/{id}/metrics/{request_id}` | 204; repeated IDs are deduplicated |
| `POST /uploads` (tus) | Creates a resumable upload from `Upload-Length` and `Upload-Metadata` (`id`, `spec`), storing any first bytes; 201 with `Upload-Offset` |
| `PATCH /uploads/{id}` (tus) | Appends at `Upload-Offset`; 204 with the stored offset; the last byte completes and publishes the upload |
| `HEAD /uploads/{id}` (tus) | 200 with the stored `Upload-Offset` and `Upload-Length`; completes and publishes a fully stored upload |
| `OPTIONS /uploads` (tus) | 204 with `Tus-Version` and `Tus-Extension` |
| `PUT /uploads/{id}` | Streams multipart `spec` JSON followed by `file` directly into the final R2 object in one request; 200 after publication. Kept for older clients |

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
local upload files or intermediate payload objects. Existing final objects are
checked against the upload's content and metadata.

The client uploads with the [tus 1.0.0](https://tus.io/protocols/resumable-upload)
resumable-upload protocol (core, `creation` and `creation-with-upload`), so no
request lasts longer than one chunk however large the file is (a proxy limit on
request duration, such as a Traefik `readTimeout`, no longer cuts large
checkpoints). `POST /uploads` declares the upload: `Upload-Metadata` carries the
client-chosen upload `id` and its JSON `spec` (size, SHA-256 and asset or artifact
metadata), both base64, and the body carries the first chunk. A file up to one
chunk is therefore a single request, as before. `PATCH /uploads/{id}` sends each
further chunk at `Upload-Offset`. Chunks are whole parts of about 64 MiB
(`upload_chunk_bytes`); the server stores whole parts (`upload_part_bytes`), so a
request cut inside a part stores nothing of that part. After any failure the
client reads the stored offset with `HEAD /uploads/{id}` and continues from there;
a failure resends at most one chunk. A `409` at an offset that is still the stored
one (for example a cancelled run) fails the upload, as do other 4xx responses.

Any replica can serve any request. The S3 multipart upload ID and, for each stored
part, the SHA-256 state after it are immutable R2 state records (`tus/{id}` and
`tus/{id}/parts/{n}`), so the server still checks the whole file's SHA-256 before
storing its last part. Storing the last byte completes the object and publishes
its row; a `HEAD` or `PATCH` of a fully stored upload publishes again, so a lost
completion is finished by the client's next request. A resumable upload is not
aborted when a request fails; uploads abandoned by a crashed client stay as
incomplete S3 multipart uploads, so the bucket needs a lifecycle rule that aborts
incomplete multipart uploads after a few days.

Small immutable R2 records hold upload fingerprints, checkpoint
lineage, and metric retry receipts. They contain control metadata, never staged
payload bytes. ClickHouse insert tokens deduplicate metric and artifact retries
between insertion and receipt publication; the deduplication window must cover
how long a client keeps retrying.

## Recovery and limits

Requests retry network failures, incomplete bodies, 408, 429, 499 (a proxy that
dropped the request), and server errors forever, with jittered backoff up to 10 seconds. The daemon logs a failing
request once a minute and logs when it recovers. 202 (plan or batch still being
prepared) is polled with backoff up to 2 seconds. Other 4xx responses fail
immediately. There is no per-request or total deadline; dead connections are
detected by TCP keepalive (15 s idle, 3 probes) and a 10-second connect timeout.
`init(timeout=...)`, `lane.batches(timeout=...)`, `metric(timeout=...)` and
`flush(timeout=...)` are opt-in and unset by default. A flush timeout leaves its
upload running.

A lost connection does not fail a run. Server startup and shutdown leave run
status intact. Cached query results can be regenerated. Explicit run completion
changes a run to succeeded or failed; initialization reopens it. Clients send a
heartbeat every 30 seconds, which keeps the run
alive. A run silent for 2 minutes (the client crashed, was OOM-killed or lost its node) is marked failed by the next sweep, at
most 5 minutes later; if its client comes back, its next heartbeat returns it to running. Each server serves an active run's
config from memory for 30 seconds, so a cancellation written by another replica takes effect within 30 seconds.

Server batch-loading admission uses `config.tensorlane.max_load_memory_bytes`
(default 256 MiB per run per server process), divided into independent stream
pools. A batch exceeding its stream's share may run alone within that stream;
its reservation lasts through cache publication. This is an estimated working-set
target. Query preparation uses independent stream locks. Each server process
allows two streaming uploads and 128 asset range requests at once. Upload queues
have bounded capacity. There are no application byte or item-count caps on query
plans, batches, samples, blobs, uploads, metadata, JSON request/response bodies,
asset ranges, metrics, or IPC messages. Query plans stream to disk; the filesystem
must have enough space. Client asset downloads still use 4 MiB chunks, which do
not limit the total size. Storage-provider limits and available disk/RAM apply.

Remaining timeouts, each kept for a concrete reason:

| Timeout | Why it stays |
| --- | --- |
| Client connect 10 s | A blackholed host otherwise blocks each attempt for about 2 minutes; the attempt is retried. |
| Client TCP keepalive / `tcp_user_timeout` (reqwest defaults) | Detects dead peers and half-open connections; the request is retried. |
| Server incoming upload chunk idle 30 s | Frees one of two upload slots from a dead uploader; the client resumes from the stored offset. |
| S3 connect 10 s, 600 s per attempt, 5 attempts, SDK stalled-stream protection | The S3 SDK has no TCP keepalive; these are the only dead-peer detection for S3. Failures are retried by the client. |
| `/readyz` 2 s | A probe must answer; it checks only the local cache (free space and writability), not ClickHouse or S3. Never affects a run. |
| Server shutdown drain 30 s + 30 s | Lets in-flight requests finish; clients retry against the restarted server. |
| Silent-run sweep every 5 min, 2 min of silence (after 2 min of server uptime) | Marks crashed or killed clients failed, reading only runs with status in the last day; a returning client is revived. |
| Two uploads and 128 asset ranges per server | Concurrency limits, not deadlines; waiters queue. |

ClickHouse plan queries run with `max_execution_time=0`; other ClickHouse
queries use the server's profile defaults.
