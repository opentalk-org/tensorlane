# Authentication and transport

The server runs without authentication by default. Set `TENSORLANE_API_KEY` to
require a shared key. Generate one with `openssl rand -hex 32`. Keys must contain
at least 32 printable ASCII characters without whitespace. All key holders have
access to every run, query, asset, and metric.

Application endpoints use `Authorization: Bearer <key>` when a key is configured.
Missing, invalid, or duplicate authorization headers return HTTP 401. This also
covers downloads, uploads, metrics, legacy heartbeats, run completion, unknown routes,
and unsupported methods. Keys are compared using constant-time comparison of
their SHA-256 digests.

`GET /healthz` and `GET /readyz` are unauthenticated probe endpoints. They return
only fixed status text, without dependency errors or credentials.

The Python client reads `TENSORLANE_API_KEY`, or accepts `api_key=` in
`tensorlane.init`. Secrets are not written to IPC metadata.

The server listens on HTTP port 8180 by default. Both `http://` and `https://`
client addresses are supported. A bare `host:port` address uses HTTP. For HTTPS,
terminate TLS at a reverse proxy and forward every route to the same HTTP port.
TensorLane does not need certificate files. HTTPS clients verify the certificate
against their trust store. HTTP sends the key and request data in plaintext.

Authentication and TLS are independent. Empty or invalid configured keys fail
startup. `AWS_REGION` sets the S3 signing region and defaults to `auto` for R2.
