# Authentication and transport

The server requires `TENSORLANE_API_KEY` by default. Generate a shared key with
`openssl rand -hex 32`. Keys must contain at least 32 printable ASCII characters
without whitespace. All key holders have access to every run, query, asset, and
metric; the shared key does not provide user or project isolation.

HTTP requests use `Authorization: Bearer <key>`. Missing, invalid, or duplicate
authorization headers return HTTP 401. Every gRPC method, including streaming
methods, uses the same header in request metadata and returns `UNAUTHENTICATED`
when credentials are missing or invalid. Keys are compared using constant-time
comparison of their SHA-256 digests.

The Python client reads `TENSORLANE_API_KEY`, or accepts `api_key=` in
`tensorlane.init`. It attaches the key to all RPCs, including downloads, uploads,
metrics, heartbeats, and run completion. Secrets are not written to IPC metadata.
Both `http://` and `https://` addresses are supported with or without an API key.
A bare `host:port` address uses HTTP. HTTPS verifies the server certificate against
the system trust store. Use HTTPS for public connections: HTTP transmits the API
key and request data in plaintext.

The Nix development shell explicitly sets `TENSORLANE_ALLOW_UNAUTHENTICATED=true`
for local development. Outside that shell, use `--allow-unauthenticated` only for
local testing. A configured key always enables authentication, even with the
local opt-out. Empty or invalid configured keys fail startup.

For native gRPC TLS, set both `GRPC_TLS_CERT_FILE` and `GRPC_TLS_KEY_FILE` to PEM
files. The server loads these files at startup. After certificate renewal,
restart the server between runs to load the new certificate. HTTP can be served
through a TLS-terminating reverse proxy while gRPC uses TLS passthrough.

`AWS_REGION` sets the S3 signing region and defaults to `auto` for R2. Use the
region required by your S3 provider.
