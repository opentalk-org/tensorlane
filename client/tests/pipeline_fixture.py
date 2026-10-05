from __future__ import annotations

import hashlib
from email.parser import BytesParser
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import tarfile
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from urllib.parse import unquote, urlsplit
import uuid

import tensorlane
from fixture_transforms import transform


def wait_for(check, timeout=15):
    deadline = time.monotonic() + timeout
    while not check():
        if time.monotonic() >= deadline:
            raise TimeoutError("condition did not become true")
        time.sleep(0.02)


def varint(value):
    output = bytearray()
    while value > 127:
        output.append((value & 127) | 128)
        value >>= 7
    output.append(value)
    return bytes(output)


def field(number, value):
    return varint(number * 8 + 2) + varint(len(value)) + value


def batch_bytes(stream, index, samples):
    payload = b"".join(field(1, sample) for sample in samples)
    return (
        payload
        + field(2, stream.encode())
        + b"\x18"
        + varint(index)
        + b"\x20"
        + varint(index * 4 + 7)
    )


class Authorization:
    def __init__(self, key):
        self.key = key
        self.methods = set()


class Handler(BaseHTTPRequestHandler):
    def __init__(self, fixture, request, client_address, server):
        self.fixture = fixture
        super().__init__(request, client_address, server)

    def do_GET(self):
        self.fixture.handle(self)

    do_POST = do_GET
    do_PUT = do_GET

    def log_message(self, *args):
        pass


class Fixture:
    def __init__(self, authorization=None):
        self.streams = {"training": 5, "validation": 3, "evaluation": 2}
        self.config = {
            "queries": {key: {"sql": "SELECT ..."} for key in self.streams},
            "app": {"optimizer": {"lr": 0.1}},
            "tensorlane": {"assets": {}},
        }
        self.requests = {name: [] for name in self.streams}
        self.init_requests = []
        self.end_requests = []
        self.assets = {}
        self.asset_requests = []
        self.metrics = []
        self.artifacts = []
        self.saved = []
        self.heads = {}
        self.uploads = {}
        self.transient_batches = 0
        self.truncate_asset_once = False
        self.drop_metric_reply = False
        self.drop_upload_reply = False
        self.metric_receipts = set()
        self.fail_init = False
        self.fail_asset = False
        self.fail_save = False
        self.fail_metrics = False
        self.fail_stream = None
        self.empty_batch = False
        self.blob_size = None
        self.returned_run_id = None
        self.asset_started = threading.Event()
        self.asset_gate = threading.Event()
        self.asset_gate.set()
        self.upload_started = threading.Event()
        self.upload_gate = threading.Event()
        self.upload_gate.set()
        self.end_gate = threading.Event()
        self.end_gate.set()
        self.lock = threading.Lock()
        self.authorization = authorization
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, self))
        self.port = self.server.server_port
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def reply(self, handler, status, payload=b"", headers=None):
        if not isinstance(payload, bytes):
            payload = json.dumps(payload).encode()
        handler.send_response(status)
        handler.send_header("Content-Length", str(len(payload)))
        for name, value in (headers or {}).items():
            handler.send_header(name, value)
        handler.end_headers()
        try:
            handler.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def handle(self, handler):
        parts = [
            unquote(part) for part in urlsplit(handler.path).path.split("/") if part
        ]
        if (
            self.authorization
            and handler.headers.get("Authorization")
            != f"Bearer {self.authorization.key}"
        ):
            self.reply(handler, 401, {"message": "Unauthorized"})
            return
        content = handler.rfile.read(int(handler.headers.get("Content-Length", 0)))
        data = (
            json.loads(content)
            if handler.headers.get("Content-Type") == "application/json"
            else None
        )
        action = parts[-1] if parts[-1] in {"init", "heartbeat", "end"} else parts[-2]
        if self.authorization:
            self.authorization.methods.add(action)
        match parts:
            case ["runs", run, "init"]:
                self.init_requests.append(SimpleNamespace(run_id=run))
                if self.fail_init:
                    self.reply(handler, 422, {"message": "fixture init failure"})
                    return
                self.reply(
                    handler,
                    200,
                    {
                        "run_id": self.returned_run_id or run,
                        "config": json.dumps(self.config),
                        "assets": list(self.assets),
                        "streams": list(self.streams),
                    },
                )
            case ["runs", run, "heartbeat"]:
                self.reply(handler, 204)
            case ["runs", run, "end"]:
                self.end_gate.wait(20)
                self.end_requests.append(
                    SimpleNamespace(run_id=run, failed=data["failed"])
                )
                self.reply(handler, 204)
            case ["runs", run, "streams", stream, "batches", index]:
                self.data(handler, run, stream, int(index))
            case ["runs", run, "inputs", name]:
                self.asset_requests.append(SimpleNamespace(run_id=run, name=name))
                self.asset_started.set()
                self.asset_gate.wait(20)
                asset_id, body = self.assets[name]
                digest = hashlib.sha256(body).hexdigest()
                self.reply(
                    handler,
                    200,
                    {
                        "size": len(body),
                        "etag": digest,
                        "sha256": digest,
                        "metadata": {
                            "asset_id": asset_id,
                            "entrypoint": None,
                            "metadata_json": "{}",
                            "kind": "file",
                            "asset_type": "generic",
                        },
                    },
                )
            case ["runs", run, "inputs", name, "bytes"]:
                if self.fail_asset:
                    self.reply(handler, 422, {"message": "fixture asset failure"})
                    return
                body = self.assets[name][1]
                start, end = map(int, handler.headers["Range"][6:].split("-"))
                if self.truncate_asset_once:
                    self.truncate_asset_once = False
                    handler.send_response(206)
                    handler.send_header("Content-Length", str(end - start + 1))
                    handler.send_header(
                        "Content-Range", f"bytes {start}-{end}/{len(body)}"
                    )
                    handler.send_header("ETag", hashlib.sha256(body).hexdigest())
                    handler.end_headers()
                    handler.wfile.write(body[start : start + (end - start + 1) // 2])
                    handler.close_connection = True
                    return
                self.reply(
                    handler,
                    206,
                    body[start : end + 1],
                    {
                        "Content-Range": f"bytes {start}-{end}/{len(body)}",
                        "ETag": hashlib.sha256(body).hexdigest(),
                    },
                )
            case ["runs", run, "metrics", request]:
                self.upload_started.set()
                self.upload_gate.wait(20)
                if self.fail_metrics:
                    self.reply(handler, 422, {"message": "fixture metrics failure"})
                    return
                if request not in self.metric_receipts:
                    self.metrics.extend(
                        (run, SimpleNamespace(**metric)) for metric in data["scalars"]
                    )
                    self.metric_receipts.add(request)
                if self.drop_metric_reply:
                    self.drop_metric_reply = False
                    handler.close_connection = True
                    return
                self.reply(handler, 204)
            case ["uploads", upload]:
                message = BytesParser().parsebytes(
                    f"Content-Type: {handler.headers['Content-Type']}\r\n\r\n".encode()
                    + content
                )
                fields = message.get_payload()
                spec = json.loads(fields[0].get_payload(decode=True))
                body = fields[1].get_payload(decode=True)
                if (
                    len(body) != spec["size"]
                    or hashlib.sha256(body).hexdigest() != spec["sha256"]
                ):
                    self.reply(
                        handler, 400, {"message": "upload has unexpected content"}
                    )
                    return
                self.save(handler, upload, spec, body)
            case _:
                self.reply(handler, 404, {"message": "unknown fixture endpoint"})

    def data(self, handler, run, stream, index):
        with self.lock:
            self.requests[stream].append(
                SimpleNamespace(run_id=run, stream=stream, index=index)
            )
        if self.transient_batches:
            self.transient_batches -= 1
            self.reply(handler, 503, {"message": "temporary failure"})
            return
        if stream == self.fail_stream:
            self.reply(handler, 422, {"message": "fixture data failure"})
            return
        if index >= self.streams[stream]:
            self.reply(handler, 204)
            return
        value = index + 1000 * list(self.streams).index(stream)
        samples = []
        for part in range(0 if self.empty_batch else index % 3 + 1):
            blobs = {
                "payload": b"x" * self.blob_size
                if self.blob_size is not None
                else bytes([part, index % 256]),
                "context": b"arbitrary bytes",
            }
            sample = field(6, f"sample-{value}-{part}".encode()) + field(
                7, json.dumps({"position": value, "label": f"label-{part}"}).encode()
            )
            sample += b"".join(
                field(8, field(1, name.encode()) + field(2, content))
                for name, content in blobs.items()
            )
            samples.append(sample)
        self.reply(handler, 200, batch_bytes(stream, index, samples))

    def save(self, handler, upload, spec, body):
        self.upload_started.set()
        self.upload_gate.wait(20)
        metadata = spec["metadata"]["metadata"]
        if self.fail_save or self.fail_metrics:
            self.reply(handler, 422, {"message": "fixture save failure"})
            return
        with self.lock:
            state = self.uploads.get(upload)
            if state is not None:
                if state != (spec, body):
                    self.reply(
                        handler, 409, {"message": "conflicting retry of upload ID"}
                    )
                    return
            else:
                if spec["metadata"]["kind"] == "asset":
                    parent = self.heads.get(
                        metadata["name"],
                        self.assets.get(metadata["name"], (None, b""))[0],
                    )
                    self.saved.append((SimpleNamespace(**metadata), parent, body))
                    self.heads[metadata["name"]] = metadata["asset_id"]
                else:
                    self.artifacts.append((SimpleNamespace(**metadata), body))
                self.uploads[upload] = (spec, body)
            drop_reply = self.drop_upload_reply
            self.drop_upload_reply = False
        if drop_reply:
            handler.close_connection = True
            return
        self.reply(handler, 200, {"committed": True})

    def close(self):
        self.upload_gate.set()
        self.asset_gate.set()
        self.end_gate.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


class PipelineCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="tlt-", dir="/tmp")
        self.run_id = str(uuid.uuid4())
        self.service = Fixture()
        self.daemon = None

    def tearDown(self):
        self.service.upload_gate.set()
        self.service.asset_gate.set()
        self.service.end_gate.set()
        if self.daemon:
            try:
                self.daemon.close()
            except RuntimeError:
                pass
        self.service.close()
        self.temp.cleanup()

    def start(
        self,
        ranks=1,
        factor=2,
        workers=2,
        transform_fn=transform,
        collate_fn=None,
        performance_metrics=True,
    ):
        self.daemon = tensorlane.init(
            self.run_id,
            transform_fn,
            ranks,
            factor,
            rank=0,
            start_daemon=True,
            num_workers=workers,
            collate_fn=collate_fn,
            performance_metrics=performance_metrics,
            addr=f"localhost:{self.service.port}",
            ipc_dir=self.temp.name,
            timeout=20,
        )
        return self.daemon

    def attach(self, rank=0):
        return tensorlane.init(
            self.run_id, ranks=1, rank=rank, start_daemon=False, ipc_dir=self.temp.name
        )

    def file(self, name="weights", content=b"opaque bytes"):
        path = Path(self.temp.name) / name
        path.write_bytes(content)
        return path

    @staticmethod
    def archive_contents(content):
        with tarfile.open(fileobj=io.BytesIO(content), mode="r:") as archive:
            return {
                item.name: archive.extractfile(item).read()
                for item in archive.getmembers()
                if item.isfile()
            }
