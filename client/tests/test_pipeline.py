from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial
import importlib
import io
import json
import multiprocessing
from pathlib import Path
from unittest.mock import patch
import sys
import tarfile
import tempfile
import threading
import time
import unittest
import uuid

import grpc
from grpc_tools import protoc
import tensorlane
from fixture_transforms import (
    transform,
    collate,
    fail,
    fail_collate,
    slow,
    identify_worker,
    invalid,
)

GENERATED = tempfile.TemporaryDirectory(prefix="tl-proto-")
PROTO = Path(__file__).resolve().parents[2] / "proto"
protoc.main(
    [
        "grpc_tools.protoc",
        f"-I{PROTO}",
        f"--python_out={GENERATED.name}",
        f"--grpc_python_out={GENERATED.name}",
        str(PROTO / "tensorlane.proto"),
    ]
)
sys.path.insert(0, GENERATED.name)
pb = importlib.import_module("tensorlane_pb2")
rpc = importlib.import_module("tensorlane_pb2_grpc")


def wait_for(check, timeout=15):
    deadline = time.monotonic() + timeout
    while not check():
        if time.monotonic() >= deadline:
            raise TimeoutError("condition did not become true")
        time.sleep(0.02)


class Fixture(rpc.TensorLaneServicer):
    def __init__(self):
        self.streams = {"training": 5, "validation": 3, "evaluation": 2}
        self.config = {
            "queries": {name: "SELECT ..." for name in self.streams},
            "optimizer": {"lr": 0.1},
            "assets": {},
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
        self.fail_init = False
        self.fail_asset = False
        self.fail_save = False
        self.fail_metrics = False
        self.fail_stream = None
        self.empty_batch = False
        self.blob_size = None
        self.returned_run_id = None
        self.upload_started = threading.Event()
        self.upload_gate = threading.Event()
        self.upload_gate.set()
        self.end_gate = threading.Event()
        self.end_gate.set()
        self.lock = threading.Lock()
        self.server = grpc.server(
            ThreadPoolExecutor(max_workers=12),
            options=[("grpc.max_send_message_length", 80 * 1024 * 1024)],
        )
        rpc.add_TensorLaneServicer_to_server(self, self.server)
        self.port = self.server.add_insecure_port("127.0.0.1:0")
        self.server.start()

    def Init(self, request, context):
        self.init_requests.append(request)
        if self.fail_init:
            context.abort(grpc.StatusCode.INTERNAL, "fixture init failure")
        return pb.InitResponse(
            run_id=self.returned_run_id or request.run_id,
            config=json.dumps(self.config),
            assets=list(self.assets),
            streams=list(self.streams),
        )

    def End(self, request, context):
        self.end_gate.wait(20)
        self.end_requests.append(request)
        return pb.EndResponse()

    def Asset(self, request, context):
        self.asset_requests.append(request)
        asset_id, content = self.assets[request.name]
        yield pb.AssetResponse(
            metadata=pb.AssetMetadata(
                asset_id=asset_id, metadata_json="{}", kind="file", asset_type="generic"
            )
        )
        for offset in range(0, len(content), 127):
            if self.fail_asset:
                context.abort(grpc.StatusCode.INTERNAL, "fixture asset failure")
            yield pb.AssetResponse(chunk=content[offset : offset + 127])

    def Data(self, requests, context):
        for index, request in enumerate(requests):
            with self.lock:
                self.requests[request.stream].append(request)
            if request.stream == self.fail_stream:
                context.abort(grpc.StatusCode.INTERNAL, "fixture data failure")
            if index >= self.streams[request.stream]:
                return
            value = index + 1000 * list(self.streams).index(request.stream)
            samples = (
                []
                if self.empty_batch
                else [
                    pb.Sample(
                        sample_id=f"sample-{value}-{part}",
                        metadata_json=json.dumps(
                            {"position": value, "label": f"label-{part}"}
                        ),
                        blobs={
                            "payload": b"x" * self.blob_size
                            if self.blob_size is not None
                            else bytes([part, index % 256]),
                            "context": b"arbitrary bytes",
                        },
                    )
                    for part in range(index % 3 + 1)
                ]
            )
            yield pb.DataResponse(
                batch=samples,
                stream=request.stream,
                batch_id=index,
                query_batch_idx=index * 4 + 7,
            )

    def SaveAsset(self, requests, context):
        first = next(requests)
        if first.WhichOneof("payload") != "metadata":
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "missing metadata")
        self.upload_started.set()
        self.upload_gate.wait(20)
        metadata = first.metadata
        content = b"".join(message.chunk for message in requests)
        if self.fail_save:
            context.abort(grpc.StatusCode.INTERNAL, "fixture save failure")
        with self.lock:
            parent = self.heads.get(
                metadata.name, self.assets.get(metadata.name, (None, b""))[0]
            )
            self.saved.append((metadata, parent, content))
            self.heads[metadata.name] = metadata.asset_id
        return pb.SaveAssetResponse(asset_id=metadata.asset_id)

    def Metrics(self, requests, context):
        iterator = iter(requests)
        first = next(iterator)
        self.upload_started.set()
        self.upload_gate.wait(20)
        if self.fail_metrics:
            context.abort(grpc.StatusCode.INTERNAL, "fixture metrics failure")
        pending = None
        chunks = []
        received = 0
        for message in iterator:
            kind = message.WhichOneof("payload")
            if kind == "metric":
                self.metrics.append((first.metadata.run_id, message.metric))
            elif kind == "artifact":
                pending = message.artifact
                chunks = []
            elif kind == "artifact_chunk":
                chunks.append(message.artifact_chunk.data)
                received += len(message.artifact_chunk.data)
                if sum(map(len, chunks)) == pending.size_bytes:
                    self.artifacts.append((pending, b"".join(chunks)))
                    pending = None
        return pb.MetricsResponse(
            metrics_received=len(self.metrics),
            artifacts_received=len(self.artifacts),
            artifact_bytes_received=received,
        )

    def close(self):
        self.upload_gate.set()
        self.end_gate.set()
        self.server.stop(0).wait()


def read_rank(run_id, rank, root, output, stream="training"):
    try:
        with tensorlane.init(
            run_id, ranks=2, rank=rank, start_daemon=False, ipc_dir=root, timeout=20
        ) as lane:
            with lane.batches(stream, timeout=20) as batches:
                result = [
                    (batch.batch_id, batch.samples[0]["value"].tolist())
                    for batch in batches
                ]
            output.put((rank, result, None))
    except Exception as error:
        output.put((rank, None, str(error)))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="tlt-", dir="/tmp")
        self.run_id = str(uuid.uuid4())
        self.service = Fixture()
        self.daemon = None

    def tearDown(self):
        self.service.upload_gate.set()
        self.service.end_gate.set()
        if self.daemon:
            try:
                self.daemon.close()
            except RuntimeError:
                pass
        self.service.close()
        self.temp.cleanup()

    def start(
        self, ranks=1, factor=2, workers=2, transform_fn=transform, collate_fn=None
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

    def test_config_callbacks_and_three_streams(self):
        self.start(collate_fn={"training": collate})
        self.assertEqual(self.daemon.config, self.service.config)
        self.assertEqual(self.daemon.streams, tuple(self.service.streams))
        for stream, count in self.service.streams.items():
            with self.daemon.batches(stream) as reader:
                batches = list(reader)
            self.assertEqual(len(batches), count)
            for index, batch in enumerate(batches):
                self.assertEqual(
                    (batch.stream, batch.batch_id, batch.query_batch_idx),
                    (stream, index, index * 4 + 7),
                )
                self.assertEqual(len(batch), index % 3 + 1)
                if stream == "training":
                    self.assertEqual(batch.data["values"].shape[0], len(batch))
                else:
                    self.assertEqual(batch.data, batch.samples)

    def test_runtime_parameters_from_json_are_shared_with_followers(self):
        self.service.config.update(ranks=2, num_workers=2, prefetch_factor=1)
        self.daemon = tensorlane.init(
            self.run_id, addr=f"localhost:{self.service.port}", ipc_dir=self.temp.name
        )
        self.assertEqual(
            (
                self.daemon.rank,
                self.daemon.ranks,
                self.daemon.num_workers,
                self.daemon.prefetch_factor,
            ),
            (0, 2, 2, 1),
        )
        self.assertEqual(len(self.daemon._processes), 3)
        with tensorlane.init(self.run_id, rank=1, ipc_dir=self.temp.name) as follower:
            self.assertEqual(
                (
                    follower.rank,
                    follower.ranks,
                    follower.num_workers,
                    follower.prefetch_factor,
                ),
                (1, 2, 2, 1),
            )

    def test_explicit_runtime_parameters_override_json(self):
        self.service.config.update(ranks=2, num_workers=2, prefetch_factor=1)
        self.start(workers=1)
        self.assertEqual(
            (self.daemon.ranks, self.daemon.num_workers, self.daemon.prefetch_factor),
            (1, 1, 2),
        )
        self.assertEqual(self.daemon.config, self.service.config)

    def test_existing_run_id_from_environment_needs_no_argument(self):
        self.service.config.update(num_workers=1)
        with patch.dict("os.environ", {"TENSORLANE_RUN_ID": self.run_id}):
            self.daemon = tensorlane.init(
                addr=f"localhost:{self.service.port}", ipc_dir=self.temp.name
            )
        self.assertEqual(self.daemon.run_id, self.run_id)

    def test_raw_samples_and_mapping_defaults(self):
        self.start(transform_fn={"training": transform})
        with self.daemon.batches("evaluation") as reader:
            sample = next(reader).samples[0]
            self.assertIsInstance(sample, tensorlane.RawSample)
            self.assertEqual(sample.blobs["context"], b"arbitrary bytes")
            self.assertEqual(sample.metadata["position"], 2000)

    def test_other_streams_drain_while_training_is_full(self):
        self.start(factor=1, workers=3)
        wait_for(lambda: len(self.service.requests["training"]) == 1)
        for name in ("validation", "evaluation"):
            with self.daemon.batches(name) as reader:
                self.assertEqual(len(list(reader)), self.service.streams[name])
        self.assertEqual(len(self.service.requests["training"]), 1)
        with self.daemon.batches() as reader:
            self.assertEqual(len(list(reader)), 5)

    def test_credit_budget_covers_unconsumed_batches(self):
        self.start(factor=1)
        wait_for(lambda: len(self.service.requests["training"]) == 1)
        time.sleep(0.1)
        self.assertEqual(len(self.service.requests["training"]), 1)
        with self.daemon.batches() as reader:
            next(reader)
            wait_for(lambda: len(self.service.requests["training"]) == 2)
            time.sleep(0.1)
            self.assertEqual(len(self.service.requests["training"]), 2)

    def test_multiple_workers_preserve_order(self):
        self.service.streams["training"] = 10
        self.start(factor=4, workers=3, transform_fn=identify_worker)
        with self.daemon.batches() as reader:
            batches = list(reader)
        self.assertEqual(
            [batch.samples[0]["value"][0].item() for batch in batches], list(range(10))
        )
        self.assertEqual(
            len({sample["value"][1].item() for batch in batches for sample in batch}), 3
        )

    def test_independent_ranks_attach_before_owner(self):
        context = multiprocessing.get_context("spawn")
        output = context.Queue()
        processes = [
            context.Process(
                target=read_rank, args=(self.run_id, rank, self.temp.name, output)
            )
            for rank in range(2)
        ]
        try:
            for process in processes:
                process.start()
            self.start(ranks=2, workers=3)
            for _ in processes:
                rank, batches, error = output.get(timeout=30)
                self.assertIsNone(error)
                self.assertEqual(
                    [batch[0] for batch in batches], list(range(rank, 5, 2))
                )
            for process in processes:
                process.join(10)
                self.assertEqual(process.exitcode, 0)
        finally:
            for process in processes:
                if process.is_alive():
                    process.kill()
                process.join()
            output.close()

    def test_assets_are_prefetched_once_and_shared(self):
        asset_id = str(uuid.uuid4())
        self.service.assets = {"model": (asset_id, b"weights"), "empty": (None, b"")}
        self.start()
        with self.attach() as follower:
            self.assertEqual(follower.asset("model"), self.daemon.asset("model"))
            self.assertEqual(follower.asset("model").read_bytes(), b"weights")
            self.assertEqual(follower.asset_metadata["model"]["asset_id"], asset_id)
        self.assertEqual(len(self.service.asset_requests), 2)

    def test_saved_files_directories_and_automatic_lineage(self):
        source_id = str(uuid.uuid4())
        self.service.assets = {"model": (source_id, b"input")}
        self.start()
        path = self.file()
        a = self.daemon.save_asset(
            "model", path, step=1, kind="checkpoint", metadata={"dataset_offset": 6}
        )
        b = self.daemon.save_asset("model", path, step=2)
        directory = Path(self.temp.name) / "bundle"
        directory.mkdir()
        (directory / "config.json").write_text("{}")
        other = self.daemon.save_asset("other", directory)
        self.daemon.flush()
        self.assertEqual([row[0].asset_id for row in self.service.saved], [a, b, other])
        self.assertEqual([row[1] for row in self.service.saved], [source_id, a, None])
        self.assertEqual(
            json.loads(self.service.saved[0][0].metadata_json), {"dataset_offset": 6}
        )
        self.assertEqual(
            self.archive_contents(self.service.saved[0][2]),
            {"weights": b"opaque bytes"},
        )
        self.assertEqual(
            self.archive_contents(self.service.saved[2][2]),
            {"bundle/config.json": b"{}"},
        )

    def test_saves_queue_promptly_and_flush_waits_for_commit(self):
        self.start()
        path = self.file()
        self.service.upload_gate.clear()
        started = time.monotonic()
        self.daemon.save_asset("model", path)
        self.assertLess(time.monotonic() - started, 1)
        self.assertTrue(self.service.upload_started.wait(5))
        with self.daemon.batches() as reader:
            self.assertEqual(len(list(reader)), 5)
        with ThreadPoolExecutor() as executor:
            flushed = executor.submit(self.daemon.flush)
            time.sleep(0.1)
            self.assertFalse(flushed.done())
            self.assertEqual(self.service.saved, [])
            self.service.upload_gate.set()
            flushed.result(timeout=15)
        self.assertEqual(len(self.service.saved), 1)

    def test_save_failure_and_missing_source_reach_flush(self):
        self.start()
        with self.attach() as follower:
            follower.save_asset("model", Path(self.temp.name) / "missing")
            with self.assertRaisesRegex(RuntimeError, "opening asset"):
                follower.flush()
        self.service.fail_save = True
        with self.attach() as follower:
            follower.save_asset("model", self.file())
            with self.assertRaisesRegex(RuntimeError, "fixture save failure"):
                follower.flush()

    def test_metrics_and_artifacts_still_work(self):
        self.start()
        self.daemon.metric(3, "loss", 0.25)
        self.daemon.metric_artifact(
            3, self.file("report.json", b"{}"), "report", "application/json"
        )
        self.daemon.flush()
        self.assertEqual(self.service.metrics[0][1].name, "loss")
        self.assertEqual(
            self.archive_contents(self.service.artifacts[0][1]), {"report.json": b"{}"}
        )

    def test_shutdown_drains_save_before_end_and_cleans_resources(self):
        self.start()
        self.service.upload_gate.clear()
        self.daemon.save_asset("model", self.file())
        self.assertTrue(self.service.upload_started.wait(5))
        with ThreadPoolExecutor() as executor:
            closed = executor.submit(self.daemon.close)
            time.sleep(0.1)
            self.assertFalse(closed.done())
            self.assertEqual(self.service.end_requests, [])
            self.service.upload_gate.set()
            closed.result(timeout=15)
        self.assertEqual(len(self.service.saved), 1)
        self.assertEqual(len(self.service.end_requests), 1)
        self.assertEqual({path.name for path in self.daemon._root.iterdir()}, {"lock"})

    def test_follower_close_does_not_end_run(self):
        self.start()
        follower = self.attach()
        follower.close()
        self.assertEqual(self.service.end_requests, [])
        self.daemon.close()
        self.assertEqual(len(self.service.end_requests), 1)

    def test_empty_streams_and_partial_transform(self):
        self.service.streams = {name: 0 for name in self.service.streams}
        self.start(transform_fn=partial(transform))
        for stream in self.daemon.streams:
            with self.daemon.batches(stream) as reader:
                self.assertEqual(list(reader), [])

    def test_unknown_stream_and_duplicate_reader_are_rejected(self):
        self.start()
        with self.assertRaisesRegex(ValueError, "unknown stream"):
            self.daemon.batches("missing")
        reader = self.daemon.batches()
        try:
            with self.assertRaisesRegex(RuntimeError, "already connected"):
                self.daemon.batches()
        finally:
            self.daemon.close()
            reader.close()

    def test_duplicate_daemon_is_rejected(self):
        self.start()
        with self.assertRaisesRegex(RuntimeError, "already active"):
            tensorlane.init(
                self.run_id, ranks=1, rank=0, start_daemon=True, ipc_dir=self.temp.name
            )

    def test_callback_and_stream_errors_reach_readers(self):
        for transform_fn, collate_fn, error in [
            (fail, None, "transform"),
            (invalid, None, "transform"),
            (transform, fail_collate, "collate"),
        ]:
            with (
                self.subTest(error=error),
                self.assertRaisesRegex(RuntimeError, "exited|disconnected"),
            ):
                self.start(transform_fn=transform_fn, collate_fn=collate_fn)
                with self.daemon.batches() as reader:
                    next(reader)
            if self.daemon:
                self.daemon.close()
        self.service.fail_stream = "evaluation"
        with self.assertRaisesRegex(RuntimeError, "fixture data failure"):
            self.start()
            with self.daemon.batches("evaluation") as reader:
                next(reader)

    def test_worker_and_collator_crashes_wake_readers(self):
        for name in ("tensorlane-transform-0", "tensorlane-collate"):
            self.start(transform_fn=slow)
            with self.daemon.batches() as reader:
                next(
                    process
                    for process in self.daemon._processes
                    if process.name == name
                ).kill()
                with self.assertRaisesRegex(RuntimeError, "exited|disconnected"):
                    next(reader)
            self.daemon.close()

    def test_asset_and_init_failures_clean_up(self):
        self.service.assets = {"model": (None, b"weights")}
        self.service.fail_asset = True
        with self.assertRaisesRegex(RuntimeError, "fixture asset failure"):
            self.start()
        self.assertEqual(len(self.service.end_requests), 1)
        self.assertFalse(list(Path(self.temp.name).rglob("init.json")))
        self.service.fail_asset = False
        self.service.fail_init = True
        with self.assertRaisesRegex(RuntimeError, "fixture init failure"):
            self.start()
        self.assertEqual(len(self.service.end_requests), 1)

    def test_invalid_worker_count_and_callback_mapping(self):
        for workers in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                self.start(workers=workers)
        with self.assertRaisesRegex(ValueError, "unknown stream"):
            self.start(transform_fn={"missing": transform})
        with self.assertRaises(TypeError):
            self.start(transform_fn="invalid")

    def test_follower_callback_names_are_validated(self):
        self.start()
        for argument in ("transform", "collate_fn"):
            with self.assertRaisesRegex(ValueError, "unknown stream"):
                tensorlane.init(
                    self.run_id,
                    rank=0,
                    start_daemon=False,
                    ipc_dir=self.temp.name,
                    **{argument: {"missing": transform}},
                )

    def test_flush_timeout_closes_connection(self):
        self.start()
        self.service.upload_gate.clear()
        self.daemon.save_asset("model", self.file())
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            self.daemon.flush(timeout=0.05)
        self.service.upload_gate.set()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.daemon.metric(1, "after", 1)

    def test_cleaned_directory_can_be_reused(self):
        self.start()
        self.daemon.close()
        self.start()
        self.daemon.close()

    def test_large_raw_payload_and_empty_batch_failure(self):
        self.service.streams = {"training": 1, "validation": 0, "evaluation": 0}
        self.service.blob_size = 9 * 1024 * 1024
        self.start(transform_fn=None)
        with self.daemon.batches() as reader:
            self.assertEqual(
                len(next(reader).samples[0].blobs["payload"]), self.service.blob_size
            )
            self.assertEqual(list(reader), [])
        self.daemon.close()
        self.service.empty_batch = True
        with self.assertRaisesRegex(RuntimeError, "empty batches"):
            self.start()
            with self.daemon.batches() as reader:
                next(reader)

    def test_nonzero_owner_and_returned_run_id(self):
        self.service.returned_run_id = str(uuid.uuid4())
        self.daemon = tensorlane.init(
            self.run_id,
            ranks=2,
            rank=1,
            start_daemon=True,
            num_workers=1,
            addr=f"localhost:{self.service.port}",
            ipc_dir=self.temp.name,
        )
        with self.daemon.batches() as reader:
            self.assertEqual([batch.batch_id for batch in reader], [1, 3])
        self.assertEqual(self.daemon.run_id, self.service.returned_run_id)
        self.daemon.close()
        self.daemon.close()
        self.assertEqual(
            [message.run_id for message in self.service.end_requests],
            [self.service.returned_run_id],
        )

    def test_close_surfaces_upload_failure_and_follower_can_close_after_owner(self):
        self.start()
        follower = self.attach()
        follower.save_asset("other", self.file())
        follower.flush()
        self.service.fail_save = True
        self.daemon.save_asset("model", self.file())
        with self.assertRaisesRegex(RuntimeError, "fixture save failure"):
            self.daemon.close()
        follower.close()
        self.assertEqual({path.name for path in self.daemon._root.iterdir()}, {"lock"})
        self.assertEqual(len(self.service.end_requests), 1)

    def test_second_process_start_failure_does_not_publish_readiness(self):
        original = multiprocessing.process.BaseProcess.start
        count = 0

        def start(process):
            nonlocal count
            count += 1
            if count == 2:
                raise RuntimeError("injected start failure")
            original(process)

        with patch("multiprocessing.process.BaseProcess.start", start):
            with self.assertRaisesRegex(RuntimeError, "injected start failure"):
                self.start()
        self.assertFalse(list(Path(self.temp.name).rglob("init.json")))
        self.assertFalse(list(Path(self.temp.name).rglob("work.sock")))
        self.assertEqual(len(self.service.end_requests), 1)

    def test_default_five_workers_and_busy_close(self):
        self.daemon = tensorlane.init(
            self.run_id,
            slow,
            rank=0,
            start_daemon=True,
            addr=f"localhost:{self.service.port}",
            ipc_dir=self.temp.name,
        )
        self.assertEqual(len(self.daemon._processes), 6)
        with self.daemon.batches() as reader:
            started = time.monotonic()
            self.daemon.close()
            self.assertLess(time.monotonic() - started, 10)
            with self.assertRaises((StopIteration, RuntimeError)):
                next(reader)


if __name__ == "__main__":
    unittest.main()
