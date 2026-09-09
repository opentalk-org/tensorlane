from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial
import importlib
import multiprocessing
import socket
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import uuid

import grpc
from grpc_tools import protoc
import tensorlane
from fixture_transforms import double, fail, slow, invalid, identify_worker

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


class Fixture(rpc.TensorLaneServicer):
    def __init__(self, count=5, empty=False):
        self.count = count
        self.empty = empty
        self.fail_data = False
        self.requests = []
        self.lock = threading.Lock()
        self.server = grpc.server(ThreadPoolExecutor(max_workers=4))
        rpc.add_TensorLaneServicer_to_server(self, self.server)
        self.port = self.server.add_insecure_port("127.0.0.1:0")
        self.server.start()

    def Init(self, request, context):
        return pb.InitResponse(run_id=request.run_id, train_config="opaque config")

    def Data(self, requests, context):
        for index, request in enumerate(requests):
            with self.lock:
                self.requests.append(request)
            if self.fail_data:
                context.abort(grpc.StatusCode.INTERNAL, "fixture gRPC failure")
            if index >= self.count:
                return
            samples = (
                []
                if self.empty and index == 1
                else [
                    pb.Sample(
                        wave=struct.pack("<hh", index, -index),
                        text=struct.pack("<q", sample),
                        duration=0.5,
                        speaker_id=index,
                        language_id=3,
                    )
                    for sample in range(index % 3 + 1)
                ]
            )
            yield pb.DataResponse(batch=samples)

    def close(self):
        self.server.stop(0).wait()


def read_rank(run_id, rank, root, output):
    try:
        with tensorlane.batches(run_id, rank, ipc_dir=root, timeout=20) as reader:
            result = []
            for batch in reader:
                if any(
                    not sample.wave.is_shared() or not sample.text.is_shared()
                    for sample in batch
                ):
                    raise AssertionError("expected shared CPU tensor storage")
                result.append(
                    [
                        (
                            sample.wave.tolist(),
                            sample.text.tolist(),
                            sample.speaker_id,
                            sample.language_id,
                            sample.duration,
                        )
                        for sample in batch
                    ]
                )
        output.put((rank, result, None))
    except Exception as error:
        output.put((rank, None, repr(error)))


def wait_for(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition timed out")


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="tlt-", dir="/tmp")
        self.run_id = str(uuid.uuid4())
        self.service = Fixture()
        self.daemon = None

    def tearDown(self):
        if self.daemon:
            self.daemon.close()
            self.assertEqual(list(Path(self.temp.name).rglob("*.sock")), [])
        self.service.close()
        self.temp.cleanup()

    def start(self, ranks=2, factor=2, transform=double, workers=1):
        self.daemon = tensorlane.init(
            self.run_id,
            transform,
            ranks,
            factor,
            addr=f"localhost:{self.service.port}",
            ipc_dir=self.temp.name,
            **({"num_workers": workers} if workers is not None else {}),
        )
        return self.daemon

    def test_independent_ranks_attach_before_init(self):
        context = multiprocessing.get_context("spawn")
        output = context.Queue()
        ranks = [
            context.Process(
                target=read_rank, args=(self.run_id, rank, self.temp.name, output)
            )
            for rank in range(2)
        ]
        try:
            for rank in ranks:
                rank.start()
            self.start(workers=3)
            self.assertEqual(self.daemon.train_config, "opaque config")
            results = [output.get(timeout=30) for _ in ranks]
            for rank, batches, error in results:
                self.assertIsNone(error, error)
                expected = list(range(rank, 5, 2))
                self.assertEqual(len(batches), len(expected))
                for index, batch in zip(expected, batches):
                    self.assertEqual(len(batch), index % 3 + 1)
                    for sample_index, sample in enumerate(batch):
                        self.assertEqual(
                            sample,
                            (
                                [index * 2, -index * 2],
                                [sample_index],
                                index,
                                3,
                                0.5,
                            ),
                        )
            for rank in ranks:
                rank.join(5)
                self.assertEqual(rank.exitcode, 0)
        finally:
            for rank in ranks:
                if rank.is_alive():
                    rank.kill()
                rank.join(5)
            output.close()

    def test_prefetch_credits_cover_unconsumed_batches(self):
        self.start(factor=1)
        wait_for(lambda: len(self.service.requests) == 2)
        time.sleep(0.2)
        self.assertEqual(len(self.service.requests), 2)
        with tensorlane.batches(self.run_id, 0, ipc_dir=self.temp.name) as reader:
            self.assertEqual(next(reader).samples[0].speaker_id, 0)
            wait_for(lambda: len(self.service.requests) == 3)
            time.sleep(0.2)
            self.assertEqual(len(self.service.requests), 3)
            self.daemon.close()

    def test_global_budget_allows_one_rank_to_release_any_slot(self):
        self.service.count = 20
        self.start(factor=1, workers=3)
        wait_for(lambda: len(self.service.requests) == 2)
        with tensorlane.batches(self.run_id, 0, ipc_dir=self.temp.name) as reader:
            self.assertEqual(next(reader).samples[0].speaker_id, 0)
            wait_for(lambda: len(self.service.requests) == 3)
            self.assertEqual(next(reader).samples[0].speaker_id, 2)
            wait_for(lambda: len(self.service.requests) == 4)
            time.sleep(0.2)
            self.assertEqual(len(self.service.requests), 4)
            self.daemon.close()

    def test_shutdown_cancels_sem_wait_and_unlinks_semaphore(self):
        self.service.count = 20
        self.start(ranks=1, factor=1)
        wait_for(lambda: len(self.service.requests) == 1)
        name = next(Path(self.temp.name).rglob("semaphore")).read_text()
        started = time.monotonic()
        self.daemon.close()
        self.assertLess(time.monotonic() - started, 8)
        with self.assertRaises(RuntimeError):
            tensorlane._native.Semaphore(name)

    def test_duplicate_init_and_rank_are_rejected(self):
        self.start()
        with self.assertRaisesRegex(Exception, "already active"):
            tensorlane.init(self.run_id, double, 2, ipc_dir=self.temp.name)
        reader = tensorlane.batches(self.run_id, 0, ipc_dir=self.temp.name)
        try:
            with self.assertRaisesRegex(RuntimeError, "already connected"):
                tensorlane.batches(self.run_id, 0, ipc_dir=self.temp.name)
            with self.assertRaisesRegex(RuntimeError, "invalid"):
                tensorlane.batches(self.run_id, 2, ipc_dir=self.temp.name)
        finally:
            self.daemon.close()
            reader.close()

    def test_transform_failure_reaches_rank(self):
        with self.assertRaisesRegex(RuntimeError, "tensorlane-transform-0 exited"):
            self.start(transform=fail)
            with tensorlane.batches(self.run_id, 0, ipc_dir=self.temp.name) as reader:
                next(reader)

    def test_invalid_transform_output_reaches_rank(self):
        with self.assertRaisesRegex(RuntimeError, "tensorlane-transform-0 exited"):
            self.start(transform=invalid)
            with tensorlane.batches(self.run_id, 0, ipc_dir=self.temp.name) as reader:
                next(reader)

    def test_startup_timeout_and_callable_validation(self):
        with self.assertRaises(TimeoutError):
            tensorlane.batches(self.run_id, 0, ipc_dir=self.temp.name, timeout=0.1)
        with self.assertRaises(AttributeError):
            self.start(transform=lambda wave: wave)

    def test_close_interrupts_busy_transform_and_is_idempotent(self):
        self.start(transform=slow)
        wait_for(lambda: len(self.service.requests) > 0)
        started = time.monotonic()
        self.daemon.close()
        self.assertLess(time.monotonic() - started, 8)
        self.daemon.close()

    def test_new_init_reuses_cleaned_directory(self):
        self.start()
        self.daemon.close()
        self.start()
        self.daemon.close()

    def test_empty_stream(self):
        self.service.count = 0
        self.start()
        for rank in range(2):
            with tensorlane.batches(
                self.run_id, rank, ipc_dir=self.temp.name
            ) as reader:
                self.assertEqual(list(reader), [])

    def test_empty_batch_fails_without_leaking_prefetch_slots(self):
        self.service.empty = True
        with self.assertRaisesRegex(RuntimeError, "empty batches are unsupported"):
            self.start(ranks=1, factor=2)
            with tensorlane.batches(self.run_id, 0, ipc_dir=self.temp.name) as reader:
                list(reader)

    def test_grpc_error_reaches_rank_without_retry(self):
        self.service.fail_data = True
        with self.assertRaisesRegex(RuntimeError, "fixture gRPC failure"):
            self.start()
            with tensorlane.batches(self.run_id, 0, ipc_dir=self.temp.name) as reader:
                next(reader)
        self.assertEqual(len(self.service.requests), 1)

    def test_collater_crash_wakes_reader(self):
        self.start(transform=slow)
        reader = tensorlane.batches(self.run_id, 0, ipc_dir=self.temp.name)
        try:
            collater = next(
                process
                for process in self.daemon._processes
                if process.name == "tensorlane-collate"
            )
            collater.kill()
            with self.assertRaisesRegex(RuntimeError, "disconnected|exited"):
                next(reader)
        finally:
            reader.close()

    def test_rank_disconnect_fails_pipeline(self):
        self.service.count = 100
        self.start()
        reader = tensorlane.batches(self.run_id, 0, ipc_dir=self.temp.name)
        reader.close()
        with self.assertRaises(RuntimeError):
            with tensorlane.batches(self.run_id, 1, ipc_dir=self.temp.name) as other:
                list(other)

    def test_collator_crash_stops_daemon_before_ranks_attach(self):
        self.service.count = 20
        self.start(ranks=1, factor=1)
        wait_for(lambda: len(self.service.requests) == 1)
        collator = next(
            process
            for process in self.daemon._processes
            if process.name == "tensorlane-collate"
        )
        collator.kill()
        wait_for(self.daemon._stopped.is_set)
        with self.assertRaisesRegex(RuntimeError, "tensorlane-collate exited"):
            tensorlane.batches(self.run_id, 0, ipc_dir=self.temp.name)
        self.assertEqual(len(self.service.requests), 1)

    def test_worker_connection_is_ready_without_control_messages(self):
        from tensorlane.client import Daemon, _root

        root = _root(self.run_id, self.temp.name)
        self.daemon = Daemon(
            tensorlane._native.Daemon(
                self.run_id, f"localhost:{self.service.port}", root, 1, 1, 1
            )
        )
        with socket.socket(socket.AF_UNIX) as connection:
            connection.connect(str(root / "work.sock"))
            receiver = tensorlane._native.Listener(connection)
            try:
                self.assertFalse(hasattr(receiver, "ready"))
                self.assertFalse(hasattr(receiver, "error"))
                message = receiver.recv()
                self.assertEqual(message["kind"], "sample")
                self.assertEqual(message["batch"], (0, 1))
                self.assertEqual(message["index"], 0)
            finally:
                receiver.close()

    def test_transform_from_callers_search_path(self):
        module_path = Path(self.temp.name) / "custom_transform.py"
        module_path.write_text("def transform(wave):\n    return wave.clone()\n")
        sys.path.insert(0, self.temp.name)
        try:
            transform = importlib.import_module("custom_transform").transform
            self.service.count = 1
            self.start(ranks=1, transform=transform)
            with tensorlane.batches(self.run_id, 0, ipc_dir=self.temp.name) as reader:
                batch = next(reader)
                self.assertEqual(batch.samples[0].wave.tolist(), [0, 0])
                self.assertEqual(list(reader), [])
        finally:
            sys.path.remove(self.temp.name)
            sys.modules.pop("custom_transform", None)

    def test_transform_defined_in_main_script(self):
        self.service.count = 1
        script = Path(self.temp.name) / "train.py"
        script.write_text(
            "import tensorlane\n"
            "def transform(wave):\n"
            "    return wave.float() + 7\n"
            "if __name__ == '__main__':\n"
            f"    with tensorlane.init({self.run_id!r}, transform, 1, "
            f"addr='localhost:{self.service.port}', ipc_dir={self.temp.name!r}):\n"
            f"        with tensorlane.batches({self.run_id!r}, 0, ipc_dir={self.temp.name!r}) as reader:\n"
            "            batches = list(reader)\n"
            "            assert len(batches) == 1\n"
            "            assert batches[0].samples[0].wave.tolist() == [7, 7]\n"
        )
        subprocess.run([sys.executable, str(script)], check=True, timeout=40)

    def test_multiple_workers_preserve_order(self):
        self.service.count = 6
        self.start(ranks=1, factor=4, transform=identify_worker, workers=3)
        with tensorlane.batches(self.run_id, 0, ipc_dir=self.temp.name) as reader:
            batches = list(reader)
        self.assertEqual(
            [[int(sample.wave[0]) for sample in batch] for batch in batches],
            [[batch_id] * (batch_id % 3 + 1) for batch_id in range(6)],
        )
        self.assertEqual(
            len({int(sample.wave[1]) for batch in batches for sample in batch}), 3
        )

    def test_default_five_workers(self):
        self.service.count = 6
        self.start(ranks=1, factor=4, transform=identify_worker, workers=None)
        workers = [
            process
            for process in self.daemon._processes
            if process.name.startswith("tensorlane-transform-")
        ]
        self.assertEqual(len(workers), 5)
        self.assertEqual(
            {path.name for path in Path(self.temp.name).rglob("*.sock")},
            {"work.sock"},
        )
        with tensorlane.batches(self.run_id, 0, ipc_dir=self.temp.name) as reader:
            batches = list(reader)
        self.assertEqual(
            len({int(sample.wave[1]) for batch in batches for sample in batch}), 5
        )

    def test_invalid_worker_count(self):
        for workers in (0, -1, 1.5, True):
            with (
                self.subTest(workers=workers),
                self.assertRaisesRegex(ValueError, "num_workers"),
            ):
                self.start(workers=workers)

    def test_transformation_worker_crash_wakes_reader(self):
        self.start(transform=slow, workers=3)
        with tensorlane.batches(self.run_id, 0, ipc_dir=self.temp.name) as reader:
            self.daemon._processes[1].kill()
            with self.assertRaisesRegex(RuntimeError, "disconnected|exited"):
                next(reader)

    def test_empty_stream_with_multiple_workers(self):
        self.service.count = 0
        self.start(ranks=1, workers=3)
        with tensorlane.batches(self.run_id, 0, ipc_dir=self.temp.name) as reader:
            self.assertEqual(list(reader), [])

    def test_partial_transform(self):
        self.service.count = 1
        self.start(ranks=1, transform=partial(double))
        with tensorlane.batches(self.run_id, 0, ipc_dir=self.temp.name) as reader:
            self.assertEqual(len(list(reader)), 1)

    def test_second_process_start_failure_cleans_up(self):
        original = multiprocessing.process.BaseProcess.start
        started = []

        def start(process):
            if started:
                raise OSError("fixture spawn failure")
            original(process)
            started.append(process)

        with patch.object(multiprocessing.process.BaseProcess, "start", start):
            with self.assertRaisesRegex(OSError, "fixture spawn failure"):
                self.start()
        self.assertEqual(len(started), 1)
        self.assertTrue(started[0]._closed)
        self.start()


if __name__ == "__main__":
    unittest.main()
