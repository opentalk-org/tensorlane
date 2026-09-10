from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial
import importlib
import io
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import struct
import subprocess
import sys
import tarfile
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
        self.init_requests = []
        self.returned_run_id = None
        self.train_config = "opaque config"
        self.assets = {}
        self.asset_requests = []
        self.fail_asset = False
        self.validation_count = 0
        self.validation_requests = []
        self.fail_validation = False
        self.lock = threading.Lock()
        self.server = grpc.server(ThreadPoolExecutor(max_workers=4))
        rpc.add_TensorLaneServicer_to_server(self, self.server)
        self.port = self.server.add_insecure_port("127.0.0.1:0")
        self.server.start()

    def Init(self, request, context):
        with self.lock:
            self.init_requests.append(request)
        return pb.InitResponse(
            run_id=self.returned_run_id or request.run_id,
            train_config=self.train_config,
            assets=list(self.assets),
        )

    def Asset(self, request, context):
        with self.lock:
            self.asset_requests.append(request)
        entrypoint, data = self.assets[request.name]
        yield pb.AssetResponse(metadata=pb.AssetMetadata(entrypoint=entrypoint))
        for offset in range(0, len(data), 127):
            if self.fail_asset and offset > 0:
                context.abort(grpc.StatusCode.INTERNAL, "fixture asset failure")
            yield pb.AssetResponse(chunk=data[offset : offset + 127])

    def Data(self, requests, context):
        for index, request in enumerate(requests):
            validation = request.split == pb.VALIDATION
            with self.lock:
                (self.validation_requests if validation else self.requests).append(
                    request
                )
            if self.fail_validation if validation else self.fail_data:
                context.abort(grpc.StatusCode.INTERNAL, "fixture gRPC failure")
            if index >= (self.validation_count if validation else self.count):
                return
            value = index + (1000 if validation else 0)
            samples = (
                []
                if self.empty and index == 1
                else [
                    pb.Sample(
                        wave=struct.pack("<hh", value, -value),
                        text=struct.pack("<q", sample),
                        duration=0.5,
                        speaker_id=value,
                        language_id=3,
                    )
                    for sample in range(index % 3 + 1)
                ]
            )
            yield pb.DataResponse(batch=samples)

    def close(self):
        self.server.stop(0).wait()


def read_rank(run_id, rank, root, output, validation=False):
    try:
        with (
            tensorlane.init(
                run_id,
                double,
                2,
                rank=rank,
                start_daemon=False,
                ipc_dir=root,
                timeout=20,
            ) as lane,
            lane.batches(validation=validation, timeout=20) as reader,
        ):
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


def initialize_rank(run_id, rank, addr, root, output, finished):
    try:
        with tensorlane.init(
            run_id,
            double,
            2,
            rank=rank,
            start_daemon=rank == 1,
            addr=addr,
            ipc_dir=root,
            num_workers=1,
            timeout=20,
        ) as lane:
            output.put((lane.rank, lane.run_id, lane.train_config, None))
            if not finished.wait(20):
                raise TimeoutError("test did not release ranks")
    except Exception as error:
        output.put((rank, None, None, repr(error)))


class PipelineTests(unittest.TestCase):
    @staticmethod
    def archive(files):
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode="w") as archive:
            for name, content in files.items():
                info = tarfile.TarInfo(name)
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
        return output.getvalue()

    def test_assets_are_prefetched_once_and_shared_with_followers(self):
        self.service.returned_run_id = str(uuid.uuid4())
        self.service.assets = {
            "asr": (
                "../ignored-entrypoint.pth",
                b"synthetic asset asr\n" * 1024,
            ),
            "checkpoint": (
                None,
                self.archive({"config.yml": b"config", "model.pth": b"model"}),
            ),
            "binary": (None, bytes(range(256)) * 5),
            "empty": (None, b""),
        }
        original = multiprocessing.process.BaseProcess.start

        def start(process):
            self.assertEqual(len(self.service.asset_requests), 4)
            self.assertEqual(len(list(Path(self.temp.name).rglob("assets/*/data"))), 4)
            original(process)

        with patch.object(multiprocessing.process.BaseProcess, "start", start):
            self.start()
        with self.attach(1) as follower:
            for lane in (self.daemon, follower):
                for name, (_, data) in self.service.assets.items():
                    self.assertIsInstance(lane.asset(name), Path)
                    self.assertTrue(lane.asset(name).is_file())
                    self.assertEqual(lane.asset(name).read_bytes(), data)
                    self.assertEqual(lane.asset(name), self.daemon.asset(name))
                with self.assertRaises(KeyError):
                    lane.asset("missing")
        self.assertEqual(len(self.service.asset_requests), 4)
        self.assertTrue(
            all(
                request.run_id == self.service.returned_run_id
                for request in self.service.asset_requests
            )
        )
        self.assertFalse(list(self.daemon._root.rglob("*.part")))
        self.assertFalse(list(self.daemon._root.rglob("model.pth")))
        self.daemon.close()
        self.assertEqual({path.name for path in self.daemon._root.iterdir()}, {"lock"})

    def test_asset_failure_does_not_publish_readiness_and_cleans_downloads(self):
        data = b"weights" * 128
        self.service.assets = {"model": (None, data)}
        self.service.fail_asset = True
        with self.assertRaisesRegex(RuntimeError, "fixture asset failure"):
            self.start()
        self.assertFalse(list(Path(self.temp.name).rglob("init.json")))
        self.assertFalse(list(Path(self.temp.name).rglob("*.part")))
        self.service.fail_asset = False
        self.start()
        self.assertEqual(self.daemon.asset("model").read_bytes(), data)

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
            rank=0,
            start_daemon=True,
            addr=f"localhost:{self.service.port}",
            ipc_dir=self.temp.name,
            **({"num_workers": workers} if workers is not None else {}),
        )
        return self.daemon

    def attach(self, rank):
        return tensorlane.init(
            self.run_id,
            double,
            2,
            rank=rank,
            start_daemon=False,
            ipc_dir=self.temp.name,
        )

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

    def test_nonzero_rank_owns_daemon_and_publishes_config(self):
        self.service.returned_run_id = str(uuid.uuid4())
        self.service.train_config = 'name: "hello 🦀"\nsteps: 5\n'
        context = multiprocessing.get_context("spawn")
        output = context.Queue()
        finished = context.Event()
        ranks = [
            context.Process(
                target=initialize_rank,
                args=(
                    self.run_id,
                    rank,
                    f"localhost:{self.service.port}",
                    self.temp.name,
                    output,
                    finished,
                ),
            )
            for rank in (0, 1)
        ]
        try:
            ranks[0].start()
            time.sleep(0.2)
            self.assertEqual(self.service.init_requests, [])
            ranks[1].start()
            for _ in ranks:
                rank, run_id, config, error = output.get(timeout=30)
                self.assertIsNone(error, error)
                self.assertEqual(run_id, self.service.returned_run_id)
                self.assertEqual(config, self.service.train_config)
            self.assertEqual(len(self.service.init_requests), 1)
            finished.set()
            for rank in ranks:
                rank.join(10)
                self.assertEqual(rank.exitcode, 0)
            self.assertEqual(list(Path(self.temp.name).rglob("init.json")), [])
        finally:
            finished.set()
            for rank in ranks:
                if rank.is_alive():
                    rank.kill()
                rank.join(5)
            output.close()

    def test_follower_reads_metadata_once_without_starting_anything(self):
        from tensorlane.client import _root

        root = _root(self.run_id, self.temp.name)
        root.mkdir()
        metadata = {
            "run_id": "returned-run-id",
            "train_config": 'opaque\n"config"',
            "assets": {},
        }
        (root / "init.json").write_text(json.dumps(metadata))
        (root / "ranks").write_text("2")
        with (root / "lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with (
                patch("tensorlane.client.json.loads", wraps=json.loads) as decode,
                patch(
                    "tensorlane._native.Daemon",
                    side_effect=AssertionError("daemon was started"),
                ),
                tensorlane.init(
                    self.run_id,
                    None,
                    0,
                    rank=0,
                    start_daemon=False,
                    ipc_dir=self.temp.name,
                ) as lane,
            ):
                self.assertIsInstance(lane, tensorlane.TensorLane)
                decode.assert_called_once_with(json.dumps(metadata))
                self.assertEqual(lane.run_id, metadata["run_id"])
                self.assertEqual(lane.train_config, metadata["train_config"])
        self.assertTrue((root / "init.json").exists())

    def test_nonowner_close_does_not_stop_daemon_and_preserves_directory(self):
        self.service.returned_run_id = str(uuid.uuid4())
        self.start(ranks=1)
        follower = tensorlane.init(
            self.run_id, double, 1, rank=0, start_daemon=False, ipc_dir=self.temp.name
        )
        self.assertEqual(follower.run_id, self.daemon.run_id)
        self.assertEqual(follower.train_config, self.daemon.train_config)
        with follower.batches() as reader:
            follower.close()
            follower.close()
            self.assertEqual(len(list(reader)), 5)
        self.assertTrue((self.daemon._root / "init.json").exists())
        self.assertEqual(len(self.service.init_requests), 1)
        self.daemon.close()
        self.assertFalse((self.daemon._root / "init.json").exists())

    def test_stale_metadata_is_rejected_and_replaced(self):
        from tensorlane.client import _root

        root = _root(self.run_id, self.temp.name)
        root.mkdir()
        (root / "lock").touch()
        (root / "init.json").write_text(
            json.dumps({"run_id": "stale", "train_config": "stale"})
        )
        with self.assertRaisesRegex(RuntimeError, "stopped unexpectedly"):
            tensorlane.init(
                self.run_id,
                double,
                2,
                rank=0,
                start_daemon=False,
                ipc_dir=self.temp.name,
            )
        self.start()
        self.assertEqual(
            json.loads((root / "init.json").read_text()),
            {
                "run_id": self.run_id,
                "train_config": self.service.train_config,
                "assets": {},
            },
        )

    def test_failed_startup_never_publishes_readiness(self):
        from tensorlane.client import _root

        with patch.object(
            multiprocessing.process.BaseProcess,
            "start",
            side_effect=OSError("spawn failed"),
        ):
            with self.assertRaisesRegex(OSError, "spawn failed"):
                self.start()
        root = _root(self.run_id, self.temp.name)
        self.assertFalse((root / "init.json").exists())
        with patch.object(
            Path, "read_text", side_effect=AssertionError("waiting read a file")
        ):
            with self.assertRaises(TimeoutError):
                tensorlane.init(
                    self.run_id,
                    double,
                    2,
                    rank=0,
                    start_daemon=False,
                    ipc_dir=self.temp.name,
                    timeout=0.05,
                )

    def test_metadata_is_published_only_after_workers_are_ready(self):
        original = multiprocessing.process.BaseProcess.start
        observations = []

        def start(process):
            from tensorlane.client import _root

            observations.append(
                (_root(self.run_id, self.temp.name) / "init.json").exists()
            )
            original(process)

        with patch.object(multiprocessing.process.BaseProcess, "start", start):
            self.start()
        self.assertEqual(observations, [False, False])
        self.assertTrue((self.daemon._root / "init.json").exists())
        self.assertFalse((self.daemon._root / "init.tmp").exists())
        self.assertEqual(
            {path.name for path in self.daemon._root.iterdir()},
            {
                "init.json",
                "auth",
                "ranks",
                "lock",
                "semaphore",
                "validation-semaphore",
                "work.sock",
            },
        )
        self.daemon.close()
        self.assertEqual({path.name for path in self.daemon._root.iterdir()}, {"lock"})

    def test_cpu_example_uses_custom_ipc_dir_for_every_rank(self):
        self.service.validation_count = self.service.count
        example = Path(__file__).resolve().parents[1] / "examples" / "cpu.py"
        result = subprocess.run(
            [sys.executable, str(example), self.run_id, "--ipc-dir", "custom-ipc"],
            cwd=self.temp.name,
            env={**os.environ, "TENSORLANE_ADDR": f"localhost:{self.service.port}"},
            capture_output=True,
            text=True,
            timeout=45,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        lines = [
            line for line in result.stdout.splitlines() if line.startswith("rank=")
        ]
        self.assertEqual(len(lines), self.service.count * 2)
        for split in ("training", "validation"):
            split_lines = [line for line in lines if f"split={split} " in line]
            self.assertEqual(len(split_lines), self.service.count)
            self.assertEqual(sum(line.startswith("rank=0 ") for line in split_lines), 3)
            self.assertEqual(sum(line.startswith("rank=1 ") for line in split_lines), 2)
        self.assertTrue((Path(self.temp.name) / "custom-ipc").is_dir())
        self.assertEqual(list(Path(self.temp.name).rglob("*.sock")), [])

    @unittest.skipUnless(importlib.util.find_spec("accelerate"), "requires accelerate")
    def test_accelerate_cpu_example_initializes_once_for_two_ranks(self):
        self.service.validation_count = self.service.count
        example = Path(__file__).resolve().parents[1] / "examples" / "accelerate_cpu.py"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--rdzv-backend=c10d",
                "--rdzv-endpoint=127.0.0.1:0",
                "--local-addr=127.0.0.1",
                "--nproc_per_node=2",
                str(example),
                self.run_id,
                "--ipc-dir",
                "custom-ipc",
            ],
            cwd=self.temp.name,
            env={**os.environ, "TENSORLANE_ADDR": f"localhost:{self.service.port}"},
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(self.service.init_requests), 1)
        for rank in range(2):
            self.assertIn(
                f"rank={rank} run_id={self.run_id} train_config='opaque config'",
                result.stdout,
            )
        lines = [
            line
            for line in result.stdout.splitlines()
            if line.startswith("rank=") and "split=" in line
        ]
        self.assertEqual(len(lines), self.service.count * 2)
        for split in ("training", "validation"):
            for rank, count in ((0, 3), (1, 2)):
                self.assertEqual(
                    sum(
                        f"rank={rank} device=cpu" in line and f"split={split} " in line
                        for line in lines
                    ),
                    count,
                )
        self.assertEqual(list(Path(self.temp.name).rglob("*.sock")), [])

    def test_prefetch_credits_cover_unconsumed_batches(self):
        self.start(factor=1)
        wait_for(lambda: len(self.service.requests) == 2)
        time.sleep(0.2)
        self.assertEqual(len(self.service.requests), 2)
        with self.daemon.batches() as reader:
            self.assertEqual(next(reader).samples[0].speaker_id, 0)
            wait_for(lambda: len(self.service.requests) == 3)
            time.sleep(0.2)
            self.assertEqual(len(self.service.requests), 3)
            self.daemon.close()

    def test_validation_drains_while_training_buffer_is_full(self):
        self.service.validation_count = 4
        self.start(ranks=1, factor=1, workers=3)
        wait_for(lambda: len(self.service.requests) == 1)
        with self.daemon.batches(validation=True) as reader:
            batches = list(reader)
        self.assertEqual(
            [batch.samples[0].speaker_id for batch in batches], list(range(1000, 1004))
        )
        self.assertEqual(
            [batch.samples[0].wave.tolist() for batch in batches],
            [[value * 2, -value * 2] for value in range(1000, 1004)],
        )
        self.assertEqual(len(self.service.requests), 1)
        with self.daemon.batches() as reader:
            self.assertEqual(
                [batch.samples[0].speaker_id for batch in reader], list(range(5))
            )

    def test_training_drains_while_validation_buffer_is_full(self):
        self.service.validation_count = 4
        self.start(ranks=1, factor=1, workers=3)
        wait_for(lambda: len(self.service.validation_requests) == 1)
        with self.daemon.batches() as reader:
            self.assertEqual(
                [batch.samples[0].speaker_id for batch in reader], list(range(5))
            )
        self.assertEqual(len(self.service.validation_requests), 1)
        with self.daemon.batches(validation=True) as reader:
            self.assertEqual(
                [batch.samples[0].speaker_id for batch in reader],
                list(range(1000, 1004)),
            )

    def test_training_and_validation_readers_can_alternate(self):
        self.service.validation_count = 3
        self.start(ranks=1, factor=2, workers=3)
        with (
            self.daemon.batches() as training,
            self.daemon.batches(validation=True) as validation,
        ):
            for index in range(3):
                self.assertEqual(next(training).samples[0].speaker_id, index)
                self.assertEqual(next(validation).samples[0].speaker_id, 1000 + index)
            self.assertEqual(list(validation), [])
            self.assertEqual(
                [batch.samples[0].speaker_id for batch in training], [3, 4]
            )

    def test_validation_is_distributed_between_independent_ranks(self):
        self.service.validation_count = 5
        self.start(workers=3)
        context = multiprocessing.get_context("spawn")
        output = context.Queue()
        ranks = [
            context.Process(
                target=read_rank, args=(self.run_id, rank, self.temp.name, output, True)
            )
            for rank in range(2)
        ]
        try:
            for rank in ranks:
                rank.start()
            for _ in ranks:
                rank, batches, error = output.get(timeout=30)
                self.assertIsNone(error, error)
                self.assertEqual(
                    [batch[0][2] for batch in batches],
                    list(range(1000 + rank, 1005, 2)),
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

    def test_validation_rpc_failure_reaches_reader(self):
        self.service.fail_validation = True
        with self.assertRaisesRegex(RuntimeError, "fixture gRPC failure"):
            self.start(ranks=1)
            with self.daemon.batches(validation=True) as reader:
                next(reader)

    def test_global_budget_allows_one_rank_to_release_any_slot(self):
        self.service.count = 20
        self.start(factor=1, workers=3)
        wait_for(lambda: len(self.service.requests) == 2)
        with self.daemon.batches() as reader:
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
        names = [path.read_text() for path in Path(self.temp.name).rglob("*semaphore")]
        self.assertEqual(len(names), 2)
        started = time.monotonic()
        self.daemon.close()
        self.assertLess(time.monotonic() - started, 8)
        for name in names:
            with self.assertRaises(RuntimeError):
                tensorlane._native.Semaphore(name)

    def test_duplicate_init_and_rank_are_rejected(self):
        self.start()
        with self.assertRaisesRegex(Exception, "already active"):
            tensorlane.init(
                self.run_id,
                double,
                2,
                rank=0,
                start_daemon=True,
                ipc_dir=self.temp.name,
            )
        reader = self.daemon.batches()
        try:
            with self.assertRaisesRegex(RuntimeError, "already connected"):
                self.daemon.batches()
            with self.assertRaisesRegex(ValueError, "invalid"):
                self.attach(2)
        finally:
            self.daemon.close()
            reader.close()

    def test_transform_failure_reaches_rank(self):
        with self.assertRaisesRegex(RuntimeError, "tensorlane-transform-0 exited"):
            self.start(transform=fail)
            with self.daemon.batches() as reader:
                next(reader)

    def test_invalid_transform_output_reaches_rank(self):
        with self.assertRaisesRegex(RuntimeError, "tensorlane-transform-0 exited"):
            self.start(transform=invalid)
            with self.daemon.batches() as reader:
                next(reader)

    def test_startup_timeout_and_callable_validation(self):
        with self.assertRaises(TimeoutError):
            tensorlane.init(
                self.run_id,
                double,
                2,
                rank=0,
                start_daemon=False,
                ipc_dir=self.temp.name,
                timeout=0.1,
            )
        with self.assertRaises(AttributeError):
            self.start(transform=lambda wave: wave)

    def test_init_rejects_invalid_rank_before_starting_or_waiting(self):
        for owner in (False, True):
            for rank in (-1, True, 1.5):
                with self.subTest(owner=owner, rank=rank):
                    with self.assertRaisesRegex(ValueError, "rank"):
                        tensorlane.init(
                            self.run_id,
                            double,
                            2,
                            rank=rank,
                            start_daemon=owner,
                            ipc_dir=self.temp.name,
                        )
        with self.assertRaisesRegex(ValueError, "invalid rank"):
            tensorlane.init(
                self.run_id,
                double,
                2,
                rank=2,
                start_daemon=True,
                ipc_dir=self.temp.name,
            )
        self.assertEqual(self.service.init_requests, [])

    def test_handle_uses_its_initialized_rank_for_both_readers(self):
        self.service.validation_count = 5
        self.start(factor=3)
        with self.attach(1) as lane:
            self.assertEqual(lane.rank, 1)
            self.assertEqual(self.daemon.rank, 0)
            with (
                lane.batches() as training,
                lane.batches(validation=True) as validation,
            ):
                self.assertEqual(
                    [batch.samples[0].speaker_id for batch in training], [1, 3]
                )
                self.assertEqual(
                    [batch.samples[0].speaker_id for batch in validation], [1001, 1003]
                )

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
            with self.attach(rank).batches() as reader:
                self.assertEqual(list(reader), [])

    def test_empty_batch_fails_without_leaking_prefetch_slots(self):
        self.service.empty = True
        with self.assertRaisesRegex(RuntimeError, "empty batches are unsupported"):
            self.start(ranks=1, factor=2)
            with self.daemon.batches() as reader:
                list(reader)

    def test_grpc_error_reaches_rank_without_retry(self):
        self.service.fail_data = True
        with self.assertRaisesRegex(RuntimeError, "fixture gRPC failure"):
            self.start()
            with self.daemon.batches() as reader:
                next(reader)
        self.assertEqual(len(self.service.requests), 1)

    def test_collater_crash_wakes_reader(self):
        self.start(transform=slow)
        reader = self.daemon.batches()
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
        reader = self.daemon.batches()
        with self.attach(1) as lane:
            reader.close()
            with self.assertRaises(RuntimeError):
                with lane.batches() as other:
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
        wait_for(lambda: not (self.daemon._root / "init.json").exists())
        with self.assertRaisesRegex(RuntimeError, "tensorlane-collate exited"):
            self.daemon.batches()
        self.assertEqual(len(self.service.requests), 1)

    def test_worker_connection_is_ready_without_control_messages(self):
        from tensorlane.client import TensorLane, _root

        root = _root(self.run_id, self.temp.name)
        native = tensorlane._native.Daemon(
            self.run_id, f"localhost:{self.service.port}", root, 1, 1, 1
        )
        self.daemon = TensorLane(root, native.run_id, native.train_config, 0, native)
        receiver = tensorlane._native.Listener(root / "work.sock")
        try:
            self.assertFalse(hasattr(receiver, "ready"))
            self.assertFalse(hasattr(receiver, "error"))
            message = receiver.recv()
            if message["kind"] == "end":
                self.assertTrue(message["validation"])
                message = receiver.recv()
            self.assertEqual(message["kind"], "sample")
            self.assertFalse(message["validation"])
            self.assertEqual(message["batch"], (0, 1))
            self.assertEqual(message["index"], 0)
        finally:
            del receiver

    def test_native_listener_reports_connection_failure(self):
        with self.assertRaisesRegex(RuntimeError, "No such file or directory"):
            tensorlane._native.Listener(Path(self.temp.name) / "missing.sock")

    def test_transform_from_callers_search_path(self):
        module_path = Path(self.temp.name) / "custom_transform.py"
        module_path.write_text("def transform(wave):\n    return wave.clone()\n")
        sys.path.insert(0, self.temp.name)
        try:
            transform = importlib.import_module("custom_transform").transform
            self.service.count = 1
            self.start(ranks=1, transform=transform)
            with self.daemon.batches() as reader:
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
            f"    with tensorlane.init({self.run_id!r}, transform, 1, rank=0, start_daemon=True, "
            f"addr='localhost:{self.service.port}', ipc_dir={self.temp.name!r}) as lane:\n"
            "        with lane.batches() as reader:\n"
            "            batches = list(reader)\n"
            "            assert len(batches) == 1\n"
            "            assert batches[0].samples[0].wave.tolist() == [7, 7]\n"
        )
        subprocess.run([sys.executable, str(script)], check=True, timeout=40)

    def test_multiple_workers_preserve_order(self):
        self.service.count = 6
        self.start(ranks=1, factor=4, transform=identify_worker, workers=3)
        with self.daemon.batches() as reader:
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
        with self.daemon.batches() as reader:
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
        with self.daemon.batches() as reader:
            self.daemon._processes[1].kill()
            with self.assertRaisesRegex(RuntimeError, "disconnected|exited"):
                next(reader)

    def test_empty_stream_with_multiple_workers(self):
        self.service.count = 0
        self.start(ranks=1, workers=3)
        with self.daemon.batches() as reader:
            self.assertEqual(list(reader), [])

    def test_partial_transform(self):
        self.service.count = 1
        self.start(ranks=1, transform=partial(double))
        with self.daemon.batches() as reader:
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
