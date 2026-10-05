from __future__ import annotations
import multiprocessing
from pathlib import Path
from unittest.mock import patch
import time
import uuid
import tensorlane
from fixture_transforms import transform, fail, fail_collate, slow, invalid
from pipeline_fixture import PipelineCase


class PipelineTests(PipelineCase):
    def test_reader_handshake_disconnect_is_reported_and_socket_is_cleaned(self):
        self.start()
        with patch("tensorlane.reader.Listener.accept", side_effect=ConnectionResetError):
            with self.assertRaisesRegex(RuntimeError, "collater disconnected"):
                self.daemon.batches("training")
        self.assertFalse(list(self.daemon._root.rglob("rank-0.sock")))

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
                with self.daemon.batches("training") as reader:
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
            with self.daemon.batches("training") as reader:
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
        self.assertTrue(self.service.end_requests[0].failed)
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

    def test_flush_timeout_keeps_upload_and_connection_alive(self):
        self.start()
        self.service.upload_gate.clear()
        self.daemon.save_asset("model", self.file())
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            self.daemon.flush(timeout=0.05)
        self.service.upload_gate.set()
        self.daemon.metric(1, "after", 1)
        self.daemon.flush()
        self.assertEqual(len(self.service.saved), 1)
        self.assertTrue(
            any(metric.name == "after" for _, metric in self.service.metrics)
        )

    def test_cleaned_directory_can_be_reused(self):
        self.start()
        self.daemon.close()
        self.start()
        self.daemon.close()

    def test_large_raw_payload_and_empty_batch_failure(self):
        self.service.streams = {"training": 1, "validation": 0, "evaluation": 0}
        self.service.blob_size = 9 * 1024 * 1024
        self.start(transform_fn=None)
        with self.daemon.batches("training") as reader:
            self.assertEqual(
                len(next(reader).samples[0].blobs["payload"]), self.service.blob_size
            )
            self.assertEqual(list(reader), [])
        self.daemon.close()
        self.service.empty_batch = True
        with self.assertRaisesRegex(RuntimeError, "empty batches"):
            self.start()
            with self.daemon.batches("training") as reader:
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
        with self.daemon.batches("training") as reader:
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
        self.assertTrue(self.service.end_requests[-1].failed)
        follower.close()
        self.assertEqual(
            {path.name for path in self.daemon._root.iterdir()}, {"lock", "session"}
        )
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
        with self.daemon.batches("training") as reader:
            started = time.monotonic()
            self.daemon.close()
            self.assertLess(time.monotonic() - started, 10)
            with self.assertRaises((StopIteration, RuntimeError)):
                next(reader)
