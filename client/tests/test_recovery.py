from pathlib import Path
import time
import threading
import tensorlane
from pipeline_fixture import PipelineCase, wait_for


class RecoveryTests(PipelineCase):
    def test_parallel_batch_requests_preserve_order_and_credit_limit(self):
        self.service.streams.update(training=18, validation=0, evaluation=0)
        gate = threading.Event()
        started = set()
        completed = set()
        data = self.service.data

        def delayed_data(handler, run, stream, index):
            if stream == "training":
                with self.service.lock:
                    started.add(index)
                if index == 0:
                    gate.wait(20)
            data(handler, run, stream, index)
            if stream == "training":
                with self.service.lock:
                    completed.add(index)

        self.service.data = delayed_data
        try:
            self.start(
                factor=16, workers=1, transform_fn=None, performance_metrics=False
            )
            with self.daemon.batches("training") as reader:
                wait_for(lambda: set(range(1, 16)).issubset(completed))
                self.assertEqual(started, set(range(16)))
                self.assertNotIn(0, completed)
                gate.set()
                batches = list(reader)
            self.assertEqual([batch.batch_id for batch in batches], list(range(18)))
            self.assertEqual(
                [batch.samples[0].metadata["position"] for batch in batches],
                list(range(18)),
            )
        finally:
            gate.set()

    def test_pending_batches_poll_without_exponential_backoff(self):
        attempts = []
        data = self.service.data

        def pending_data(handler, run, stream, index):
            if stream == "training" and index == 0 and len(attempts) < 6:
                attempts.append(time.monotonic())
                self.service.reply(handler, 202, {})
                return
            data(handler, run, stream, index)

        self.service.data = pending_data
        self.start(factor=1, workers=1, transform_fn=None, performance_metrics=False)
        with self.daemon.batches("training") as reader:
            self.assertEqual([batch.batch_id for batch in reader], list(range(5)))
        self.assertEqual(len(attempts), 6)
        self.assertLess(attempts[-1] - attempts[0], 1)

    def test_eof_discards_errors_from_speculative_requests(self):
        self.service.streams.update(training=0, validation=0, evaluation=0)
        data = self.service.data

        def empty_data(handler, run, stream, index):
            if index > 0:
                self.service.reply(handler, 422, {"message": "past EOF"})
            else:
                data(handler, run, stream, index)

        self.service.data = empty_data
        self.start(factor=3, workers=1, transform_fn=None, performance_metrics=False)
        with self.daemon.batches("training") as reader:
            self.assertEqual(list(reader), [])
        self.daemon.close()
        self.assertFalse(self.service.end_requests[-1].failed)

    def test_truncated_asset_range_is_retried_and_verified(self):
        body = bytes(range(251)) * 20000
        self.service.assets = {"model": (None, body)}
        self.service.truncate_asset_once = True
        daemon = self.start(transform_fn=None, workers=1)
        self.assertEqual(daemon.asset("model").read_bytes(), body)
        self.assertFalse(self.service.truncate_asset_once)

    def test_transient_batch_failures_preserve_order(self):
        self.service.transient_batches = 3
        self.start(transform_fn=None, workers=1)
        with self.daemon.batches("training") as batches:
            self.assertEqual([batch.batch_id for batch in batches], list(range(5)))
        self.assertEqual(self.service.transient_batches, 0)
        self.assertEqual(self.service.end_requests, [])

    def test_lost_upload_reply_retries_the_whole_file_without_duplicate_asset(self):
        self.start(workers=1, performance_metrics=False)
        body = bytes(range(251)) * 20000
        self.service.drop_upload_reply = True
        asset_id = self.daemon.save_asset("model", self.file(content=body))
        self.daemon.flush()
        self.assertEqual(len(self.service.saved), 1)
        self.assertEqual(self.service.saved[0][0].asset_id, asset_id)
        self.assertEqual(self.service.saved[0][2], body)
        self.assertFalse(self.service.drop_upload_reply)

    def test_lost_metric_reply_reuses_the_request_id(self):
        self.start(transform_fn=None, workers=1, performance_metrics=False)
        self.service.drop_metric_reply = True
        self.daemon.metric(1, "loss", 0.25)
        self.daemon.flush()
        self.assertEqual(len(self.service.metrics), 1)
        self.assertFalse(self.service.drop_metric_reply)

    def test_startup_timeout_covers_asset_download(self):
        self.service.assets = {"model": (None, b"weights")}
        self.service.asset_gate.clear()
        started = time.monotonic()
        try:
            with self.assertRaisesRegex(RuntimeError, "startup timed out"):
                tensorlane.init(
                    self.run_id,
                    rank=0,
                    start_daemon=True,
                    num_workers=1,
                    addr=f"localhost:{self.service.port}",
                    ipc_dir=self.temp.name,
                    timeout=0.5,
                )
        finally:
            self.service.asset_gate.set()
        self.assertLess(time.monotonic() - started, 3)
        self.assertFalse(list(Path(self.temp.name).rglob("init.json")))

    def test_flush_deadline_includes_automatic_metrics_lock(self):
        self.start(transform_fn=None, workers=1)
        self.daemon.metric(0, "loss", 0.5)
        self.daemon.flush(timeout=5)
        performance = self.daemon._performance_collector()
        performance.event("test", 0.1)
        performance._sending.acquire()
        started = time.monotonic()
        try:
            self.daemon.flush(timeout=0.05)
        finally:
            performance._sending.release()
        self.assertLess(time.monotonic() - started, 0.5)
        self.daemon.flush(timeout=5)
