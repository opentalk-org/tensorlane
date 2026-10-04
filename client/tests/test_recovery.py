from pathlib import Path
import time
import tensorlane
from pipeline_fixture import PipelineCase


class RecoveryTests(PipelineCase):
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
