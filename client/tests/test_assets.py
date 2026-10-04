from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import asyncio
import json
from pathlib import Path
import time
import uuid
import tensorlane
from fixture_transforms import transform
from pipeline_fixture import PipelineCase


class PipelineTests(PipelineCase):
    def test_end_reports_success_and_application_failure(self):
        self.start()
        self.daemon.close()
        self.assertFalse(self.service.end_requests[-1].failed)
        with self.assertRaisesRegex(ValueError, "training failed"):
            with self.start():
                raise ValueError("training failed")
        self.assertTrue(self.service.end_requests[-1].failed)
        with self.start():
            with self.assertRaisesRegex(ValueError, "follower failed"):
                with self.attach():
                    raise ValueError("follower failed")
        self.assertTrue(self.service.end_requests[-1].failed)

    def test_assets_are_prefetched_once_and_shared(self):
        asset_id = str(uuid.uuid4())
        self.service.assets = {"model": (asset_id, b"weights"), "empty": (None, b"")}
        self.start()
        with self.attach() as follower:
            self.assertEqual(follower.asset("model"), self.daemon.asset("model"))
            self.assertEqual(follower.asset("model").read_bytes(), b"weights")
            self.assertEqual(follower.asset_metadata["model"]["asset_id"], asset_id)
        self.assertEqual(len(self.service.asset_requests), 2)

    def test_async_init_waits_for_assets_without_blocking_event_loop(self):
        self.service.assets = {"model": (str(uuid.uuid4()), b"weights")}
        self.service.asset_gate.clear()

        async def initialize():
            pending = asyncio.create_task(
                tensorlane.init_async(
                    self.run_id,
                    transform,
                    ranks=1,
                    num_workers=1,
                    addr=f"localhost:{self.service.port}",
                    ipc_dir=self.temp.name,
                    performance_metrics=False,
                )
            )
            try:
                await asyncio.to_thread(self.service.asset_started.wait, 10)
                await asyncio.sleep(0.05)
                self.assertTrue(self.service.asset_started.is_set())
                self.assertFalse(pending.done())
            finally:
                self.service.asset_gate.set()
            self.daemon = await pending
            self.assertEqual(self.daemon.asset("model").read_bytes(), b"weights")

        asyncio.run(initialize())

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
            self.service.saved[0][2],
            b"opaque bytes",
        )
        self.assertEqual(
            self.archive_contents(self.service.saved[2][2]),
            {"bundle/config.json": b"{}"},
        )

    def test_saved_assets_are_automatically_extracted_on_load(self):
        self.start()
        directory = Path(self.temp.name) / "bundle"
        directory.mkdir()
        (directory / "config.json").write_text("{}")
        self.daemon.save_asset("model", self.file())
        self.daemon.save_asset("bundle", directory)
        self.daemon.flush()
        self.service.assets = {
            row[0].name: (row[0].asset_id, row[2]) for row in self.service.saved
        }
        self.daemon.close()
        self.daemon = None
        self.start()
        self.assertEqual(self.daemon.asset("model").read_bytes(), b"opaque bytes")
        self.assertEqual(
            (self.daemon.asset("bundle") / "config.json").read_text(), "{}"
        )
        with self.attach() as follower:
            self.assertEqual(follower.asset("bundle"), self.daemon.asset("bundle"))

    def test_saves_queue_promptly_and_flush_waits_for_commit(self):
        self.start()
        path = self.file()
        self.service.upload_gate.clear()
        started = time.monotonic()
        self.daemon.save_asset("model", path)
        self.assertLess(time.monotonic() - started, 1)
        self.assertTrue(self.service.upload_started.wait(5))
        with self.daemon.batches("training") as reader:
            self.assertEqual(len(list(reader)), 5)
        with ThreadPoolExecutor() as executor:
            flushed = executor.submit(self.daemon.flush)
            time.sleep(0.1)
            self.assertFalse(flushed.done())
            self.assertEqual(self.service.saved, [])
            self.service.upload_gate.set()
            flushed.result(timeout=15)
        self.assertEqual(len(self.service.saved), 1)

    def test_enqueue_and_batches_continue_while_flush_waits(self):
        self.start(workers=1, performance_metrics=False)
        self.service.upload_gate.clear()
        self.daemon.save_asset("model", self.file())
        self.assertTrue(self.service.upload_started.wait(5))
        with ThreadPoolExecutor() as executor:
            flushed = executor.submit(self.daemon.flush)
            time.sleep(0.05)
            queued = executor.submit(
                self.daemon.save_asset, "other", self.file("other")
            )
            try:
                queued.result(timeout=1)
                with self.daemon.batches("training") as reader:
                    self.assertEqual(len(list(reader)), 5)
                self.assertFalse(flushed.done())
            finally:
                self.service.upload_gate.set()
            flushed.result(timeout=15)
        self.daemon.flush()
        self.assertEqual(len(self.service.saved), 2)

    def test_async_flush_keeps_event_loop_responsive(self):
        self.start(workers=1, performance_metrics=False)
        self.service.upload_gate.clear()
        self.daemon.save_asset("model", self.file())
        self.assertTrue(self.service.upload_started.wait(5))

        async def run():
            flushed = asyncio.create_task(self.daemon.flush_async())
            try:
                await asyncio.sleep(0.05)
                self.assertFalse(flushed.done())
                self.daemon.metric(0, "while-uploading", 1)
            finally:
                self.service.upload_gate.set()
            await flushed

        asyncio.run(run())
        self.daemon.flush()
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
