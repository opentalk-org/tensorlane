from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import io
import json
from pathlib import Path
from unittest.mock import patch
import time
import torch
from fixture_transforms import transform, timed_transform
from pipeline_fixture import PipelineCase, wait_for


class PipelineTests(PipelineCase):
    def test_state_dict_save_snapshots_model_and_optimizer_and_cleans_source(self):
        self.start(workers=1, performance_metrics=False)
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters())
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        expected = model.weight.detach().clone()
        self.service.upload_gate.clear()
        with ThreadPoolExecutor() as executor:
            saved = executor.submit(
                self.daemon.save_asset,
                "model",
                {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
                kind="checkpoint",
                step=3,
                metadata={"note": "state dictionary"},
            )
            try:
                self.assertTrue(self.service.upload_started.wait(5))
                self.assertFalse(saved.done())
                self.assertTrue(list(self.daemon._root.glob("*.pt")))
                with torch.no_grad():
                    model.weight.add_(10)
            finally:
                self.service.upload_gate.set()
            asset_id = saved.result(timeout=15)
        self.assertFalse(list(self.daemon._root.glob("*.pt")))
        metadata, _, body = self.service.saved[0]
        self.assertEqual(metadata.asset_id, asset_id)
        self.assertEqual(metadata.kind, "checkpoint")
        self.assertEqual(metadata.step, 3)
        self.assertEqual(metadata.content_type, "application/octet-stream")
        self.assertEqual(
            json.loads(metadata.metadata_json),
            {
                "note": "state dictionary",
                "_tensorlane": {"next_batches": dict.fromkeys(self.daemon.streams, 0)},
            },
        )
        state = torch.load(io.BytesIO(body), weights_only=True)
        restored = torch.nn.Linear(2, 1)
        restored.load_state_dict(state["model"])
        restored_optimizer = torch.optim.Adam(restored.parameters())
        restored_optimizer.load_state_dict(state["optimizer"])
        self.assertTrue(torch.equal(restored.weight, expected))
        self.assertTrue(restored_optimizer.state)

    def test_state_dict_save_cleans_source_when_serialization_or_enqueue_fails(self):
        self.start(workers=1, performance_metrics=False)
        with patch("torch.save", side_effect=ValueError("serialization failed")):
            with self.assertRaisesRegex(ValueError, "serialization failed"):
                self.daemon.save_asset("model", {"weight": torch.ones(2)})
        self.assertFalse(list(self.daemon._root.glob("*.pt")))
        with self.assertRaisesRegex(RuntimeError, "asset kind"):
            self.daemon.save_asset("model", {"weight": torch.ones(2)}, kind="invalid")
        self.assertFalse(list(self.daemon._root.glob("*.pt")))
        self.service.fail_save = True
        with self.assertRaisesRegex(RuntimeError, "fixture save failure"):
            self.daemon.save_asset("model", {"weight": torch.ones(2)})
        self.assertFalse(list(self.daemon._root.glob("*.pt")))

    def test_metrics_and_artifacts_still_work(self):
        self.start()
        self.daemon.metric(3, "loss", 0.25)
        self.daemon.metric_artifact(
            3, self.file("report.json", b"{}"), "report", "application/json"
        )
        self.daemon.flush()
        self.assertEqual(self.service.metrics[0][1].name, "loss")
        self.assertEqual(self.service.artifacts[0][1], b"{}")

    def test_file_artifacts_are_raw_and_directory_artifacts_are_tar(self):
        self.start()
        for filename, body, content_type in [
            ("payload.bin", b"opaque bytes", "application/octet-stream"),
            ("config.json", b'{"model":1}', "application/json"),
            ("weights.pt", b"plain tensor bytes", "application/octet-stream"),
        ]:
            self.daemon.metric_artifact(0, self.file(filename, body), filename)
            self.daemon.save_asset(filename, Path(self.temp.name) / filename)
        directory = Path(self.temp.name) / "reports"
        directory.mkdir()
        (directory / "config.json").write_text("{}")
        self.daemon.metric_artifact(0, directory, "reports", "application/json")
        self.daemon.flush()
        for artifact, saved, (_, body, content_type) in zip(
            self.service.artifacts,
            self.service.saved,
            [
                ("payload.bin", b"opaque bytes", "application/octet-stream"),
                ("config.json", b'{"model":1}', "application/json"),
                ("weights.pt", b"plain tensor bytes", "application/octet-stream"),
            ],
        ):
            self.assertEqual(artifact[1], body)
            self.assertEqual(saved[2], body)
            self.assertEqual(artifact[0].content_type, content_type)
            self.assertEqual(saved[0].content_type, content_type)
        metadata, body = self.service.artifacts[-1]
        self.assertEqual(metadata.content_type, "application/x-tar")
        self.assertEqual(self.archive_contents(body), {"reports/config.json": b"{}"})

    def test_automatic_performance_metrics_flush_during_run(self):
        with patch("tensorlane._performance.INTERVAL", 0.05):
            self.start(workers=1, transform_fn=timed_transform)
            with self.daemon.batches("training") as reader:
                next(reader)
                time.sleep(0.03)
                next(reader)
                wait_for(
                    lambda: any(
                        "/training/application_seconds_mean" in row[1].name
                        for row in self.service.metrics
                    )
                )
            self.daemon.flush()
        values = {metric.name: metric.value for _, metric in self.service.metrics}
        prefix = "tensorlane/training/"
        self.assertGreater(values[prefix + "samples_per_second"], 0)
        self.assertGreater(values[prefix + "transform_work_seconds_mean"], 0)
        self.assertGreater(values[prefix + "application_seconds_mean"], 0)
        self.assertFalse(self.service.end_requests)

    def test_automatic_performance_metrics_can_be_disabled(self):
        self.start(workers=1, performance_metrics=False)
        with self.daemon.batches("training") as reader:
            list(reader)
        self.daemon.flush()
        self.daemon.close()
        self.assertFalse(self.service.metrics)

    def test_automatic_metrics_failure_does_not_fail_training(self):
        self.start(workers=1)
        with self.daemon.batches("training") as reader:
            next(reader)
            self.service.fail_metrics = True
            with self.assertLogs("tensorlane", level="WARNING"):
                self.daemon.flush()
            self.service.fail_metrics = False
            list(reader)
        self.daemon.close()
        self.assertFalse(self.service.end_requests[-1].failed)

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
        reader = self.daemon.batches("training")
        try:
            with self.assertRaisesRegex(RuntimeError, "already connected"):
                self.daemon.batches("training")
        finally:
            self.daemon.close()
            reader.close()
