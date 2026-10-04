import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from tensorlane._performance import Performance
from tensorlane.client import TensorLane
from tensorlane.data import Batch


class PerformanceTests(TestCase):
    def test_window_rates_timings_and_no_idle_duplicates(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch("tensorlane._performance.threading.Thread"), \
             patch("tensorlane._performance.time.monotonic", return_value=0) as clock:
            performance = Performance(Path(directory))
            performance.open("training/a")
            self.assertFalse(performance.snapshot())
            clock.return_value = 2
            performance.batch(Batch("training/a", 42, 7, (1, 2, 3, 4), None,
                                    (.1, .2, .3, .4, .5)), .6)
            performance.application("training/a", 2)
            clock.return_value = 5
            values = {name: value for _, name, value in performance.snapshot()}
            prefix = "tensorlane/training%2Fa/"
            self.assertEqual(set(values), {prefix + name for name in (
                "samples_per_second", "server_load_seconds_mean",
                "transform_work_seconds_mean", "collate_seconds_mean",
                "data_wait_seconds_mean", "application_seconds_mean",
            )})
            self.assertAlmostEqual(values[prefix + "samples_per_second"], .8)
            self.assertAlmostEqual(values[prefix + "server_load_seconds_mean"], .1)
            self.assertAlmostEqual(values[prefix + "collate_seconds_mean"], .5)
            self.assertAlmostEqual(values[prefix + "data_wait_seconds_mean"], .6)
            self.assertEqual(values[prefix + "application_seconds_mean"], 2)
            self.assertAlmostEqual(values[prefix + "transform_work_seconds_mean"], .4)
            clock.return_value = 10
            self.assertFalse(performance.snapshot())
            performance.application("training/a", .5)
            self.assertEqual(performance.snapshot(), [(42, prefix + "application_seconds_mean", .5)])
            self.assertFalse(performance.snapshot())

    def test_errors_and_upload_timings_are_only_reported_when_recorded(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch("tensorlane._performance.threading.Thread"):
            performance = Performance(Path(directory))
            performance.open("training")
            self.assertFalse(performance.snapshot())
            performance.error("training")
            performance.event("uploads/flush_seconds", 1)
            performance.event("uploads/flush_seconds", 3)
            self.assertEqual(performance.snapshot(), [
                (0, "tensorlane/training/errors_total", 1),
                (0, "tensorlane/uploads/flush_seconds", 2),
            ])
            self.assertFalse(performance.snapshot())
            performance.error("training")
            self.assertEqual(performance.snapshot(), [(0, "tensorlane/training/errors_total", 2)])

    def test_only_rank_zero_collects_automatic_metrics(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch("tensorlane.client.Performance") as collector:
            root = Path(directory)
            for rank in (1, 2):
                lane = TensorLane(root, "run", {}, rank)
                self.assertIsNone(lane._performance_collector())
            collector.assert_not_called()
            lane = TensorLane(root, "run", {}, 0)
            self.assertIs(lane._performance_collector(), collector.return_value)
            self.assertIs(lane._performance_collector(), collector.return_value)
            collector.assert_called_once_with(root)
