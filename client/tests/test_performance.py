import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from tensorlane._performance import Performance
from tensorlane.data import Batch


class PerformanceTests(TestCase):
    def test_window_rates_timings_and_cumulative_totals(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch("tensorlane._performance.threading.Thread"), \
             patch("tensorlane._performance.time.monotonic", return_value=0) as clock:
            performance = Performance(Path(directory), 2)
            performance.open("training/a")
            clock.return_value = 2
            performance.batch(Batch("training/a", 42, 7, (1, 2, 3, 4), None,
                                    (.1, .2, .3, .4, .5)), .6)
            performance.application("training/a", 2)
            clock.return_value = 5
            values = {name: value for _, name, value in performance.snapshot()}
            prefix = "tensorlane/rank/2/training%2Fa/"
            self.assertAlmostEqual(values[prefix + "batches_per_second"], .2)
            self.assertAlmostEqual(values[prefix + "samples_per_second"], .8)
            self.assertAlmostEqual(values[prefix + "data_wait_fraction"], .6 / 2.6)
            self.assertEqual(values[prefix + "application_seconds_mean"], 2)
            self.assertAlmostEqual(values[prefix + "transform_work_seconds_mean"], .4)
            clock.return_value = 10
            values = {name: value for _, name, value in performance.snapshot()}
            self.assertEqual(values[prefix + "batches_total"], 1)
            self.assertEqual(values[prefix + "samples_per_second"], 0)
            self.assertEqual(values[prefix + "seconds_since_last_batch"], 8)
            performance.finish("training/a")
            performance.snapshot()
            self.assertFalse(performance.snapshot())
