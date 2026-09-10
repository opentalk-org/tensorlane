import multiprocessing
from pathlib import Path
import queue
import tempfile
import threading
import unittest
from unittest.mock import patch

import torch

from tensorlane._process import collate_worker
from tensorlane.data import Sample


class CollationTests(unittest.TestCase):
    def test_first_sample_sets_size_and_batches_remain_ordered(self):
        samples = [
            Sample(torch.tensor([value]), 0.5, value, 0, torch.tensor([value]))
            for value in range(3)
        ]
        incoming = queue.Queue()
        incoming.put(("sample", False, ((1, 2), 1, samples[2])))
        incoming.put(("sample", True, ((0, 1), 0, samples[2])))
        incoming.put(("sample", False, ((0, 1), 0, samples[0])))
        incoming.put(("end", True, None))
        incoming.put(("sample", False, ((1, 999), 0, samples[1])))
        incoming.put(("end", False, None))
        outgoing = queue.Queue()
        validation = queue.Queue()
        errors = queue.Queue()
        stopped = threading.Event()
        ready = threading.Event()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "auth").write_bytes(multiprocessing.current_process().authkey)
            with (
                patch("tensorlane._process.threading.Thread"),
                patch(
                    "tensorlane._process.queue.Queue",
                    side_effect=[outgoing, validation, errors],
                ),
                patch.object(stopped, "wait", side_effect=lambda _: stopped.set()),
            ):
                collate_worker(root, 1, 1, incoming, stopped, ready)
        self.assertTrue(ready.is_set())
        for batch_id, expected in [(0, [samples[0]]), (1, samples[1:])]:
            kind, (received_id, batch) = outgoing.get_nowait()
            self.assertEqual(kind, "batch")
            self.assertEqual(received_id, batch_id)
            self.assertEqual(len(batch.samples), len(expected))
            for received, sample in zip(batch.samples, expected):
                self.assertIs(received, sample)
        self.assertEqual(outgoing.get_nowait(), ("end", None))
        self.assertTrue(outgoing.empty())
        kind, (batch_id, batch) = validation.get_nowait()
        self.assertEqual((kind, batch_id), ("batch", 0))
        self.assertIs(batch.samples[0], samples[2])
        self.assertEqual(validation.get_nowait(), ("end", None))
        self.assertTrue(validation.empty())
