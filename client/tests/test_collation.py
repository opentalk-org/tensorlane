from pathlib import Path
import queue
import tempfile
import threading
import unittest
from unittest.mock import patch
import torch
from tensorlane._process import collate_worker, share


class Reader:
    def __init__(self):
        self.received = []

    def recv(self):
        return ("next", None)

    def receive(self, message):
        self.received.append(message)

    def close(self):
        pass


class CollationTests(unittest.TestCase):
    streams = ("training", "validation", "evaluation")

    def connect(self, path, key):
        return self.readers[self.streams[int(path.parent.name)]]

    def available(self, connections, timeout):
        if not self.incoming.empty():
            return connections
        self.idle += 1
        if self.idle > 2:
            self.stopped.set()
        return connections[1:]

    def run_collator(self, messages, collate_fn=None):
        self.incoming = queue.Queue()
        for message in messages:
            kind, stream, value = message
            if kind == "sample":
                batch, query_idx, index, sample, timings = value
                message = (
                    "samples",
                    stream,
                    (batch, query_idx, [(index, sample, timings[3])], timings[:3], 0),
                )
            self.incoming.put(message)
        self.readers = {stream: Reader() for stream in self.streams}
        self.stopped = threading.Event()
        self.idle = 0
        ready = threading.Event()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "auth").write_bytes(b"key")
            for index in range(3):
                directory = root / "streams" / str(index)
                directory.mkdir(parents=True)
                (directory / "start").write_text("0")
                (directory / "semaphore").write_text("unused")
            with (
                patch("tensorlane._process.threading.Thread"),
                patch("tensorlane._process.signal.pthread_sigmask"),
                patch("tensorlane._process._native.Semaphore"),
                patch("tensorlane._process._connect", side_effect=self.connect),
                patch("tensorlane._process.wait", side_effect=self.available),
                patch(
                    "tensorlane._process._ipc.recv", side_effect=queue.Queue.get_nowait
                ),
                patch("tensorlane._process._ipc.send", side_effect=Reader.receive),
            ):
                collate_worker(
                    0,
                    root,
                    1,
                    1,
                    1,
                    self.streams,
                    collate_fn,
                    [self.incoming],
                    self.stopped,
                    ready,
                )
        self.assertTrue(ready.is_set())
        return [self.readers[stream].received for stream in self.streams]

    def test_order_sparse_query_indices_and_custom_collation(self):
        messages = [
            ("sample", "training", ((1, 2), 40, 1, 3, (0, 0, 0, 0))),
            ("sample", "training", ((0, 1), 7, 0, 1, (0, 0, 0, 0))),
            ("sample", "training", ((1, 2), 40, 0, 2, (0, 0, 0, 0))),
            ("sample", "evaluation", ((0, 1), 90, 0, 4, (0, 0, 0, 0))),
        ]
        messages.extend(
            ("end", stream, None) for stream in ("training", "validation", "evaluation")
        )
        outputs = self.run_collator(messages, {"training": sum})
        for batch_id, index, samples, data in [(0, 7, (1,), 1), (1, 40, (2, 3), 5)]:
            kind, batch = outputs[0].pop(0)
            self.assertEqual(kind, "batch")
            self.assertEqual(
                (batch.batch_id, batch.query_batch_idx, batch.samples, batch.data),
                (batch_id, index, samples, data),
            )
        self.assertEqual(outputs[0][0], ("end", None))
        self.assertEqual(outputs[1][0], ("end", None))
        self.assertEqual(outputs[2][0][1].data, (4,))

    def test_grouped_parts_preserve_sample_order_and_timings(self):
        outputs = self.run_collator(
            [
                (
                    "samples",
                    "training",
                    ((0, 4), 9, [(3, 4, 0.4), (1, 2, 0.2)], (1, 2, 3), 0),
                ),
                (
                    "samples",
                    "training",
                    ((0, 4), 9, [(2, 3, 0.3), (0, 1, 0.1)], (1, 2, 3), 0),
                ),
                ("end", "training", None),
            ],
            {"training": sum},
        )
        batch = outputs[0][0][1]
        self.assertEqual(batch.samples, (1, 2, 3, 4))
        self.assertEqual(batch.data, 10)
        self.assertEqual(batch.query_batch_idx, 9)
        self.assertAlmostEqual(batch._timings[3], 1)

    def test_nested_tensors_are_detached_and_shared(self):
        value = torch.tensor([1.0], requires_grad=True)
        output = share({"list": [value], "tuple": (value, b"bytes"), "none": None})
        self.assertTrue(output["list"][0].is_shared())
        self.assertFalse(output["list"][0].requires_grad)
        with self.assertRaises(TypeError):
            share(object())

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_tensors_are_rejected(self):
        with self.assertRaisesRegex(TypeError, "CPU"):
            share(torch.ones(1, device="cuda"))
