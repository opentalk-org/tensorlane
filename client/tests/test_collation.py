import multiprocessing
from pathlib import Path
import queue
import tempfile
import threading
import unittest
from unittest.mock import patch
import torch
from tensorlane._process import collate_worker, share


class CollationTests(unittest.TestCase):
    def run_collator(self, messages, collate_fn=None):
        incoming = queue.Queue()
        for message in messages:
            kind, stream, value = message
            if kind == "sample":
                batch, query_idx, index, sample, timings = value
                message = (
                    "samples",
                    stream,
                    (batch, query_idx, [(index, sample, timings[3])], timings[:3], 0),
                )
            incoming.put(message)
        outputs = [queue.Queue() for _ in range(3)]
        errors = queue.Queue()
        stopped = threading.Event()
        ready = threading.Event()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "auth").write_bytes(multiprocessing.current_process().authkey)
            with (
                patch("tensorlane._process.threading.Thread"),
                patch(
                    "tensorlane._process.queue.Queue", side_effect=[*outputs, errors]
                ),
            ):
                incoming_get = incoming.get

                def get(timeout):
                    if incoming.empty():
                        stopped.set()
                        raise queue.Empty
                    return incoming_get(timeout=timeout)

                with patch.object(incoming, "get", side_effect=get):
                    collate_worker(
                        root,
                        1,
                        1,
                        ("training", "validation", "evaluation"),
                        collate_fn,
                        incoming,
                        stopped,
                        ready,
                    )
        self.assertTrue(ready.is_set())
        return outputs

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
            kind, batch = outputs[0].get_nowait()
            self.assertEqual(kind, "batch")
            self.assertEqual(
                (batch.batch_id, batch.query_batch_idx, batch.samples, batch.data),
                (batch_id, index, samples, data),
            )
        self.assertEqual(outputs[0].get_nowait(), ("end", None))
        self.assertEqual(outputs[1].get_nowait(), ("end", None))
        self.assertEqual(outputs[2].get_nowait()[1].data, (4,))

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
        batch = outputs[0].get_nowait()[1]
        self.assertEqual(batch.samples, (1, 2, 3, 4))
        self.assertEqual(batch.data, 10)
        self.assertEqual(batch.query_batch_idx, 9)
        self.assertAlmostEqual(batch._timings[3], 1)

    def test_duplicate_positions_and_inconsistent_sizes_fail(self):
        for second in [((0, 2), 1, 0, 2), ((0, 3), 1, 1, 2), ((0, 2), 2, 1, 2)]:
            with (
                self.subTest(second=second),
                self.assertRaisesRegex(RuntimeError, "inconsistent"),
            ):
                self.run_collator(
                    [
                        ("sample", "training", ((0, 2), 1, 0, 1, (0, 0, 0, 0))),
                        ("sample", "training", (*second, (0, 0, 0, 0))),
                    ]
                )

    def test_incomplete_batch_on_end_fails(self):
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            self.run_collator(
                [
                    ("sample", "training", ((0, 2), 1, 0, 1, (0, 0, 0, 0))),
                    ("end", "training", None),
                ]
            )

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
