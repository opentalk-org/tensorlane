import multiprocessing
import unittest

import torch

from tensorlane import RawSample
from tensorlane import _ipc


class IPCTests(unittest.TestCase):
    def test_shared_batch_preserves_bytes_and_tensor_storage(self):
        sender, receiver = multiprocessing.Pipe()
        tensor = torch.arange(32).share_memory_()
        sample = RawSample(
            "sample", "audio", {"position": 7}, {"audio": bytes(range(256)) * 8192}
        )
        with sender, receiver:
            _ipc.send(sender, (sample, tensor))
            received, shared = _ipc.recv(receiver)
        self.assertEqual(received, sample)
        self.assertTrue(shared.is_shared())
        shared[0] = 99
        self.assertEqual(tensor[0].item(), 99)

    def test_repeated_messages_and_end_preserve_order(self):
        sender, receiver = multiprocessing.Pipe()
        with sender, receiver:
            for value in [("batch", [1, None, b"bytes"]), ("end", None)]:
                _ipc.send(sender, value)
                self.assertEqual(_ipc.recv(receiver), value)
