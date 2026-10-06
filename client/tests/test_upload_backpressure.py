from pathlib import Path
import socket
import struct
import tempfile
import threading
import time
import unittest

from tensorlane import _native
from pipeline_fixture import wait_for


class UploadBackpressureTests(unittest.TestCase):
    def test_flush_timeout_includes_waiting_for_queue_space(self):
        with tempfile.TemporaryDirectory(prefix="tl-backpressure-") as root:
            listener = socket.socket(socket.AF_UNIX)
            path = Path(root) / "upload.sock"
            listener.bind(str(path))
            listener.listen(1)
            gate = threading.Event()
            received = []

            def serve():
                connection, _ = listener.accept()
                connection.settimeout(10)
                with connection, connection.makefile("rb") as reader:
                    gate.wait()
                    while header := reader.read(8):
                        payload = reader.read(struct.unpack("!Q", header)[0])
                        if payload == b"\x04":
                            connection.sendall(b"\x00\x00\x00\x00\x00\x00\x00\x01\x00")
                        else:
                            received.append(payload)

            server = threading.Thread(target=serve, daemon=True)
            server.start()
            client = _native.UploadClient(path)
            progress = []

            def produce():
                for index in range(2048):
                    client.metric(index, "x" * 4096, 0.5)
                    progress.append(index)

            producer = threading.Thread(target=produce, daemon=True)
            producer.start()
            try:
                wait_for(lambda: len(progress) >= 1024)
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    previous = len(progress)
                    time.sleep(0.1)
                    if len(progress) == previous:
                        break
                self.assertEqual(len(progress), previous)
                self.assertTrue(producer.is_alive())
                started = time.monotonic()
                with self.assertRaisesRegex(RuntimeError, "flush timed out"):
                    client.flush(0.05)
                self.assertLess(time.monotonic() - started, 0.5)
                started = time.monotonic()
                with self.assertRaisesRegex(RuntimeError, "enqueue timed out"):
                    client.metric(2048, "deadline", 0.5, timeout=0.05)
                self.assertLess(time.monotonic() - started, 0.5)
            finally:
                gate.set()
                producer.join(10)
                try:
                    client.flush(5)
                finally:
                    try:
                        client.close()
                    finally:
                        server.join(5)
                        listener.close()
            self.assertFalse(producer.is_alive())
            self.assertFalse(server.is_alive())
            self.assertEqual(len(received), 2048)
