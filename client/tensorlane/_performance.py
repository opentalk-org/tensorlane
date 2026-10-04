from __future__ import annotations

import logging
import threading
import time
from urllib.parse import quote

from . import _native

INTERVAL = 10.0
STAGES = ("server_load", "server_wait", "http_receive_wait", "transform_work", "collate", "data_wait")
REPORTED_STAGES = {"server_load", "transform_work", "collate", "data_wait"}


class Performance:
    """Aggregate timings in memory; upload off the batch-consumption path."""

    def __init__(self, root):
        self._root = root
        self._lock = threading.Lock()
        self._sending = threading.Lock()
        self._streams = {}
        self._events = {}
        self._client = None
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _state(self, stream):
        if stream not in self._streams:
            self._streams[stream] = dict(
                batches=0, samples=0, errors=0, reported_errors=0, step=0,
                application=0.0, intervals=0, timings=[0.0] * len(STAGES),
                since=time.monotonic(),
            )
        return self._streams[stream]

    def open(self, stream):
        with self._lock:
            self._state(stream)

    def application(self, stream, seconds):
        with self._lock:
            state = self._state(stream)
            state["application"] += seconds
            state["intervals"] += 1

    def batch(self, batch, wait):
        with self._lock:
            state = self._state(batch.stream)
            state["step"] = batch.batch_id
            state["batches"] += 1
            state["samples"] += len(batch)
            for index, seconds in enumerate((*batch._timings, wait)):
                state["timings"][index] += seconds

    def error(self, stream):
        with self._lock:
            self._state(stream)["errors"] += 1

    def event(self, name, seconds):
        with self._lock:
            count, total = self._events.get(name, (0, 0.0))
            self._events[name] = (count + 1, total + seconds)

    def snapshot(self):
        now = time.monotonic()
        result = []
        prefix = "tensorlane/"
        with self._lock:
            for stream, state in self._streams.items():
                batches, samples = state["batches"], state["samples"]
                elapsed = max(now - state["since"], 1e-9)
                if not (batches or state["intervals"] or state["errors"] != state["reported_errors"]):
                    continue
                values = {}
                if batches:
                    values["samples_per_second"] = samples / elapsed
                    for name, total in zip(STAGES, state["timings"]):
                        if name in REPORTED_STAGES:
                            values[f"{name}_seconds_mean"] = total / batches
                if state["intervals"]:
                    values["application_seconds_mean"] = state["application"] / state["intervals"]
                if state["errors"] != state["reported_errors"]:
                    values["errors_total"] = state["errors"]
                for name, value in values.items():
                    result.append((state["step"], prefix + quote(stream, safe="") + "/" + name, value))
                state.update(batches=0, samples=0, application=0.0, intervals=0,
                             reported_errors=state["errors"], timings=[0.0] * len(STAGES), since=now)
            for name, (count, total) in self._events.items():
                result.append((0, prefix + name, total / count))
            self._events.clear()
        return result

    def flush(self):
        with self._sending:
            values = self.snapshot()
            if not values:
                return
            try:
                if self._client is None:
                    self._client = _native.UploadClient(self._root / "uploads.sock", automatic=True)
                for step, name, value in values:
                    self._client.metric(step, name, value)
                self._client.flush(timeout=5)
            except Exception as error:
                logging.getLogger("tensorlane").warning("performance metrics upload failed: %s", error)
                self._client = None

    def _run(self):
        while not self._stopped.wait(INTERVAL):
            self.flush()

    def close(self):
        self._stopped.set()
        self._thread.join()
        self.flush()
        if self._client is not None:
            self._client.close()
