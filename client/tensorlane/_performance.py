from __future__ import annotations

import logging
import threading
import time
from urllib.parse import quote

from . import _native

INTERVAL = 10.0
STAGES = ("server_load", "server_wait", "grpc_receive_wait", "transform_work", "collate", "data_wait")


class Performance:
    """Aggregate timings in memory; upload off the batch-consumption path."""

    def __init__(self, root, rank):
        self._root = root
        self._rank = rank
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
                batches=0, samples=0, errors=0, step=0, active=0,
                window_batches=0, window_samples=0, application=0.0, intervals=0,
                dirty=True, last_batch=time.monotonic(), timings=[0.0] * len(STAGES),
                wait_max=0.0, since=time.monotonic(),
            )
        return self._streams[stream]

    def open(self, stream):
        with self._lock:
            self._state(stream)["active"] += 1

    def finish(self, stream):
        with self._lock:
            self._state(stream)["active"] -= 1
            self._state(stream)["dirty"] = True

    def application(self, stream, seconds):
        with self._lock:
            state = self._state(stream)
            state["dirty"] = True
            state["application"] += seconds
            state["intervals"] += 1

    def batch(self, batch, wait):
        with self._lock:
            state = self._state(batch.stream)
            state["dirty"] = True
            state["last_batch"] = time.monotonic()
            state["step"] = batch.batch_id
            state["batches"] += 1
            state["samples"] += len(batch)
            state["window_batches"] += 1
            state["window_samples"] += len(batch)
            for index, seconds in enumerate((*batch._timings, wait)):
                state["timings"][index] += seconds
            state["wait_max"] = max(state["wait_max"], wait)

    def error(self, stream):
        with self._lock:
            self._state(stream)["errors"] += 1
            self._state(stream)["dirty"] = True

    def event(self, name, seconds):
        with self._lock:
            count, total = self._events.get(name, (0, 0.0))
            self._events[name] = (count + 1, total + seconds)

    def snapshot(self):
        now = time.monotonic()
        result = []
        prefix = f"tensorlane/rank/{self._rank}/"
        with self._lock:
            for stream, state in self._streams.items():
                batches, samples = state["window_batches"], state["window_samples"]
                elapsed = max(now - state["since"], 1e-9)
                if not (state["active"] or state["dirty"]):
                    continue
                values = dict(
                    batches_total=state["batches"], samples_total=state["samples"],
                    errors_total=state["errors"], seconds_since_last_batch=now - state["last_batch"],
                    batches_per_second=batches / elapsed,
                    samples_per_second=samples / elapsed,
                    application_seconds_mean=state["application"] / max(state["intervals"], 1),
                    loop_seconds_mean=(state["application"] + state["timings"][-1]) / max(batches, state["intervals"], 1),
                    data_wait_seconds_max=state["wait_max"],
                    data_wait_fraction=state["timings"][-1] / max(state["timings"][-1] + state["application"], 1e-9),
                )
                if batches:
                    for name, total in zip(STAGES, state["timings"]):
                        values[f"{name}_seconds_mean"] = total / batches
                for name, value in values.items():
                    result.append((state["step"], prefix + quote(stream, safe="") + "/" + name, value))
                state.update(window_batches=0, window_samples=0, application=0.0,
                             intervals=0, dirty=False, timings=[0.0] * len(STAGES), wait_max=0.0, since=now)
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
