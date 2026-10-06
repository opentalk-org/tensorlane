from __future__ import annotations
import errno
import time
from pathlib import Path
from collections.abc import Callable, Iterator
from multiprocessing.connection import Listener
import socket
from . import _native
from . import _ipc
from .data import Batch
from ._performance import Performance


class BatchReader(Iterator[Batch]):
    def __init__(
        self,
        root: Path,
        rank: int,
        timeout: float,
        check: Callable[[], None],
        stream_index: int,
        stream: str,
        performance: Performance | None = None,
    ) -> None:
        if rank < 0 or timeout <= 0:
            raise ValueError("rank must be nonnegative and timeout must be positive")
        self._check = check
        self._performance = performance
        self._stream = stream
        self._returned = None
        if performance is not None:
            performance.open(stream)
        self._closed = False
        self._connection = None
        self._listener = None
        self._semaphore = None
        self._memory_semaphore = None
        deadline = time.monotonic() + timeout
        try:
            self._check()
            self._ranks = int((root / "ranks").read_text())
            if rank >= self._ranks:
                raise RuntimeError("invalid rank")
            key = (root / "auth").read_bytes()
            directory = root / "streams" / str(stream_index)
            self._progress = directory / f"progress-{rank}"
            self._semaphore = _native.Semaphore((directory / "semaphore").read_text())
            self._memory_semaphore = _native.Semaphore(
                (directory / "memory").read_text()
            )
            try:
                self._listener = Listener(
                    str(directory / f"rank-{rank}.sock"),
                    family="AF_UNIX",
                    authkey=key,
                )
            except OSError as error:
                if error.errno == errno.EADDRINUSE:
                    raise RuntimeError("rank already connected") from error
                raise
            self._listener._listener._socket.settimeout(0.1)
            while self._connection is None:
                self._check()
                if time.monotonic() >= deadline:
                    raise TimeoutError("collator did not connect to rank")
                try:
                    self._connection = self._listener.accept()
                except socket.timeout:
                    pass
            while not self._connection.poll(0.1):
                self._check()
                if time.monotonic() >= deadline:
                    raise TimeoutError("rank handshake timed out")
            kind, value = self._connection.recv()
            if kind != "ready":
                raise RuntimeError(value)
        except (EOFError, OSError) as error:
            self.close()
            self._check()
            raise RuntimeError("TensorLane collater disconnected") from error
        except BaseException:
            self.close()
            raise

    def __next__(self) -> Batch:
        if self._closed:
            raise StopIteration
        connection = self._connection
        if connection is None:
            raise RuntimeError("rank reader is not connected")
        started = time.monotonic()
        self._record_application(started)
        try:
            while not connection.poll(0.1):
                self._check()
            kind, value = _ipc.recv(connection)
            if kind == "end":
                self.close()
                raise StopIteration
            if kind == "error":
                raise RuntimeError(value)
            if kind != "batch":
                raise RuntimeError(f"unexpected rank message: {kind}")
            batch = value
            temporary = self._progress.with_suffix(".tmp")
            temporary.write_text(str(batch.batch_id + self._ranks))
            temporary.replace(self._progress)
            self._memory_semaphore.post(batch._memory_units)
            self._semaphore.post()
            self._returned = time.monotonic()
            if self._performance is not None:
                self._performance.batch(batch, self._returned - started)
            return batch
        except (EOFError, OSError) as error:
            if self._performance is not None:
                self._performance.error(self._stream)
            self._check()
            raise RuntimeError("TensorLane collater disconnected") from error
        except StopIteration:
            raise
        except Exception:
            if self._performance is not None:
                self._performance.error(self._stream)
            raise

    def _record_application(self, now):
        if self._returned is not None and self._performance is not None:
            self._performance.application(self._stream, now - self._returned)
        self._returned = None

    def close(self) -> None:
        if self._closed:
            return
        self._record_application(time.monotonic())
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        if self._listener is not None:
            try:
                self._listener.close()
            except FileNotFoundError:
                pass
            self._listener = None
        self._semaphore = None
        self._memory_semaphore = None
        self._closed = True

    def __enter__(self) -> BatchReader:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
