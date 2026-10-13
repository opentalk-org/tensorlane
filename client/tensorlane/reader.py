from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import suppress
import errno
import math
from multiprocessing.connection import Listener
from pathlib import Path
import socket
import time

from . import _native
from . import _ipc
from .data import Batch
from ._performance import Performance


class BatchReader(Iterator[Batch]):
    def __init__(
        self,
        root: Path,
        rank: int,
        timeout: float | None,
        check: Callable[[], None],
        stream_index: int,
        stream: str,
        performance: Performance | None,
    ) -> None:
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive")
        self._check = check
        self._performance = performance
        self._stream = stream
        self._returned = None
        self._closed = False
        self._connection = None
        self._listener = None
        if performance is not None:
            performance.open(stream)
        deadline = time.monotonic() + (math.inf if timeout is None else timeout)
        check()
        self._ranks = int((root / "ranks").read_text())
        directory = root / "streams" / str(stream_index)
        self._progress = directory / f"progress-{rank}"
        self._semaphore = _native.Semaphore((directory / "semaphore").read_text())
        self._memory_semaphore = _native.Semaphore((directory / "memory").read_text())
        try:
            self._listener = Listener(
                str(directory / f"rank-{rank}.sock"),
                family="AF_UNIX",
                authkey=(root / "auth").read_bytes(),
            )
        except OSError as error:
            if error.errno != errno.EADDRINUSE:
                raise
            raise RuntimeError("rank already connected") from error
        try:
            self._listener._listener._socket.settimeout(0.1)
            while self._connection is None:
                check()
                if time.monotonic() >= deadline:
                    raise TimeoutError("collator did not connect to rank")
                with suppress(socket.timeout):
                    self._connection = self._listener.accept()
            while not self._connection.poll(0.1):
                check()
                if time.monotonic() >= deadline:
                    raise TimeoutError("rank handshake timed out")
            self._connection.recv()
        except (EOFError, OSError) as error:
            self.close()
            check()
            raise RuntimeError("TensorLane collater disconnected") from error
        except BaseException:
            self.close()
            raise

    def __next__(self) -> Batch:
        if self._closed:
            raise StopIteration
        started = time.monotonic()
        if self._returned is not None and self._performance is not None:
            self._performance.application(self._stream, started - self._returned)
        self._returned = None
        try:
            self._connection.send(("next", None))
            while not self._connection.poll(0.1):
                self._check()
            kind, batch = _ipc.recv(self._connection)
        except Exception as error:
            if self._performance is not None:
                self._performance.error(self._stream)
            if not isinstance(error, (EOFError, OSError)):
                raise
            self._check()
            raise RuntimeError("TensorLane collater disconnected") from error
        if kind == "end":
            self.close()
            raise StopIteration
        temporary = self._progress.with_suffix(".tmp")
        temporary.write_text(str(batch.batch_id + self._ranks))
        temporary.replace(self._progress)
        self._memory_semaphore.post(batch._memory_units)
        self._semaphore.post()
        self._returned = time.monotonic()
        if self._performance is not None:
            self._performance.batch(batch, self._returned - started)
        return batch

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._returned is not None and self._performance is not None:
            self._performance.application(
                self._stream, time.monotonic() - self._returned
            )
        if self._connection is not None:
            self._connection.close()
        if self._listener is not None:
            with suppress(FileNotFoundError):
                self._listener.close()

    def __enter__(self) -> BatchReader:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
