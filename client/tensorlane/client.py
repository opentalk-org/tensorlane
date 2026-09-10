from __future__ import annotations

import hashlib
import errno
import fcntl
import json
import os
from pathlib import Path
import socket
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from multiprocessing.connection import Listener
from torch import Tensor
from typing import Any

import torch.multiprocessing as multiprocessing

from . import _native
from .data import Batch


def _root(run_id: str, ipc_dir: str | Path | None) -> Path:
    base = Path(ipc_dir) if ipc_dir is not None else Path(tempfile.gettempdir())
    identifier = hashlib.sha256(run_id.encode()).hexdigest()[:16]
    return base / f"tl-{os.getuid()}-{identifier}"


def _status(root: Path) -> dict[str, Any]:
    try:
        return json.loads((root / "status.json").read_text())
    except FileNotFoundError:
        return {"state": "starting"}


def _failure(root: Path) -> None:
    status = _status(root)
    if status["state"] in ("failed", "closed", "stopping"):
        raise RuntimeError(status.get("error") or "TensorLane daemon is closed")
    if status["state"] == "ready":
        try:
            with (root / "lock").open("rb") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return
                raise RuntimeError("TensorLane daemon stopped unexpectedly")
        except FileNotFoundError as error:
            raise RuntimeError("TensorLane daemon disappeared") from error


class Daemon:
    def __init__(self, native: _native.Daemon) -> None:
        self._native = native
        self._processes = []
        self._queue = None
        self._stopped = None
        self._monitor = None

    def _supervise_processes(self, root: Path) -> None:
        while not self._stopped.wait(0.05):
            if _status(root)["state"] in ("stopping", "closed", "failed"):
                self._stopped.set()
                return
            for process in self._processes:
                if process.exitcode is not None:
                    self._native.stop(
                        f"{process.name} exited unexpectedly: {process.exitcode}"
                    )
                    self._stopped.set()
                    return

    @property
    def run_id(self) -> str:
        return self._native.run_id

    @property
    def train_config(self) -> str:
        return self._native.train_config

    def close(self) -> None:
        if self._stopped is not None:
            self._stopped.set()
        if self._monitor is not None:
            self._monitor.join()
            self._monitor = None
        self._native.stop()
        try:
            deadline = time.monotonic() + 5
            for process in self._processes:
                process.join(max(0, deadline - time.monotonic()))
            for process in self._processes:
                if process.is_alive():
                    process.kill()
                process.join()
                process.close()
            self._processes.clear()
        finally:
            if self._queue is not None:
                self._queue.close()
                self._queue.join_thread()
                self._queue = None
            self._native.close()

    def __enter__(self) -> Daemon:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def init(
    run_id: str,
    transform: Callable[[Tensor], Tensor],
    ranks: int,
    prefetch_factor: int = 2,
    *,
    num_workers: int = 5,
    addr: str | None = None,
    ipc_dir: str | Path | None = None,
) -> Daemon:
    """Start once, in the main rank. Transform must accept and return a CPU tensor."""
    if ranks <= 0 or prefetch_factor <= 0:
        raise ValueError("ranks and prefetch_factor must be positive")
    if not callable(transform):
        raise TypeError("transform must be callable")
    if (
        not isinstance(num_workers, int)
        or isinstance(num_workers, bool)
        or num_workers <= 0
    ):
        raise ValueError("num_workers must be a positive integer")

    root = _root(run_id, ipc_dir)

    daemon = Daemon(
        _native.Daemon(
            run_id,
            addr or os.environ.get("TENSORLANE_ADDR", "localhost:8181"),
            root,
            ranks,
            prefetch_factor,
            num_workers,
        )
    )

    from ._process import collate_worker, transform_worker

    context = multiprocessing.get_context("spawn")
    try:
        daemon._queue = context.Queue()
        daemon._stopped = context.Event()
        collator_ready = context.Event()
        for worker_index in range(num_workers):
            process = context.Process(
                name=f"tensorlane-transform-{worker_index}",
                target=transform_worker,
                args=(root, transform, daemon._queue, daemon._stopped),
            )
            process.start()
            daemon._processes.append(process)
        process = context.Process(
            name="tensorlane-collate",
            target=collate_worker,
            args=(
                root,
                ranks,
                num_workers,
                daemon._queue,
                daemon._stopped,
                collator_ready,
            ),
        )
        process.start()
        daemon._processes.append(process)
        daemon._monitor = threading.Thread(
            target=daemon._supervise_processes,
            args=(root,),
            name="tensorlane-process-monitor",
            daemon=True,
        )
        daemon._monitor.start()
        deadline = time.monotonic() + 120
        while _status(root)["state"] != "ready" or not collator_ready.is_set():
            _failure(root)
            for process in daemon._processes:
                if process.exitcode is not None:
                    raise RuntimeError(
                        f"{process.name} exited during startup: {process.exitcode}"
                    )
            if time.monotonic() >= deadline:
                raise TimeoutError("TensorLane workers did not become ready")
            time.sleep(0.02)
        return daemon
    except BaseException:
        daemon.close()
        raise


class BatchReader(Iterator[Batch]):
    def __init__(
        self,
        run_id: str,
        rank: int,
        ipc_dir: str | Path | None,
        timeout: float,
        validation: bool = False,
    ) -> None:
        if rank < 0 or timeout <= 0:
            raise ValueError("rank must be nonnegative and timeout must be positive")
        self._root = _root(run_id, ipc_dir)
        self._rank = rank
        self._closed = False
        self._connection = None
        self._listener = None
        self._semaphore = None
        deadline = time.monotonic() + timeout
        try:
            while _status(self._root)["state"] != "ready":
                _failure(self._root)
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"TensorLane init did not become ready for run {run_id!r}"
                    )
                time.sleep(0.02)
            _failure(self._root)
            if rank >= int((self._root / "ranks").read_text()):
                raise RuntimeError("invalid rank")
            key = (self._root / "auth").read_bytes()
            multiprocessing.current_process().authkey = key
            prefix = "validation-" if validation else ""
            self._semaphore = _native.Semaphore(
                (self._root / f"{prefix}semaphore").read_text()
            )
            try:
                self._listener = Listener(
                    str(self._root / f"{prefix}rank-{rank}.sock"),
                    family="AF_UNIX",
                    authkey=key,
                )
            except OSError as error:
                if error.errno == errno.EADDRINUSE:
                    raise RuntimeError("rank already connected") from error
                raise
            self._listener._listener._socket.settimeout(0.1)
            while self._connection is None:
                _failure(self._root)
                if time.monotonic() >= deadline:
                    raise TimeoutError("collator did not connect to rank")
                try:
                    self._connection = self._listener.accept()
                except socket.timeout:
                    pass
            while not self._connection.poll(0.1):
                _failure(self._root)
                if time.monotonic() >= deadline:
                    raise TimeoutError("rank handshake timed out")
            kind, value = self._connection.recv()
            if kind != "ready":
                raise RuntimeError(value)
        except BaseException:
            self.close()
            raise

    def __iter__(self) -> BatchReader:
        return self

    def __next__(self) -> Batch:
        if self._closed:
            raise StopIteration
        connection = self._connection
        if connection is None:
            raise RuntimeError("rank reader is not connected")
        try:
            while not connection.poll(0.1):
                _failure(self._root)
            kind, value = connection.recv()
            if kind == "end":
                self.close()
                raise StopIteration
            if kind == "error":
                raise RuntimeError(value)
            if kind != "batch":
                raise RuntimeError(f"unexpected rank message: {kind}")
            _batch_id, batch = value
            self._semaphore.post()
            return batch
        except (EOFError, OSError) as error:
            _failure(self._root)
            raise RuntimeError("TensorLane collater disconnected") from error

    def close(self) -> None:
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
        self._closed = True

    def __enter__(self) -> BatchReader:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def batches(
    run_id: str,
    rank: int,
    validation: bool = False,
    *,
    ipc_dir: str | Path | None = None,
    timeout: float = 120,
) -> BatchReader:
    """Read training or validation batches from an existing daemon."""
    return BatchReader(run_id, rank, ipc_dir, timeout, validation)
