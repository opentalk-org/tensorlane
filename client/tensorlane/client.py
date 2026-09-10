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

import torch.multiprocessing as multiprocessing

from . import _native
from .data import Batch


def _root(run_id: str, ipc_dir: str | Path | None) -> Path:
    base = Path(ipc_dir) if ipc_dir is not None else Path(tempfile.gettempdir())
    identifier = hashlib.sha256(run_id.encode()).hexdigest()[:16]
    return base / f"tl-{os.getuid()}-{identifier}"


def _check_alive(root: Path) -> None:
    try:
        with (root / "lock").open("rb") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
            raise RuntimeError("TensorLane daemon stopped unexpectedly")
    except FileNotFoundError as error:
        raise RuntimeError("TensorLane daemon disappeared") from error


class TensorLane:
    def __init__(
        self,
        root: Path,
        run_id: str,
        train_config: str,
        rank: int,
        native: _native.Daemon | None = None,
        assets: dict[str, Path] | None = None,
    ) -> None:
        self._root = root
        self.run_id = run_id
        self.train_config = train_config
        self._rank = rank
        self._assets = assets if assets is not None else {}
        self._closed = False
        self._native = native
        self._processes = []
        self._queue = None
        self._stopped = None
        self._monitor = None
        self._uploads = None

    def _supervise_processes(self, root: Path) -> None:
        while not self._stopped.wait(0.05):
            try:
                self._native.check()
            except RuntimeError:
                (root / "init.json").unlink(missing_ok=True)
                self._stopped.set()
                return
            for process in self._processes:
                if process.exitcode is not None:
                    self._native.stop(
                        f"{process.name} exited unexpectedly: {process.exitcode}"
                    )
                    self._stopped.set()
                    return

    def _check(self) -> None:
        if self._native is not None:
            self._native.check()
        else:
            if not (self._root / "init.json").exists():
                raise RuntimeError("TensorLane daemon is closed")
            _check_alive(self._root)

    def batches(self, validation: bool = False, *, timeout: float = 120) -> BatchReader:
        if self._closed:
            raise RuntimeError("TensorLane handle is closed")
        return BatchReader(self._root, self._rank, timeout, self._check, validation)

    def asset(self, name: str) -> Path:
        """Return the downloaded file unchanged; decoding or unpacking is up to the caller."""
        if self._closed:
            raise RuntimeError("TensorLane handle is closed")
        return self._assets[name]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._uploads is not None:
                self._uploads.close()
        finally:
            self._uploads = None
            self._close_daemon()

    def _close_daemon(self) -> None:
        if self._native is None:
            return
        (self._root / "init.json").unlink(missing_ok=True)
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

    def _upload_client(self):
        if self._closed:
            raise RuntimeError("TensorLane handle is closed")
        if self._uploads is None:
            self._check()
            self._uploads = _native.UploadClient(self._root / "uploads.sock")
        return self._uploads

    def metric(self, step: int, name: str, value: float) -> None:
        """Queue a scalar metric; the daemon supplies its timestamp."""
        self._upload_client().metric(step, name, value)

    def metric_artifact(
        self,
        step: int,
        path: str | Path,
        name: str,
        content_type: str = "application/octet-stream",
    ) -> None:
        """Queue a file unchanged. Keep it available and unchanged until flush completes."""
        self._upload_client().metric_artifact(
            step, Path(path).absolute(), name, content_type
        )

    def checkpoint(self, step: int, path: str | Path) -> None:
        """Queue a file unchanged. Keep it available and unchanged until flush completes."""
        self._upload_client().checkpoint(step, Path(path).absolute())

    def flush(self, *, timeout: float = 300) -> None:
        """Wait for this rank's uploads and surface errors. Flush all ranks before the shutdown barrier."""
        if self._closed:
            raise RuntimeError("TensorLane handle is closed")
        if self._uploads is not None:
            self._uploads.flush(timeout)

    def __enter__(self) -> TensorLane:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def init(
    run_id: str,
    transform: Callable[[Tensor], Tensor],
    ranks: int,
    prefetch_factor: int = 2,
    *,
    rank: int,
    start_daemon: bool,
    num_workers: int = 5,
    addr: str | None = None,
    ipc_dir: str | Path | None = None,
    timeout: float = 120,
) -> TensorLane:
    """Initialize on every rank; exactly one caller must start the daemon."""
    if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0:
        raise ValueError("rank must be a nonnegative integer")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    root = _root(run_id, ipc_dir)
    if not start_daemon:
        deadline = time.monotonic() + timeout
        metadata_path = root / "init.json"
        while not metadata_path.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"TensorLane init did not become ready for run {run_id!r}"
                )
            time.sleep(0.02)
        metadata = json.loads(metadata_path.read_text())
        _check_alive(root)
        if rank >= int((root / "ranks").read_text()):
            raise ValueError("invalid rank")
        return TensorLane(
            root,
            metadata["run_id"],
            metadata["train_config"],
            rank,
            assets={name: Path(path) for name, path in metadata["assets"].items()},
        )

    if ranks <= 0 or prefetch_factor <= 0:
        raise ValueError("ranks and prefetch_factor must be positive")
    if rank >= ranks:
        raise ValueError("invalid rank")
    if not callable(transform):
        raise TypeError("transform must be callable")
    if (
        not isinstance(num_workers, int)
        or isinstance(num_workers, bool)
        or num_workers <= 0
    ):
        raise ValueError("num_workers must be a positive integer")

    native = _native.Daemon(
        run_id,
        addr or os.environ.get("TENSORLANE_ADDR", "localhost:8181"),
        root,
        ranks,
        prefetch_factor,
        num_workers,
    )
    daemon = TensorLane(
        root,
        native.run_id,
        native.train_config,
        rank,
        native,
        assets={name: Path(path) for name, path in native.assets.items()},
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
        deadline = time.monotonic() + timeout
        while not native.ready() or not collator_ready.is_set():
            for process in daemon._processes:
                if process.exitcode is not None:
                    raise RuntimeError(
                        f"{process.name} exited during startup: {process.exitcode}"
                    )
            if time.monotonic() >= deadline:
                raise TimeoutError("TensorLane workers did not become ready")
            time.sleep(0.02)
        temporary = root / "init.tmp"
        temporary.write_text(
            json.dumps(
                {
                    "run_id": daemon.run_id,
                    "train_config": daemon.train_config,
                    "assets": {
                        name: str(path) for name, path in daemon._assets.items()
                    },
                }
            )
        )
        temporary.replace(root / "init.json")
        native.check()
        return daemon
    except BaseException:
        daemon.close()
        raise


class BatchReader(Iterator[Batch]):
    def __init__(
        self,
        root: Path,
        rank: int,
        timeout: float,
        check: Callable[[], None],
        validation: bool = False,
    ) -> None:
        if rank < 0 or timeout <= 0:
            raise ValueError("rank must be nonnegative and timeout must be positive")
        self._root = root
        self._check = check
        self._rank = rank
        self._closed = False
        self._connection = None
        self._listener = None
        self._semaphore = None
        deadline = time.monotonic() + timeout
        try:
            self._check()
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
                self._check()
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
            self._check()
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
