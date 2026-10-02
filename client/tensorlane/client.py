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
from collections.abc import Callable, Iterator, Mapping
from multiprocessing.connection import Listener

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
        config: str | dict,
        rank: int,
        native: _native.Daemon | None = None,
        assets: dict[str, Path] | None = None,
        streams: list[str] | tuple[str, ...] = (),
        asset_metadata: dict | None = None,
        ranks: int = 1,
        num_workers: int = 5,
        prefetch_factor: int = 2,
    ) -> None:
        self._root = root
        self.run_id = run_id
        self.config = json.loads(config) if isinstance(config, str) else config
        if not isinstance(self.config, dict):
            raise TypeError("config must be a JSON object")
        self.streams = tuple(streams)
        self.asset_metadata = asset_metadata or {}
        self._rank = rank
        self.ranks = ranks
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self._assets = assets if assets is not None else {}
        self._closed = False
        self._native = native
        self._processes = []
        self._queue = None
        self._stopped = None
        self._monitor = None
        self._uploads = None

    @property
    def rank(self) -> int:
        return self._rank

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

    def batches(self, stream: str = "training", *, timeout: float = 120) -> BatchReader:
        if self._closed:
            raise RuntimeError("TensorLane handle is closed")
        if stream not in self.streams:
            raise ValueError(f"unknown stream: {stream}")
        return BatchReader(
            self._root, self._rank, timeout, self._check, self.streams.index(stream)
        )

    def asset(self, name: str) -> Path:
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
        except BaseException as error:
            self._fail(error)
            raise
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
        self._upload_client().metric(step, name, value)

    def metric_artifact(
        self,
        step: int,
        path: str | Path,
        name: str,
        content_type: str = "application/octet-stream",
    ) -> None:
        self._upload_client().metric_artifact(
            step, Path(path).absolute(), name, content_type
        )

    def save_asset(
        self,
        name: str,
        path: str | Path,
        *,
        step: int = 0,
        kind: str = "file",
        asset_type: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("asset metadata must be an object")
        return self._upload_client().save_asset(
            name,
            Path(path).absolute(),
            step,
            kind,
            asset_type,
            json.dumps(metadata or {}, allow_nan=False),
        )

    def flush(self, *, timeout: float = 300) -> None:
        if self._closed:
            raise RuntimeError("TensorLane handle is closed")
        if self._uploads is not None:
            self._uploads.flush(timeout)

    def __enter__(self) -> TensorLane:
        return self

    def _fail(self, error: BaseException) -> None:
        (self._root / "failed").write_text(str(error))

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is not None:
            self._fail(exc_value)
        self.close()


def _validate_callbacks(spec, streams=None):
    if spec is None:
        return
    if isinstance(spec, Mapping):
        if streams is not None and any(name not in streams for name in spec):
            raise ValueError("callback mapping contains an unknown stream")
        if any(
            not isinstance(name, str) or not callable(fn) for name, fn in spec.items()
        ):
            raise TypeError("callback mapping must contain callables")
    elif not callable(spec):
        raise TypeError("callback must be callable or a mapping")


def init(
    run_id: str | None = None,
    transform=None,
    ranks: int | None = None,
    prefetch_factor: int | None = None,
    *,
    rank: int | None = None,
    start_daemon: bool | None = None,
    num_workers: int | None = None,
    collate_fn=None,
    addr: str | None = None,
    ipc_dir: str | Path | None = None,
    timeout: float = 120,
) -> TensorLane:
    run_id = run_id or os.environ.get("TENSORLANE_RUN_ID")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("provide run_id or set TENSORLANE_RUN_ID")
    if rank is None:
        rank = int(os.environ.get("RANK", "0"))
    if start_daemon is None:
        start_daemon = rank == 0
    for name, value in (
        ("ranks", ranks),
        ("prefetch_factor", prefetch_factor),
        ("num_workers", num_workers),
    ):
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer")
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
        _validate_callbacks(transform, metadata["streams"])
        _validate_callbacks(collate_fn, metadata["streams"])
        if rank >= int((root / "ranks").read_text()):
            raise ValueError("invalid rank")
        return TensorLane(
            root,
            metadata["run_id"],
            metadata["config"],
            rank,
            assets={name: Path(path) for name, path in metadata["assets"].items()},
            streams=metadata["streams"],
            asset_metadata=metadata["asset_metadata"],
            ranks=metadata["ranks"],
            num_workers=metadata["num_workers"],
            prefetch_factor=metadata["prefetch_factor"],
        )

    if ranks is not None and rank >= ranks:
        raise ValueError("invalid rank")
    _validate_callbacks(transform)
    _validate_callbacks(collate_fn)

    native = _native.Daemon(
        run_id,
        addr or os.environ.get("TENSORLANE_ADDR", "localhost:8181"),
        root,
        ranks,
        prefetch_factor,
        num_workers,
        rank,
    )
    daemon = TensorLane(
        root,
        native.run_id,
        native.config,
        rank,
        native,
        assets={name: Path(path) for name, path in native.assets.items()},
        streams=native.streams,
        asset_metadata={
            name: json.loads(value) for name, value in native.asset_metadata.items()
        },
        ranks=native.ranks,
        num_workers=native.num_workers,
        prefetch_factor=native.prefetch_factor,
    )

    ranks = daemon.ranks
    num_workers = daemon.num_workers
    from ._process import collate_worker, transform_worker

    context = multiprocessing.get_context("spawn")
    try:
        _validate_callbacks(transform, daemon.streams)
        _validate_callbacks(collate_fn, daemon.streams)
        daemon._queue = context.Queue()
        daemon._stopped = context.Event()
        collator_ready = context.Event()
        for worker_index in range(num_workers):
            process = context.Process(
                name=f"tensorlane-transform-{worker_index}",
                target=transform_worker,
                args=(root, transform, daemon.streams, daemon._queue, daemon._stopped),
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
                daemon.streams,
                collate_fn,
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
                    "ranks": daemon.ranks,
                    "num_workers": daemon.num_workers,
                    "prefetch_factor": daemon.prefetch_factor,
                    "config": daemon.config,
                    "streams": daemon.streams,
                    "asset_metadata": daemon.asset_metadata,
                    "assets": {
                        name: str(path) for name, path in daemon._assets.items()
                    },
                }
            )
        )
        temporary.replace(root / "init.json")
        native.check()
        return daemon
    except BaseException as error:
        daemon._fail(error)
        daemon.close()
        raise


class BatchReader(Iterator[Batch]):
    def __init__(
        self,
        root: Path,
        rank: int,
        timeout: float,
        check: Callable[[], None],
        stream_index: int = 0,
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
            directory = self._root / "streams" / str(stream_index)
            self._semaphore = _native.Semaphore((directory / "semaphore").read_text())
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
