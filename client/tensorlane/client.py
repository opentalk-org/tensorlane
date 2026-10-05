from __future__ import annotations

import hashlib
import asyncio
import fcntl
import json
import mimetypes
import os
from pathlib import Path
import tempfile
import time


from . import _native
from .reader import BatchReader
from ._performance import Performance


def _content_type(path):
    suffix = Path(path).suffix.lower()
    if suffix in {".pt", ".pth", ".ckpt", ".safetensors"}:
        return "application/octet-stream"
    return mimetypes.guess_type(str(path))[0] or "application/octet-stream"


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
        performance_metrics: bool = True,
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
        self._performance = None
        self._performance_enabled = performance_metrics

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

    def batches(self, stream: str, *, timeout: float = 120) -> BatchReader:
        if self._closed:
            raise RuntimeError("TensorLane handle is closed")
        if stream not in self.streams:
            raise ValueError(f"unknown stream: {stream}")
        return BatchReader(
            self._root,
            self._rank,
            timeout,
            self._check,
            self.streams.index(stream),
            stream,
            self._performance_collector(),
        )

    def _performance_collector(self):
        if self._closed:
            raise RuntimeError("TensorLane handle is closed")
        if self._performance is None and self._performance_enabled and self._rank == 0:
            self._performance = Performance(self._root)
        return self._performance

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
                started = time.monotonic()
                try:
                    self._uploads.close()
                finally:
                    if self._performance is not None:
                        self._performance.event(
                            "uploads/flush_seconds", time.monotonic() - started
                        )
        except BaseException as error:
            self._fail(error)
            raise
        finally:
            self._uploads = None
            try:
                if self._performance is not None:
                    self._performance.close()
            finally:
                self._close_daemon()

    async def close_async(self) -> None:
        await asyncio.to_thread(self.close)

    def _close_daemon(self) -> None:
        if self._native is None:
            return
        (self._root / "init.json").unlink(missing_ok=True)
        self._native.stop()
        if self._stopped is not None:
            self._stopped.set()
        if self._monitor is not None:
            self._monitor.join()
            self._monitor = None
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
        content_type: str | None = None,
    ) -> None:
        self._performance_collector()
        self._upload_client().metric_artifact(
            step, Path(path).absolute(), name, content_type or _content_type(path)
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
        self._performance_collector()
        return self._upload_client().save_asset(
            name,
            Path(path).absolute(),
            step,
            kind,
            asset_type,
            json.dumps(metadata or {}, allow_nan=False),
            _content_type(path),
        )

    def flush(self, *, timeout: float | None = None) -> None:
        if self._closed:
            raise RuntimeError("TensorLane handle is closed")
        deadline = None if timeout is None else time.monotonic() + timeout
        if self._uploads is not None:
            started = time.monotonic()
            remaining = None if deadline is None else max(0, deadline - started)
            try:
                self._uploads.flush(remaining)
            finally:
                if self._performance is not None:
                    self._performance.event(
                        "uploads/flush_seconds", time.monotonic() - started
                    )
        if self._performance is not None:
            self._performance.flush(deadline=deadline)

    async def flush_async(self, *, timeout: float | None = None) -> None:
        await asyncio.to_thread(self.flush, timeout=timeout)

    def __enter__(self) -> TensorLane:
        return self

    def _fail(self, error: BaseException) -> None:
        (self._root / "failed").write_text(str(error))

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is not None:
            self._fail(exc_value)
        self.close()
