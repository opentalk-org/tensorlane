from __future__ import annotations

import asyncio
import atexit
from collections.abc import Mapping
import functools
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import threading
import time

from . import _native
from ._process import collate_worker, transform_worker
from .client import TensorLane, _check_alive, _root


def _stream_callbacks(spec, streams):
    if not isinstance(spec, Mapping):
        return dict.fromkeys(streams, spec)
    if spec.keys() - set(streams):
        raise ValueError("callback mapping contains an unknown stream")
    return dict(spec)


def _excepthook(daemon, previous, kind, value, traceback):
    if not daemon._closed:
        daemon._fail(value)
    previous(kind, value, traceback)


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
    api_key: str | None = None,
    ipc_dir: str | Path | None = None,
    timeout: float | None = None,
    performance_metrics: bool = True,
) -> TensorLane:
    run_id = run_id or os.environ.get("TENSORLANE_RUN_ID")
    if not run_id:
        raise ValueError("provide run_id or set TENSORLANE_RUN_ID")
    if rank is None:
        rank = int(os.environ.get("RANK", "0"))
    if start_daemon is None:
        start_daemon = rank == 0
    for name, value, minimum in (
        ("ranks", ranks, 1),
        ("prefetch_factor", prefetch_factor, 1),
        ("num_workers", num_workers, 1),
        ("rank", rank, 0),
    ):
        if value is not None and (type(value) is not int or value < minimum):
            raise ValueError(f"{name} must be an integer of at least {minimum}")
    if timeout is not None and timeout <= 0:
        raise ValueError("timeout must be positive")
    for spec in (transform, collate_fn):
        callbacks = spec.values() if isinstance(spec, Mapping) else [spec]
        if not all(callback is None or callable(callback) for callback in callbacks):
            raise TypeError("callback must be callable or a mapping of callables")
    deadline = time.monotonic() + (math.inf if timeout is None else timeout)
    root = _root(run_id, ipc_dir)
    native = None
    if start_daemon:
        native = _native.Daemon(
            run_id,
            addr or os.environ.get("TENSORLANE_ADDR", "localhost:8180"),
            root,
            ranks,
            prefetch_factor,
            num_workers,
            rank,
            api_key if api_key is not None else os.environ.get("TENSORLANE_API_KEY"),
            timeout,
        )
        (root / "auth").write_bytes(multiprocessing.current_process().authkey)
        metadata = {
            "run_id": native.run_id,
            "config": json.loads(native.config),
            "assets": {name: str(path) for name, path in native.assets.items()},
            "streams": native.streams,
            "asset_metadata": {
                name: json.loads(value) for name, value in native.asset_metadata.items()
            },
            "ranks": native.ranks,
            "num_workers": native.num_workers,
            "prefetch_factor": native.prefetch_factor,
        }
    else:
        while not (root / "init.json").exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"TensorLane init did not become ready for run {run_id!r}"
                )
            time.sleep(0.02)
        metadata = json.loads((root / "init.json").read_text())
        multiprocessing.current_process().authkey = (root / "auth").read_bytes()
        _check_alive(root)
        _stream_callbacks(transform, metadata["streams"])
        _stream_callbacks(collate_fn, metadata["streams"])
        if rank >= metadata["ranks"]:
            raise ValueError("invalid rank")
    daemon = TensorLane(
        root,
        rank=rank,
        native=native,
        performance_metrics=performance_metrics,
        **metadata,
    )
    sys.excepthook = functools.partial(_excepthook, daemon, sys.excepthook)
    atexit.register(daemon.close)
    if native is None:
        return daemon
    context = multiprocessing.get_context("forkserver")
    context.set_forkserver_preload(["torch.multiprocessing"])
    connections = []
    try:
        transform = _stream_callbacks(transform, daemon.streams)
        collate_fn = _stream_callbacks(collate_fn, daemon.streams)
        daemon._stopped = context.Event()
        collator_ready = context.Event()
        for index in range(daemon.num_workers):
            incoming, output = context.Pipe()
            connections.append(incoming)
            process = context.Process(
                name=f"tensorlane-transform-{index}",
                target=transform_worker,
                args=(
                    os.getpid(),
                    root,
                    transform,
                    daemon.streams,
                    output,
                    daemon._stopped,
                ),
            )
            with output:
                process.start()
            daemon._processes.append(process)
        process = context.Process(
            name="tensorlane-collate",
            target=collate_worker,
            args=(
                os.getpid(),
                root,
                daemon.ranks,
                daemon.num_workers,
                daemon.prefetch_factor,
                daemon.streams,
                collate_fn,
                connections,
                daemon._stopped,
                collator_ready,
            ),
        )
        process.start()
        daemon._processes.append(process)
        daemon._monitor = threading.Thread(
            target=daemon._supervise_processes,
            name="tensorlane-process-monitor",
            daemon=True,
        )
        daemon._monitor.start()
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
        temporary.write_text(json.dumps(metadata))
        temporary.replace(root / "init.json")
        native.check()
        return daemon
    except BaseException as error:
        daemon._fail(error)
        daemon.close()
        raise
    finally:
        for connection in connections:
            connection.close()


async def init_async(*args, **kwargs) -> TensorLane:
    return await asyncio.to_thread(init, *args, **kwargs)
