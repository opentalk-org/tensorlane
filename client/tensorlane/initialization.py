from __future__ import annotations
import asyncio
import json
import os
import time
import threading
from pathlib import Path
from collections.abc import Mapping
import multiprocessing
from . import _native
from .client import TensorLane, _root, _check_alive


def _validate_callbacks(spec):
    if isinstance(spec, Mapping):
        for name, fn in spec.items():
            if not isinstance(name, str) or not callable(fn):
                raise TypeError("callback mapping must contain callables")
        return dict(spec)
    if spec is not None and not callable(spec):
        raise TypeError("callback must be callable or a mapping")
    return spec


def _stream_callbacks(spec, streams):
    if isinstance(spec, dict):
        if spec.keys() - set(streams):
            raise ValueError("callback mapping contains an unknown stream")
        return spec
    return dict.fromkeys(streams, spec)


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
    if timeout is not None and timeout <= 0:
        raise ValueError("timeout must be positive")
    deadline = None if timeout is None else time.monotonic() + timeout
    if start_daemon and ranks is not None and rank >= ranks:
        raise ValueError("invalid rank")
    transform = _validate_callbacks(transform)
    collate_fn = _validate_callbacks(collate_fn)
    root = _root(run_id, ipc_dir)
    native = None
    if not start_daemon:
        metadata_path = root / "init.json"
        while not metadata_path.exists():
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"TensorLane init did not become ready for run {run_id!r}"
                )
            time.sleep(0.02)
        metadata = json.loads(metadata_path.read_text())
        _check_alive(root)
        _stream_callbacks(transform, metadata["streams"])
        _stream_callbacks(collate_fn, metadata["streams"])
        if rank >= int((root / "ranks").read_text()):
            raise ValueError("invalid rank")
    else:
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
    daemon = TensorLane(
        root,
        metadata["run_id"],
        metadata["config"],
        rank,
        native,
        assets={name: Path(path) for name, path in metadata["assets"].items()},
        streams=metadata["streams"],
        asset_metadata=metadata["asset_metadata"],
        ranks=metadata["ranks"],
        num_workers=metadata["num_workers"],
        prefetch_factor=metadata["prefetch_factor"],
        performance_metrics=performance_metrics,
    )
    if native is None:
        return daemon

    ranks = daemon.ranks
    num_workers = daemon.num_workers
    from ._process import collate_worker, transform_worker

    context = multiprocessing.get_context("forkserver")
    context.set_forkserver_preload(["torch.multiprocessing"])
    try:
        transform = _stream_callbacks(transform, daemon.streams)
        collate_fn = _stream_callbacks(collate_fn, daemon.streams)
        daemon._stopped = context.Event()
        collator_ready = context.Event()
        for worker_index in range(num_workers):
            incoming, output = context.Pipe()
            daemon._connections.append(incoming)
            process = context.Process(
                name=f"tensorlane-transform-{worker_index}",
                target=transform_worker,
                args=(root, transform, daemon.streams, output, daemon._stopped),
            )
            with output:
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
                daemon._connections,
                daemon._stopped,
                collator_ready,
            ),
        )
        process.start()
        daemon._processes.append(process)
        for connection in daemon._connections:
            connection.close()
        daemon._connections.clear()
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
            if deadline is not None and time.monotonic() >= deadline:
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


async def init_async(*args, **kwargs) -> TensorLane:
    return await asyncio.to_thread(init, *args, **kwargs)
