from __future__ import annotations

from collections import deque
import json
from multiprocessing.connection import Client, wait
from pathlib import Path
import os
import select
import signal
import threading
import time

from . import _native
from . import _ipc
from .data import Batch, RawSample

SIGNALS = {signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGUSR1, signal.SIGUSR2}


def _exit_with(owner: int) -> None:
    select.select([os.pidfd_open(owner)], [], [])
    os._exit(1)


def _connect(path: Path, key: bytes):
    try:
        connection = Client(str(path), family="AF_UNIX", authkey=key)
    except (FileNotFoundError, ConnectionRefusedError):
        return None
    connection.send(("ready", None))
    return connection


def share(value):
    if value is None or isinstance(value, (str, bytes, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {key: share(item) for key, item in value.items()}
    if isinstance(value, list):
        return [share(item) for item in value]
    if isinstance(value, tuple):
        return tuple(share(item) for item in value)
    if isinstance(value, RawSample):
        return RawSample(
            value.sample_id, value.stream, share(value.metadata), share(value.blobs)
        )
    import torch

    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise TypeError("workers must return CPU tensors")
        return value.detach().share_memory_()
    raise TypeError(f"unsupported worker output: {type(value).__name__}")


def transform_worker(
    owner: int, root: Path, transform, streams, output, stopped
) -> None:
    signal.pthread_sigmask(signal.SIG_BLOCK, SIGNALS)
    threading.Thread(target=_exit_with, args=(owner,), daemon=True).start()
    import torch

    torch.set_num_threads(1)
    receiver = _native.Listener(root / "work.sock")
    ended = set()
    with torch.no_grad(), output:
        while (message := receiver.recv()) is not None:
            stream = message["stream"]
            if message["kind"] == "end":
                _ipc.send(output, ("end", stream, None))
                ended.add(stream)
                continue
            function = transform.get(stream)
            parts = []
            for item in message["samples"]:
                started = time.monotonic()
                sample = RawSample(
                    item["sample_id"],
                    stream,
                    json.loads(item["metadata_json"]),
                    item["blobs"],
                )
                try:
                    value = share(function(sample) if function else sample)
                except Exception as error:
                    raise RuntimeError(
                        f"stream {stream} batch {message['batch']} sample {item['index']}: {error}"
                    ) from error
                parts.append((item["index"], value, time.monotonic() - started))
            batch = (
                message["batch"],
                message["query_batch_idx"],
                parts,
                message["timings"],
                message["memory_units"],
            )
            _ipc.send(output, ("samples", stream, batch))
    if len(ended) < len(streams) and not stopped.is_set():
        raise RuntimeError("data stream ended before End")


def collate_worker(
    owner: int,
    root: Path,
    ranks: int,
    num_workers: int,
    prefetch_factor: int,
    streams,
    collate_fn,
    incoming,
    stopped,
    ready,
) -> None:
    signal.pthread_sigmask(signal.SIG_BLOCK, SIGNALS)
    threading.Thread(target=_exit_with, args=(owner,), daemon=True).start()
    collate_fn = collate_fn or {}
    key = (root / "auth").read_bytes()
    directories = [root / "streams" / str(index) for index in range(len(streams))]
    slots = [(stream, rank) for stream in streams for rank in range(ranks)]
    readers = dict.fromkeys(slots)
    outbox = {slot: deque() for slot in slots}
    requested = set()
    activated = set()
    assembling = {stream: {} for stream in streams}
    finished = {stream: {} for stream in streams}
    next_batch = {
        stream: int((directory / "start").read_text())
        for stream, directory in zip(streams, directories)
    }
    ended = dict.fromkeys(streams, 0)
    ready.set()
    while not stopped.is_set():
        for stream, rank in slots:
            if readers[stream, rank] is not None:
                continue
            directory = directories[streams.index(stream)]
            readers[stream, rank] = _connect(directory / f"rank-{rank}.sock", key)
            if readers[stream, rank] is not None and stream not in activated:
                _native.Semaphore((directory / "semaphore").read_text()).post(
                    ranks * prefetch_factor
                )
                activated.add(stream)
        owners = {
            reader: slot for slot, reader in readers.items() if reader is not None
        }
        for connection in wait([*incoming, *owners], timeout=0.1):
            slot = owners.get(connection)
            if slot is not None:
                try:
                    connection.recv()
                    requested.add(slot)
                except (EOFError, OSError):
                    connection.close()
                    readers[slot] = None
                    requested.discard(slot)
                continue
            try:
                kind, stream, value = _ipc.recv(connection)
            except (EOFError, OSError):
                incoming.remove(connection)
                continue
            if kind == "end":
                ended[stream] += 1
                if ended[stream] == num_workers:
                    for rank in range(ranks):
                        outbox[stream, rank].append(("end", None))
                continue
            (batch_id, size), query_idx, parts, timings, memory_units = value
            batch = assembling[stream].setdefault(batch_id, {})
            batch.update({index: (sample, seconds) for index, sample, seconds in parts})
            if len(batch) < size:
                continue
            del assembling[stream][batch_id]
            samples = tuple(batch[index][0] for index in range(size))
            timings = (*timings, sum(seconds for _, seconds in batch.values()))
            started = time.monotonic()
            collator = collate_fn.get(stream)
            data = share(collator(samples)) if collator else samples
            timings = (*timings, time.monotonic() - started)
            finished[stream][batch_id] = Batch(
                stream, batch_id, query_idx, samples, data, timings, memory_units
            )
            while next_batch[stream] in finished[stream]:
                rank = next_batch[stream] % ranks
                outbox[stream, rank].append(
                    ("batch", finished[stream].pop(next_batch[stream]))
                )
                next_batch[stream] += 1
        for slot in [slot for slot in requested if outbox[slot]]:
            message = outbox[slot][0]
            try:
                _ipc.send(readers[slot], message)
            except OSError:
                readers[slot].close()
                readers[slot] = None
            requested.discard(slot)
            if readers[slot] is not None and message[0] != "end":
                outbox[slot].popleft()
