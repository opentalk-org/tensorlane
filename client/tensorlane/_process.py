from __future__ import annotations

import json
from multiprocessing.connection import Client, wait
from pathlib import Path
import queue
import threading
import traceback
import time

from . import _native
from . import _ipc
from .data import Batch, RawSample


def _connect(path: Path, key: bytes, stopped):
    while not stopped.is_set():
        try:
            return Client(str(path), family="AF_UNIX", authkey=key)
        except (FileNotFoundError, ConnectionRefusedError):
            time.sleep(0.02)
    raise RuntimeError("daemon stopped during connection")


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


def transform_worker(root: Path, transform, streams, output, stopped) -> None:
    receiver = _native.Listener(root / "work.sock")
    try:
        import torch

        torch.set_num_threads(1)
        ended = set()
        with torch.no_grad():
            while not stopped.is_set():
                message = receiver.recv() if len(ended) < len(streams) else None
                if len(ended) == len(streams):
                    stopped.wait(0.1)
                    continue
                if message is None:
                    if stopped.is_set():
                        return
                    raise RuntimeError("data stream ended before End")
                stream = message["stream"]
                if stream not in streams or stream in ended:
                    raise RuntimeError("unexpected or already ended stream")
                if message["kind"] == "batch":
                    parts = []
                    for item in message["samples"]:
                        try:
                            started = time.monotonic()
                            metadata = json.loads(item["metadata_json"])
                            if not isinstance(metadata, dict):
                                raise TypeError("sample metadata must be an object")
                            sample = RawSample(
                                item["sample_id"], stream, metadata, item["blobs"]
                            )
                            transform_fn = transform.get(stream)
                            transformed = share(
                                transform_fn(sample) if transform_fn else sample
                            )
                            parts.append(
                                (item["index"], transformed, time.monotonic() - started)
                            )
                        except Exception as error:
                            raise RuntimeError(
                                f"stream {stream} batch {message['batch']} sample {item['index']}: {error}"
                            ) from error
                    _ipc.send(
                        output,
                        (
                            "samples",
                            stream,
                            (
                                message["batch"],
                                message["query_batch_idx"],
                                parts,
                                message["timings"],
                                message["memory_units"],
                            ),
                        ),
                    )
                elif message["kind"] == "end":
                    _ipc.send(output, ("end", stream, None))
                    ended.add(stream)
                else:
                    raise RuntimeError("unexpected work message")
    except Exception:
        if not stopped.is_set():
            raise
    finally:
        output.close()


def collate_worker(
    root: Path,
    ranks: int,
    num_workers: int,
    streams,
    collate_fn,
    incoming,
    stopped,
    ready,
) -> None:
    collate_fn = collate_fn or {}
    key = (root / "auth").read_bytes()
    outputs = {name: [queue.Queue() for _ in range(ranks)] for name in streams}
    errors = queue.Queue()

    def deliver(rank: int, stream: str, stream_index: int) -> None:
        connection = None
        try:
            connection = _connect(
                root / "streams" / str(stream_index) / f"rank-{rank}.sock", key, stopped
            )
            connection.send(("ready", None))
            while not stopped.is_set():
                try:
                    message = outputs[stream][rank].get(timeout=0.1)
                except queue.Empty:
                    if connection.poll():
                        raise RuntimeError(f"rank {rank} disconnected")
                    continue
                _ipc.send(connection, message)
                if message[0] == "end":
                    return
                del message
        except Exception:
            if not stopped.is_set():
                errors.put(traceback.format_exc())
        finally:
            if connection is not None:
                connection.close()

    for stream_index, stream in enumerate(streams):
        for rank in range(ranks):
            threading.Thread(
                target=deliver, args=(rank, stream, stream_index), daemon=True
            ).start()
    ready.set()
    pending = {name: {} for name in streams}
    completed = {name: {} for name in streams}
    next_batch = {name: 0 for name in streams}
    ended = {name: 0 for name in streams}
    while not stopped.is_set():
        if not errors.empty():
            raise RuntimeError(errors.get())
        available = wait(incoming, timeout=0.1)
        if not available:
            continue
        connection = available[0]
        try:
            kind, stream, value = _ipc.recv(connection)
        except (EOFError, OSError):
            if stopped.is_set():
                return
            raise
        incoming.remove(connection)
        incoming.append(connection)
        if stream not in outputs:
            raise RuntimeError("unexpected stream")
        if kind == "end":
            ended[stream] += 1
            if ended[stream] > num_workers:
                raise RuntimeError("duplicate stream end")
            if ended[stream] == num_workers:
                if pending[stream] or completed[stream]:
                    raise RuntimeError(
                        "stream ended with incomplete or missing batches"
                    )
                for output in outputs[stream]:
                    output.put(("end", None))
            continue
        if kind != "samples" or ended[stream] == num_workers:
            raise RuntimeError(f"unexpected transformation message: {kind}")
        (batch_id, batch_size), query_idx, incoming_parts, timings, memory_units = value
        if batch_id < next_batch[stream] or batch_id in completed[stream]:
            raise RuntimeError("message for an already completed batch")
        if batch_size <= 0 or not incoming_parts:
            raise RuntimeError("invalid sample position")
        size, expected_query_idx, expected_units, parts = pending[stream].setdefault(
            batch_id, (batch_size, query_idx, memory_units, {})
        )
        if (
            size != batch_size
            or expected_query_idx != query_idx
            or expected_units != memory_units
        ):
            raise RuntimeError("inconsistent batch or duplicate sample position")
        for index, sample, seconds in incoming_parts:
            if index < 0 or index >= batch_size:
                raise RuntimeError("invalid sample position")
            if index in parts:
                raise RuntimeError("inconsistent batch or duplicate sample position")
            parts[index] = (sample, seconds)
        if len(parts) == size:
            samples = tuple(parts[index][0] for index in range(size))
            timings = (*timings, sum(part[1] for part in parts.values()))
            started = time.monotonic()
            collator = collate_fn.get(stream)
            data = share(collator(samples)) if collator else samples
            completed[stream][batch_id] = Batch(
                stream,
                batch_id,
                query_idx,
                samples,
                data,
                (*timings, time.monotonic() - started),
                memory_units,
            )
            del pending[stream][batch_id]
        while next_batch[stream] in completed[stream]:
            batch_id = next_batch[stream]
            outputs[stream][batch_id % ranks].put(
                ("batch", completed[stream].pop(batch_id))
            )
            next_batch[stream] += 1
