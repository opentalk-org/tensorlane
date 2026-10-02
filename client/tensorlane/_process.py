from __future__ import annotations

import json
from collections.abc import Mapping
from multiprocessing.connection import Client
from pathlib import Path
import queue
import threading
import traceback
import time

import torch.multiprocessing as multiprocessing

from . import _native
from .data import Batch, RawSample


def _connect(path: Path, key: bytes, stopped):
    while not stopped.is_set():
        try:
            return Client(str(path), family="AF_UNIX", authkey=key)
        except (FileNotFoundError, ConnectionRefusedError):
            time.sleep(0.02)
    raise RuntimeError("daemon stopped during connection")


def callback(spec, stream):
    return spec.get(stream) if isinstance(spec, Mapping) else spec


def share(value):
    import torch

    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise TypeError("workers must return CPU tensors")
        return value.detach().share_memory_()
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
    if value is None or isinstance(value, (str, bytes, bool, int, float)):
        return value
    raise TypeError(f"unsupported worker output: {type(value).__name__}")


def transform_worker(root: Path, transform, streams, output, stopped) -> None:
    multiprocessing.current_process().authkey = (root / "auth").read_bytes()
    receiver = _native.Listener(root / "work.sock")
    try:
        import torch

        output.cancel_join_thread()
        errors = queue.SimpleQueue()
        output._on_queue_feeder_error = lambda error, _message: errors.put(error)
        ended = set()
        with torch.no_grad():
            while not stopped.is_set():
                message = receiver.recv()
                if not errors.empty():
                    raise RuntimeError(
                        "transformed sample queue failed"
                    ) from errors.get()
                if message is None:
                    if stopped.is_set():
                        return
                    raise RuntimeError("data stream ended before End")
                stream = message["stream"]
                if stream not in streams or stream in ended:
                    raise RuntimeError("unexpected or already ended stream")
                if message["kind"] == "sample":
                    try:
                        metadata = json.loads(message["metadata_json"])
                        if not isinstance(metadata, dict):
                            raise TypeError("sample metadata must be an object")
                        sample = RawSample(
                            message["sample_id"], stream, metadata, message["blobs"]
                        )
                        transform_fn = callback(transform, stream)
                        transformed = share(
                            transform_fn(sample) if transform_fn else sample
                        )
                        output.put(
                            (
                                "sample",
                                stream,
                                (
                                    message["batch"],
                                    message["query_batch_idx"],
                                    message["index"],
                                    transformed,
                                ),
                            )
                        )
                    except Exception as error:
                        raise RuntimeError(
                            f"stream {stream} batch {message['batch']} sample {message['index']}: {error}"
                        ) from error
                elif message["kind"] == "end":
                    output.put(("end", stream, None))
                    ended.add(stream)
                else:
                    raise RuntimeError("unexpected work message")
                if len(ended) == len(streams):
                    while not stopped.wait(0.1):
                        if not errors.empty():
                            raise RuntimeError(
                                "transformed sample queue failed"
                            ) from errors.get()
                    return
    except Exception:
        if not stopped.is_set():
            raise


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
    key = (root / "auth").read_bytes()
    multiprocessing.current_process().authkey = key
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
                connection.send(message)
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
        try:
            kind, stream, value = incoming.get(timeout=0.1)
        except queue.Empty:
            continue
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
        if kind != "sample" or ended[stream] == num_workers:
            raise RuntimeError(f"unexpected transformation message: {kind}")
        (batch_id, batch_size), query_idx, index, sample = value
        if batch_id < next_batch[stream] or batch_id in completed[stream]:
            raise RuntimeError("message for an already completed batch")
        if batch_size <= 0 or index < 0 or index >= batch_size:
            raise RuntimeError("invalid sample position")
        size, expected_query_idx, parts = pending[stream].setdefault(
            batch_id, (batch_size, query_idx, {})
        )
        if size != batch_size or expected_query_idx != query_idx or index in parts:
            raise RuntimeError("inconsistent batch or duplicate sample position")
        parts[index] = sample
        if len(parts) == size:
            samples = tuple(parts[index] for index in range(size))
            collator = callback(collate_fn, stream)
            data = share(collator(samples)) if collator else samples
            completed[stream][batch_id] = Batch(
                stream, batch_id, query_idx, samples, data
            )
            del pending[stream][batch_id]
        while next_batch[stream] in completed[stream]:
            batch_id = next_batch[stream]
            outputs[stream][batch_id % ranks].put(
                ("batch", (batch_id, completed[stream].pop(batch_id)))
            )
            next_batch[stream] += 1
