from __future__ import annotations

from multiprocessing.connection import Client
from pathlib import Path
import queue
import sys
import threading
import traceback
import time

import torch.multiprocessing as multiprocessing

from . import _native


def _connect(path: Path, key: bytes, stopped):
    while not stopped.is_set():
        try:
            return Client(str(path), family="AF_UNIX", authkey=key)
        except (FileNotFoundError, ConnectionRefusedError):
            time.sleep(0.02)
    raise RuntimeError("daemon stopped during connection")


def transform_worker(
    root: Path,
    transform,
    output,
    stopped,
) -> None:
    multiprocessing.current_process().authkey = (root / "auth").read_bytes()
    receiver = _native.Listener(root / "work.sock")
    try:
        import torch

        output.cancel_join_thread()

        errors = queue.SimpleQueue()

        def feeder_error(error, _message):
            errors.put(error)

        output._on_queue_feeder_error = feeder_error
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
                if message["kind"] == "sample":
                    from .data import Sample

                    try:
                        if sys.byteorder != "little":
                            raise RuntimeError(
                                "PCM decoding currently requires a little-endian host"
                            )
                        wave = (
                            torch.frombuffer(
                                bytearray(message["wave"]), dtype=torch.int16
                            )
                            if message["wave"]
                            else torch.empty(0, dtype=torch.int16)
                        )
                        text = (
                            torch.frombuffer(
                                bytearray(message["text"]), dtype=torch.int64
                            )
                            if message["text"]
                            else torch.empty(0, dtype=torch.int64)
                        )
                        transformed = transform(wave)
                        if (
                            not isinstance(transformed, torch.Tensor)
                            or transformed.device.type != "cpu"
                        ):
                            raise TypeError("transform must return a CPU torch.Tensor")
                        sample = Sample(
                            transformed.detach().share_memory_(),
                            message["duration"],
                            message["speaker_id"],
                            message["language_id"],
                            text.share_memory_(),
                        )
                        output.put(
                            ("sample", (message["batch"], message["index"], sample))
                        )
                    except Exception as error:
                        raise RuntimeError(
                            f"batch {message['batch']} sample {message['index']}: {error}"
                        ) from error
                else:
                    output.put(("end", None))
                if message["kind"] == "end":
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
    incoming,
    stopped,
    ready,
) -> None:
    from .data import Batch

    key = (root / "auth").read_bytes()
    multiprocessing.current_process().authkey = key
    outputs = [queue.Queue() for _ in range(ranks)]
    errors = queue.Queue()

    def deliver(rank: int) -> None:
        connection = None
        try:
            connection = _connect(root / f"rank-{rank}.sock", key, stopped)
            connection.send(("ready", None))
            while not stopped.is_set():
                try:
                    message = outputs[rank].get(timeout=0.1)
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

    for rank in range(ranks):
        threading.Thread(target=deliver, args=(rank,), daemon=True).start()
    ready.set()
    pending = {}
    completed = {}
    next_batch = 0
    ended = 0
    while not stopped.is_set():
        try:
            error = errors.get_nowait()
        except queue.Empty:
            pass
        else:
            raise RuntimeError(error)
        if ended == num_workers:
            stopped.wait(0.1)
            continue
        try:
            kind, value = incoming.get(timeout=0.1)
        except queue.Empty:
            continue
        if kind == "error":
            raise RuntimeError(value)
        if kind == "end":
            ended += 1
            if ended == num_workers:
                if pending or completed:
                    raise RuntimeError(
                        "stream ended with incomplete or missing batches"
                    )
                for output in outputs:
                    output.put(("end", None))
            continue
        if kind != "sample":
            raise RuntimeError(f"unexpected transformation message: {kind}")
        (batch_id, batch_size), index, sample = value
        if batch_id < next_batch or batch_id in completed:
            raise RuntimeError("message for an already completed batch")
        size, parts = pending.setdefault(batch_id, (batch_size, {}))
        parts[index] = sample
        if len(parts) == size:
            completed[batch_id] = Batch(tuple(parts[index] for index in range(size)))
            del pending[batch_id]
        while next_batch in completed:
            outputs[next_batch % ranks].put(
                ("batch", (next_batch, completed.pop(next_batch)))
            )
            next_batch += 1
