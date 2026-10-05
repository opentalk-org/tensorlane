from concurrent.futures import ThreadPoolExecutor, wait, FIRST_EXCEPTION
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import threading
import time

import tensorlane
import torch

from server_stress import BATCHES, BATCH_SIZE, transform


def consume(reader, stream, deadline, stopped):
    count = 0
    longest_wait = 0
    while time.monotonic() < deadline and not stopped.is_set():
        started = time.monotonic()
        batch = next(reader)
        longest_wait = max(longest_wait, time.monotonic() - started)
        assert batch.batch_id == count
        assert batch.query_batch_idx == count % BATCHES
        assert len(batch) == BATCH_SIZE
        offset = (count % BATCHES) * BATCH_SIZE
        expected = torch.arange(offset, offset + BATCH_SIZE).float()
        torch.testing.assert_close(
            batch.data.flatten(), expected / (BATCHES * BATCH_SIZE)
        )
        count += 1
    assert count > BATCHES, f"stream did not complete a replay: {stream}"
    return {"stream": stream, "batches": count, "max_wait_seconds": longest_wait}


def main():
    address, run_id, directory, seconds = sys.argv[1:]
    duration = int(seconds)
    stopped = threading.Event()
    with (
        ExitStack() as readers,
        tensorlane.init(
            run_id,
            transform,
            num_workers=2,
            collate_fn=torch.stack,
            prefetch_factor=8,
            addr=address,
            ipc_dir=Path(directory),
            timeout=120,
            performance_metrics=False,
        ) as lane,
    ):
        connections = [
            (stream, readers.enter_context(lane.batches(stream)))
            for stream in lane.streams
        ]
        started = time.monotonic()
        deadline = started + duration
        with ThreadPoolExecutor(max_workers=len(lane.streams)) as executor:
            futures = [
                executor.submit(consume, reader, stream, deadline, stopped)
                for stream, reader in connections
            ]
            try:
                for minute in range(1, duration // 60 + 1):
                    done, _ = wait(
                        futures,
                        timeout=max(0, started + minute * 60 - time.monotonic()),
                        return_when=FIRST_EXCEPTION,
                    )
                    for future in done:
                        future.result()
                    print(
                        json.dumps(
                            {
                                "soak_elapsed_seconds": round(
                                    time.monotonic() - started, 1
                                )
                            }
                        ),
                        flush=True,
                    )
                reports = [future.result(timeout=30) for future in futures]
            finally:
                stopped.set()
        elapsed = time.monotonic() - started
        assert elapsed >= duration
    batches = sum(report["batches"] for report in reports)
    print(
        json.dumps(
            {
                "seconds": elapsed,
                "batches": batches,
                "samples": batches * BATCH_SIZE,
                "batches_per_second": batches / elapsed,
                "streams": reports,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
