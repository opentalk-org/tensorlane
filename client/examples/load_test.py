"""Drain the configured training schedule with no transform or training computation.

Use a long server-side schedule for meaningful throughput measurements.
Consumption timing includes rank socket setup, but excludes process startup and Init.
"""

import argparse
import multiprocessing
from multiprocessing.connection import wait
import time

import tensorlane


def identity(wave):
    return wave


def consume(args, rank, barrier, results):
    with tensorlane.init(
        args.run_id,
        identity,
        args.ranks,
        rank=rank,
        start_daemon=False,
        ipc_dir=args.ipc_dir,
    ) as lane:
        barrier.wait(timeout=120)
        started = time.perf_counter()
        batch_count = 0
        sample_count = 0
        with lane.batches() as batches:
            for batch in batches:
                batch_count += 1
                sample_count += len(batch)
                print(
                    f"rank={rank} batch={batch_count} samples={len(batch)} "
                    f"elapsed_seconds={time.perf_counter() - started:.3f}",
                    flush=True,
                )
        finished = time.perf_counter()
        results.put((rank, batch_count, sample_count, started, finished))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--ranks", type=int, default=4)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--addr", default=None)
    parser.add_argument("--ipc-dir", default=".client-cache")
    args = parser.parse_args()
    if min(args.ranks, args.workers, args.prefetch_factor) < 1:
        parser.error("ranks, workers and prefetch-factor must be positive")

    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(args.ranks)
    results = context.Queue()
    started = time.perf_counter()
    try:
        with tensorlane.init(
            args.run_id,
            identity,
            args.ranks,
            rank=0,
            start_daemon=True,
            num_workers=args.workers,
            prefetch_factor=args.prefetch_factor,
            addr=args.addr,
            ipc_dir=args.ipc_dir,
        ):
            print(f"init_seconds={time.perf_counter() - started:.3f}", flush=True)
            processes = [
                context.Process(target=consume, args=(args, rank, barrier, results))
                for rank in range(args.ranks)
            ]
            try:
                for process in processes:
                    process.start()
                pending = {process.sentinel: process for process in processes}
                while pending:
                    for sentinel in wait(pending):
                        process = pending.pop(sentinel)
                        process.join()
                        if process.exitcode != 0:
                            raise RuntimeError(
                                f"{process.name} failed: {process.exitcode}"
                            )
                measurements = [results.get(timeout=5) for _ in processes]
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                    if process.pid is not None:
                        process.join()

            for rank, batches, samples, begin, end in sorted(measurements):
                elapsed = end - begin
                print(
                    f"rank={rank} batches={batches} samples={samples} "
                    f"seconds={elapsed:.3f} batches/s={batches / elapsed:.2f} "
                    f"samples/s={samples / elapsed:.2f}"
                )
            elapsed = max(row[4] for row in measurements) - min(
                row[3] for row in measurements
            )
            batches = sum(row[1] for row in measurements)
            samples = sum(row[2] for row in measurements)
            print(
                f"total batches={batches} samples={samples} seconds={elapsed:.3f} "
                f"batches/s={batches / elapsed:.2f} samples/s={samples / elapsed:.2f}"
            )
    finally:
        results.close()
        results.join_thread()


if __name__ == "__main__":
    main()
