import argparse
import multiprocessing

import tensorlane
from audio_transforms import transform_audio


def consume(run_id, rank, ranks, ipc_dir):
    with (
        tensorlane.init(
            run_id,
            transform_audio,
            ranks,
            rank=rank,
            start_daemon=False,
            ipc_dir=ipc_dir,
        ) as lane,
        lane.batches() as training,
        lane.batches(validation=True) as validation,
    ):
        for index, batch in enumerate(training):
            print(
                f"rank={rank} split=training batch={index} samples={len(batch)}",
                flush=True,
            )
            validation_batch = next(validation, None)
            if validation_batch is not None:
                print(
                    f"rank={rank} split=validation batch={index} samples={len(validation_batch)}",
                    flush=True,
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--ranks", type=int, default=2)
    parser.add_argument("--ipc-dir", default=".client-cache")
    args = parser.parse_args()
    context = multiprocessing.get_context("spawn")
    with tensorlane.init(
        args.run_id,
        transform_audio,
        args.ranks,
        rank=0,
        start_daemon=True,
        ipc_dir=args.ipc_dir,
    ):
        processes = [
            context.Process(
                target=consume, args=(args.run_id, rank, args.ranks, args.ipc_dir)
            )
            for rank in range(args.ranks)
        ]
        try:
            for process in processes:
                process.start()
            for process in processes:
                process.join()
            if any(process.exitcode != 0 for process in processes):
                raise RuntimeError("a rank failed")
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                if process.pid is not None:
                    process.join()


if __name__ == "__main__":
    main()
