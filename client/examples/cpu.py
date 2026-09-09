import argparse
import multiprocessing

import tensorlane
from audio_transforms import transform_audio


def consume(run_id, rank):
    with tensorlane.batches(run_id, rank) as reader:
        for index, batch in enumerate(reader):
            print(f"rank={rank} batch={index} samples={len(batch)}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--ranks", type=int, default=2)
    args = parser.parse_args()
    context = multiprocessing.get_context("spawn")
    with tensorlane.init(args.run_id, transform_audio, args.ranks):
        processes = [
            context.Process(target=consume, args=(args.run_id, rank))
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
