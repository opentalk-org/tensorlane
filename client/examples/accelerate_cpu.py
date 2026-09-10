import argparse

import torch.distributed as distributed
from accelerate import Accelerator

import tensorlane
from audio_transforms import transform_audio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--ipc-dir", default=".client-cache")
    args = parser.parse_args()
    accelerator = Accelerator(cpu=True)
    try:
        with tensorlane.init(
            args.run_id,
            transform_audio,
            accelerator.num_processes,
            rank=accelerator.process_index,
            start_daemon=accelerator.is_main_process,
            ipc_dir=args.ipc_dir,
        ) as lane:
            rank = lane.rank
            print(
                f"rank={rank} run_id={lane.run_id} train_config={lane.train_config!r}",
                flush=True,
            )
            with (
                lane.batches() as training,
                lane.batches(validation=True) as validation,
            ):
                for index, batch in enumerate(training):
                    print(
                        f"rank={rank} device={accelerator.device} split=training "
                        f"batch={index} samples={len(batch)}",
                        flush=True,
                    )
                    validation_batch = next(validation, None)
                    if validation_batch is not None:
                        print(
                            f"rank={rank} device={accelerator.device} split=validation "
                            f"batch={index} samples={len(validation_batch)}",
                            flush=True,
                        )

            if accelerator.num_processes > 1:
                distributed.barrier()
    finally:
        accelerator.end_training()


if __name__ == "__main__":
    main()
