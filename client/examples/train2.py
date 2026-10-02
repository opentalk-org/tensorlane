from contextlib import nullcontext
import json
from pathlib import Path
import tarfile

import torch
from accelerate import Accelerator
from accelerate.utils import send_to_device, set_seed
import tensorlane


def transform(sample):
    value = sample.metadata["position"] / 1000
    return torch.tensor([value]), torch.tensor([2 * value + 1])


def collate(samples):
    return tuple(torch.stack(items) for items in zip(*samples))


def main():
    accelerator = Accelerator()
    try:
        with tensorlane.init(
            transform=transform,
            collate_fn=collate,
            ranks=accelerator.num_processes,
            rank=accelerator.local_process_index,
            start_daemon=accelerator.is_local_main_process,
        ) as lane:
            set_seed(lane.config["seed"])
            model = torch.nn.Linear(1, 1)
            with tarfile.open(lane.asset("model")) as archive:
                model.load_state_dict(
                    torch.load(
                        archive.extractfile("weights.pt"),
                        map_location="cpu",
                        weights_only=True,
                    )
                )
            optimizer = torch.optim.SGD(
                model.parameters(), lr=lane.config["optimizer"]["learning_rate"]
            )
            model, optimizer = accelerator.prepare(model, optimizer)
            completed = 0
            with (
                lane.batches() as batches,
                model.join() if accelerator.num_processes > 1 else nullcontext(),
            ):
                for batch in batches:
                    features, targets = send_to_device(batch.data, accelerator.device)
                    optimizer.zero_grad()
                    loss = torch.nn.functional.mse_loss(model(features), targets)
                    accelerator.backward(loss)
                    optimizer.step()
                    completed += len(batch)
                    lane.metric(batch.batch_id, f"rank/{lane.rank}/loss", loss.item())
            count = torch.tensor(completed, device=accelerator.device)
            offset = lane.config["params"]["dataset_offset"]
            offset += accelerator.reduce(count, reduction="sum").item()
            if accelerator.is_main_process:
                output = Path(lane.config["output_dir"])
                output.mkdir(parents=True, exist_ok=True)
                weights = output / "weights.pt"
                accelerator.save(accelerator.unwrap_model(model).state_dict(), weights)
                asset_id = lane.save_asset(
                    "model",
                    weights,
                    step=offset,
                    kind="checkpoint",
                    metadata={"dataset_offset": offset},
                )
                lane.flush()
                (output / "progress.json").write_text(
                    json.dumps({"asset_id": asset_id, "dataset_offset": offset}) + "\n"
                )
            lane.flush()
            accelerator.wait_for_everyone()
    finally:
        accelerator.end_training()


if __name__ == "__main__":
    main()
