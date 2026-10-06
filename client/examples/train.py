from contextlib import nullcontext
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
            set_seed(lane.config["app"]["seed"])
            model = torch.nn.Linear(1, 1)
            optimizer = torch.optim.SGD(
                model.parameters(), lr=lane.config["app"]["optimizer"]["learning_rate"]
            )
            if "model" in lane.asset_metadata:
                state = torch.load(
                    lane.asset("model"), map_location="cpu", weights_only=True
                )
                model.load_state_dict(state["model"])
                optimizer.load_state_dict(state["optimizer"])
            model, optimizer = accelerator.prepare(model, optimizer)
            with (
                lane.batches("training") as batches,
                model.join() if accelerator.num_processes > 1 else nullcontext(),
            ):
                for batch in batches:
                    features, targets = send_to_device(batch.data, accelerator.device)
                    optimizer.zero_grad()
                    loss = torch.nn.functional.mse_loss(model(features), targets)
                    accelerator.backward(loss)
                    optimizer.step()
                    lane.metric(batch.batch_id, f"rank/{lane.rank}/loss", loss.item())
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                lane.save_asset(
                    "model",
                    {
                        "model": accelerator.unwrap_model(model).state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    kind="checkpoint",
                )
            lane.flush()
            accelerator.wait_for_everyone()
    finally:
        accelerator.end_training()


if __name__ == "__main__":
    main()
