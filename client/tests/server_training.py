from pathlib import Path
import sys
import tempfile

import tensorlane
import torch


def transform(sample):
    return torch.tensor([sample.metadata["position"]], dtype=torch.float32)


def main():
    address, run_id = sys.argv[1:]
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as root:
        with tensorlane.init(
            run_id,
            transform,
            num_workers=1,
            collate_fn=torch.stack,
            addr=address,
            ipc_dir=root,
            timeout=20,
        ) as lane:
            assert lane.asset("model").read_bytes() == b"initial weights"
            model = torch.nn.Linear(1, 1)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            steps = 0
            with lane.batches("training") as batches:
                for batch in batches:
                    prediction = model(batch.data)
                    loss = (prediction - batch.data * 2).square().mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    lane.metric(batch.batch_id, "loss", loss.item())
                    steps += 1
            assert steps == 4
            path = Path(root) / "model.pt"
            torch.save(model.state_dict(), path)
            lane.save_asset("model", path, step=steps, kind="checkpoint")
            lane.flush()
    print("training passed")


if __name__ == "__main__":
    main()
