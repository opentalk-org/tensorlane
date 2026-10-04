from pathlib import Path
import hashlib
import json
import sys
import time

import tensorlane
import torch


BATCHES = 64
BATCH_SIZE = 8
BLOB_BYTES = 32768


def transform(sample):
    position = sample.metadata["position"]
    assert len(sample.blobs["payload"]) == BLOB_BYTES
    assert sample.blobs["payload"] == bytes([23]) * BLOB_BYTES
    return torch.tensor([position / (BATCHES * BATCH_SIZE)], dtype=torch.float32)


def wait(path):
    deadline = time.monotonic() + 180
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"stress barrier timed out: {path.name}")
        time.sleep(0.025)


def main():
    address, run_id, directory = sys.argv[1:]
    root = Path(directory)
    ipc = root / run_id[:8]
    ipc.mkdir()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    with tensorlane.init(
        run_id,
        transform,
        num_workers=1,
        collate_fn=torch.stack,
        addr=address,
        ipc_dir=ipc,
        timeout=120,
        performance_metrics=False,
    ) as lane:
        asset = lane.asset("model").read_bytes()
        assert asset == bytes([17]) * (16 * 1024 * 1024)
        (root / f"{run_id}.ready").touch()
        wait(root / "go")
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        steps = 0
        with lane.batches("training") as batches:
            for batch in batches:
                assert batch.batch_id == steps
                assert batch.query_batch_idx == steps
                assert len(batch) == BATCH_SIZE
                expected = torch.arange(steps * BATCH_SIZE, (steps + 1) * BATCH_SIZE)
                torch.testing.assert_close(
                    batch.data.flatten(), expected.float() / (BATCHES * BATCH_SIZE)
                )
                prediction = model(batch.data)
                loss = (prediction - batch.data * 2).square().mean()
                assert torch.isfinite(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lane.metric(steps, "loss", loss.item())
                steps += 1
                if steps == 8:
                    (root / f"{run_id}.restart").touch()
                    wait(root / "resume")
        assert steps == BATCHES
        checkpoint = ipc / "model.pt"
        torch.save(
            {"model": model.state_dict(), "padding": bytes(8 * 1024 * 1024)}, checkpoint
        )
        asset_id = lane.save_asset("model", checkpoint, step=steps, kind="checkpoint")
        lane.flush(timeout=120)
        report = {
            "run_id": run_id,
            "asset_id": asset_id,
            "batches": steps,
            "samples": steps * BATCH_SIZE,
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "seconds": round(time.monotonic() - started, 2),
        }
    (root / f"{run_id}.json").write_text(json.dumps(report))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
