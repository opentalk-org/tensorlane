"""Uploads a small and a multi-chunk file to a real server (resumable.rs)."""

import sys
import tempfile
from pathlib import Path

import tensorlane


def main():
    address, run_id, size = sys.argv[1], sys.argv[2], int(sys.argv[3])
    with tempfile.TemporaryDirectory() as root:
        small = Path(root) / "small.bin"
        small.write_bytes(b"small checkpoint")
        large = Path(root) / "large.bin"
        large.write_bytes(bytes(i % 251 for i in range(size)))
        with tensorlane.init(
            run_id,
            num_workers=1,
            addr=address,
            ipc_dir=root,
            timeout=20,
            performance_metrics=False,
        ) as lane:
            print(lane.save_asset("small", small))
            print(lane.save_asset("large", large, kind="checkpoint"))
            lane.flush()


if __name__ == "__main__":
    main()
