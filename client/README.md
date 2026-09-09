# Client

The client runs one Rust/Tokio daemon thread in the process that calls `tensorlane.init()`.
Python starts five transform processes by default and one collator process using multiprocessing's
`spawn` context, passing connected Unix sockets and the transform callable directly.
Training ranks connect independently; the library does not launch them.

## Install

From the repository's development shell:

```sh
uv venv
uv pip install torch==2.11.0 maturin
maturin develop --manifest-path client/Cargo.toml
```

The extension uses PyO3 but does not link libtorch. Python workers create CPU tensors.
Linux and macOS with little-endian CPUs are supported.

## With Accelerate

Put the transform in an importable module, for example `audio_transforms.py`:

```python
import torch

def transform_audio(wave):
    return wave.to(torch.float32) / 32768.0
```

Then in the training script:

```python
from accelerate import Accelerator
import tensorlane
from audio_transforms import transform_audio

def main(run_id):
    accelerator = Accelerator()
    daemon = None
    try:
        if accelerator.is_main_process:
            daemon = tensorlane.init(
                run_id=run_id,
                transform=transform_audio,
                ranks=accelerator.num_processes,
                prefetch_factor=2,
            )

        with tensorlane.batches(run_id, accelerator.process_index) as reader:
            for batch in reader:
                for sample in batch:
                    wave = sample.wave.to(accelerator.device)

        accelerator.wait_for_everyone()
    finally:
        if daemon is not None:
            daemon.close()

if __name__ == "__main__":
    main("your-run-id")
```

Keep the initializing process and daemon alive until every rank finishes.
Use the completion barrier only on successful completion; delegate distributed error
handling to your launcher. The daemon handle also supports a context manager for
single-process use.

`init(run_id, transform, ranks, prefetch_factor=2, *, num_workers=5, addr=None, ipc_dir=None)`
returns a handle with `run_id`, `train_config`, and `close()`.
The gRPC address defaults to `TENSORLANE_ADDR` or `localhost:8181`.
Set `num_workers` to a positive integer to control the transformation process count.
Initialization waits for GMD and worker readiness (at most 120 seconds), not for
the data buffer to fill.

`batches(run_id, rank, *, ipc_dir=None, timeout=120)` waits for initialization
and attaches the rank. All ranks use the same run ID and IPC base directory.
The default is the local temporary directory, with a private per-user/per-run
subdirectory. This is a single-machine interface. Duplicate initialization and
duplicate rank attachment are rejected. Readers cannot reconnect after failure.

## Data and scheduling

A transform receives one unpadded, one-dimensional CPU `torch.int16` waveform
and returns a CPU tensor. It runs without autograd, in a separate interpreter.
The callable is passed through standard multiprocessing serialization, not resolved
by a module/name string. Top-level functions in the training script, picklable
callable objects, and partial functions work. Lambdas and local closures generally
do not. Guard initialization with `if __name__ == "__main__":` as with any spawn-based
multiprocessing program. Workers inherit Python's executable and module search path.

Each `Batch` contains ordered `Sample` objects with `wave`, `duration`,
`speaker_id`, `language_id`, and CPU int64 `text` tokens.
The collater groups samples; it does not pad or stack them.
The waveform's dtype and shape are whatever the transform returns.

Raw samples go round-robin to transformation workers. The collator restores sample
and batch order even when workers finish out of order, and waits for every worker's
EOF before ending the rank streams. Whole server batches go round-robin to ranks.
Rust creates one named POSIX semaphore with initial value `prefetch_factor * ranks`.
The request producer calls `sem_wait` before every gRPC batch request; each rank
calls `sem_post` directly after receiving and reconstructing a batch, before
returning it to training code. There are no per-rank quotas or batch acknowledgments.
The global limit includes fetching, raw data, transformation, collator queues, and
rank socket buffers; it is not a separate limit for every pipeline stage. A slow
rank can occupy more than `prefetch_factor` slots and eventually stall fetching.

All final nonempty batches are delivered. Empty server batches are rejected because
the sample-only protocol cannot represent them. Uneven final rank
counts are not balanced: choose a schedule compatible with your distributed
training. Normal stream exhaustion ends iteration. This version handles only
training data, without validation, checkpoints, or metrics.

## IPC and lifetime

Rust serializes sample messages with Serde/Postcard and sends them over
`work.sock` using Tokio's `LengthDelimitedCodec`: a four-byte big-endian length
followed by the payload, with a 256 MiB frame limit on both ends. There are no
shared-memory mappings or ownership acknowledgments on this link. The worker's
Rust listener deserializes each frame while releasing the Python GIL during the wait.
Each worker receives data on one connection and sends nothing back. A connected
socket is ready; there is no worker readiness handshake or control protocol.
The collator has no connection to Rust. Python supervises all child processes,
uses an event for collator readiness, and a shared event for shutdown.
Data EOF does not close the connection: workers stay alive until daemon shutdown
so ranks can finish receiving shared tensors.

Python creates one shared `multiprocessing.Queue` before launching the workers and
collator. Workers put transformed samples and EOF messages into it. Every sample
carries `(batch_id, batch_size)` and its sample index; there are no batch headers.
The collator groups by batch ID, takes the size from the first arriving sample
without checking later sizes, and restores sample and batch order.
PyTorch multiprocessing reducers and
`Tensor.share_memory_()` transfer tensor storage. Each rank owns a
`multiprocessing.connection.Listener` at `rank-<rank>.sock`, and the collator opens
a `Client` connection to it. There are no `samples.sock` or shared `ranks.sock`
listeners, and no data acknowledgments on these links. Rust manages daemon socket
endpoints. Python owns process startup,
joining, and termination. The daemon starts listening before Python launches the
workers; `init()` returns after all worker sockets connect and the collator is ready.

Participating Python processes use the job's multiprocessing authentication
key. Keep this in mind if your training also configures multiprocessing
authentication. Tensor senders remain alive after EOF so receivers can finish
reconstructing shared storage.

`close()` interrupts fetching and IPC, then waits up to five seconds for child
shutdown before killing and reaping remaining children. It is idempotent.
Cancellation wakes a blocked `sem_wait` without issuing another request. The named
semaphore and socket files are cleaned up; the lock file and final
status remain for stale-daemon detection. Errors wake rank readers rather than
retrying a server batch whose cursor may already have advanced.
If a Python child fails, ranks receive a process-exit error; the child prints the
original exception traceback to stderr rather than forwarding it over the worker socket.

## Checks and CPU example

```sh
uv pip install grpcio-tools
cargo test -p tensorlane-client --lib
PYTHONPATH=client/tests python -m unittest discover -s client/tests -v
python client/examples/cpu.py RUN_ID
```

The integration tests launch a local mock gRPC service and real Python workers
and training-rank processes. They need no GPU or running TensorLane service.
The CPU example connects to a real running service.
