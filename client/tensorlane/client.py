import os
from types import TracebackType
from typing import Self

# Load libtorch's shared libraries before the extension.
import torch as _torch  # noqa: F401

from . import _native
from .data import Batch, Sample


class Client:
    """Handle to a dedicated Rust thread running a Tokio command loop."""

    def __init__(self, run_id: str, addr: str | None = None) -> None:
        """Connect and call Init on the worker, releasing the GIL while waiting."""
        address = addr or os.environ.get("TENSORLANE_ADDR", "localhost:8181")
        self._native = _native.Client(run_id, address)

    @property
    def run_id(self) -> str:
        return self._native.run_id

    @property
    def train_config(self) -> str:
        return self._native.train_config

    def next_batch(self, validation: bool = False) -> Batch | None:
        """Request one batch, or return None when the selected split is exhausted."""
        parts = self._native.next_batch(validation)
        if parts is None:
            return None
        return Batch(tuple(Sample(*sample) for sample in parts))

    def close(self) -> None:
        """Finish queued requests and join the worker thread. Idempotent."""
        self._native.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
