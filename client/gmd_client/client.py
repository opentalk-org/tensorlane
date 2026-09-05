import os
from pathlib import Path

import torch as _torch
from . import _native
from .metrics import MetricsStream

DEFAULT_ADDR = "localhost:8181"


class GiveMeDataClient:
    def __init__(self, training_id: str, addr: str | None = None) -> None:
        address = addr or os.environ.get("GIVEMEDATA_ADDR", DEFAULT_ADDR)
        self._native = _native.Client(training_id, address)
        self.training_id: str = self._native.training_id
        self.train_config: str = self._native.train_config
        self._metrics: MetricsStream | None = None
        self._closed = False

    def download_asset(self, name: str, dest_dir: Path) -> Path:
        return Path(self._native.download_asset(name, Path(dest_dir)))

    def upload_checkpoint(self, step: int, source_dir: Path) -> None:
        """Queue a checkpoint directory for background archiving and upload."""
        self._native.upload_checkpoint(step, Path(source_dir))

    def metrics(self) -> MetricsStream:
        if self._closed:
            raise RuntimeError("givemedata client is closed")
        if self._metrics is None:
            self._metrics = MetricsStream(self._native.metrics())
        return self._metrics

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._native.close()
