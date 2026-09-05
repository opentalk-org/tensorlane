from dataclasses import dataclass, fields

import torch
from torch.utils.data import DataLoader, IterableDataset

from .client import Client


@dataclass(frozen=True)
class Batch:
    waves: tuple[torch.Tensor, ...]
    audio_durations: tuple[float, ...]
    speaker_ids: torch.Tensor
    language_ids: torch.Tensor
    modality_ids: torch.Tensor
    texts: torch.Tensor
    input_lengths: torch.Tensor
    mels: torch.Tensor
    mel_lengths: torch.Tensor

    def to(self, device: torch.device) -> "Batch":
        cpu_fields = {"input_lengths", "mel_lengths"}
        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name == "waves":
                value = tuple(wave.to(device, non_blocking=True) for wave in value)
            elif isinstance(value, torch.Tensor) and field.name not in cpu_fields:
                value = value.to(device, non_blocking=True)
            values[field.name] = value
        return Batch(**values)


class _StreamDataset(IterableDataset):
    def __init__(
        self,
        client: Client,
        validation: bool,
        prefetch: int,
        samples_per_epoch: int | None,
        modality_id: int,
        pin_memory: bool,
    ) -> None:
        self.client = client
        self.validation = validation
        self.prefetch = prefetch
        self.samples_per_epoch = samples_per_epoch
        self.modality_id = modality_id
        self.pin_memory = pin_memory
        self._stream = None

    def __iter__(self):
        if self._stream is None:
            self._stream = self.client._native.batches(
                self.validation,
                self.prefetch,
                self.modality_id,
                self.pin_memory,
            )
        if self.samples_per_epoch is None:
            for parts in self._stream:
                yield _batch(parts)
            return
        served = 0
        while served < self.samples_per_epoch:
            parts = next(self._stream)
            batch = _batch(parts)
            served += len(batch.audio_durations)
            yield batch


def _batch(parts) -> Batch:
    waves, durations, *tensors = parts
    return Batch(tuple(waves), tuple(durations), *tensors)


def dataloader(
    client: Client,
    validation: bool = False,
    prefetch: int = 4,
    device: str = "cpu",
    samples_per_epoch: int | None = None,
    modality_id: int = 0,
) -> DataLoader:
    dataset = _StreamDataset(
        client,
        validation,
        prefetch,
        samples_per_epoch,
        modality_id,
        pin_memory=device.startswith("cuda"),
    )
    return DataLoader(
        dataset,
        batch_size=None,
        num_workers=0,
        collate_fn=lambda batch: batch,
        pin_memory=False,
    )
