from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar, cast

from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset

if TYPE_CHECKING:
    from .client import Client


@dataclass(frozen=True)
class Sample:
    """CPU tensors: int16 PCM waveform and int64 text token IDs."""

    wave: Tensor
    duration: float
    speaker_id: int
    language_id: int
    text: Tensor


@dataclass(frozen=True)
class Batch:
    samples: tuple[Sample, ...]

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.samples)


class _Dataset(IterableDataset[Batch]):
    def __init__(self, client: Client, validation: bool) -> None:
        self.client = client
        self.validation = validation

    def __iter__(self) -> Iterator[Batch]:
        while True:
            batch = self.client.next_batch(validation=self.validation)
            if batch is None:
                return
            yield batch


class BatchLoader(DataLoader[Batch]):
    def __iter__(self) -> Iterator[Batch]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return cast(Iterator[Batch], super().__iter__())


_T = TypeVar("_T")


def _identity(value: _T) -> _T:
    return value


def dataloader(client: Client, validation: bool = False) -> BatchLoader:
    return BatchLoader(
        _Dataset(client, validation),
        batch_size=None,
        num_workers=0,
        prefetch_factor=None,
        persistent_workers=False,
        pin_memory=False,
        collate_fn=_identity,
    )
