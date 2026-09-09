from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


@dataclass(frozen=True)
class Sample:
    """Transformed CPU waveform, original metadata, and int64 text tokens."""

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
