from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RawSample:
    sample_id: str
    stream: str
    metadata: dict[str, Any]
    blobs: dict[str, bytes]


@dataclass(frozen=True)
class Batch:
    stream: str
    batch_id: int
    query_batch_idx: int
    samples: tuple[Any, ...]
    data: Any
    _timings: tuple[float, ...] = field(default=(), repr=False, compare=False)
    _memory_units: int = field(default=0, repr=False, compare=False)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[Any]:
        return iter(self.samples)
