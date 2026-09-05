from collections.abc import Mapping
from pathlib import Path, PurePosixPath


class MetricsStream:
    def __init__(self, native) -> None:
        self._native = native

    def log_metric(
        self,
        name: str,
        value: float,
        step: int,
        timestamp_unix_ms: int | None = None,
    ) -> None:
        self._native.log_metric(name, value, step, timestamp_unix_ms)

    def log_metrics(
        self,
        metrics: Mapping[str, float],
        step: int,
        timestamp_unix_ms: int | None = None,
    ) -> None:
        for name, value in metrics.items():
            self.log_metric(name, float(value), step, timestamp_unix_ms)

    def log_artifact(
        self,
        path: Path,
        name: str,
        step: int,
        content_type: str | None = None,
        timestamp_unix_ms: int | None = None,
    ) -> None:
        self._native.log_artifact(
            Path(path),
            _artifact_name(name),
            step,
            content_type,
            timestamp_unix_ms,
        )

    def log_artifacts(self, path: Path, name: str, step: int) -> None:
        self._native.log_artifacts(Path(path), _artifact_name(name), step)

    def close(self) -> None:
        self._native.close()


def _artifact_name(name: str) -> str:
    path = PurePosixPath(name)
    if not path.parts or path.is_absolute() or ".." in path.parts:
        raise ValueError(
            "artifact name must be a relative path without parent components"
        )
    return path.as_posix()
