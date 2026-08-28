"""Standalone machine-readable environment reports for reproducible experiments."""

from __future__ import annotations

import json
import platform
import socket
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import torch

from tiny_qwen_coder.reporting.manifest import DependencyVersions, collect_dependency_versions

_ENVIRONMENT_REPORT_SCHEMA_VERSION = 1


class EnvironmentReportError(ValueError):
    """Raised when an environment report cannot be created safely."""


@dataclass(frozen=True, slots=True)
class PythonEnvironment:
    """Python interpreter identity."""

    version: str
    implementation: str


@dataclass(frozen=True, slots=True)
class PyTorchEnvironment:
    """PyTorch and CUDA runtime metadata."""

    version: str
    cuda_available: bool
    cuda_runtime: str | None
    cudnn_version: int | None
    deterministic_algorithms: bool


@dataclass(frozen=True, slots=True)
class HostEnvironment:
    """Host operating-system metadata."""

    hostname: str
    system: str
    release: str
    machine: str


@dataclass(frozen=True, slots=True)
class GpuEnvironment:
    """One CUDA-visible GPU and its current VRAM snapshot."""

    index: int
    name: str
    compute_capability: str
    total_vram_bytes: int
    free_vram_bytes: int


@dataclass(frozen=True, slots=True)
class EnvironmentReport:
    """Standalone software/runtime/hardware report generated without training."""

    schema_version: int
    generated_at_utc: str
    python: PythonEnvironment
    pytorch: PyTorchEnvironment
    dependencies: DependencyVersions
    host: HostEnvironment
    gpus: tuple[GpuEnvironment, ...]

    def __post_init__(self) -> None:
        if self.schema_version != _ENVIRONMENT_REPORT_SCHEMA_VERSION:
            raise EnvironmentReportError(
                f"unsupported environment report schema_version {self.schema_version}; "
                f"expected {_ENVIRONMENT_REPORT_SCHEMA_VERSION}"
            )
        try:
            generated_at = datetime.fromisoformat(self.generated_at_utc.replace("Z", "+00:00"))
        except ValueError as exc:
            raise EnvironmentReportError("generated_at_utc must be an ISO-8601 timestamp") from exc
        if generated_at.tzinfo is None or generated_at.utcoffset() != UTC.utcoffset(generated_at):
            raise EnvironmentReportError("generated_at_utc must be UTC")
        if self.pytorch.cuda_available != bool(self.gpus):
            raise EnvironmentReportError("CUDA availability and GPU inventory disagree")
        for gpu in self.gpus:
            if gpu.total_vram_bytes <= 0:
                raise EnvironmentReportError("GPU total VRAM must be greater than zero")
            if not 0 <= gpu.free_vram_bytes <= gpu.total_vram_bytes:
                raise EnvironmentReportError("GPU free VRAM must be between zero and total VRAM")


def _utc_timestamp(now: datetime | None = None) -> str:
    instant = now or datetime.now(UTC)
    if instant.tzinfo is None:
        raise EnvironmentReportError("environment report timestamp must be timezone-aware")
    return instant.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def collect_environment_report(*, now: datetime | None = None) -> EnvironmentReport:
    """Collect software and hardware metadata without loading a model or starting training."""

    cuda_available = torch.cuda.is_available()
    gpus: list[GpuEnvironment] = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            free_memory, total_memory = torch.cuda.mem_get_info(index)
            gpus.append(
                GpuEnvironment(
                    index=index,
                    name=properties.name,
                    compute_capability=f"{properties.major}.{properties.minor}",
                    total_vram_bytes=total_memory,
                    free_vram_bytes=free_memory,
                )
            )

    cudnn_version = cast(Callable[[], int | None], torch.backends.cudnn.version)()

    return EnvironmentReport(
        schema_version=_ENVIRONMENT_REPORT_SCHEMA_VERSION,
        generated_at_utc=_utc_timestamp(now),
        python=PythonEnvironment(
            version=platform.python_version(),
            implementation=platform.python_implementation(),
        ),
        pytorch=PyTorchEnvironment(
            version=str(torch.__version__),
            cuda_available=cuda_available,
            cuda_runtime=torch.version.cuda,
            cudnn_version=cudnn_version,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        ),
        dependencies=collect_dependency_versions(),
        host=HostEnvironment(
            hostname=socket.gethostname(),
            system=platform.system(),
            release=platform.release(),
            machine=platform.machine(),
        ),
        gpus=tuple(gpus),
    )


def environment_report_json(report: EnvironmentReport) -> str:
    """Serialize an environment report deterministically."""

    return json.dumps(asdict(report), indent=2, sort_keys=True) + "\n"


def write_environment_report(report: EnvironmentReport, path: Path) -> Path:
    """Atomically write an environment report to ``path``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(environment_report_json(report), encoding="utf-8")
    temporary.replace(path)
    return path


def main() -> None:
    """Print the current environment report as JSON without starting training."""

    print(environment_report_json(collect_environment_report()), end="")


if __name__ == "__main__":
    main()
