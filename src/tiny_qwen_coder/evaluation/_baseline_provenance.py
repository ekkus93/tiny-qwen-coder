"""Dependency-light provenance collection for the canonical Python baseline."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import re
import socket
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch
import yaml

from tiny_qwen_coder.evaluation._baseline_types import PythonBaselineError
from tiny_qwen_coder.identities import AdapterIdentity, BaseModelIdentity

_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_TRACKED_DISTRIBUTIONS = (
    "accelerate",
    "datasets",
    "numpy",
    "peft",
    "pyyaml",
    "tiny-qwen-coder",
    "torch",
    "transformers",
    "trl",
)


@dataclass(frozen=True, slots=True)
class BaselineGpuProvenance:
    """One CUDA-visible GPU recorded by the baseline run."""

    index: int
    name: str
    total_memory_bytes: int
    compute_capability: str


@dataclass(frozen=True, slots=True)
class BaselineProvenance:
    """Source, dependency, host, and identity envelope for one P6-005 run."""

    schema_version: int
    created_at_utc: str
    source_git_sha: str
    source_git_dirty: bool
    base_model: BaseModelIdentity
    adapter: AdapterIdentity
    language: str
    seed: int
    hostname: str
    system: str
    release: str
    machine: str
    python_version: str
    cuda_available: bool
    cuda_runtime: str | None
    gpus: tuple[BaselineGpuProvenance, ...]
    dependencies: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise PythonBaselineError("unsupported baseline provenance schema version")
        if not _SHA_PATTERN.fullmatch(self.source_git_sha):
            raise PythonBaselineError("baseline provenance Git SHA is invalid")
        if self.language != "python":
            raise PythonBaselineError("baseline provenance must use language='python'")
        if self.adapter.adapter_id is not None:
            raise PythonBaselineError("baseline provenance must represent unchanged base only")
        if not self.created_at_utc.endswith("Z"):
            raise PythonBaselineError("baseline provenance timestamp must be UTC")
        if not self.hostname or not self.system or not self.machine or not self.python_version:
            raise PythonBaselineError("baseline provenance host metadata is incomplete")
        if not self.dependencies:
            raise PythonBaselineError("baseline provenance dependency versions are empty")
        names = tuple(name for name, _ in self.dependencies)
        if len(names) != len(set(names)):
            raise PythonBaselineError("baseline provenance dependency names must be unique")


def _strict_mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise PythonBaselineError(f"{context} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise PythonBaselineError(f"{context} keys must be strings")
    return dict(value)


def _expect_str(mapping: dict[str, object], key: str, *, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PythonBaselineError(f"{context}.{key} must be a non-empty string")
    return value


def load_baseline_base_model_identity(path: Path) -> BaseModelIdentity:
    """Read the exact model/tokenizer identity without importing reporting services."""

    try:
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PythonBaselineError(f"could not read base-model config {path}") from exc
    mapping = _strict_mapping(raw, context="base config")
    model = _strict_mapping(mapping.get("model"), context="base config.model")
    tokenizer = _strict_mapping(mapping.get("tokenizer"), context="base config.tokenizer")
    return BaseModelIdentity(
        repository=_expect_str(model, "repository", context="base config.model"),
        revision=_expect_str(model, "revision", context="base config.model"),
        tokenizer_repository=_expect_str(
            tokenizer,
            "repository",
            context="base config.tokenizer",
        ),
        tokenizer_revision=_expect_str(
            tokenizer,
            "revision",
            context="base config.tokenizer",
        ),
    )


def _git_metadata(repo_root: Path) -> tuple[str, bool]:
    github_sha = os.environ.get("GITHUB_SHA")
    if github_sha is not None and _SHA_PATTERN.fullmatch(github_sha):
        sha = github_sha
    else:
        try:
            sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise PythonBaselineError("could not resolve source Git commit") from exc
    if not _SHA_PATTERN.fullmatch(sha):
        raise PythonBaselineError("source Git commit is not a full lowercase SHA")
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PythonBaselineError("could not inspect source working tree") from exc
    return sha, bool(status.strip())


def _dependency_versions() -> tuple[tuple[str, str], ...]:
    versions: list[tuple[str, str]] = []
    for distribution in _TRACKED_DISTRIBUTIONS:
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise PythonBaselineError(
                f"required baseline distribution is not installed: {distribution}"
            ) from exc
        versions.append((distribution, version))
    return tuple(versions)


def collect_baseline_provenance(
    *,
    base_model: BaseModelIdentity,
    seed: int,
    repo_root: Path = Path("."),
    now: datetime | None = None,
) -> BaselineProvenance:
    """Collect complete unchanged-base run provenance before artifact freezing."""

    sha, dirty = _git_metadata(repo_root)
    instant = (now or datetime.now(UTC)).astimezone(UTC)
    created_at_utc = instant.isoformat(timespec="microseconds").replace("+00:00", "Z")
    cuda_available = torch.cuda.is_available()
    gpus: list[BaselineGpuProvenance] = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            gpus.append(
                BaselineGpuProvenance(
                    index=index,
                    name=props.name,
                    total_memory_bytes=props.total_memory,
                    compute_capability=f"{props.major}.{props.minor}",
                )
            )
    return BaselineProvenance(
        schema_version=1,
        created_at_utc=created_at_utc,
        source_git_sha=sha,
        source_git_dirty=dirty,
        base_model=base_model,
        adapter=AdapterIdentity(family=None, adapter_id=None),
        language="python",
        seed=seed,
        hostname=socket.gethostname(),
        system=platform.system(),
        release=platform.release(),
        machine=platform.machine(),
        python_version=platform.python_version(),
        cuda_available=cuda_available,
        cuda_runtime=torch.version.cuda,
        gpus=tuple(gpus),
        dependencies=_dependency_versions(),
    )


def baseline_provenance_json(provenance: BaselineProvenance) -> str:
    """Serialize baseline provenance deterministically."""

    return json.dumps(asdict(provenance), indent=2, sort_keys=True) + "\n"


def write_baseline_provenance(provenance: BaselineProvenance, output_dir: Path) -> Path:
    """Atomically write baseline provenance."""

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "provenance.json"
    temporary = output_dir / ".provenance.json.tmp"
    temporary.write_text(baseline_provenance_json(provenance), encoding="utf-8")
    temporary.replace(destination)
    return destination
