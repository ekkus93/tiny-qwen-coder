"""Machine-readable provenance manifests for training and evaluation runs."""

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
from typing import Literal, TypeAlias
from uuid import uuid4

import torch
import yaml

from tiny_qwen_coder.reproducibility import SeedError, validate_seed

RunKind: TypeAlias = Literal["training", "evaluation"]

_MANIFEST_SCHEMA_VERSION = 1
_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_LANGUAGE_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")
_TRACKED_DISTRIBUTIONS = (
    "accelerate",
    "bitsandbytes",
    "datasets",
    "numpy",
    "peft",
    "pyyaml",
    "tiny-qwen-coder",
    "torch",
    "transformers",
    "trl",
)


class ManifestError(ValueError):
    """Raised when a run manifest cannot be created safely."""


def _require_non_empty(value: str, *, field_name: str) -> None:
    if not value.strip():
        raise ManifestError(f"{field_name} must not be empty")


def _require_sha(value: str, *, field_name: str) -> None:
    if not _SHA_PATTERN.fullmatch(value):
        raise ManifestError(f"{field_name} must be a lowercase 40-character Git SHA")


def _require_language(value: str) -> None:
    if not _LANGUAGE_PATTERN.fullmatch(value):
        raise ManifestError("language must match ^[a-z][a-z0-9_-]*$")


@dataclass(frozen=True, slots=True)
class BaseModelIdentity:
    """Exact model/tokenizer identity required by every run."""

    repository: str
    revision: str
    tokenizer_repository: str
    tokenizer_revision: str

    def __post_init__(self) -> None:
        _require_non_empty(self.repository, field_name="base_model.repository")
        _require_sha(self.revision, field_name="base_model.revision")
        _require_non_empty(self.tokenizer_repository, field_name="base_model.tokenizer_repository")
        _require_sha(self.tokenizer_revision, field_name="base_model.tokenizer_revision")


@dataclass(frozen=True, slots=True)
class AdapterIdentity:
    """Adapter identity; both fields are null only for base-only evaluation."""

    family: str | None
    adapter_id: str | None

    def __post_init__(self) -> None:
        if (self.family is None) != (self.adapter_id is None):
            raise ManifestError("adapter family and adapter_id must be defined together")
        if self.family is not None:
            _require_non_empty(self.family, field_name="adapter.family")
        if self.adapter_id is not None:
            _require_non_empty(self.adapter_id, field_name="adapter.adapter_id")


@dataclass(frozen=True, slots=True)
class DependencyVersions:
    """Versions of the project and core ML runtime dependencies."""

    accelerate: str
    bitsandbytes: str | None
    datasets: str
    numpy: str
    peft: str
    pyyaml: str
    tiny_qwen_coder: str
    torch: str
    transformers: str
    trl: str


@dataclass(frozen=True, slots=True)
class GpuMetadata:
    """One CUDA-visible GPU."""

    index: int
    name: str
    total_memory_bytes: int
    compute_capability: str


@dataclass(frozen=True, slots=True)
class HostMetadata:
    """Execution host metadata collected without requiring a CUDA device."""

    hostname: str
    system: str
    release: str
    machine: str
    python_version: str
    cuda_available: bool
    cuda_runtime: str | None
    gpus: tuple[GpuMetadata, ...]


@dataclass(frozen=True, slots=True)
class GitMetadata:
    """Source-tree identity for the run."""

    sha: str
    dirty: bool

    def __post_init__(self) -> None:
        _require_sha(self.sha, field_name="git.sha")


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Complete provenance envelope emitted by one training/evaluation run."""

    schema_version: int
    run_id: str
    run_kind: RunKind
    created_at_utc: str
    git: GitMetadata
    base_model: BaseModelIdentity
    language: str
    adapter: AdapterIdentity
    seed: int
    dependencies: DependencyVersions
    host: HostMetadata

    def __post_init__(self) -> None:
        if self.schema_version != _MANIFEST_SCHEMA_VERSION:
            raise ManifestError(
                f"unsupported manifest schema_version {self.schema_version}; "
                f"expected {_MANIFEST_SCHEMA_VERSION}"
            )
        _require_non_empty(self.run_id, field_name="run_id")
        _require_language(self.language)
        try:
            validate_seed(self.seed)
        except SeedError as exc:
            raise ManifestError(str(exc)) from exc
        if self.run_kind == "training" and self.adapter.adapter_id is None:
            raise ManifestError("training manifests require an adapter identity")
        try:
            parsed = datetime.fromisoformat(self.created_at_utc.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ManifestError("created_at_utc must be an ISO-8601 timestamp") from exc
        if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
            raise ManifestError("created_at_utc must be UTC")


def load_base_model_identity(path: Path) -> BaseModelIdentity:
    """Read exact model/tokenizer identity from the canonical base YAML."""

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ManifestError(f"could not read base-model config {path}") from exc
    if not isinstance(raw, dict):
        raise ManifestError("base-model config must be a mapping")

    model = raw.get("model")
    tokenizer = raw.get("tokenizer")
    if not isinstance(model, dict) or not isinstance(tokenizer, dict):
        raise ManifestError("base-model config must contain model and tokenizer mappings")

    fields = (
        (model, "repository", "model.repository"),
        (model, "revision", "model.revision"),
        (tokenizer, "repository", "tokenizer.repository"),
        (tokenizer, "revision", "tokenizer.revision"),
    )
    values: list[str] = []
    for mapping, key, field_name in fields:
        value = mapping.get(key)
        if not isinstance(value, str):
            raise ManifestError(f"{field_name} must be a string")
        values.append(value)

    return BaseModelIdentity(
        repository=values[0],
        revision=values[1],
        tokenizer_repository=values[2],
        tokenizer_revision=values[3],
    )


def collect_dependency_versions() -> DependencyVersions:
    """Collect the dependency versions required for experiment reproduction."""

    versions: dict[str, str] = {}
    for distribution in _TRACKED_DISTRIBUTIONS:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            if distribution == "bitsandbytes":
                versions[distribution] = ""
                continue
            raise ManifestError(f"required distribution is not installed: {distribution}") from exc

    return DependencyVersions(
        accelerate=versions["accelerate"],
        bitsandbytes=versions["bitsandbytes"] or None,
        datasets=versions["datasets"],
        numpy=versions["numpy"],
        peft=versions["peft"],
        pyyaml=versions["pyyaml"],
        tiny_qwen_coder=versions["tiny-qwen-coder"],
        torch=versions["torch"],
        transformers=versions["transformers"],
        trl=versions["trl"],
    )


def collect_host_metadata() -> HostMetadata:
    """Collect host and CUDA-visible GPU metadata; CPU-only hosts are valid."""

    cuda_available = torch.cuda.is_available()
    gpus: list[GpuMetadata] = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpus.append(
                GpuMetadata(
                    index=index,
                    name=properties.name,
                    total_memory_bytes=properties.total_memory,
                    compute_capability=f"{properties.major}.{properties.minor}",
                )
            )

    return HostMetadata(
        hostname=socket.gethostname(),
        system=platform.system(),
        release=platform.release(),
        machine=platform.machine(),
        python_version=platform.python_version(),
        cuda_available=cuda_available,
        cuda_runtime=torch.version.cuda,
        gpus=tuple(gpus),
    )


def collect_git_metadata(repo_root: Path = Path(".")) -> GitMetadata:
    """Resolve the exact source commit and whether the working tree is dirty."""

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
            raise ManifestError("could not resolve Git commit SHA") from exc

    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ManifestError("could not inspect Git working-tree state") from exc

    return GitMetadata(sha=sha, dirty=bool(status.strip()))


def _utc_timestamp(now: datetime | None = None) -> str:
    instant = now or datetime.now(UTC)
    if instant.tzinfo is None:
        raise ManifestError("run timestamp must be timezone-aware")
    return instant.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _generate_run_id(run_kind: RunKind, language: str, created_at_utc: str, git_sha: str) -> str:
    compact_time = created_at_utc.replace("-", "").replace(":", "").replace(".", "")
    return f"{run_kind}-{language}-{compact_time}-{git_sha[:8]}-{uuid4().hex[:8]}"


def create_run_manifest(
    *,
    run_kind: RunKind,
    base_model: BaseModelIdentity,
    language: str,
    adapter: AdapterIdentity,
    seed: int,
    repo_root: Path = Path("."),
    run_id: str | None = None,
    created_at: datetime | None = None,
    git: GitMetadata | None = None,
    dependencies: DependencyVersions | None = None,
    host: HostMetadata | None = None,
) -> RunManifest:
    """Create one complete provenance manifest without writing it."""

    _require_language(language)
    git_metadata = git or collect_git_metadata(repo_root)
    timestamp = _utc_timestamp(created_at)
    resolved_run_id = run_id or _generate_run_id(run_kind, language, timestamp, git_metadata.sha)
    return RunManifest(
        schema_version=_MANIFEST_SCHEMA_VERSION,
        run_id=resolved_run_id,
        run_kind=run_kind,
        created_at_utc=timestamp,
        git=git_metadata,
        base_model=base_model,
        language=language,
        adapter=adapter,
        seed=seed,
        dependencies=dependencies or collect_dependency_versions(),
        host=host or collect_host_metadata(),
    )


def manifest_json(manifest: RunManifest) -> str:
    """Serialize a manifest deterministically for storage and hashing."""

    return json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n"


def write_run_manifest(manifest: RunManifest, output_dir: Path) -> Path:
    """Atomically write ``run-manifest.json`` into a run output directory."""

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "run-manifest.json"
    temporary = output_dir / ".run-manifest.json.tmp"
    temporary.write_text(manifest_json(manifest), encoding="utf-8")
    temporary.replace(destination)
    return destination


def emit_training_run_manifest(
    *,
    output_dir: Path,
    base_config: Path,
    language: str,
    adapter_family: str,
    adapter_id: str,
    seed: int,
    repo_root: Path = Path("."),
) -> Path:
    """Create and write the provenance manifest required by a training run."""

    manifest = create_run_manifest(
        run_kind="training",
        base_model=load_base_model_identity(base_config),
        language=language,
        adapter=AdapterIdentity(family=adapter_family, adapter_id=adapter_id),
        seed=seed,
        repo_root=repo_root,
    )
    return write_run_manifest(manifest, output_dir)


def emit_evaluation_run_manifest(
    *,
    output_dir: Path,
    base_config: Path,
    language: str,
    seed: int,
    adapter_family: str | None = None,
    adapter_id: str | None = None,
    repo_root: Path = Path("."),
) -> Path:
    """Create and write provenance for a base-only or adapted evaluation run."""

    manifest = create_run_manifest(
        run_kind="evaluation",
        base_model=load_base_model_identity(base_config),
        language=language,
        adapter=AdapterIdentity(family=adapter_family, adapter_id=adapter_id),
        seed=seed,
        repo_root=repo_root,
    )
    return write_run_manifest(manifest, output_dir)
