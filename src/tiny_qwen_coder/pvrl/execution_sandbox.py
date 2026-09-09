"""Hardened PVRL execution boundary with grader/source-of-truth isolation.

PVRL-104 intentionally keeps protected grader material out of every candidate
execution request.  Trusted grading code runs in the orchestrator and may drive
the candidate only through ``HardenedCandidateExecutor``.  The candidate OCI
container therefore receives a disposable copy of public workspace material,
never hidden tests, references, host credentials, container-engine sockets, or
personal host storage.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Protocol

from tiny_qwen_coder.config import ExecutionConfig
from tiny_qwen_coder.evaluation.execution import (
    ConstrainedExecutionHarness,
    DirectExecutionHarness,
    ExecutionFile,
    ExecutionLimits,
    ExecutionRequest,
    ExecutionResult,
)
from tiny_qwen_coder.pvrl.environment_manifest import (
    ArtifactKind,
    ArtifactVisibility,
    EnvironmentArtifact,
    EnvironmentManifest,
    NetworkMode,
    environment_manifest_sha256,
)
from tiny_qwen_coder.pvrl.reference_validation import (
    ReferenceValidationInfrastructureError,
    ReferenceValidationSandbox,
    ReferenceValidationSandboxFactory,
    ValidationExecutionClass,
    ValidationExecutionEvidence,
    ValidationPhase,
)

_MIB = 1024 * 1024
_WORKSPACE_PREFIX = "workspace/"
_REFERENCE_PREFIX = "reference/"
_TRUSTED_PREFIX = ".pvrl/"
_LIMIT_RUNNER_PATH = ".pvrl/limit_runner.py"
_LIMIT_RUNNER = b"""from __future__ import annotations

import os
import resource
import sys

cpu_seconds = int(sys.argv[1])
max_file_bytes = int(sys.argv[2])
command = sys.argv[3:]
if not command:
    raise SystemExit(125)
resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
resource.setrlimit(resource.RLIMIT_FSIZE, (max_file_bytes, max_file_bytes))
os.execvp(command[0], command)
"""


class HardenedSandboxError(ValueError):
    """Raised when sandbox material or configuration violates the PVRL boundary."""


class HardenedSandboxClosedError(RuntimeError):
    """Raised when a disposed validation sandbox is reused."""


@dataclass(frozen=True, slots=True)
class SandboxArtifactContent:
    """Exact immutable bytes for one manifest artifact."""

    path: str
    content: bytes

    def __post_init__(self) -> None:
        _require_relative_path(self.path, field_name="sandbox artifact path")
        if not isinstance(self.content, bytes):
            raise TypeError("sandbox artifact content must be bytes")

    @classmethod
    def from_text(cls, path: str, content: str) -> SandboxArtifactContent:
        """Create UTF-8 artifact content without newline rewriting."""

        return cls(path=path, content=content.encode("utf-8"))

    @property
    def sha256(self) -> str:
        """Return the exact byte-level identity used by the environment manifest."""

        return hashlib.sha256(self.content).hexdigest()


@dataclass(frozen=True, slots=True)
class TrustedGraderArtifacts:
    """Pristine grader/public-test source of truth unavailable to candidate execution."""

    files: tuple[SandboxArtifactContent, ...]

    def __post_init__(self) -> None:
        paths = tuple(item.path for item in self.files)
        if tuple(sorted(paths)) != paths:
            raise HardenedSandboxError("trusted grader artifacts must be sorted by path")
        if len(paths) != len(set(paths)):
            raise HardenedSandboxError("trusted grader artifact paths must be unique")

    @property
    def paths(self) -> tuple[str, ...]:
        """Return protected artifact names for trusted grader introspection."""

        return tuple(item.path for item in self.files)

    def read_bytes(self, path: str) -> bytes:
        """Read one pristine grader-controlled artifact by manifest path."""

        _require_relative_path(path, field_name="trusted grader artifact path")
        for item in self.files:
            if item.path == path:
                return item.content
        raise KeyError(path)

    def read_text(self, path: str) -> str:
        """Read one pristine UTF-8 grader-controlled artifact."""

        return self.read_bytes(path).decode("utf-8")


@dataclass(frozen=True, slots=True)
class HardenedEnvironmentMaterial:
    """Manifest plus exact bytes, partitioned at the trusted boundary."""

    manifest: EnvironmentManifest
    artifacts: tuple[SandboxArtifactContent, ...]

    def __post_init__(self) -> None:
        if self.manifest.network.mode is not NetworkMode.DISABLED:
            raise HardenedSandboxError("PVRL hardened sandbox requires outbound network disabled")

        paths = tuple(item.path for item in self.artifacts)
        if tuple(sorted(paths)) != paths:
            raise HardenedSandboxError("sandbox artifact contents must be sorted by path")
        if len(paths) != len(set(paths)):
            raise HardenedSandboxError("sandbox artifact content paths must be unique")

        manifest_by_path = {item.path: item for item in self.manifest.artifacts}
        if set(paths) != set(manifest_by_path):
            raise HardenedSandboxError(
                "sandbox material must provide exactly one byte payload for every manifest artifact"
            )
        content_by_path = {item.path: item for item in self.artifacts}
        for path, artifact in manifest_by_path.items():
            content = content_by_path[path]
            if content.sha256 != artifact.sha256:
                raise HardenedSandboxError(
                    f"sandbox artifact {path!r} does not match manifest SHA-256"
                )
            if len(content.content) > self.manifest.resources.file_bytes:
                raise HardenedSandboxError(
                    f"sandbox artifact {path!r} exceeds the manifest file-size limit"
                )
            _validate_artifact_namespace(artifact)

        candidate_paths = tuple(path for path, _ in self.initial_candidate_files())
        if len(candidate_paths) != len(set(candidate_paths)):
            raise HardenedSandboxError(
                "public manifest artifacts collide in the candidate workspace namespace"
            )
        reference_targets = tuple(path for path, _ in self.reference_replacements())
        if len(reference_targets) != len(set(reference_targets)):
            raise HardenedSandboxError("reference artifacts target the same candidate path")

        initial_workspace = dict(self.initial_candidate_files())
        if (
            sum(len(content) for content in initial_workspace.values())
            > self.manifest.resources.disk_bytes
        ):
            raise HardenedSandboxError(
                "initial candidate workspace exceeds the manifest disk limit"
            )
        for path, content in self.reference_replacements():
            initial_workspace[path] = content
        if (
            sum(len(content) for content in initial_workspace.values())
            > self.manifest.resources.disk_bytes
        ):
            raise HardenedSandboxError("reference workspace exceeds the manifest disk limit")

        _execution_limits(self.manifest)
        _exact_image_reference(self.manifest)

    def initial_candidate_files(self) -> tuple[tuple[str, bytes], ...]:
        """Return only public bytes that may enter a solver/candidate container."""

        manifest_by_path = {item.path: item for item in self.manifest.artifacts}
        visible: list[tuple[str, bytes]] = []
        for content in self.artifacts:
            artifact = manifest_by_path[content.path]
            candidate_path = _candidate_visible_path(artifact)
            if candidate_path is not None:
                visible.append((candidate_path, content.content))
        return tuple(sorted(visible))

    def reference_replacements(self) -> tuple[tuple[str, bytes], ...]:
        """Map hidden reference artifacts onto candidate paths by namespace convention."""

        manifest_by_path = {item.path: item for item in self.manifest.artifacts}
        replacements: list[tuple[str, bytes]] = []
        for content in self.artifacts:
            artifact = manifest_by_path[content.path]
            if artifact.kind is ArtifactKind.REFERENCE:
                target = _strip_required_prefix(
                    artifact.path,
                    _REFERENCE_PREFIX,
                    field_name="reference artifact path",
                )
                replacements.append((target, content.content))
        return tuple(sorted(replacements))

    def trusted_grader_artifacts(self) -> TrustedGraderArtifacts:
        """Return pristine grader/support bytes while withholding reference solutions."""

        manifest_by_path = {item.path: item for item in self.manifest.artifacts}
        protected = tuple(
            content
            for content in self.artifacts
            if manifest_by_path[content.path].kind in {ArtifactKind.GRADER, ArtifactKind.SUPPORT}
        )
        return TrustedGraderArtifacts(files=protected)


class CandidateExecutionHarness(Protocol):
    """Minimal execution backend used by the hardened candidate boundary."""

    def run(
        self,
        request: ExecutionRequest,
        execution: ExecutionConfig,
        *,
        limits: ExecutionLimits | None = None,
    ) -> ExecutionResult:
        """Execute one isolated candidate request."""
        ...


class HardenedCandidateExecutor:
    """Execute a public candidate workspace with no protected grader material mounted."""

    def __init__(
        self,
        manifest: EnvironmentManifest,
        workspace: tuple[tuple[str, bytes], ...],
        *,
        harness: CandidateExecutionHarness | None = None,
    ) -> None:
        if manifest.network.mode is not NetworkMode.DISABLED:
            raise HardenedSandboxError("candidate execution cannot enable outbound network")
        resolved_harness: CandidateExecutionHarness = (
            harness if harness is not None else ConstrainedExecutionHarness()
        )
        if isinstance(resolved_harness, DirectExecutionHarness):
            raise HardenedSandboxError(
                "PVRL hardened execution refuses the reduced-isolation direct backend"
            )
        self._manifest = manifest
        self._workspace = _normalize_workspace(workspace)
        for path, content in self._workspace:
            if path.startswith(_TRUSTED_PREFIX):
                raise HardenedSandboxError(
                    f"candidate workspace path {path!r} uses reserved PVRL namespace"
                )
            if len(content) > manifest.resources.file_bytes:
                raise HardenedSandboxError(
                    f"candidate workspace file {path!r} exceeds manifest file-size limit"
                )
        if sum(len(content) for _, content in self._workspace) > manifest.resources.disk_bytes:
            raise HardenedSandboxError("candidate workspace exceeds manifest disk limit")
        self._harness = resolved_harness

    @property
    def visible_paths(self) -> tuple[str, ...]:
        """Return exactly the file names visible inside the candidate workspace."""

        return tuple(path for path, _ in self._workspace)

    def read_visible_bytes(self, path: str) -> bytes:
        """Inspect candidate-visible bytes from trusted orchestration code."""

        _require_relative_path(path, field_name="candidate workspace path")
        for candidate_path, content in self._workspace:
            if candidate_path == path:
                return content
        raise KeyError(path)

    def run(self, command: tuple[str, ...]) -> ExecutionResult:
        """Run one disposable OCI attempt under the manifest's fail-closed limits."""

        wrapped_command = (
            "python",
            "-I",
            "-B",
            f"/input/{_LIMIT_RUNNER_PATH}",
            str(self._manifest.resources.cpu_seconds),
            str(self._manifest.resources.file_bytes),
            *command,
        )
        request = ExecutionRequest(
            image=_exact_image_reference(self._manifest),
            command=wrapped_command,
            files=(
                *(ExecutionFile(path=path, content=content) for path, content in self._workspace),
                ExecutionFile(path=_LIMIT_RUNNER_PATH, content=_LIMIT_RUNNER),
            ),
        )
        execution = ExecutionConfig(
            timeout_seconds=float(self._manifest.resources.wall_clock_seconds),
            network_enabled=False,
        )
        return self._harness.run(
            request,
            execution,
            limits=_execution_limits(self._manifest),
        )


class HardenedReferenceGrader(Protocol):
    """Trusted host-side grader that never passes source-of-truth files to the candidate."""

    def run_hidden(
        self,
        candidate: HardenedCandidateExecutor,
        protected: TrustedGraderArtifacts,
        phase: ValidationPhase,
    ) -> ValidationExecutionEvidence:
        """Drive hidden validation through the candidate executor."""
        ...

    def run_preservation(
        self,
        candidate: HardenedCandidateExecutor,
        protected: TrustedGraderArtifacts,
    ) -> ValidationExecutionEvidence:
        """Drive pristine public/preservation checks through the candidate executor."""
        ...


class HardenedReferenceValidationSandbox(ReferenceValidationSandbox):
    """One disposable PVRL-103 sandbox backed by the PVRL-104 execution boundary."""

    def __init__(
        self,
        material: HardenedEnvironmentMaterial,
        grader: HardenedReferenceGrader,
        *,
        harness: CandidateExecutionHarness | None = None,
    ) -> None:
        self._manifest = material.manifest
        self._workspace = dict(material.initial_candidate_files())
        self._references = material.reference_replacements()
        self._material = material
        self._grader = grader
        self._harness = harness
        self._closed = False

    def _candidate(self) -> HardenedCandidateExecutor:
        self._require_open()
        return HardenedCandidateExecutor(
            self._manifest,
            tuple(sorted(self._workspace.items())),
            harness=self._harness,
        )

    def _require_open(self) -> None:
        if self._closed:
            raise HardenedSandboxClosedError("PVRL validation sandbox is already disposed")

    def run_hidden(self, phase: ValidationPhase) -> ValidationExecutionEvidence:
        """Run trusted hidden grading without exposing hidden files to the candidate."""

        self._require_open()
        if phase not in {ValidationPhase.INITIAL_HIDDEN, ValidationPhase.REFERENCE_HIDDEN}:
            raise HardenedSandboxError("run_hidden requires an initial/reference hidden phase")
        return self._grader.run_hidden(
            self._candidate(),
            self._material.trusted_grader_artifacts(),
            phase,
        )

    def apply_reference(self) -> str:
        """Apply pristine hidden reference replacements in trusted orchestration code."""

        self._require_open()
        for path, content in self._references:
            self._workspace[path] = content
        return _workspace_sha256(tuple(sorted(self._workspace.items())))

    def run_preservation(self) -> ValidationExecutionEvidence:
        """Run pristine preservation checks, ignoring candidate copies of public tests."""

        self._require_open()
        return self._grader.run_preservation(
            self._candidate(),
            self._material.trusted_grader_artifacts(),
        )

    def close(self) -> None:
        """Dispose all mutable candidate state; protected material remains immutable."""

        self._workspace.clear()
        self._closed = True


class HardenedReferenceValidationSandboxFactory(ReferenceValidationSandboxFactory):
    """Create fresh PVRL-104 sandboxes for PVRL-103 reference-first validation."""

    def __init__(
        self,
        material: HardenedEnvironmentMaterial,
        grader: HardenedReferenceGrader,
        *,
        harness: CandidateExecutionHarness | None = None,
    ) -> None:
        if isinstance(harness, DirectExecutionHarness):
            raise HardenedSandboxError(
                "PVRL hardened execution refuses the reduced-isolation direct backend"
            )
        self._material = material
        self._grader = grader
        self._harness = harness
        self._manifest_sha256 = environment_manifest_sha256(material.manifest)

    def create(
        self,
        manifest: EnvironmentManifest,
        *,
        repetition: int,
    ) -> HardenedReferenceValidationSandbox:
        """Create an identity-checked fresh sandbox with pristine grader/reference bytes."""

        if repetition <= 0:
            raise HardenedSandboxError("sandbox repetition must be positive")
        if manifest.environment_id != self._material.manifest.environment_id:
            raise ReferenceValidationInfrastructureError(
                ValidationExecutionClass.ENVIRONMENT_IDENTITY_MISMATCH,
                "hardened sandbox material targets another environment_id",
            )
        if manifest.material_sha256 != self._material.manifest.material_sha256:
            raise ReferenceValidationInfrastructureError(
                ValidationExecutionClass.ENVIRONMENT_IDENTITY_MISMATCH,
                "hardened sandbox material targets different environment material",
            )
        if environment_manifest_sha256(manifest) != self._manifest_sha256:
            raise ReferenceValidationInfrastructureError(
                ValidationExecutionClass.ENVIRONMENT_IDENTITY_MISMATCH,
                "hardened sandbox manifest evidence changed from validated material",
            )
        return HardenedReferenceValidationSandbox(
            self._material,
            self._grader,
            harness=self._harness,
        )


def _require_relative_path(value: str, *, field_name: str) -> None:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\x00" in value
    ):
        raise HardenedSandboxError(
            f"{field_name} must be a normalized relative POSIX path without traversal"
        )


def _strip_required_prefix(path: str, prefix: str, *, field_name: str) -> str:
    if not path.startswith(prefix) or path == prefix:
        raise HardenedSandboxError(f"{field_name} must live under {prefix!r}")
    candidate = path.removeprefix(prefix)
    _require_relative_path(candidate, field_name=field_name)
    return candidate


def _validate_artifact_namespace(artifact: EnvironmentArtifact) -> None:
    if artifact.kind is ArtifactKind.WORKSPACE:
        _strip_required_prefix(
            artifact.path,
            _WORKSPACE_PREFIX,
            field_name="workspace artifact path",
        )
    elif artifact.kind is ArtifactKind.REFERENCE:
        _strip_required_prefix(
            artifact.path,
            _REFERENCE_PREFIX,
            field_name="reference artifact path",
        )


def _candidate_visible_path(artifact: EnvironmentArtifact) -> str | None:
    if artifact.kind is ArtifactKind.WORKSPACE:
        return _strip_required_prefix(
            artifact.path,
            _WORKSPACE_PREFIX,
            field_name="workspace artifact path",
        )
    if artifact.visibility is ArtifactVisibility.PUBLIC and artifact.kind is ArtifactKind.SUPPORT:
        _require_relative_path(artifact.path, field_name="public support artifact path")
        return artifact.path
    return None


def _normalize_workspace(
    workspace: tuple[tuple[str, bytes], ...],
) -> tuple[tuple[str, bytes], ...]:
    normalized = tuple(sorted(workspace))
    paths = tuple(path for path, _ in normalized)
    if len(paths) != len(set(paths)):
        raise HardenedSandboxError("candidate workspace paths must be unique")
    for path, content in normalized:
        _require_relative_path(path, field_name="candidate workspace path")
        if not isinstance(content, bytes):
            raise TypeError("candidate workspace content must be bytes")
    return normalized


def _workspace_sha256(workspace: tuple[tuple[str, bytes], ...]) -> str:
    payload = [
        {"path": path, "sha256": hashlib.sha256(content).hexdigest()}
        for path, content in _normalize_workspace(workspace)
    ]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode(
            "ascii"
        )
    ).hexdigest()


def _exact_image_reference(manifest: EnvironmentManifest) -> str:
    image = manifest.runtime.image
    digest = manifest.runtime.image_digest
    if "@" not in image:
        return f"{image}@{digest}"
    base, actual_digest = image.rsplit("@", 1)
    if not base or actual_digest != digest:
        raise HardenedSandboxError(
            "runtime image digest embedded in image reference does not match image_digest"
        )
    return image


def _execution_limits(manifest: EnvironmentManifest) -> ExecutionLimits:
    resources = manifest.resources
    # The reusable OCI harness permits swap up to 2x its configured memory value.
    # Use half of the manifest memory budget so RAM+swap cannot exceed that budget.
    memory_mebibytes = resources.memory_bytes // (2 * _MIB)
    disk_mebibytes = resources.disk_bytes // _MIB
    if memory_mebibytes < 1:
        raise HardenedSandboxError(
            "PVRL memory limit must allow at least two MiB for bounded RAM and swap"
        )
    if disk_mebibytes < 2:
        raise HardenedSandboxError(
            "PVRL disk limit must allow at least two MiB for workspace and temporary storage"
        )

    temp_mebibytes = max(1, min(64, disk_mebibytes // 4))
    workspace_mebibytes = disk_mebibytes - temp_mebibytes
    if workspace_mebibytes < 1:  # pragma: no cover - guarded by disk check
        raise HardenedSandboxError("PVRL disk split left no candidate workspace capacity")

    return ExecutionLimits(
        cpus=1.0,
        memory_mebibytes=memory_mebibytes,
        pids=resources.pids_limit,
        workspace_mebibytes=workspace_mebibytes,
        temp_mebibytes=temp_mebibytes,
        max_output_bytes=resources.output_bytes,
        max_input_bytes=resources.disk_bytes + len(_LIMIT_RUNNER),
    )
