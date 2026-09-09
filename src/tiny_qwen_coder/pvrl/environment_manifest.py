"""Immutable executable-environment manifests for PVRL.

PVRL environments are content-addressed at the task-material boundary. Evidence
such as reference-first validation and contamination checks is recorded in the
manifest but does not alter the environment ID, so validation can be attached
without pretending that the task itself changed.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import TypeVar, cast

_SCHEMA_VERSION = 1
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_COMPONENT_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]*$")
_IMAGE_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_ENVIRONMENT_ID_PATTERN = re.compile(r"^pvrl-env-v1-g[0-9]{4,}-[0-9a-f]{20}$")

_EnumT = TypeVar("_EnumT", bound=StrEnum)


class EnvironmentManifestError(ValueError):
    """Raised when a PVRL environment manifest is incomplete or inconsistent."""


class ArtifactVisibility(StrEnum):
    """Whether an environment artifact is visible to the policy."""

    PUBLIC = "public"
    HIDDEN = "hidden"


class ArtifactKind(StrEnum):
    """Role of one immutable artifact in the executable environment."""

    WORKSPACE = "workspace"
    GRADER = "grader"
    REFERENCE = "reference"
    SUPPORT = "support"


class ProvenanceRole(StrEnum):
    """Producer role kept separate in the environment trust model."""

    AUTHOR = "author"
    ORACLE = "oracle"


class EvidenceStatus(StrEnum):
    """State of a validation process that can run after material is authored."""

    NOT_RUN = "not_run"
    PASSED = "passed"
    FAILED = "failed"


class ContaminationStatus(StrEnum):
    """Explicit protected-data contamination state for one environment."""

    NOT_RUN = "not_run"
    CLEAN = "clean"
    FINDINGS = "findings"


class NetworkMode(StrEnum):
    """Network policy supported by PVRL schema generation 1."""

    DISABLED = "disabled"


@dataclass(frozen=True, slots=True)
class EnvironmentArtifact:
    """Content-addressed file or contract material used by an environment."""

    path: str
    sha256: str
    visibility: ArtifactVisibility
    kind: ArtifactKind

    def __post_init__(self) -> None:
        normalized = PurePosixPath(self.path)
        if (
            not self.path
            or normalized.is_absolute()
            or ".." in normalized.parts
            or "." in normalized.parts
            or str(normalized) != self.path
        ):
            raise EnvironmentManifestError(
                "artifact path must be a normalized relative POSIX path without traversal"
            )
        _require_sha256(self.sha256, field_name=f"artifact[{self.path}].sha256")
        if self.kind is ArtifactKind.WORKSPACE and self.visibility is not ArtifactVisibility.PUBLIC:
            raise EnvironmentManifestError("workspace artifacts must be public")
        if self.kind is ArtifactKind.GRADER and self.visibility is not ArtifactVisibility.HIDDEN:
            raise EnvironmentManifestError("grader artifacts must be hidden")
        if self.kind is ArtifactKind.REFERENCE and self.visibility is not ArtifactVisibility.HIDDEN:
            raise EnvironmentManifestError("reference artifacts must be hidden")


@dataclass(frozen=True, slots=True)
class RuntimeIdentity:
    """Exact runtime/dependency identity required to construct an environment."""

    image: str
    image_digest: str
    dependency_lock_sha256: str
    python_version: str

    def __post_init__(self) -> None:
        if not self.image.strip():
            raise EnvironmentManifestError("runtime image must not be empty")
        if not _IMAGE_DIGEST_PATTERN.fullmatch(self.image_digest):
            raise EnvironmentManifestError(
                "runtime image_digest must be a sha256:<64-lowercase-hex> digest"
            )
        _require_sha256(
            self.dependency_lock_sha256,
            field_name="runtime.dependency_lock_sha256",
        )
        if not self.python_version.strip():
            raise EnvironmentManifestError("runtime python_version must not be empty")


@dataclass(frozen=True, slots=True)
class ProvenanceIdentity:
    """Exact producer identity for author or oracle material."""

    role: ProvenanceRole
    producer_type: str
    producer_id: str
    revision: str
    config_sha256: str
    output_sha256: str

    def __post_init__(self) -> None:
        if not _COMPONENT_ID_PATTERN.fullmatch(self.producer_type):
            raise EnvironmentManifestError(
                "provenance producer_type must be a stable lowercase component identifier"
            )
        if not self.producer_id.strip():
            raise EnvironmentManifestError("provenance producer_id must not be empty")
        if not self.revision.strip():
            raise EnvironmentManifestError("provenance revision must not be empty")
        _require_sha256(self.config_sha256, field_name="provenance.config_sha256")
        _require_sha256(self.output_sha256, field_name="provenance.output_sha256")


@dataclass(frozen=True, slots=True)
class ResourcePolicy:
    """Hard resource limits that are part of task material identity."""

    wall_clock_seconds: int
    cpu_seconds: int
    memory_bytes: int
    pids_limit: int
    output_bytes: int
    file_bytes: int
    disk_bytes: int

    def __post_init__(self) -> None:
        for field_name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise EnvironmentManifestError(
                    f"resource policy {field_name} must be a positive integer"
                )


@dataclass(frozen=True, slots=True)
class NetworkPolicy:
    """Network policy. Schema generation 1 permits no outbound network."""

    mode: NetworkMode

    def __post_init__(self) -> None:
        if self.mode is not NetworkMode.DISABLED:
            raise EnvironmentManifestError("PVRL schema v1 requires network mode disabled")


@dataclass(frozen=True, slots=True)
class ReferenceValidationEvidence:
    """Reference-first validation state without embedding execution output."""

    status: EvidenceStatus
    validator_id: str | None = None
    evidence_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.status is EvidenceStatus.NOT_RUN:
            if self.validator_id is not None or self.evidence_sha256 is not None:
                raise EnvironmentManifestError(
                    "not_run reference validation must not carry validator/evidence identity"
                )
            return
        if self.validator_id is None or not _COMPONENT_ID_PATTERN.fullmatch(self.validator_id):
            raise EnvironmentManifestError(
                "completed reference validation requires a stable validator_id"
            )
        if self.evidence_sha256 is None:
            raise EnvironmentManifestError(
                "completed reference validation requires evidence_sha256"
            )
        _require_sha256(
            self.evidence_sha256,
            field_name="reference_validation.evidence_sha256",
        )


@dataclass(frozen=True, slots=True)
class ContaminationEvidence:
    """Protected-data check state with explicit not-run semantics."""

    status: ContaminationStatus
    checker_ids: tuple[str, ...]
    evidence_sha256: str | None = None

    def __post_init__(self) -> None:
        if tuple(sorted(self.checker_ids)) != self.checker_ids:
            raise EnvironmentManifestError("contamination checker_ids must be sorted")
        if len(self.checker_ids) != len(set(self.checker_ids)):
            raise EnvironmentManifestError("contamination checker_ids must be unique")
        if any(not _COMPONENT_ID_PATTERN.fullmatch(item) for item in self.checker_ids):
            raise EnvironmentManifestError(
                "contamination checker_ids must be stable lowercase component identifiers"
            )
        if self.status is ContaminationStatus.NOT_RUN:
            if self.checker_ids or self.evidence_sha256 is not None:
                raise EnvironmentManifestError(
                    "not_run contamination state must contain no checks/evidence"
                )
            return
        if not self.checker_ids or self.evidence_sha256 is None:
            raise EnvironmentManifestError(
                "completed contamination state requires checker_ids and evidence_sha256"
            )
        _require_sha256(
            self.evidence_sha256,
            field_name="contamination.evidence_sha256",
        )


@dataclass(frozen=True, slots=True)
class EnvironmentManifest:
    """Complete immutable PVRL environment envelope for schema generation 1."""

    schema_version: int
    environment_id: str
    generation: int
    material_sha256: str
    task_category: str
    specification_sha256: str
    artifacts: tuple[EnvironmentArtifact, ...]
    runtime: RuntimeIdentity
    author_provenance: ProvenanceIdentity
    oracle_provenance: ProvenanceIdentity
    resources: ResourcePolicy
    network: NetworkPolicy
    contamination: ContaminationEvidence
    reference_validation: ReferenceValidationEvidence

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise EnvironmentManifestError(
                f"unsupported environment schema_version {self.schema_version}; "
                f"expected {_SCHEMA_VERSION}"
            )
        if self.generation <= 0:
            raise EnvironmentManifestError("environment generation must be positive")
        if not _COMPONENT_ID_PATTERN.fullmatch(self.task_category):
            raise EnvironmentManifestError(
                "task_category must be a stable lowercase component identifier"
            )
        _require_sha256(self.specification_sha256, field_name="specification_sha256")
        if not self.artifacts:
            raise EnvironmentManifestError("environment must contain artifacts")
        paths = tuple(item.path for item in self.artifacts)
        if tuple(sorted(paths)) != paths:
            raise EnvironmentManifestError("environment artifacts must be sorted by path")
        if len(paths) != len(set(paths)):
            raise EnvironmentManifestError("environment artifact paths must be unique")
        if not any(item.kind is ArtifactKind.WORKSPACE for item in self.artifacts):
            raise EnvironmentManifestError(
                "environment must contain at least one public workspace artifact"
            )
        if not any(item.kind is ArtifactKind.GRADER for item in self.artifacts):
            raise EnvironmentManifestError(
                "environment must contain at least one hidden grader artifact"
            )
        if self.author_provenance.role is not ProvenanceRole.AUTHOR:
            raise EnvironmentManifestError("author_provenance must have author role")
        if self.oracle_provenance.role is not ProvenanceRole.ORACLE:
            raise EnvironmentManifestError("oracle_provenance must have oracle role")

        expected_material = _material_sha256(self)
        _require_sha256(self.material_sha256, field_name="material_sha256")
        if self.material_sha256 != expected_material:
            raise EnvironmentManifestError(
                "environment material changed without recomputing material_sha256"
            )
        expected_id = _environment_id(
            schema_version=self.schema_version,
            generation=self.generation,
            material_sha256=expected_material,
        )
        if not _ENVIRONMENT_ID_PATTERN.fullmatch(self.environment_id):
            raise EnvironmentManifestError("environment_id does not match the PVRL v1 format")
        if self.environment_id != expected_id:
            raise EnvironmentManifestError(
                "environment material changed while retaining a stale environment_id"
            )


def _require_sha256(value: str, *, field_name: str) -> None:
    if not _SHA256_PATTERN.fullmatch(value):
        raise EnvironmentManifestError(f"{field_name} must be a lowercase SHA-256 digest")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _material_payload(
    *,
    schema_version: int,
    generation: int,
    task_category: str,
    specification_sha256: str,
    artifacts: tuple[EnvironmentArtifact, ...],
    runtime: RuntimeIdentity,
    author_provenance: ProvenanceIdentity,
    oracle_provenance: ProvenanceIdentity,
    resources: ResourcePolicy,
    network: NetworkPolicy,
) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "generation": generation,
        "task_category": task_category,
        "specification_sha256": specification_sha256,
        "artifacts": [asdict(item) for item in artifacts],
        "runtime": asdict(runtime),
        "author_provenance": asdict(author_provenance),
        "oracle_provenance": asdict(oracle_provenance),
        "resources": asdict(resources),
        "network": asdict(network),
    }


def _material_sha256(manifest: EnvironmentManifest) -> str:
    return _sha256(
        _material_payload(
            schema_version=manifest.schema_version,
            generation=manifest.generation,
            task_category=manifest.task_category,
            specification_sha256=manifest.specification_sha256,
            artifacts=manifest.artifacts,
            runtime=manifest.runtime,
            author_provenance=manifest.author_provenance,
            oracle_provenance=manifest.oracle_provenance,
            resources=manifest.resources,
            network=manifest.network,
        )
    )


def _environment_id(*, schema_version: int, generation: int, material_sha256: str) -> str:
    return f"pvrl-env-v{schema_version}-g{generation:04d}-{material_sha256[:20]}"


def create_environment_manifest(
    *,
    generation: int,
    task_category: str,
    specification_sha256: str,
    artifacts: tuple[EnvironmentArtifact, ...],
    runtime: RuntimeIdentity,
    author_provenance: ProvenanceIdentity,
    oracle_provenance: ProvenanceIdentity,
    resources: ResourcePolicy,
    network: NetworkPolicy,
    contamination: ContaminationEvidence,
    reference_validation: ReferenceValidationEvidence,
) -> EnvironmentManifest:
    """Create one content-addressed environment manifest."""

    ordered_artifacts = tuple(sorted(artifacts, key=lambda item: item.path))
    material = _sha256(
        _material_payload(
            schema_version=_SCHEMA_VERSION,
            generation=generation,
            task_category=task_category,
            specification_sha256=specification_sha256,
            artifacts=ordered_artifacts,
            runtime=runtime,
            author_provenance=author_provenance,
            oracle_provenance=oracle_provenance,
            resources=resources,
            network=network,
        )
    )
    return EnvironmentManifest(
        schema_version=_SCHEMA_VERSION,
        environment_id=_environment_id(
            schema_version=_SCHEMA_VERSION,
            generation=generation,
            material_sha256=material,
        ),
        generation=generation,
        material_sha256=material,
        task_category=task_category,
        specification_sha256=specification_sha256,
        artifacts=ordered_artifacts,
        runtime=runtime,
        author_provenance=author_provenance,
        oracle_provenance=oracle_provenance,
        resources=resources,
        network=network,
        contamination=contamination,
        reference_validation=reference_validation,
    )


def environment_manifest_json(manifest: EnvironmentManifest) -> str:
    """Serialize a manifest deterministically."""

    return json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n"


def environment_manifest_sha256(manifest: EnvironmentManifest) -> str:
    """Return SHA-256 of the exact deterministic manifest JSON bytes."""

    return hashlib.sha256(environment_manifest_json(manifest).encode("utf-8")).hexdigest()


def write_environment_manifest(manifest: EnvironmentManifest, destination: Path) -> Path:
    """Atomically write one deterministic environment manifest."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(environment_manifest_json(manifest), encoding="utf-8")
    temporary.replace(destination)
    return destination


def _strict_mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise EnvironmentManifestError(f"{context} must be a JSON object")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise EnvironmentManifestError(f"{context} keys must be strings")
        result[key] = item
    return result


def _required_str(mapping: Mapping[str, object], key: str, *, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise EnvironmentManifestError(f"{context}.{key} must be a non-empty string")
    return value


def _required_int(mapping: Mapping[str, object], key: str, *, context: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise EnvironmentManifestError(f"{context}.{key} must be an integer")
    return value


def _optional_str(mapping: Mapping[str, object], key: str, *, context: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise EnvironmentManifestError(f"{context}.{key} must be a non-empty string or null")
    return value


def _enum_value(
    enum_type: type[_EnumT],
    mapping: Mapping[str, object],
    key: str,
    *,
    context: str,
) -> _EnumT:
    raw = _required_str(mapping, key, context=context)
    try:
        return enum_type(raw)
    except ValueError as exc:
        raise EnvironmentManifestError(f"{context}.{key} has unsupported value {raw!r}") from exc


def environment_manifest_from_json(text: str) -> EnvironmentManifest:
    """Parse and fully validate one environment manifest JSON document."""

    try:
        raw: object = json.loads(text)
    except json.JSONDecodeError as exc:
        raise EnvironmentManifestError("environment manifest is not valid JSON") from exc
    payload = _strict_mapping(raw, context="environment manifest")

    artifacts_raw = payload.get("artifacts")
    if not isinstance(artifacts_raw, list):
        raise EnvironmentManifestError("environment manifest.artifacts must be a list")
    artifacts: list[EnvironmentArtifact] = []
    for index, item in enumerate(artifacts_raw):
        mapping = _strict_mapping(item, context=f"artifacts[{index}]")
        artifacts.append(
            EnvironmentArtifact(
                path=_required_str(mapping, "path", context=f"artifacts[{index}]"),
                sha256=_required_str(mapping, "sha256", context=f"artifacts[{index}]"),
                visibility=_enum_value(
                    ArtifactVisibility,
                    mapping,
                    "visibility",
                    context=f"artifacts[{index}]",
                ),
                kind=_enum_value(
                    ArtifactKind,
                    mapping,
                    "kind",
                    context=f"artifacts[{index}]",
                ),
            )
        )

    runtime_raw = _strict_mapping(payload.get("runtime"), context="runtime")
    author_raw = _strict_mapping(payload.get("author_provenance"), context="author_provenance")
    oracle_raw = _strict_mapping(payload.get("oracle_provenance"), context="oracle_provenance")
    resources_raw = _strict_mapping(payload.get("resources"), context="resources")
    network_raw = _strict_mapping(payload.get("network"), context="network")
    contamination_raw = _strict_mapping(payload.get("contamination"), context="contamination")
    validation_raw = _strict_mapping(
        payload.get("reference_validation"),
        context="reference_validation",
    )

    checker_ids_raw = contamination_raw.get("checker_ids")
    if not isinstance(checker_ids_raw, list) or any(
        not isinstance(item, str) for item in checker_ids_raw
    ):
        raise EnvironmentManifestError("contamination.checker_ids must be a string list")
    checker_ids = cast(tuple[str, ...], tuple(checker_ids_raw))

    return EnvironmentManifest(
        schema_version=_required_int(payload, "schema_version", context="environment manifest"),
        environment_id=_required_str(payload, "environment_id", context="environment manifest"),
        generation=_required_int(payload, "generation", context="environment manifest"),
        material_sha256=_required_str(
            payload,
            "material_sha256",
            context="environment manifest",
        ),
        task_category=_required_str(payload, "task_category", context="environment manifest"),
        specification_sha256=_required_str(
            payload,
            "specification_sha256",
            context="environment manifest",
        ),
        artifacts=tuple(artifacts),
        runtime=RuntimeIdentity(
            image=_required_str(runtime_raw, "image", context="runtime"),
            image_digest=_required_str(runtime_raw, "image_digest", context="runtime"),
            dependency_lock_sha256=_required_str(
                runtime_raw,
                "dependency_lock_sha256",
                context="runtime",
            ),
            python_version=_required_str(runtime_raw, "python_version", context="runtime"),
        ),
        author_provenance=ProvenanceIdentity(
            role=_enum_value(ProvenanceRole, author_raw, "role", context="author_provenance"),
            producer_type=_required_str(
                author_raw,
                "producer_type",
                context="author_provenance",
            ),
            producer_id=_required_str(author_raw, "producer_id", context="author_provenance"),
            revision=_required_str(author_raw, "revision", context="author_provenance"),
            config_sha256=_required_str(
                author_raw,
                "config_sha256",
                context="author_provenance",
            ),
            output_sha256=_required_str(
                author_raw,
                "output_sha256",
                context="author_provenance",
            ),
        ),
        oracle_provenance=ProvenanceIdentity(
            role=_enum_value(ProvenanceRole, oracle_raw, "role", context="oracle_provenance"),
            producer_type=_required_str(
                oracle_raw,
                "producer_type",
                context="oracle_provenance",
            ),
            producer_id=_required_str(oracle_raw, "producer_id", context="oracle_provenance"),
            revision=_required_str(oracle_raw, "revision", context="oracle_provenance"),
            config_sha256=_required_str(
                oracle_raw,
                "config_sha256",
                context="oracle_provenance",
            ),
            output_sha256=_required_str(
                oracle_raw,
                "output_sha256",
                context="oracle_provenance",
            ),
        ),
        resources=ResourcePolicy(
            wall_clock_seconds=_required_int(
                resources_raw,
                "wall_clock_seconds",
                context="resources",
            ),
            cpu_seconds=_required_int(resources_raw, "cpu_seconds", context="resources"),
            memory_bytes=_required_int(resources_raw, "memory_bytes", context="resources"),
            pids_limit=_required_int(resources_raw, "pids_limit", context="resources"),
            output_bytes=_required_int(resources_raw, "output_bytes", context="resources"),
            file_bytes=_required_int(resources_raw, "file_bytes", context="resources"),
            disk_bytes=_required_int(resources_raw, "disk_bytes", context="resources"),
        ),
        network=NetworkPolicy(
            mode=_enum_value(NetworkMode, network_raw, "mode", context="network"),
        ),
        contamination=ContaminationEvidence(
            status=_enum_value(
                ContaminationStatus,
                contamination_raw,
                "status",
                context="contamination",
            ),
            checker_ids=checker_ids,
            evidence_sha256=_optional_str(
                contamination_raw,
                "evidence_sha256",
                context="contamination",
            ),
        ),
        reference_validation=ReferenceValidationEvidence(
            status=_enum_value(
                EvidenceStatus,
                validation_raw,
                "status",
                context="reference_validation",
            ),
            validator_id=_optional_str(
                validation_raw,
                "validator_id",
                context="reference_validation",
            ),
            evidence_sha256=_optional_str(
                validation_raw,
                "evidence_sha256",
                context="reference_validation",
            ),
        ),
    )


def read_environment_manifest(path: Path) -> EnvironmentManifest:
    """Read and fully validate one environment manifest file."""

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EnvironmentManifestError(f"could not read environment manifest: {path}") from exc
    return environment_manifest_from_json(text)
