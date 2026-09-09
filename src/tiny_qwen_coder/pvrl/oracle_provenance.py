"""Oracle/reference provenance and independence grading for PVRL environments."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import TypeVar

from tiny_qwen_coder.pvrl.environment_manifest import (
    EnvironmentManifest,
    EnvironmentManifestError,
    ProvenanceIdentity,
    ProvenanceRole,
    environment_manifest_sha256,
)

_SCHEMA_VERSION = 1
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_COMPONENT_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]*$")
_EnumT = TypeVar("_EnumT", bound=StrEnum)


class OracleProvenanceError(EnvironmentManifestError):
    """Raised when PVRL oracle/reference provenance is incomplete or inconsistent."""


class OracleIndependenceGrade(StrEnum):
    """Declared independence class for the executable correctness contract."""

    EXISTING_HUMAN_SOURCE_TESTS = "existing_human_source_tests"
    DETERMINISTIC_PROGRAMMATIC_CONTRACT = "deterministic_programmatic_contract"
    REPOSITORY_NATIVE_TESTS = "repository_native_tests"
    INDEPENDENTLY_GENERATED_CONTRACT_REFERENCE = "independently_generated_contract_reference"
    SAME_FAMILY_GENERATED_CONTRACT_REFERENCE = "same_family_generated_contract_reference"


@dataclass(frozen=True, slots=True)
class ReferenceProvenanceIdentity:
    """Exact producer identity for the reference implementation or patch."""

    producer_type: str
    producer_id: str
    revision: str
    config_sha256: str
    output_sha256: str
    producer_family: str | None = None

    def __post_init__(self) -> None:
        _validate_producer_identity(
            producer_type=self.producer_type,
            producer_id=self.producer_id,
            revision=self.revision,
            config_sha256=self.config_sha256,
            output_sha256=self.output_sha256,
            producer_family=self.producer_family,
            context="reference_provenance",
        )


@dataclass(frozen=True, slots=True)
class OracleProvenanceAssessment:
    """Content-addressed trust envelope bound to one immutable environment manifest."""

    schema_version: int
    environment_id: str
    environment_manifest_sha256: str
    author_provenance: ProvenanceIdentity
    oracle_provenance: ProvenanceIdentity
    reference_provenance: ReferenceProvenanceIdentity
    author_family: str | None
    oracle_family: str | None
    independence_grade: OracleIndependenceGrade
    correlation_caveat: str | None
    assessment_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise OracleProvenanceError(
                f"unsupported oracle provenance schema_version {self.schema_version}; "
                f"expected {_SCHEMA_VERSION}"
            )
        if not self.environment_id.strip():
            raise OracleProvenanceError("environment_id must not be empty")
        _require_sha256(
            self.environment_manifest_sha256,
            field_name="environment_manifest_sha256",
        )
        if self.author_provenance.role is not ProvenanceRole.AUTHOR:
            raise OracleProvenanceError("author_provenance must have author role")
        if self.oracle_provenance.role is not ProvenanceRole.ORACLE:
            raise OracleProvenanceError("oracle_provenance must have oracle role")
        _validate_family_binding(
            self.author_provenance,
            self.author_family,
            context="author_provenance",
        )
        _validate_family_binding(
            self.oracle_provenance,
            self.oracle_family,
            context="oracle_provenance",
        )
        _validate_independence_claim(
            author=self.author_provenance,
            oracle=self.oracle_provenance,
            author_family=self.author_family,
            oracle_family=self.oracle_family,
            grade=self.independence_grade,
            correlation_caveat=self.correlation_caveat,
        )
        expected = _assessment_sha256(self)
        _require_sha256(self.assessment_sha256, field_name="assessment_sha256")
        if self.assessment_sha256 != expected:
            raise OracleProvenanceError(
                "oracle provenance material changed while retaining a stale assessment_sha256"
            )


def _require_sha256(value: str, *, field_name: str) -> None:
    if not _SHA256_PATTERN.fullmatch(value):
        raise OracleProvenanceError(f"{field_name} must be a lowercase SHA-256 digest")


def _validate_component_id(value: str, *, field_name: str) -> None:
    if not _COMPONENT_ID_PATTERN.fullmatch(value):
        raise OracleProvenanceError(f"{field_name} must be a stable lowercase component identifier")


def _validate_producer_identity(
    *,
    producer_type: str,
    producer_id: str,
    revision: str,
    config_sha256: str,
    output_sha256: str,
    producer_family: str | None,
    context: str,
) -> None:
    _validate_component_id(producer_type, field_name=f"{context}.producer_type")
    if not producer_id.strip():
        raise OracleProvenanceError(f"{context}.producer_id must not be empty")
    if not revision.strip():
        raise OracleProvenanceError(f"{context}.revision must not be empty")
    _require_sha256(config_sha256, field_name=f"{context}.config_sha256")
    _require_sha256(output_sha256, field_name=f"{context}.output_sha256")
    if producer_type == "model":
        if producer_family is None:
            raise OracleProvenanceError(
                f"{context}.producer_family is required for model-produced material"
            )
        _validate_component_id(
            producer_family,
            field_name=f"{context}.producer_family",
        )
    elif producer_family is not None:
        raise OracleProvenanceError(
            f"{context}.producer_family is only valid for model-produced material"
        )


def _validate_family_binding(
    provenance: ProvenanceIdentity,
    family: str | None,
    *,
    context: str,
) -> None:
    if provenance.producer_type == "model":
        if family is None:
            raise OracleProvenanceError(f"{context} family is required for model-produced material")
        _validate_component_id(family, field_name=f"{context}.family")
    elif family is not None:
        raise OracleProvenanceError(f"{context} family is only valid for model-produced material")


def _validate_independence_claim(
    *,
    author: ProvenanceIdentity,
    oracle: ProvenanceIdentity,
    author_family: str | None,
    oracle_family: str | None,
    grade: OracleIndependenceGrade,
    correlation_caveat: str | None,
) -> None:
    if correlation_caveat is not None and not correlation_caveat.strip():
        raise OracleProvenanceError("correlation_caveat must not be blank when provided")

    if grade is OracleIndependenceGrade.EXISTING_HUMAN_SOURCE_TESTS:
        if oracle.producer_type not in {"human", "source"}:
            raise OracleProvenanceError(
                "existing_human_source_tests requires human/source oracle provenance"
            )
    elif grade is OracleIndependenceGrade.DETERMINISTIC_PROGRAMMATIC_CONTRACT:
        if oracle.producer_type != "programmatic":
            raise OracleProvenanceError(
                "deterministic_programmatic_contract requires programmatic oracle provenance"
            )
    elif grade is OracleIndependenceGrade.REPOSITORY_NATIVE_TESTS:
        if oracle.producer_type != "repository":
            raise OracleProvenanceError(
                "repository_native_tests requires repository oracle provenance"
            )
    elif grade is OracleIndependenceGrade.INDEPENDENTLY_GENERATED_CONTRACT_REFERENCE:
        if oracle.producer_type != "model":
            raise OracleProvenanceError(
                "independently_generated_contract_reference requires model oracle provenance"
            )
        if (
            author.producer_type == "model"
            and author_family is not None
            and oracle_family == author_family
        ):
            raise OracleProvenanceError(
                "same-family generated oracle cannot be labeled independently generated"
            )
    elif grade is OracleIndependenceGrade.SAME_FAMILY_GENERATED_CONTRACT_REFERENCE:
        if author.producer_type != "model" or oracle.producer_type != "model":
            raise OracleProvenanceError(
                "same_family_generated_contract_reference requires model author and oracle"
            )
        if author_family is None or oracle_family is None or author_family != oracle_family:
            raise OracleProvenanceError(
                "same_family_generated_contract_reference requires matching model families"
            )
        if correlation_caveat is None:
            raise OracleProvenanceError(
                "same-family generated oracle requires an explicit correlation_caveat"
            )


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _assessment_payload(
    *,
    schema_version: int,
    environment_id: str,
    environment_manifest_sha256: str,
    author_provenance: ProvenanceIdentity,
    oracle_provenance: ProvenanceIdentity,
    reference_provenance: ReferenceProvenanceIdentity,
    author_family: str | None,
    oracle_family: str | None,
    independence_grade: OracleIndependenceGrade,
    correlation_caveat: str | None,
) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "environment_id": environment_id,
        "environment_manifest_sha256": environment_manifest_sha256,
        "author_provenance": asdict(author_provenance),
        "oracle_provenance": asdict(oracle_provenance),
        "reference_provenance": asdict(reference_provenance),
        "author_family": author_family,
        "oracle_family": oracle_family,
        "independence_grade": independence_grade,
        "correlation_caveat": correlation_caveat,
    }


def _assessment_sha256(assessment: OracleProvenanceAssessment) -> str:
    return _sha256(
        _assessment_payload(
            schema_version=assessment.schema_version,
            environment_id=assessment.environment_id,
            environment_manifest_sha256=assessment.environment_manifest_sha256,
            author_provenance=assessment.author_provenance,
            oracle_provenance=assessment.oracle_provenance,
            reference_provenance=assessment.reference_provenance,
            author_family=assessment.author_family,
            oracle_family=assessment.oracle_family,
            independence_grade=assessment.independence_grade,
            correlation_caveat=assessment.correlation_caveat,
        )
    )


def create_oracle_provenance_assessment(
    *,
    manifest: EnvironmentManifest,
    reference_provenance: ReferenceProvenanceIdentity,
    author_family: str | None,
    oracle_family: str | None,
    independence_grade: OracleIndependenceGrade,
    correlation_caveat: str | None = None,
) -> OracleProvenanceAssessment:
    """Create one immutable provenance assessment bound to an environment manifest."""

    payload = _assessment_payload(
        schema_version=_SCHEMA_VERSION,
        environment_id=manifest.environment_id,
        environment_manifest_sha256=environment_manifest_sha256(manifest),
        author_provenance=manifest.author_provenance,
        oracle_provenance=manifest.oracle_provenance,
        reference_provenance=reference_provenance,
        author_family=author_family,
        oracle_family=oracle_family,
        independence_grade=independence_grade,
        correlation_caveat=correlation_caveat,
    )
    return OracleProvenanceAssessment(
        schema_version=_SCHEMA_VERSION,
        environment_id=manifest.environment_id,
        environment_manifest_sha256=environment_manifest_sha256(manifest),
        author_provenance=manifest.author_provenance,
        oracle_provenance=manifest.oracle_provenance,
        reference_provenance=reference_provenance,
        author_family=author_family,
        oracle_family=oracle_family,
        independence_grade=independence_grade,
        correlation_caveat=correlation_caveat,
        assessment_sha256=_sha256(payload),
    )


def oracle_provenance_assessment_json(assessment: OracleProvenanceAssessment) -> str:
    """Serialize an oracle provenance assessment deterministically."""

    return json.dumps(asdict(assessment), indent=2, sort_keys=True) + "\n"


def _strict_mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise OracleProvenanceError(f"{context} must be a JSON object")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise OracleProvenanceError(f"{context} keys must be strings")
        result[key] = item
    return result


def _required_str(mapping: Mapping[str, object], key: str, *, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise OracleProvenanceError(f"{context}.{key} must be a non-empty string")
    return value


def _required_int(mapping: Mapping[str, object], key: str, *, context: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise OracleProvenanceError(f"{context}.{key} must be an integer")
    return value


def _optional_str(mapping: Mapping[str, object], key: str, *, context: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise OracleProvenanceError(f"{context}.{key} must be a non-empty string or null")
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
        raise OracleProvenanceError(f"{context}.{key} has unsupported value {raw!r}") from exc


def _parse_provenance(mapping: Mapping[str, object], *, context: str) -> ProvenanceIdentity:
    return ProvenanceIdentity(
        role=_enum_value(ProvenanceRole, mapping, "role", context=context),
        producer_type=_required_str(mapping, "producer_type", context=context),
        producer_id=_required_str(mapping, "producer_id", context=context),
        revision=_required_str(mapping, "revision", context=context),
        config_sha256=_required_str(mapping, "config_sha256", context=context),
        output_sha256=_required_str(mapping, "output_sha256", context=context),
    )


def oracle_provenance_assessment_from_json(text: str) -> OracleProvenanceAssessment:
    """Parse and fully validate one oracle provenance assessment."""

    try:
        raw: object = json.loads(text)
    except json.JSONDecodeError as exc:
        raise OracleProvenanceError("oracle provenance assessment is not valid JSON") from exc
    payload = _strict_mapping(raw, context="oracle provenance assessment")
    author_raw = _strict_mapping(payload.get("author_provenance"), context="author_provenance")
    oracle_raw = _strict_mapping(payload.get("oracle_provenance"), context="oracle_provenance")
    reference_raw = _strict_mapping(
        payload.get("reference_provenance"),
        context="reference_provenance",
    )

    return OracleProvenanceAssessment(
        schema_version=_required_int(
            payload,
            "schema_version",
            context="oracle provenance assessment",
        ),
        environment_id=_required_str(
            payload,
            "environment_id",
            context="oracle provenance assessment",
        ),
        environment_manifest_sha256=_required_str(
            payload,
            "environment_manifest_sha256",
            context="oracle provenance assessment",
        ),
        author_provenance=_parse_provenance(author_raw, context="author_provenance"),
        oracle_provenance=_parse_provenance(oracle_raw, context="oracle_provenance"),
        reference_provenance=ReferenceProvenanceIdentity(
            producer_type=_required_str(
                reference_raw,
                "producer_type",
                context="reference_provenance",
            ),
            producer_id=_required_str(
                reference_raw,
                "producer_id",
                context="reference_provenance",
            ),
            revision=_required_str(
                reference_raw,
                "revision",
                context="reference_provenance",
            ),
            config_sha256=_required_str(
                reference_raw,
                "config_sha256",
                context="reference_provenance",
            ),
            output_sha256=_required_str(
                reference_raw,
                "output_sha256",
                context="reference_provenance",
            ),
            producer_family=_optional_str(
                reference_raw,
                "producer_family",
                context="reference_provenance",
            ),
        ),
        author_family=_optional_str(
            payload,
            "author_family",
            context="oracle provenance assessment",
        ),
        oracle_family=_optional_str(
            payload,
            "oracle_family",
            context="oracle provenance assessment",
        ),
        independence_grade=_enum_value(
            OracleIndependenceGrade,
            payload,
            "independence_grade",
            context="oracle provenance assessment",
        ),
        correlation_caveat=_optional_str(
            payload,
            "correlation_caveat",
            context="oracle provenance assessment",
        ),
        assessment_sha256=_required_str(
            payload,
            "assessment_sha256",
            context="oracle provenance assessment",
        ),
    )
