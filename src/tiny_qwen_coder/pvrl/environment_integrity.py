"""PVRL-G1 environment-integrity admission boundary.

PVRL-101 through PVRL-105 deliberately allow intermediate candidate states so
that authored environments can be validated incrementally.  This module is the
single fail-closed boundary that turns those independently frozen artifacts into
one G1 admission decision.  Admission requires the final manifest, exact sandbox
material, oracle-independence assessment, reference-first report, and clean
contamination report to agree on the same immutable environment.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from enum import StrEnum

from tiny_qwen_coder.pvrl.contamination import PVRLContaminationReport
from tiny_qwen_coder.pvrl.environment_manifest import (
    ContaminationStatus,
    EnvironmentManifest,
    EnvironmentManifestError,
    EvidenceStatus,
    environment_manifest_sha256,
)
from tiny_qwen_coder.pvrl.execution_sandbox import HardenedEnvironmentMaterial
from tiny_qwen_coder.pvrl.oracle_provenance import (
    OracleIndependenceGrade,
    OracleProvenanceAssessment,
)
from tiny_qwen_coder.pvrl.reference_validation import ReferenceValidationReport

_SCHEMA_VERSION = 1
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_CHECKER_IDS = (
    "contamination-evidence",
    "hardened-sandbox",
    "immutable-manifest",
    "oracle-provenance",
    "reference-first",
)


class EnvironmentIntegrityError(EnvironmentManifestError):
    """Raised when an environment cannot cross the PVRL-G1 admission boundary."""


class EnvironmentIntegrityStatus(StrEnum):
    """Terminal PVRL-G1 status represented by a frozen admission record."""

    PASSED = "passed"


@dataclass(frozen=True, slots=True)
class EnvironmentIntegrityEvidence:
    """Content-addressed proof that one environment satisfied all PVRL-G1 checks."""

    schema_version: int
    status: EnvironmentIntegrityStatus
    environment_id: str
    environment_material_sha256: str
    environment_manifest_sha256: str
    sandbox_material_sha256: str
    oracle_assessment_sha256: str
    oracle_independence_grade: OracleIndependenceGrade
    reference_validation_evidence_sha256: str
    contamination_evidence_sha256: str
    checker_ids: tuple[str, ...]
    evidence_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise EnvironmentIntegrityError(
                f"unsupported environment-integrity schema_version {self.schema_version}; "
                f"expected {_SCHEMA_VERSION}"
            )
        if self.status is not EnvironmentIntegrityStatus.PASSED:
            raise EnvironmentIntegrityError(
                "PVRL-G1 evidence may only represent a passed gate"
            )
        if not self.environment_id.strip():
            raise EnvironmentIntegrityError("environment_id must not be empty")
        for field_name, value in (
            ("environment_material_sha256", self.environment_material_sha256),
            ("environment_manifest_sha256", self.environment_manifest_sha256),
            ("sandbox_material_sha256", self.sandbox_material_sha256),
            ("oracle_assessment_sha256", self.oracle_assessment_sha256),
            (
                "reference_validation_evidence_sha256",
                self.reference_validation_evidence_sha256,
            ),
            ("contamination_evidence_sha256", self.contamination_evidence_sha256),
            ("evidence_sha256", self.evidence_sha256),
        ):
            _require_sha256(value, field_name=field_name)
        if self.checker_ids != _CHECKER_IDS:
            raise EnvironmentIntegrityError(
                "environment-integrity evidence must declare the complete frozen G1 checker set"
            )
        if self.evidence_sha256 != _evidence_sha256(self):
            raise EnvironmentIntegrityError(
                "environment-integrity evidence changed while retaining stale evidence_sha256"
            )


def _require_sha256(value: str, *, field_name: str) -> None:
    if not _SHA256_PATTERN.fullmatch(value):
        raise EnvironmentIntegrityError(
            f"{field_name} must be a lowercase SHA-256 digest"
        )


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sandbox_material_sha256(material: HardenedEnvironmentMaterial) -> str:
    return _sha256(
        [
            {
                "path": item.path,
                "sha256": hashlib.sha256(item.content).hexdigest(),
            }
            for item in material.artifacts
        ]
    )


def _evidence_payload_fields(
    *,
    schema_version: int,
    status: EnvironmentIntegrityStatus,
    environment_id: str,
    environment_material_sha256: str,
    environment_manifest_sha256: str,
    sandbox_material_sha256: str,
    oracle_assessment_sha256: str,
    oracle_independence_grade: OracleIndependenceGrade,
    reference_validation_evidence_sha256: str,
    contamination_evidence_sha256: str,
    checker_ids: tuple[str, ...],
) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "status": status,
        "environment_id": environment_id,
        "environment_material_sha256": environment_material_sha256,
        "environment_manifest_sha256": environment_manifest_sha256,
        "sandbox_material_sha256": sandbox_material_sha256,
        "oracle_assessment_sha256": oracle_assessment_sha256,
        "oracle_independence_grade": oracle_independence_grade,
        "reference_validation_evidence_sha256": reference_validation_evidence_sha256,
        "contamination_evidence_sha256": contamination_evidence_sha256,
        "checker_ids": checker_ids,
    }


def _evidence_payload(evidence: EnvironmentIntegrityEvidence) -> dict[str, object]:
    return _evidence_payload_fields(
        schema_version=evidence.schema_version,
        status=evidence.status,
        environment_id=evidence.environment_id,
        environment_material_sha256=evidence.environment_material_sha256,
        environment_manifest_sha256=evidence.environment_manifest_sha256,
        sandbox_material_sha256=evidence.sandbox_material_sha256,
        oracle_assessment_sha256=evidence.oracle_assessment_sha256,
        oracle_independence_grade=evidence.oracle_independence_grade,
        reference_validation_evidence_sha256=evidence.reference_validation_evidence_sha256,
        contamination_evidence_sha256=evidence.contamination_evidence_sha256,
        checker_ids=evidence.checker_ids,
    )


def _evidence_sha256(evidence: EnvironmentIntegrityEvidence) -> str:
    return _sha256(_evidence_payload(evidence))


def admit_environment_integrity(
    *,
    manifest: EnvironmentManifest,
    material: HardenedEnvironmentMaterial,
    oracle_assessment: OracleProvenanceAssessment,
    reference_report: ReferenceValidationReport,
    contamination_report: PVRLContaminationReport,
) -> EnvironmentIntegrityEvidence:
    """Require every PVRL-G1 subsystem to agree before admitting an environment."""

    if manifest.reference_validation.status is not EvidenceStatus.PASSED:
        raise EnvironmentIntegrityError(
            "PVRL-G1 requires successful reference-first validation evidence"
        )
    if manifest.contamination.status is not ContaminationStatus.CLEAN:
        raise EnvironmentIntegrityError("PVRL-G1 requires clean contamination evidence")
    if material.manifest != manifest:
        raise EnvironmentIntegrityError(
            "hardened sandbox material must be bound to the exact final environment manifest"
        )

    if reference_report.environment_id != manifest.environment_id:
        raise EnvironmentIntegrityError(
            "reference validation targets another environment_id"
        )
    if reference_report.environment_material_sha256 != manifest.material_sha256:
        raise EnvironmentIntegrityError(
            "reference validation targets stale environment material"
        )
    if reference_report.validator_id != manifest.reference_validation.validator_id:
        raise EnvironmentIntegrityError(
            "manifest reference validator does not match reference-validation report"
        )
    if (
        reference_report.evidence_sha256
        != manifest.reference_validation.evidence_sha256
    ):
        raise EnvironmentIntegrityError(
            "manifest reference-validation evidence does not match the validation report"
        )

    if contamination_report.environment_id != manifest.environment_id:
        raise EnvironmentIntegrityError(
            "contamination report targets another environment_id"
        )
    if contamination_report.environment_material_sha256 != manifest.material_sha256:
        raise EnvironmentIntegrityError(
            "contamination report targets stale environment material"
        )
    if contamination_report.status is not ContaminationStatus.CLEAN:
        raise EnvironmentIntegrityError("PVRL-G1 cannot admit contamination findings")
    if contamination_report.checker_ids != manifest.contamination.checker_ids:
        raise EnvironmentIntegrityError(
            "manifest contamination checker set does not match the contamination report"
        )
    if contamination_report.evidence_sha256 != manifest.contamination.evidence_sha256:
        raise EnvironmentIntegrityError(
            "manifest contamination evidence does not match the contamination report"
        )

    final_manifest_sha256 = environment_manifest_sha256(manifest)
    if oracle_assessment.environment_id != manifest.environment_id:
        raise EnvironmentIntegrityError(
            "oracle assessment targets another environment_id"
        )
    if oracle_assessment.environment_manifest_sha256 != final_manifest_sha256:
        raise EnvironmentIntegrityError(
            "oracle assessment must bind to the exact final admitted environment manifest"
        )
    if oracle_assessment.author_provenance != manifest.author_provenance:
        raise EnvironmentIntegrityError(
            "oracle assessment author provenance does not match environment manifest"
        )
    if oracle_assessment.oracle_provenance != manifest.oracle_provenance:
        raise EnvironmentIntegrityError(
            "oracle assessment oracle provenance does not match environment manifest"
        )

    fields = _evidence_payload_fields(
        schema_version=_SCHEMA_VERSION,
        status=EnvironmentIntegrityStatus.PASSED,
        environment_id=manifest.environment_id,
        environment_material_sha256=manifest.material_sha256,
        environment_manifest_sha256=final_manifest_sha256,
        sandbox_material_sha256=_sandbox_material_sha256(material),
        oracle_assessment_sha256=oracle_assessment.assessment_sha256,
        oracle_independence_grade=oracle_assessment.independence_grade,
        reference_validation_evidence_sha256=reference_report.evidence_sha256,
        contamination_evidence_sha256=contamination_report.evidence_sha256,
        checker_ids=_CHECKER_IDS,
    )
    return EnvironmentIntegrityEvidence(
        schema_version=_SCHEMA_VERSION,
        status=EnvironmentIntegrityStatus.PASSED,
        environment_id=manifest.environment_id,
        environment_material_sha256=manifest.material_sha256,
        environment_manifest_sha256=final_manifest_sha256,
        sandbox_material_sha256=_sandbox_material_sha256(material),
        oracle_assessment_sha256=oracle_assessment.assessment_sha256,
        oracle_independence_grade=oracle_assessment.independence_grade,
        reference_validation_evidence_sha256=reference_report.evidence_sha256,
        contamination_evidence_sha256=contamination_report.evidence_sha256,
        checker_ids=_CHECKER_IDS,
        evidence_sha256=_sha256(fields),
    )


def environment_integrity_evidence_json(evidence: EnvironmentIntegrityEvidence) -> str:
    """Serialize PVRL-G1 admission evidence deterministically."""

    return json.dumps(asdict(evidence), indent=2, sort_keys=True) + "\n"


__all__ = [
    "EnvironmentIntegrityError",
    "EnvironmentIntegrityEvidence",
    "EnvironmentIntegrityStatus",
    "admit_environment_integrity",
    "environment_integrity_evidence_json",
]
