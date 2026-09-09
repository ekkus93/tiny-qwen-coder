"""Reference-first validation state machine and evidence for PVRL environments.

PVRL-103 owns validation semantics only.  The hardened sandbox implementation
is deliberately deferred to PVRL-104 and plugs in through the protocols below.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from typing import Protocol

from tiny_qwen_coder.pvrl.environment_manifest import (
    EnvironmentManifest,
    EnvironmentManifestError,
    EvidenceStatus,
    ReferenceValidationEvidence,
    environment_manifest_sha256,
)

_SCHEMA_VERSION = 1
_VALIDATOR_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ReferenceValidationError(EnvironmentManifestError):
    """Base class for invalid reference-validation material or protocol use."""


class ReferenceValidationRejectedError(ReferenceValidationError):
    """Raised when task/reference behavior fails the validation contract."""


class ValidationPhase(StrEnum):
    """Ordered execution phases inside one fresh validation sandbox."""

    INITIAL_HIDDEN = "initial_hidden"
    REFERENCE_HIDDEN = "reference_hidden"
    PRESERVATION_PUBLIC = "preservation_public"


class ValidationTimingClass(StrEnum):
    """Stable timing/resource class; exact wall-clock noise is not provenance."""

    WITHIN_BUDGET = "within_budget"
    TIMEOUT = "timeout"
    RESOURCE_LIMIT = "resource_limit"
    INFRASTRUCTURE = "infrastructure"


class ValidationExecutionClass(StrEnum):
    """Outcome taxonomy separating normal failures from invalid infrastructure."""

    PASSED = "passed"
    BEHAVIORAL_FAILURE = "candidate_behavioral_failure"
    PARSE_COMPILE_FAILURE = "candidate_parse_compile_failure"
    TIMEOUT_RESOURCE_FAILURE = "candidate_timeout_resource_failure"
    GRADER_HARNESS_FAILURE = "grader_harness_failure"
    ENVIRONMENT_CONSTRUCTION_FAILURE = "environment_construction_failure"
    DEPENDENCY_IMAGE_UNAVAILABLE = "unavailable_dependency_image"
    ENVIRONMENT_IDENTITY_MISMATCH = "corrupted_identity_mismatched_environment"

    @property
    def is_infrastructure(self) -> bool:
        """Return whether this outcome invalidates the attempt rather than grading it."""

        return self in {
            ValidationExecutionClass.GRADER_HARNESS_FAILURE,
            ValidationExecutionClass.ENVIRONMENT_CONSTRUCTION_FAILURE,
            ValidationExecutionClass.DEPENDENCY_IMAGE_UNAVAILABLE,
            ValidationExecutionClass.ENVIRONMENT_IDENTITY_MISMATCH,
        }


_INITIAL_FAILURE_CLASSES = {
    ValidationExecutionClass.BEHAVIORAL_FAILURE,
    ValidationExecutionClass.PARSE_COMPILE_FAILURE,
}


class ReferenceValidationInfrastructureError(RuntimeError):
    """Infrastructure failure that must never become a candidate/reference failure."""

    def __init__(
        self,
        kind: ValidationExecutionClass,
        message: str,
        *,
        phase: ValidationPhase | None = None,
    ) -> None:
        if not kind.is_infrastructure:
            raise ValueError("infrastructure error requires an infrastructure execution class")
        self.kind = kind
        self.phase = phase
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class ValidationExecutionEvidence:
    """Content-addressed bounded output evidence for one grader execution."""

    phase: ValidationPhase
    classification: ValidationExecutionClass
    exit_code: int | None
    timing_class: ValidationTimingClass
    stdout_sha256: str
    stderr_sha256: str
    stdout_truncated: bool
    stderr_truncated: bool
    result_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.stdout_sha256, field_name="stdout_sha256")
        _require_sha256(self.stderr_sha256, field_name="stderr_sha256")
        _require_sha256(self.result_sha256, field_name="result_sha256")
        if self.classification is ValidationExecutionClass.PASSED:
            if self.exit_code != 0 or self.timing_class is not ValidationTimingClass.WITHIN_BUDGET:
                raise ReferenceValidationError(
                    "passed validation execution must exit zero within the timing budget"
                )
        elif self.classification in _INITIAL_FAILURE_CLASSES:
            if self.exit_code is None or self.exit_code == 0:
                raise ReferenceValidationError(
                    "completed behavioral/parse failure must have a non-zero exit code"
                )
            if self.timing_class is not ValidationTimingClass.WITHIN_BUDGET:
                raise ReferenceValidationError(
                    "behavioral/parse failure must complete within the timing budget"
                )
        elif self.classification is ValidationExecutionClass.TIMEOUT_RESOURCE_FAILURE:
            if self.timing_class not in {
                ValidationTimingClass.TIMEOUT,
                ValidationTimingClass.RESOURCE_LIMIT,
            }:
                raise ReferenceValidationError(
                    "timeout/resource failure requires timeout or resource_limit timing class"
                )
        elif (
            self.classification.is_infrastructure
            and self.timing_class is not ValidationTimingClass.INFRASTRUCTURE
        ):
            raise ReferenceValidationError(
                "infrastructure failure requires infrastructure timing class"
            )
        if self.result_sha256 != _execution_sha256(self):
            raise ReferenceValidationError("execution changed while retaining stale result_sha256")


@dataclass(frozen=True, slots=True)
class ReferenceValidationRunEvidence:
    """One complete fail-to-pass sequence from one disposable sandbox."""

    repetition: int
    initial_hidden: ValidationExecutionEvidence
    reference_workspace_sha256: str
    reference_hidden: ValidationExecutionEvidence
    preservation_public: ValidationExecutionEvidence

    def __post_init__(self) -> None:
        if self.repetition not in {1, 2}:
            raise ReferenceValidationError("reference validation repetition must be 1 or 2")
        if self.initial_hidden.phase is not ValidationPhase.INITIAL_HIDDEN:
            raise ReferenceValidationError("initial_hidden evidence has wrong phase")
        if self.reference_hidden.phase is not ValidationPhase.REFERENCE_HIDDEN:
            raise ReferenceValidationError("reference_hidden evidence has wrong phase")
        if self.preservation_public.phase is not ValidationPhase.PRESERVATION_PUBLIC:
            raise ReferenceValidationError("preservation_public evidence has wrong phase")
        _require_sha256(
            self.reference_workspace_sha256,
            field_name="reference_workspace_sha256",
        )


@dataclass(frozen=True, slots=True)
class ReferenceValidationReport:
    """Immutable evidence for a successful reproducible reference-first validation."""

    schema_version: int
    validator_id: str
    environment_id: str
    environment_material_sha256: str
    source_manifest_sha256: str
    expected_initial_failure: ValidationExecutionClass
    runs: tuple[ReferenceValidationRunEvidence, ...]
    evidence_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise ReferenceValidationError(
                f"unsupported reference validation schema_version {self.schema_version}; "
                f"expected {_SCHEMA_VERSION}"
            )
        _validate_validator_id(self.validator_id)
        if not self.environment_id.strip():
            raise ReferenceValidationError("environment_id must not be empty")
        _require_sha256(
            self.environment_material_sha256,
            field_name="environment_material_sha256",
        )
        _require_sha256(self.source_manifest_sha256, field_name="source_manifest_sha256")
        if self.expected_initial_failure not in _INITIAL_FAILURE_CLASSES:
            raise ReferenceValidationError(
                "expected_initial_failure must be behavioral or parse/compile failure"
            )
        if len(self.runs) != 2 or tuple(run.repetition for run in self.runs) != (1, 2):
            raise ReferenceValidationError(
                "successful validation requires two ordered sandbox runs"
            )
        for run in self.runs:
            if run.initial_hidden.classification is not self.expected_initial_failure:
                raise ReferenceValidationError(
                    "initial workspace did not reproduce expected failure"
                )
            if run.reference_hidden.classification is not ValidationExecutionClass.PASSED:
                raise ReferenceValidationError("reference hidden validation did not pass")
            if run.preservation_public.classification is not ValidationExecutionClass.PASSED:
                raise ReferenceValidationError(
                    "reference preservation/public validation did not pass"
                )
        if _reproducibility_key(self.runs[0]) != _reproducibility_key(self.runs[1]):
            raise ReferenceValidationError(
                "reference validation is not reproducible across sandboxes"
            )
        _require_sha256(self.evidence_sha256, field_name="evidence_sha256")
        if self.evidence_sha256 != _report_sha256(self):
            raise ReferenceValidationError("report changed while retaining stale evidence_sha256")


class ReferenceValidationSandbox(Protocol):
    """Trusted operations required from the future hardened sandbox."""

    def run_hidden(self, phase: ValidationPhase) -> ValidationExecutionEvidence:
        """Execute the hidden contract for an initial or reference phase."""
        ...

    def apply_reference(self) -> str:
        """Apply trusted reference material and return the resulting workspace SHA-256."""
        ...

    def run_preservation(self) -> ValidationExecutionEvidence:
        """Execute public/preservation tests after reference application."""
        ...

    def close(self) -> None:
        """Destroy the sandbox and surface cleanup failures."""
        ...


class ReferenceValidationSandboxFactory(Protocol):
    """Create a genuinely fresh sandbox for each validation repetition."""

    def create(
        self,
        manifest: EnvironmentManifest,
        *,
        repetition: int,
    ) -> ReferenceValidationSandbox:
        """Construct one sandbox for the exact immutable environment."""
        ...


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode(
        "ascii"
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: str, *, field_name: str) -> None:
    if not _SHA256_PATTERN.fullmatch(value):
        raise ReferenceValidationError(f"{field_name} must be a lowercase SHA-256 digest")


def _validate_validator_id(value: str) -> None:
    if not _VALIDATOR_ID_PATTERN.fullmatch(value):
        raise ReferenceValidationError(
            "validator_id must be a stable lowercase component identifier"
        )


def _execution_payload(evidence: ValidationExecutionEvidence) -> dict[str, object]:
    return {
        "phase": evidence.phase,
        "classification": evidence.classification,
        "exit_code": evidence.exit_code,
        "timing_class": evidence.timing_class,
        "stdout_sha256": evidence.stdout_sha256,
        "stderr_sha256": evidence.stderr_sha256,
        "stdout_truncated": evidence.stdout_truncated,
        "stderr_truncated": evidence.stderr_truncated,
    }


def _execution_sha256(evidence: ValidationExecutionEvidence) -> str:
    return _sha256(_execution_payload(evidence))


def create_validation_execution_evidence(
    *,
    phase: ValidationPhase,
    classification: ValidationExecutionClass,
    exit_code: int | None,
    timing_class: ValidationTimingClass,
    stdout: str,
    stderr: str,
    stdout_truncated: bool = False,
    stderr_truncated: bool = False,
) -> ValidationExecutionEvidence:
    """Hash bounded stdout/stderr and freeze exact exit/result evidence."""

    stdout_sha256 = hashlib.sha256(stdout.encode("utf-8")).hexdigest()
    stderr_sha256 = hashlib.sha256(stderr.encode("utf-8")).hexdigest()
    payload: dict[str, object] = {
        "phase": phase,
        "classification": classification,
        "exit_code": exit_code,
        "timing_class": timing_class,
        "stdout_sha256": stdout_sha256,
        "stderr_sha256": stderr_sha256,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
    }
    return ValidationExecutionEvidence(
        phase=phase,
        classification=classification,
        exit_code=exit_code,
        timing_class=timing_class,
        stdout_sha256=stdout_sha256,
        stderr_sha256=stderr_sha256,
        stdout_truncated=stdout_truncated,
        stderr_truncated=stderr_truncated,
        result_sha256=_sha256(payload),
    )


def _reproducibility_key(run: ReferenceValidationRunEvidence) -> tuple[str, str, str, str]:
    return (
        run.initial_hidden.result_sha256,
        run.reference_workspace_sha256,
        run.reference_hidden.result_sha256,
        run.preservation_public.result_sha256,
    )


def _report_payload(report: ReferenceValidationReport) -> dict[str, object]:
    payload = asdict(report)
    payload.pop("evidence_sha256")
    return payload


def _report_sha256(report: ReferenceValidationReport) -> str:
    return _sha256(_report_payload(report))


def _raise_if_infrastructure(evidence: ValidationExecutionEvidence) -> None:
    if evidence.classification.is_infrastructure:
        raise ReferenceValidationInfrastructureError(
            evidence.classification,
            f"reference validation infrastructure failed during {evidence.phase.value}",
            phase=evidence.phase,
        )


def _call_hidden(
    sandbox: ReferenceValidationSandbox,
    phase: ValidationPhase,
) -> ValidationExecutionEvidence:
    try:
        evidence = sandbox.run_hidden(phase)
    except ReferenceValidationInfrastructureError:
        raise
    except Exception as exc:
        raise ReferenceValidationInfrastructureError(
            ValidationExecutionClass.GRADER_HARNESS_FAILURE,
            f"hidden grader failed during {phase.value}",
            phase=phase,
        ) from exc
    if evidence.phase is not phase:
        raise ReferenceValidationInfrastructureError(
            ValidationExecutionClass.GRADER_HARNESS_FAILURE,
            f"hidden grader returned evidence for wrong phase during {phase.value}",
            phase=phase,
        )
    _raise_if_infrastructure(evidence)
    return evidence


def _run_one_sandbox(
    manifest: EnvironmentManifest,
    factory: ReferenceValidationSandboxFactory,
    *,
    repetition: int,
    expected_initial_failure: ValidationExecutionClass,
) -> ReferenceValidationRunEvidence:
    try:
        sandbox = factory.create(manifest, repetition=repetition)
    except ReferenceValidationInfrastructureError:
        raise
    except Exception as exc:
        raise ReferenceValidationInfrastructureError(
            ValidationExecutionClass.ENVIRONMENT_CONSTRUCTION_FAILURE,
            f"could not construct fresh validation sandbox {repetition}",
        ) from exc

    try:
        initial = _call_hidden(sandbox, ValidationPhase.INITIAL_HIDDEN)
        if initial.classification is not expected_initial_failure:
            raise ReferenceValidationRejectedError(
                "initial workspace did not exhibit the declared fail/incomplete condition"
            )

        try:
            workspace_sha256 = sandbox.apply_reference()
        except ReferenceValidationInfrastructureError:
            raise
        except Exception as exc:
            raise ReferenceValidationInfrastructureError(
                ValidationExecutionClass.GRADER_HARNESS_FAILURE,
                "controlled reference application failed",
            ) from exc
        try:
            _require_sha256(workspace_sha256, field_name="reference_workspace_sha256")
        except ReferenceValidationError as exc:
            raise ReferenceValidationInfrastructureError(
                ValidationExecutionClass.ENVIRONMENT_IDENTITY_MISMATCH,
                "reference application returned invalid workspace identity",
            ) from exc

        reference_hidden = _call_hidden(sandbox, ValidationPhase.REFERENCE_HIDDEN)
        if reference_hidden.classification is not ValidationExecutionClass.PASSED:
            raise ReferenceValidationRejectedError(
                "reference implementation did not pass the hidden behavioral contract"
            )

        try:
            preservation = sandbox.run_preservation()
        except ReferenceValidationInfrastructureError:
            raise
        except Exception as exc:
            raise ReferenceValidationInfrastructureError(
                ValidationExecutionClass.GRADER_HARNESS_FAILURE,
                "preservation/public grader failed after reference application",
                phase=ValidationPhase.PRESERVATION_PUBLIC,
            ) from exc
        if preservation.phase is not ValidationPhase.PRESERVATION_PUBLIC:
            raise ReferenceValidationInfrastructureError(
                ValidationExecutionClass.GRADER_HARNESS_FAILURE,
                "preservation grader returned evidence for wrong phase",
                phase=ValidationPhase.PRESERVATION_PUBLIC,
            )
        _raise_if_infrastructure(preservation)
        if preservation.classification is not ValidationExecutionClass.PASSED:
            raise ReferenceValidationRejectedError(
                "reference implementation regressed preservation/public checks"
            )

        return ReferenceValidationRunEvidence(
            repetition=repetition,
            initial_hidden=initial,
            reference_workspace_sha256=workspace_sha256,
            reference_hidden=reference_hidden,
            preservation_public=preservation,
        )
    finally:
        try:
            sandbox.close()
        except ReferenceValidationInfrastructureError:
            raise
        except Exception as exc:
            raise ReferenceValidationInfrastructureError(
                ValidationExecutionClass.GRADER_HARNESS_FAILURE,
                "reference-validation sandbox cleanup failed",
            ) from exc


def run_reference_first_validation(
    manifest: EnvironmentManifest,
    factory: ReferenceValidationSandboxFactory,
    *,
    validator_id: str,
    expected_initial_failure: ValidationExecutionClass,
) -> ReferenceValidationReport:
    """Prove the fail-to-pass contract twice in independent fresh sandboxes."""

    _validate_validator_id(validator_id)
    if manifest.reference_validation.status is not EvidenceStatus.NOT_RUN:
        raise ReferenceValidationError("reference validation cannot overwrite completed evidence")
    if expected_initial_failure not in _INITIAL_FAILURE_CLASSES:
        raise ReferenceValidationError(
            "expected_initial_failure must be behavioral or parse/compile failure"
        )

    runs = tuple(
        _run_one_sandbox(
            manifest,
            factory,
            repetition=repetition,
            expected_initial_failure=expected_initial_failure,
        )
        for repetition in (1, 2)
    )
    if _reproducibility_key(runs[0]) != _reproducibility_key(runs[1]):
        raise ReferenceValidationRejectedError(
            "reference validation was not reproducible in the fresh sandbox"
        )

    source_manifest_sha256 = environment_manifest_sha256(manifest)
    payload: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "validator_id": validator_id,
        "environment_id": manifest.environment_id,
        "environment_material_sha256": manifest.material_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "expected_initial_failure": expected_initial_failure,
        "runs": [asdict(run) for run in runs],
    }
    return ReferenceValidationReport(
        schema_version=_SCHEMA_VERSION,
        validator_id=validator_id,
        environment_id=manifest.environment_id,
        environment_material_sha256=manifest.material_sha256,
        source_manifest_sha256=source_manifest_sha256,
        expected_initial_failure=expected_initial_failure,
        runs=runs,
        evidence_sha256=_sha256(payload),
    )


def attach_reference_validation(
    manifest: EnvironmentManifest,
    report: ReferenceValidationReport,
) -> EnvironmentManifest:
    """Attach successful evidence without changing immutable task material identity."""

    if manifest.reference_validation.status is not EvidenceStatus.NOT_RUN:
        raise ReferenceValidationError("reference validation cannot overwrite completed evidence")
    if manifest.environment_id != report.environment_id:
        raise ReferenceValidationError("validation report targets another environment_id")
    if manifest.material_sha256 != report.environment_material_sha256:
        raise ReferenceValidationError("validation report targets different environment material")
    if environment_manifest_sha256(manifest) != report.source_manifest_sha256:
        raise ReferenceValidationError(
            "environment evidence changed after validation and before report attachment"
        )
    return replace(
        manifest,
        reference_validation=ReferenceValidationEvidence(
            status=EvidenceStatus.PASSED,
            validator_id=report.validator_id,
            evidence_sha256=report.evidence_sha256,
        ),
    )


def reference_validation_report_json(report: ReferenceValidationReport) -> str:
    """Serialize successful reference-validation evidence deterministically."""

    return json.dumps(asdict(report), indent=2, sort_keys=True) + "\n"
