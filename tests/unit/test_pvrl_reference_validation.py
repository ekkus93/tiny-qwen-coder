from __future__ import annotations

from dataclasses import replace

import pytest

from tiny_qwen_coder.pvrl.environment_manifest import (
    ArtifactKind,
    ArtifactVisibility,
    ContaminationEvidence,
    ContaminationStatus,
    EnvironmentArtifact,
    EnvironmentManifest,
    EvidenceStatus,
    NetworkMode,
    NetworkPolicy,
    ProvenanceIdentity,
    ProvenanceRole,
    ReferenceValidationEvidence,
    ResourcePolicy,
    RuntimeIdentity,
    create_environment_manifest,
)
from tiny_qwen_coder.pvrl.reference_validation import (
    ReferenceValidationError,
    ReferenceValidationInfrastructureError,
    ReferenceValidationRejectedError,
    ReferenceValidationReport,
    ValidationExecutionClass,
    ValidationExecutionEvidence,
    ValidationPhase,
    ValidationTimingClass,
    attach_reference_validation,
    create_validation_execution_evidence,
    reference_validation_report_json,
    run_reference_first_validation,
)

A = "a" * 64
B = "b" * 64
C = "c" * 64
D = "d" * 64
E = "e" * 64
F = "f" * 64
REFERENCE_WORKSPACE = "9" * 64


def _manifest() -> EnvironmentManifest:
    return create_environment_manifest(
        generation=1,
        task_category="exact-spec",
        specification_sha256=A,
        artifacts=(
            EnvironmentArtifact(
                path="grader/hidden_tests.py",
                sha256=B,
                visibility=ArtifactVisibility.HIDDEN,
                kind=ArtifactKind.GRADER,
            ),
            EnvironmentArtifact(
                path="reference/solution.py",
                sha256=C,
                visibility=ArtifactVisibility.HIDDEN,
                kind=ArtifactKind.REFERENCE,
            ),
            EnvironmentArtifact(
                path="workspace/solution.py",
                sha256=D,
                visibility=ArtifactVisibility.PUBLIC,
                kind=ArtifactKind.WORKSPACE,
            ),
        ),
        runtime=RuntimeIdentity(
            image="python:3.11.14-slim",
            image_digest="sha256:" + E,
            dependency_lock_sha256=F,
            python_version="3.11.14",
        ),
        author_provenance=ProvenanceIdentity(
            role=ProvenanceRole.AUTHOR,
            producer_type="model",
            producer_id="Qwen/Qwen3.8-27B",
            revision="72a217afab8029b39e4af1c7273a829995a3dbaf",
            config_sha256=A,
            output_sha256=B,
        ),
        oracle_provenance=ProvenanceIdentity(
            role=ProvenanceRole.ORACLE,
            producer_type="programmatic",
            producer_id="pvrl-reference-checker",
            revision="v1",
            config_sha256=C,
            output_sha256=D,
        ),
        resources=ResourcePolicy(
            wall_clock_seconds=30,
            cpu_seconds=20,
            memory_bytes=1_073_741_824,
            pids_limit=64,
            output_bytes=1_048_576,
            file_bytes=16_777_216,
            disk_bytes=268_435_456,
        ),
        network=NetworkPolicy(mode=NetworkMode.DISABLED),
        contamination=ContaminationEvidence(
            status=ContaminationStatus.NOT_RUN,
            checker_ids=(),
        ),
        reference_validation=ReferenceValidationEvidence(status=EvidenceStatus.NOT_RUN),
    )


def _execution(
    phase: ValidationPhase,
    classification: ValidationExecutionClass,
    *,
    suffix: str = "",
) -> ValidationExecutionEvidence:
    if classification is ValidationExecutionClass.PASSED:
        exit_code = 0
        timing = ValidationTimingClass.WITHIN_BUDGET
    elif classification in {
        ValidationExecutionClass.BEHAVIORAL_FAILURE,
        ValidationExecutionClass.PARSE_COMPILE_FAILURE,
    }:
        exit_code = 1
        timing = ValidationTimingClass.WITHIN_BUDGET
    elif classification is ValidationExecutionClass.TIMEOUT_RESOURCE_FAILURE:
        exit_code = None
        timing = ValidationTimingClass.TIMEOUT
    else:
        exit_code = None
        timing = ValidationTimingClass.INFRASTRUCTURE
    return create_validation_execution_evidence(
        phase=phase,
        classification=classification,
        exit_code=exit_code,
        timing_class=timing,
        stdout=f"{phase.value}:{classification.value}:stdout{suffix}\n",
        stderr=f"{phase.value}:{classification.value}:stderr{suffix}\n",
    )


class _FakeSandbox:
    def __init__(
        self,
        factory: _FakeFactory,
        repetition: int,
    ) -> None:
        self._factory = factory
        self._repetition = repetition
        self._reference_applied = False

    def run_hidden(self, phase: ValidationPhase) -> ValidationExecutionEvidence:
        self._factory.events.append((self._repetition, phase.value))
        if phase is ValidationPhase.INITIAL_HIDDEN:
            classification = self._factory.initial_class
        else:
            assert self._reference_applied
            classification = self._factory.reference_class
        suffix = ""
        if self._factory.drift_second_run and self._repetition == 2:
            suffix = ":drift"
        return _execution(phase, classification, suffix=suffix)

    def apply_reference(self) -> str:
        self._factory.events.append((self._repetition, "apply_reference"))
        if self._factory.apply_error:
            raise RuntimeError("apply failed")
        self._reference_applied = True
        if self._factory.workspace_drift and self._repetition == 2:
            return "8" * 64
        return REFERENCE_WORKSPACE

    def run_preservation(self) -> ValidationExecutionEvidence:
        self._factory.events.append((self._repetition, ValidationPhase.PRESERVATION_PUBLIC.value))
        assert self._reference_applied
        return _execution(
            ValidationPhase.PRESERVATION_PUBLIC,
            self._factory.preservation_class,
        )

    def close(self) -> None:
        self._factory.events.append((self._repetition, "close"))
        if self._factory.close_error:
            raise RuntimeError("close failed")


class _FakeFactory:
    def __init__(
        self,
        *,
        initial_class: ValidationExecutionClass = ValidationExecutionClass.BEHAVIORAL_FAILURE,
        reference_class: ValidationExecutionClass = ValidationExecutionClass.PASSED,
        preservation_class: ValidationExecutionClass = ValidationExecutionClass.PASSED,
        drift_second_run: bool = False,
        workspace_drift: bool = False,
        create_error: bool = False,
        apply_error: bool = False,
        close_error: bool = False,
    ) -> None:
        self.initial_class = initial_class
        self.reference_class = reference_class
        self.preservation_class = preservation_class
        self.drift_second_run = drift_second_run
        self.workspace_drift = workspace_drift
        self.create_error = create_error
        self.apply_error = apply_error
        self.close_error = close_error
        self.events: list[tuple[int, str]] = []

    def create(self, manifest: EnvironmentManifest, *, repetition: int) -> _FakeSandbox:
        assert manifest.environment_id == _manifest().environment_id
        if self.create_error:
            raise RuntimeError("construction failed")
        self.events.append((repetition, "create"))
        return _FakeSandbox(self, repetition)


def _validate(factory: _FakeFactory | None = None) -> ReferenceValidationReport:
    return run_reference_first_validation(
        _manifest(),
        factory or _FakeFactory(),
        validator_id="reference-first-v1",
        expected_initial_failure=ValidationExecutionClass.BEHAVIORAL_FAILURE,
    )


def test_reference_first_validation_repeats_full_sequence_and_attaches_evidence() -> None:
    manifest = _manifest()
    factory = _FakeFactory()
    report = run_reference_first_validation(
        manifest,
        factory,
        validator_id="reference-first-v1",
        expected_initial_failure=ValidationExecutionClass.BEHAVIORAL_FAILURE,
    )

    assert factory.events == [
        (1, "create"),
        (1, "initial_hidden"),
        (1, "apply_reference"),
        (1, "reference_hidden"),
        (1, "preservation_public"),
        (1, "close"),
        (2, "create"),
        (2, "initial_hidden"),
        (2, "apply_reference"),
        (2, "reference_hidden"),
        (2, "preservation_public"),
        (2, "close"),
    ]
    assert len(report.runs) == 2
    assert report.runs[0].reference_workspace_sha256 == REFERENCE_WORKSPACE
    assert report.runs[0].initial_hidden.stdout_sha256 != A
    assert len(report.runs[0].initial_hidden.result_sha256) == 64

    payload = reference_validation_report_json(report)
    assert payload == reference_validation_report_json(report)

    validated = attach_reference_validation(manifest, report)
    assert validated.environment_id == manifest.environment_id
    assert validated.material_sha256 == manifest.material_sha256
    assert validated.reference_validation == ReferenceValidationEvidence(
        status=EvidenceStatus.PASSED,
        validator_id="reference-first-v1",
        evidence_sha256=report.evidence_sha256,
    )


def test_initial_workspace_must_fail_before_reference_is_applied() -> None:
    factory = _FakeFactory(initial_class=ValidationExecutionClass.PASSED)

    with pytest.raises(ReferenceValidationRejectedError, match="fail/incomplete"):
        _validate(factory)

    assert factory.events == [(1, "create"), (1, "initial_hidden"), (1, "close")]


def test_initial_failure_must_match_declared_incomplete_condition() -> None:
    factory = _FakeFactory(initial_class=ValidationExecutionClass.PARSE_COMPILE_FAILURE)

    with pytest.raises(ReferenceValidationRejectedError, match="fail/incomplete"):
        _validate(factory)


def test_reference_must_pass_hidden_contract() -> None:
    factory = _FakeFactory(reference_class=ValidationExecutionClass.BEHAVIORAL_FAILURE)

    with pytest.raises(ReferenceValidationRejectedError, match="hidden behavioral contract"):
        _validate(factory)


def test_reference_must_preserve_public_contract() -> None:
    factory = _FakeFactory(preservation_class=ValidationExecutionClass.BEHAVIORAL_FAILURE)

    with pytest.raises(ReferenceValidationRejectedError, match="preservation/public"):
        _validate(factory)


def test_reference_validation_fails_closed_on_fresh_sandbox_output_drift() -> None:
    factory = _FakeFactory(drift_second_run=True)

    with pytest.raises(ReferenceValidationRejectedError, match="not reproducible"):
        _validate(factory)


def test_reference_validation_fails_closed_on_fresh_sandbox_workspace_drift() -> None:
    factory = _FakeFactory(workspace_drift=True)

    with pytest.raises(ReferenceValidationRejectedError, match="not reproducible"):
        _validate(factory)


def test_harness_failure_is_not_converted_into_reference_failure() -> None:
    factory = _FakeFactory(reference_class=ValidationExecutionClass.GRADER_HARNESS_FAILURE)

    with pytest.raises(ReferenceValidationInfrastructureError) as raised:
        _validate(factory)

    assert raised.value.kind is ValidationExecutionClass.GRADER_HARNESS_FAILURE
    assert raised.value.phase is ValidationPhase.REFERENCE_HIDDEN
    assert not isinstance(raised.value, ReferenceValidationRejectedError)


def test_environment_construction_failure_is_infrastructure() -> None:
    factory = _FakeFactory(create_error=True)

    with pytest.raises(ReferenceValidationInfrastructureError) as raised:
        _validate(factory)

    assert raised.value.kind is ValidationExecutionClass.ENVIRONMENT_CONSTRUCTION_FAILURE


def test_reference_application_and_cleanup_failures_are_infrastructure() -> None:
    with pytest.raises(ReferenceValidationInfrastructureError, match="application"):
        _validate(_FakeFactory(apply_error=True))

    with pytest.raises(ReferenceValidationInfrastructureError, match="cleanup"):
        _validate(_FakeFactory(close_error=True))


def test_timeout_resource_failure_is_content_failure_not_harness_failure() -> None:
    factory = _FakeFactory(reference_class=ValidationExecutionClass.TIMEOUT_RESOURCE_FAILURE)

    with pytest.raises(ReferenceValidationRejectedError, match="hidden behavioral contract"):
        _validate(factory)


def test_report_hash_and_source_manifest_binding_fail_closed() -> None:
    manifest = _manifest()
    report = _validate()
    with pytest.raises(ReferenceValidationError, match="stale evidence_sha256"):
        replace(report, evidence_sha256="0" * 64)

    changed_manifest = replace(
        manifest,
        contamination=ContaminationEvidence(
            status=ContaminationStatus.CLEAN,
            checker_ids=("protected-registry",),
            evidence_sha256=E,
        ),
    )
    with pytest.raises(ReferenceValidationError, match="evidence changed"):
        attach_reference_validation(changed_manifest, report)


def test_validation_cannot_overwrite_completed_manifest_evidence() -> None:
    report = _validate()
    manifest = replace(
        _manifest(),
        reference_validation=ReferenceValidationEvidence(
            status=EvidenceStatus.PASSED,
            validator_id="reference-first-v1",
            evidence_sha256=report.evidence_sha256,
        ),
    )

    with pytest.raises(ReferenceValidationError, match="overwrite"):
        run_reference_first_validation(
            manifest,
            _FakeFactory(),
            validator_id="reference-first-v1",
            expected_initial_failure=ValidationExecutionClass.BEHAVIORAL_FAILURE,
        )
