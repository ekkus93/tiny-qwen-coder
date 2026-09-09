"""PVRL-104 hardened execution-boundary and adversarial regression tests."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from tiny_qwen_coder.config import ExecutionConfig
from tiny_qwen_coder.evaluation.execution import (
    DirectExecutionHarness,
    ExecutionLimits,
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    OciRuntime,
)
from tiny_qwen_coder.pvrl.environment_manifest import (
    ArtifactKind,
    ArtifactVisibility,
    ContaminationEvidence,
    ContaminationStatus,
    EnvironmentArtifact,
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
from tiny_qwen_coder.pvrl.execution_sandbox import (
    CandidateExecutionHarness,
    HardenedCandidateExecutor,
    HardenedEnvironmentMaterial,
    HardenedReferenceValidationSandboxFactory,
    HardenedSandboxError,
    SandboxArtifactContent,
    TrustedGraderArtifacts,
)
from tiny_qwen_coder.pvrl.reference_validation import (
    ReferenceValidationInfrastructureError,
    ValidationExecutionClass,
    ValidationExecutionEvidence,
    ValidationPhase,
    ValidationTimingClass,
    create_validation_execution_evidence,
    run_reference_first_validation,
)

A = "a" * 64
B = "b" * 64


class RecordingHarness(CandidateExecutionHarness):
    """Record exact PVRL requests without launching an OCI runtime."""

    def __init__(self) -> None:
        self.requests: list[ExecutionRequest] = []
        self.executions: list[ExecutionConfig] = []
        self.limits: list[ExecutionLimits] = []

    def run(
        self,
        request: ExecutionRequest,
        execution: ExecutionConfig,
        *,
        limits: ExecutionLimits | None = None,
    ) -> ExecutionResult:
        assert limits is not None
        self.requests.append(request)
        self.executions.append(execution)
        self.limits.append(limits)
        return ExecutionResult(
            status=ExecutionStatus.SUCCEEDED,
            runtime=OciRuntime.DOCKER,
            exit_code=0,
            duration_seconds=0.01,
            stdout="ok\n",
            stderr="",
            stdout_truncated=False,
            stderr_truncated=False,
        )


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _material() -> HardenedEnvironmentMaterial:
    contents = (
        SandboxArtifactContent.from_text("grader/hidden_tests.py", "HIDDEN_SENTINEL\n"),
        SandboxArtifactContent.from_text("reference/solution.py", "REFERENCE\n"),
        SandboxArtifactContent.from_text(
            "reference/support/public_tests.py",
            "MALICIOUS_REFERENCE_TEST_REWRITE\n",
        ),
        SandboxArtifactContent.from_text("support/public_tests.py", "PRISTINE_PUBLIC_TEST\n"),
        SandboxArtifactContent.from_text("workspace/solution.py", "INITIAL_FAIL\n"),
    )
    content_by_path = {item.path: item for item in contents}
    artifacts = (
        EnvironmentArtifact(
            path="grader/hidden_tests.py",
            sha256=content_by_path["grader/hidden_tests.py"].sha256,
            visibility=ArtifactVisibility.HIDDEN,
            kind=ArtifactKind.GRADER,
        ),
        EnvironmentArtifact(
            path="reference/solution.py",
            sha256=content_by_path["reference/solution.py"].sha256,
            visibility=ArtifactVisibility.HIDDEN,
            kind=ArtifactKind.REFERENCE,
        ),
        EnvironmentArtifact(
            path="reference/support/public_tests.py",
            sha256=content_by_path["reference/support/public_tests.py"].sha256,
            visibility=ArtifactVisibility.HIDDEN,
            kind=ArtifactKind.REFERENCE,
        ),
        EnvironmentArtifact(
            path="support/public_tests.py",
            sha256=content_by_path["support/public_tests.py"].sha256,
            visibility=ArtifactVisibility.PUBLIC,
            kind=ArtifactKind.SUPPORT,
        ),
        EnvironmentArtifact(
            path="workspace/solution.py",
            sha256=content_by_path["workspace/solution.py"].sha256,
            visibility=ArtifactVisibility.PUBLIC,
            kind=ArtifactKind.WORKSPACE,
        ),
    )
    manifest = create_environment_manifest(
        generation=1,
        task_category="exact-spec",
        specification_sha256=A,
        artifacts=artifacts,
        runtime=RuntimeIdentity(
            image="python:3.11.14-slim",
            image_digest="sha256:" + "c" * 64,
            dependency_lock_sha256="d" * 64,
            python_version="3.11.14",
        ),
        author_provenance=ProvenanceIdentity(
            role=ProvenanceRole.AUTHOR,
            producer_type="model",
            producer_id="Qwen/Qwen3.8-27B",
            revision="author-revision",
            config_sha256=A,
            output_sha256=B,
        ),
        oracle_provenance=ProvenanceIdentity(
            role=ProvenanceRole.ORACLE,
            producer_type="programmatic",
            producer_id="pvrl-test-grader",
            revision="v1",
            config_sha256=A,
            output_sha256=B,
        ),
        resources=ResourcePolicy(
            wall_clock_seconds=13,
            cpu_seconds=7,
            memory_bytes=192 * 1024 * 1024,
            pids_limit=23,
            output_bytes=777,
            file_bytes=12_345,
            disk_bytes=20 * 1024 * 1024,
        ),
        network=NetworkPolicy(mode=NetworkMode.DISABLED),
        contamination=ContaminationEvidence(
            status=ContaminationStatus.NOT_RUN,
            checker_ids=(),
        ),
        reference_validation=ReferenceValidationEvidence(status=EvidenceStatus.NOT_RUN),
    )
    return HardenedEnvironmentMaterial(manifest=manifest, artifacts=contents)


def _evidence(
    phase: ValidationPhase,
    classification: ValidationExecutionClass,
) -> ValidationExecutionEvidence:
    return create_validation_execution_evidence(
        phase=phase,
        classification=classification,
        exit_code=0 if classification is ValidationExecutionClass.PASSED else 1,
        timing_class=ValidationTimingClass.WITHIN_BUDGET,
        stdout="stable\n",
        stderr="",
    )


def test_material_partitions_public_candidate_files_from_protected_source_of_truth() -> None:
    material = _material()

    assert material.initial_candidate_files() == (
        ("solution.py", b"INITIAL_FAIL\n"),
        ("support/public_tests.py", b"PRISTINE_PUBLIC_TEST\n"),
    )
    assert material.reference_replacements() == (
        ("solution.py", b"REFERENCE\n"),
        ("support/public_tests.py", b"MALICIOUS_REFERENCE_TEST_REWRITE\n"),
    )
    protected = material.trusted_grader_artifacts()
    assert protected.paths == (
        "grader/hidden_tests.py",
        "support/public_tests.py",
    )
    assert protected.read_text("grader/hidden_tests.py") == "HIDDEN_SENTINEL\n"
    assert protected.read_text("support/public_tests.py") == "PRISTINE_PUBLIC_TEST\n"


def test_material_fails_closed_on_missing_or_identity_mismatched_bytes() -> None:
    material = _material()

    with pytest.raises(HardenedSandboxError, match="exactly one byte payload"):
        HardenedEnvironmentMaterial(
            manifest=material.manifest,
            artifacts=material.artifacts[:-1],
        )

    corrupted = tuple(
        replace(item, content=b"CORRUPTED\n") if item.path == "workspace/solution.py" else item
        for item in material.artifacts
    )
    with pytest.raises(HardenedSandboxError, match="does not match manifest SHA-256"):
        HardenedEnvironmentMaterial(manifest=material.manifest, artifacts=corrupted)


def test_candidate_execution_forces_digest_network_and_all_manifest_resource_bounds() -> None:
    material = _material()
    harness = RecordingHarness()
    executor = HardenedCandidateExecutor(
        material.manifest,
        material.initial_candidate_files(),
        harness=harness,
    )

    result = executor.run(("python", "-I", "-B", "solution.py"))

    assert result.succeeded is True
    assert len(harness.requests) == 1
    request = harness.requests[0]
    execution = harness.executions[0]
    limits = harness.limits[0]
    assert request.image == "python:3.11.14-slim@sha256:" + "c" * 64
    assert tuple(item.path for item in request.files) == (
        "solution.py",
        "support/public_tests.py",
        ".pvrl/limit_runner.py",
    )
    assert all(b"HIDDEN_SENTINEL" not in item.content for item in request.files)
    assert request.command[:6] == (
        "python",
        "-I",
        "-B",
        "/input/.pvrl/limit_runner.py",
        "7",
        "12345",
    )
    assert request.command[6:] == ("python", "-I", "-B", "solution.py")
    runner = next(item for item in request.files if item.path == ".pvrl/limit_runner.py")
    assert b"RLIMIT_CPU" in runner.content
    assert b"RLIMIT_FSIZE" in runner.content
    assert execution.timeout_seconds == 13.0
    assert execution.network_enabled is False
    assert limits.memory_mebibytes == 96
    assert limits.pids == 23
    assert limits.max_output_bytes == 777
    assert limits.workspace_mebibytes + limits.temp_mebibytes <= 20


def test_hidden_source_never_crosses_candidate_boundary_for_traversal_or_symlink_attack() -> None:
    material = _material()
    harness = RecordingHarness()
    attack = (
        b"from pathlib import Path\n"
        b"Path('leak').symlink_to('/grader/hidden_tests.py')\n"
        b"print(Path('leak').read_text())\n"
    )
    workspace = material.initial_candidate_files() + (("attack.py", attack),)
    executor = HardenedCandidateExecutor(material.manifest, workspace, harness=harness)

    executor.run(("python", "-I", "-B", "attack.py"))

    request = harness.requests[0]
    assert "grader/hidden_tests.py" not in executor.visible_paths
    assert all(item.path != "grader/hidden_tests.py" for item in request.files)
    assert all(b"HIDDEN_SENTINEL" not in item.content for item in request.files)
    assert all("docker.sock" not in item.path for item in request.files)

    with pytest.raises(HardenedSandboxError, match="without traversal"):
        HardenedCandidateExecutor(
            material.manifest,
            (("../grader/hidden_tests.py", b"attack"),),
            harness=harness,
        )

    with pytest.raises(HardenedSandboxError, match="reserved PVRL namespace"):
        HardenedCandidateExecutor(
            material.manifest,
            ((".pvrl/limit_runner.py", b"replace trusted runner"),),
            harness=harness,
        )


def test_hardened_candidate_execution_rejects_direct_backend(tmp_path: Path) -> None:
    material = _material()
    direct = DirectExecutionHarness(temp_root=tmp_path, allow_reduced_isolation=True)

    with pytest.raises(HardenedSandboxError, match="reduced-isolation direct backend"):
        HardenedCandidateExecutor(
            material.manifest,
            material.initial_candidate_files(),
            harness=direct,
        )


def test_reference_validation_uses_pristine_grader_files_after_candidate_test_rewrite() -> None:
    material = _material()

    class Grader:
        def run_hidden(
            self,
            candidate: HardenedCandidateExecutor,
            protected: TrustedGraderArtifacts,
            phase: ValidationPhase,
        ) -> ValidationExecutionEvidence:
            assert protected.read_text("grader/hidden_tests.py") == "HIDDEN_SENTINEL\n"
            assert "grader/hidden_tests.py" not in candidate.visible_paths
            classification = (
                ValidationExecutionClass.BEHAVIORAL_FAILURE
                if candidate.read_visible_bytes("solution.py") == b"INITIAL_FAIL\n"
                else ValidationExecutionClass.PASSED
            )
            return _evidence(phase, classification)

        def run_preservation(
            self,
            candidate: HardenedCandidateExecutor,
            protected: TrustedGraderArtifacts,
        ) -> ValidationExecutionEvidence:
            assert (
                candidate.read_visible_bytes("support/public_tests.py")
                == b"MALICIOUS_REFERENCE_TEST_REWRITE\n"
            )
            assert protected.read_text("support/public_tests.py") == "PRISTINE_PUBLIC_TEST\n"
            return _evidence(
                ValidationPhase.PRESERVATION_PUBLIC,
                ValidationExecutionClass.PASSED,
            )

    factory = HardenedReferenceValidationSandboxFactory(material, Grader())
    report = run_reference_first_validation(
        material.manifest,
        factory,
        validator_id="pvrl-hardened-v1",
        expected_initial_failure=ValidationExecutionClass.BEHAVIORAL_FAILURE,
    )

    assert len(report.runs) == 2
    assert report.runs[0].reference_workspace_sha256 == report.runs[1].reference_workspace_sha256
    assert (
        report.runs[0].initial_hidden.result_sha256 == report.runs[1].initial_hidden.result_sha256
    )


def test_reference_factory_rejects_environment_identity_drift_as_infrastructure_failure() -> None:
    material = _material()

    class UnusedGrader:
        def run_hidden(
            self,
            candidate: HardenedCandidateExecutor,
            protected: TrustedGraderArtifacts,
            phase: ValidationPhase,
        ) -> ValidationExecutionEvidence:
            raise AssertionError("not reached")

        def run_preservation(
            self,
            candidate: HardenedCandidateExecutor,
            protected: TrustedGraderArtifacts,
        ) -> ValidationExecutionEvidence:
            raise AssertionError("not reached")

    factory = HardenedReferenceValidationSandboxFactory(material, UnusedGrader())
    drifted = replace(
        material.manifest,
        reference_validation=ReferenceValidationEvidence(
            status=EvidenceStatus.FAILED,
            validator_id="other-validator",
            evidence_sha256=_sha(b"other evidence"),
        ),
    )

    with pytest.raises(
        ReferenceValidationInfrastructureError,
        match="manifest evidence changed",
    ) as error:
        factory.create(drifted, repetition=1)
    assert error.value.kind is ValidationExecutionClass.ENVIRONMENT_IDENTITY_MISMATCH
