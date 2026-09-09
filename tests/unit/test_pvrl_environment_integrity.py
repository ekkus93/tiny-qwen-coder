"""PVRL-G1 cross-cutting environment-integrity admission tests."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

import pytest

if TYPE_CHECKING:
    from tiny_qwen_coder.data.records import TrainingMessage

from tiny_qwen_coder.evaluation.contamination import ProtectedBenchmarkExample
from tiny_qwen_coder.evaluation.protected_benchmarks import (
    ProtectedBenchmark,
    ProtectedBenchmarkRegistry,
)
from tiny_qwen_coder.languages import (
    ConfigReferences,
    LanguageComponentRef,
    LanguageConfig,
    LanguageHookReferences,
    LanguageSpec,
    ProtectedBenchmarkRef,
    RepositoryDetectionSignals,
    StaticLanguagePlugin,
    SystemPromptSpec,
)
from tiny_qwen_coder.pvrl.contamination import (
    PVRLContaminationReport,
    TaskPartitionRole,
    admit_environment_contamination,
    check_environment_contamination,
    create_environment_contamination_input,
)
from tiny_qwen_coder.pvrl.environment_integrity import (
    EnvironmentIntegrityError,
    EnvironmentIntegrityStatus,
    admit_environment_integrity,
    environment_integrity_evidence_json,
)
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
    environment_manifest_sha256,
)
from tiny_qwen_coder.pvrl.execution_sandbox import (
    HardenedCandidateExecutor,
    HardenedEnvironmentMaterial,
    HardenedReferenceValidationSandboxFactory,
    SandboxArtifactContent,
    TrustedGraderArtifacts,
)
from tiny_qwen_coder.pvrl.oracle_provenance import (
    OracleIndependenceGrade,
    OracleProvenanceAssessment,
    ReferenceProvenanceIdentity,
    create_oracle_provenance_assessment,
)
from tiny_qwen_coder.pvrl.reference_validation import (
    ReferenceValidationReport,
    ValidationExecutionClass,
    ValidationExecutionEvidence,
    ValidationPhase,
    ValidationTimingClass,
    attach_reference_validation,
    create_validation_execution_evidence,
    run_reference_first_validation,
)

_REVISION = "a" * 40
_DIGEST = "b" * 64
_SPECIFICATION = (
    "Implement count_vowels(text), returning the number of ASCII vowels in a string. "
    "Count uppercase and lowercase vowels, preserve the input, and return an integer for "
    "the empty string as well as non-empty strings."
)
_PROTECTED_PROMPT = (
    "Implement rotate_left(values, amount) so that it returns a new list rotated to the left "
    "by amount positions, supports empty input, and leaves the original sequence unchanged."
)
_REFERENCE_CAVEAT = (
    "The environment author and generated reference share the Qwen model family; the "
    "independent programmatic grader remains authoritative, while reference errors may correlate."
)


@dataclass(frozen=True)
class _FixtureMessage:
    role: str
    content: str


def _message(role: str, content: str) -> TrainingMessage:
    return cast("TrainingMessage", _FixtureMessage(role=role, content=content))


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _artifact_contents() -> tuple[SandboxArtifactContent, ...]:
    return (
        SandboxArtifactContent.from_text(
            "grader/hidden_tests.py",
            "assert count_vowels('Audio') == 4\nassert count_vowels('rhythm') == 0\n",
        ),
        SandboxArtifactContent.from_text(
            "reference/solution.py",
            "def count_vowels(text):\n    return sum(ch.lower() in 'aeiou' for ch in text)\n",
        ),
        SandboxArtifactContent.from_text(
            "workspace/solution.py",
            "def count_vowels(text):\n    raise NotImplementedError\n",
        ),
    )


def _manifest(contents: tuple[SandboxArtifactContent, ...]) -> EnvironmentManifest:
    by_path = {item.path: item for item in contents}
    return create_environment_manifest(
        generation=1,
        task_category="self-contained-python",
        specification_sha256=_sha(_SPECIFICATION.encode()),
        artifacts=(
            EnvironmentArtifact(
                path="grader/hidden_tests.py",
                sha256=by_path["grader/hidden_tests.py"].sha256,
                visibility=ArtifactVisibility.HIDDEN,
                kind=ArtifactKind.GRADER,
            ),
            EnvironmentArtifact(
                path="reference/solution.py",
                sha256=by_path["reference/solution.py"].sha256,
                visibility=ArtifactVisibility.HIDDEN,
                kind=ArtifactKind.REFERENCE,
            ),
            EnvironmentArtifact(
                path="workspace/solution.py",
                sha256=by_path["workspace/solution.py"].sha256,
                visibility=ArtifactVisibility.PUBLIC,
                kind=ArtifactKind.WORKSPACE,
            ),
        ),
        runtime=RuntimeIdentity(
            image="python:3.11.14-slim",
            image_digest="sha256:" + _DIGEST,
            dependency_lock_sha256="c" * 64,
            python_version="3.11.14",
        ),
        author_provenance=ProvenanceIdentity(
            role=ProvenanceRole.AUTHOR,
            producer_type="model",
            producer_id="Qwen/Qwen3.8-27B",
            revision="author-revision",
            config_sha256="d" * 64,
            output_sha256="e" * 64,
        ),
        oracle_provenance=ProvenanceIdentity(
            role=ProvenanceRole.ORACLE,
            producer_type="programmatic",
            producer_id="pvrl-g1-test-grader",
            revision="v1",
            config_sha256="f" * 64,
            output_sha256="1" * 64,
        ),
        resources=ResourcePolicy(
            wall_clock_seconds=30,
            cpu_seconds=20,
            memory_bytes=256 * 1024 * 1024,
            pids_limit=32,
            output_bytes=16_384,
            file_bytes=1_048_576,
            disk_bytes=16 * 1024 * 1024,
        ),
        network=NetworkPolicy(mode=NetworkMode.DISABLED),
        contamination=ContaminationEvidence(
            status=ContaminationStatus.NOT_RUN,
            checker_ids=(),
        ),
        reference_validation=ReferenceValidationEvidence(status=EvidenceStatus.NOT_RUN),
    )


def _validation_evidence(
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


class _Grader:
    def run_hidden(
        self,
        candidate: HardenedCandidateExecutor,
        protected: TrustedGraderArtifacts,
        phase: ValidationPhase,
    ) -> ValidationExecutionEvidence:
        assert protected.read_text("grader/hidden_tests.py").startswith(
            "assert count_vowels"
        )
        assert "grader/hidden_tests.py" not in candidate.visible_paths
        classification = (
            ValidationExecutionClass.BEHAVIORAL_FAILURE
            if b"NotImplementedError" in candidate.read_visible_bytes("solution.py")
            else ValidationExecutionClass.PASSED
        )
        return _validation_evidence(phase, classification)

    def run_preservation(
        self,
        candidate: HardenedCandidateExecutor,
        protected: TrustedGraderArtifacts,
    ) -> ValidationExecutionEvidence:
        assert "grader/hidden_tests.py" not in candidate.visible_paths
        assert protected.read_text("grader/hidden_tests.py").startswith(
            "assert count_vowels"
        )
        return _validation_evidence(
            ValidationPhase.PRESERVATION_PUBLIC,
            ValidationExecutionClass.PASSED,
        )


def _plugin() -> StaticLanguagePlugin:
    validator_ref = "tests.fixtures.pipeline_hooks:validate_fixture"
    executor_ref = "tests.fixtures.pipeline_hooks:execute_fixture"
    config = LanguageConfig(
        schema_version=1,
        id="python",
        aliases=("py",),
        extensions=(".py",),
        repository_detection=RepositoryDetectionSignals(files=("pyproject.toml",)),
        system_prompt=SystemPromptSpec(version="fixture-v1", text="Write Python code."),
        config_refs=ConfigReferences(
            data_sources=("configs/data/python/fixture.yaml",),
            evaluation=("configs/eval/python/holdout.yaml",),
        ),
        hooks=LanguageHookReferences(validator=validator_ref, executor=executor_ref),
    )
    return StaticLanguagePlugin(
        LanguageSpec(
            config=config,
            execution_hook=LanguageComponentRef(id="default", import_ref=executor_ref),
            protected_benchmarks=(ProtectedBenchmarkRef(id="holdout"),),
        )
    )


def _registry() -> ProtectedBenchmarkRegistry:
    registry = ProtectedBenchmarkRegistry()
    registry.register_language(
        _plugin(),
        (
            ProtectedBenchmark(
                language="python",
                id="holdout",
                dataset_id="fixtures/holdout",
                dataset_revision=_REVISION,
                source_configs=("configs/eval/python/holdout.yaml",),
            ),
        ),
    )
    return registry


def _protected_examples() -> tuple[ProtectedBenchmarkExample, ...]:
    return (
        ProtectedBenchmarkExample(
            language="python",
            benchmark_id="holdout",
            dataset_id="fixtures/holdout",
            dataset_revision=_REVISION,
            record_id="Holdout/1",
            prompt_messages=(_message("user", _PROTECTED_PROMPT),),
            solution="def rotate_left(values, amount):\n    return list(values[amount:]) + list(values[:amount])\n",
            test_texts=("assert rotate_left([1, 2, 3], 1) == [2, 3, 1]\n",),
        ),
    )


@dataclass(frozen=True)
class _IntegratedFixture:
    base_manifest: EnvironmentManifest
    final_manifest: EnvironmentManifest
    material: HardenedEnvironmentMaterial
    reference_report: ReferenceValidationReport
    contamination_report: PVRLContaminationReport
    oracle_assessment: OracleProvenanceAssessment


def _integrated_fixture() -> _IntegratedFixture:
    contents = _artifact_contents()
    base_manifest = _manifest(contents)
    initial_material = HardenedEnvironmentMaterial(
        manifest=base_manifest, artifacts=contents
    )
    reference_report = run_reference_first_validation(
        base_manifest,
        HardenedReferenceValidationSandboxFactory(initial_material, _Grader()),
        validator_id="pvrl-g1-reference-v1",
        expected_initial_failure=ValidationExecutionClass.BEHAVIORAL_FAILURE,
    )
    reference_manifest = attach_reference_validation(base_manifest, reference_report)
    contamination_input = create_environment_contamination_input(
        reference_manifest,
        task_id="synthetic/count-vowels-001",
        role=TaskPartitionRole.TRAINING_CURRICULUM,
        specification=_SPECIFICATION,
        artifact_contents={item.path: item.content for item in contents},
    )
    contamination_report = check_environment_contamination(
        contamination_input,
        _protected_examples(),
        manifest=reference_manifest,
        language="python",
        registry=_registry(),
        author_few_shots=(),
        partition_entries=(),
    )
    final_manifest = admit_environment_contamination(
        reference_manifest, contamination_report
    )
    material = HardenedEnvironmentMaterial(manifest=final_manifest, artifacts=contents)
    oracle_assessment = create_oracle_provenance_assessment(
        manifest=final_manifest,
        reference_provenance=ReferenceProvenanceIdentity(
            producer_type="model",
            producer_id="Qwen/Qwen3.8-27B",
            revision="reference-revision",
            config_sha256="2" * 64,
            output_sha256="3" * 64,
            producer_family="qwen",
        ),
        author_family="qwen",
        oracle_family=None,
        independence_grade=OracleIndependenceGrade.DETERMINISTIC_PROGRAMMATIC_CONTRACT,
        correlation_caveat=_REFERENCE_CAVEAT,
    )
    return _IntegratedFixture(
        base_manifest=base_manifest,
        final_manifest=final_manifest,
        material=material,
        reference_report=reference_report,
        contamination_report=contamination_report,
        oracle_assessment=oracle_assessment,
    )


def test_g1_admission_composes_all_environment_integrity_subsystems() -> None:
    fixture = _integrated_fixture()

    evidence = admit_environment_integrity(
        manifest=fixture.final_manifest,
        material=fixture.material,
        oracle_assessment=fixture.oracle_assessment,
        reference_report=fixture.reference_report,
        contamination_report=fixture.contamination_report,
    )

    assert evidence.status is EnvironmentIntegrityStatus.PASSED
    assert evidence.environment_id == fixture.base_manifest.environment_id
    assert evidence.environment_material_sha256 == fixture.base_manifest.material_sha256
    assert evidence.environment_manifest_sha256 == environment_manifest_sha256(
        fixture.final_manifest
    )
    assert evidence.oracle_independence_grade is (
        OracleIndependenceGrade.DETERMINISTIC_PROGRAMMATIC_CONTRACT
    )
    assert (
        evidence.reference_validation_evidence_sha256
        == fixture.reference_report.evidence_sha256
    )
    assert (
        evidence.contamination_evidence_sha256
        == fixture.contamination_report.evidence_sha256
    )
    serialized = environment_integrity_evidence_json(evidence)
    assert _SPECIFICATION not in serialized
    assert _PROTECTED_PROMPT not in serialized


def test_g1_rejects_not_run_reference_or_contamination_state() -> None:
    fixture = _integrated_fixture()
    reference_not_run = replace(
        fixture.final_manifest,
        reference_validation=ReferenceValidationEvidence(status=EvidenceStatus.NOT_RUN),
    )
    reference_not_run_material = HardenedEnvironmentMaterial(
        manifest=reference_not_run,
        artifacts=fixture.material.artifacts,
    )

    with pytest.raises(EnvironmentIntegrityError, match="reference-first"):
        admit_environment_integrity(
            manifest=reference_not_run,
            material=reference_not_run_material,
            oracle_assessment=fixture.oracle_assessment,
            reference_report=fixture.reference_report,
            contamination_report=fixture.contamination_report,
        )

    contamination_not_run = replace(
        fixture.final_manifest,
        contamination=ContaminationEvidence(
            status=ContaminationStatus.NOT_RUN,
            checker_ids=(),
        ),
    )
    contamination_not_run_material = HardenedEnvironmentMaterial(
        manifest=contamination_not_run,
        artifacts=fixture.material.artifacts,
    )
    with pytest.raises(EnvironmentIntegrityError, match="clean contamination"):
        admit_environment_integrity(
            manifest=contamination_not_run,
            material=contamination_not_run_material,
            oracle_assessment=fixture.oracle_assessment,
            reference_report=fixture.reference_report,
            contamination_report=fixture.contamination_report,
        )


def test_g1_rejects_hardened_material_bound_before_final_evidence() -> None:
    fixture = _integrated_fixture()
    stale_material = HardenedEnvironmentMaterial(
        manifest=fixture.base_manifest,
        artifacts=fixture.material.artifacts,
    )

    with pytest.raises(
        EnvironmentIntegrityError, match="exact final environment manifest"
    ):
        admit_environment_integrity(
            manifest=fixture.final_manifest,
            material=stale_material,
            oracle_assessment=fixture.oracle_assessment,
            reference_report=fixture.reference_report,
            contamination_report=fixture.contamination_report,
        )


def test_g1_rejects_oracle_assessment_created_before_final_evidence() -> None:
    fixture = _integrated_fixture()
    stale_assessment = create_oracle_provenance_assessment(
        manifest=fixture.base_manifest,
        reference_provenance=fixture.oracle_assessment.reference_provenance,
        author_family="qwen",
        oracle_family=None,
        independence_grade=OracleIndependenceGrade.DETERMINISTIC_PROGRAMMATIC_CONTRACT,
        correlation_caveat=_REFERENCE_CAVEAT,
    )

    with pytest.raises(EnvironmentIntegrityError, match="exact final admitted"):
        admit_environment_integrity(
            manifest=fixture.final_manifest,
            material=fixture.material,
            oracle_assessment=stale_assessment,
            reference_report=fixture.reference_report,
            contamination_report=fixture.contamination_report,
        )


def test_g1_rejects_report_from_another_environment() -> None:
    fixture = _integrated_fixture()
    altered_contents = tuple(
        replace(item, content=item.content + b"# variant\n")
        if item.path == "workspace/solution.py"
        else item
        for item in _artifact_contents()
    )
    other_base = _manifest(altered_contents)
    other_material = HardenedEnvironmentMaterial(
        manifest=other_base, artifacts=altered_contents
    )
    other_report = run_reference_first_validation(
        other_base,
        HardenedReferenceValidationSandboxFactory(other_material, _Grader()),
        validator_id="pvrl-g1-reference-v1",
        expected_initial_failure=ValidationExecutionClass.BEHAVIORAL_FAILURE,
    )

    with pytest.raises(EnvironmentIntegrityError, match="another environment_id"):
        admit_environment_integrity(
            manifest=fixture.final_manifest,
            material=fixture.material,
            oracle_assessment=fixture.oracle_assessment,
            reference_report=other_report,
            contamination_report=fixture.contamination_report,
        )


def test_g1_evidence_fails_closed_on_post_admission_tampering() -> None:
    fixture = _integrated_fixture()
    evidence = admit_environment_integrity(
        manifest=fixture.final_manifest,
        material=fixture.material,
        oracle_assessment=fixture.oracle_assessment,
        reference_report=fixture.reference_report,
        contamination_report=fixture.contamination_report,
    )

    with pytest.raises(EnvironmentIntegrityError, match="stale evidence_sha256"):
        replace(evidence, environment_manifest_sha256="9" * 64)
