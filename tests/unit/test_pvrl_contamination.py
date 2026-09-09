from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

import pytest

if TYPE_CHECKING:
    from tiny_qwen_coder.data.records import TrainingMessage

from tiny_qwen_coder.evaluation.contamination import HighOverlapConfig, ProtectedBenchmarkExample
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
    AuthorFewShotExample,
    ContaminationTextComponent,
    EnvironmentContaminationInput,
    PartitionTask,
    PVRLContaminationError,
    PVRLContaminationRejectedError,
    PVRLContaminationReport,
    TaskPartitionRole,
    admit_environment_contamination,
    check_environment_contamination,
    contamination_report_json,
    create_environment_contamination_input,
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
)

_REVISION = "a" * 40
_DIGEST = "b" * 64

_PROTECTED_PROMPT = (
    "Implement a Python function named first_duplicate that receives a sequence of integer "
    "values and returns the first repeated value encountered during a left to right scan, "
    "or returns None when every value appears only once. Preserve the original scan order."
)
_PROTECTED_SOLUTION = (
    "def first_duplicate(values):\n"
    "    seen = set()\n"
    "    for value in values:\n"
    "        if value in seen:\n"
    "            return value\n"
    "        seen.add(value)\n"
    "    return None\n"
)


@dataclass(frozen=True)
class _FixtureMessage:
    role: str
    content: str


def _message(role: str, content: str) -> TrainingMessage:
    return cast("TrainingMessage", _FixtureMessage(role=role, content=content))


_PROTECTED_TEST = (
    "def test_first_duplicate():\n"
    "    assert first_duplicate([4, 7, 2, 7, 9]) == 7\n"
    "    assert first_duplicate([1, 2, 3]) is None\n"
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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
            record_id="HumanEval/42",
            prompt_messages=(_message("user", _PROTECTED_PROMPT),),
            solution=_PROTECTED_SOLUTION,
            test_texts=(_PROTECTED_TEST,),
        ),
    )


def _clean_specification() -> str:
    return (
        "Write a function count_vowels(text) that returns how many ASCII vowel characters "
        "occur in the supplied string. Count both uppercase and lowercase vowels and do not "
        "modify the input. Return an integer for every string, including the empty string."
    )


def _artifact_bytes() -> dict[str, bytes]:
    return {
        "grader/hidden_tests.py": (
            b"def test_count_vowels():\n"
            b"    assert count_vowels('Audio') == 4\n"
            b"    assert count_vowels('rhythm') == 0\n"
        ),
        "reference/solution.py": (
            b"def count_vowels(text):\n"
            b"    return sum(character.lower() in 'aeiou' for character in text)\n"
        ),
        "workspace/solution.py": b"def count_vowels(text):\n    raise NotImplementedError\n",
    }


def _manifest(
    *,
    specification: str | None = None,
    artifacts: dict[str, bytes] | None = None,
) -> EnvironmentManifest:
    selected_specification = specification or _clean_specification()
    selected_artifacts = artifacts or _artifact_bytes()
    kinds = {
        "grader/hidden_tests.py": (ArtifactKind.GRADER, ArtifactVisibility.HIDDEN),
        "reference/solution.py": (ArtifactKind.REFERENCE, ArtifactVisibility.HIDDEN),
        "workspace/solution.py": (ArtifactKind.WORKSPACE, ArtifactVisibility.PUBLIC),
    }
    manifest_artifacts = tuple(
        EnvironmentArtifact(
            path=path,
            sha256=_sha256_bytes(content),
            visibility=kinds[path][1],
            kind=kinds[path][0],
        )
        for path, content in selected_artifacts.items()
    )
    return create_environment_manifest(
        generation=1,
        task_category="self-contained-python",
        specification_sha256=_sha256_bytes(selected_specification.encode()),
        artifacts=manifest_artifacts,
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
            revision="teacher-revision",
            config_sha256="d" * 64,
            output_sha256="e" * 64,
        ),
        oracle_provenance=ProvenanceIdentity(
            role=ProvenanceRole.ORACLE,
            producer_type="programmatic",
            producer_id="pvrl-oracle",
            revision="v1",
            config_sha256="f" * 64,
            output_sha256="1" * 64,
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


def _candidate(
    *,
    task_id: str = "synthetic/count-vowels-001",
    role: TaskPartitionRole = TaskPartitionRole.TRAINING_CURRICULUM,
    specification: str | None = None,
    artifacts: dict[str, bytes] | None = None,
) -> tuple[EnvironmentManifest, EnvironmentContaminationInput]:
    selected_specification = specification or _clean_specification()
    selected_artifacts = artifacts or _artifact_bytes()
    manifest = _manifest(specification=selected_specification, artifacts=selected_artifacts)
    value = create_environment_contamination_input(
        manifest,
        task_id=task_id,
        role=role,
        specification=selected_specification,
        artifact_contents=selected_artifacts,
    )
    return manifest, value


def _check(
    manifest: EnvironmentManifest,
    candidate: EnvironmentContaminationInput,
    *,
    author_few_shots: tuple[AuthorFewShotExample, ...] = (),
    partition_entries: tuple[PartitionTask, ...] = (),
) -> PVRLContaminationReport:
    return check_environment_contamination(
        candidate,
        _protected_examples(),
        manifest=manifest,
        language="python",
        registry=_registry(),
        author_few_shots=author_few_shots,
        partition_entries=partition_entries,
        overlap=HighOverlapConfig(threshold=0.8, shingle_size=5, min_tokens=16),
    )


def test_clean_environment_freezes_evidence_without_changing_material_identity() -> None:
    manifest, candidate = _candidate()

    report = _check(manifest, candidate)
    admitted = admit_environment_contamination(manifest, report)

    assert report.status is ContaminationStatus.CLEAN
    assert report.findings == ()
    assert admitted.contamination.status is ContaminationStatus.CLEAN
    assert admitted.contamination.checker_ids == report.checker_ids
    assert admitted.contamination.evidence_sha256 == report.evidence_sha256
    assert admitted.environment_id == manifest.environment_id
    assert admitted.material_sha256 == manifest.material_sha256
    serialized = contamination_report_json(report)
    assert _PROTECTED_PROMPT not in serialized
    assert _PROTECTED_SOLUTION not in serialized
    assert _PROTECTED_TEST not in serialized


def test_exact_protected_task_id_blocks_admission() -> None:
    manifest, candidate = _candidate(task_id="HumanEval/42")

    report = _check(manifest, candidate)

    assert report.status is ContaminationStatus.FINDINGS
    assert any(item.finding_type == "exact-task-id" for item in report.findings)
    with pytest.raises(PVRLContaminationRejectedError, match="findings"):
        admit_environment_contamination(manifest, report)


def test_exact_protected_prompt_content_in_specification_is_detected() -> None:
    manifest, candidate = _candidate(specification=_PROTECTED_PROMPT)

    report = _check(manifest, candidate)

    assert any(
        item.checker_id == "exact-content"
        and item.finding_type == "exact-content"
        and item.subject_id.endswith(":specification")
        and "prompt-message:0000:user" in item.conflicting_id
        for item in report.findings
    )


def test_exact_protected_test_content_hidden_in_grader_is_detected() -> None:
    artifacts = _artifact_bytes()
    artifacts["grader/hidden_tests.py"] = _PROTECTED_TEST.encode()
    manifest, candidate = _candidate(artifacts=artifacts)

    report = _check(manifest, candidate)

    assert any(
        item.checker_id == "exact-content"
        and "artifact:grader/hidden_tests.py" in item.subject_id
        and item.finding_type == "exact-content"
        for item in report.findings
    )


def test_normalized_copy_and_high_text_overlap_are_detected() -> None:
    normalized_copy = "\ufeff" + _PROTECTED_SOLUTION.replace("\n", "\r\n")
    artifacts = _artifact_bytes()
    artifacts["reference/solution.py"] = normalized_copy.encode()
    high_overlap_spec = (
        "For a standalone exercise, " + _PROTECTED_PROMPT + " Return only Python code please."
    )
    manifest, candidate = _candidate(
        specification=high_overlap_spec,
        artifacts=artifacts,
    )

    report = _check(manifest, candidate)

    assert any(item.checker_id == "normalized-content" for item in report.findings)
    assert any(item.checker_id == "text-overlap" for item in report.findings)


def test_author_few_shot_protected_material_blocks_otherwise_clean_candidate() -> None:
    manifest, candidate = _candidate()
    few_shot = AuthorFewShotExample(
        example_id="author-example-001",
        task_id="demo-unrelated-id",
        components=(ContaminationTextComponent(component_id="prompt", text=_PROTECTED_PROMPT),),
    )

    report = _check(manifest, candidate, author_few_shots=(few_shot,))

    assert report.status is ContaminationStatus.FINDINGS
    assert any(item.checker_id == "author-few-shot" for item in report.findings)


def test_protected_material_in_reserved_partition_blocks_environment_admission() -> None:
    manifest, candidate = _candidate()
    partition = PartitionTask(
        task_id="qualification/protected-copy",
        role=TaskPartitionRole.QUALIFICATION_EVALUATION,
        components=(
            ContaminationTextComponent(component_id="specification", text=_PROTECTED_PROMPT),
        ),
    )

    report = _check(manifest, candidate, partition_entries=(partition,))

    assert report.status is ContaminationStatus.FINDINGS
    assert any(
        item.checker_id == "partition-overlap" and "protected:holdout" in item.conflicting_id
        for item in report.findings
    )


def test_training_candidate_cannot_overlap_future_qualification_material() -> None:
    qualification = PartitionTask(
        task_id="qualification/count-vowels-001",
        role=TaskPartitionRole.QUALIFICATION_EVALUATION,
        components=(
            ContaminationTextComponent(
                component_id="specification",
                text=_clean_specification(),
            ),
        ),
        environment_id="pvrl-qualification-reservation-001",
    )
    manifest, candidate = _candidate()

    report = _check(manifest, candidate, partition_entries=(qualification,))

    assert report.status is ContaminationStatus.FINDINGS
    assert any(
        item.checker_id == "partition-overlap"
        and item.finding_type in {"exact-content", "normalized-content", "text-overlap"}
        and "qualification_evaluation" in item.conflicting_id
        for item in report.findings
    )


def test_cross_role_partition_task_id_and_content_overlap_is_detected() -> None:
    shared_text = (
        "Create a stable function that sorts records by timestamp while preserving input order "
        "for records whose timestamps compare equal and returns a newly allocated list."
    )
    calibration = PartitionTask(
        task_id="shared/task-7",
        role=TaskPartitionRole.CALIBRATION_DIAGNOSTIC,
        components=(ContaminationTextComponent(component_id="specification", text=shared_text),),
    )
    development = PartitionTask(
        task_id="shared/task-7",
        role=TaskPartitionRole.DEVELOPMENT_EVALUATION,
        components=(ContaminationTextComponent(component_id="specification", text=shared_text),),
    )
    manifest, candidate = _candidate()

    report = _check(
        manifest,
        candidate,
        partition_entries=(calibration, development),
    )

    pair_findings = [item for item in report.findings if item.checker_id == "partition-overlap"]
    assert any(item.finding_type == "exact-task-id" for item in pair_findings)
    assert any(item.finding_type == "exact-content" for item in pair_findings)


def test_same_environment_id_cannot_occupy_multiple_partition_roles() -> None:
    shared_environment_id = "pvrl-env-reserved-shared"
    calibration = PartitionTask(
        task_id="calibration/task-a",
        role=TaskPartitionRole.CALIBRATION_DIAGNOSTIC,
        components=(
            ContaminationTextComponent(
                component_id="specification",
                text="Compute a deterministic checksum for a sequence of short labels.",
            ),
        ),
        environment_id=shared_environment_id,
    )
    development = PartitionTask(
        task_id="development/task-b",
        role=TaskPartitionRole.DEVELOPMENT_EVALUATION,
        components=(
            ContaminationTextComponent(
                component_id="specification",
                text="Return the longest common prefix among a collection of strings.",
            ),
        ),
        environment_id=shared_environment_id,
    )
    manifest, candidate = _candidate()

    report = _check(
        manifest,
        candidate,
        partition_entries=(calibration, development),
    )

    assert any(
        item.checker_id == "partition-overlap" and item.finding_type == "environment-role-conflict"
        for item in report.findings
    )


def test_development_candidate_must_be_disjoint_from_author_few_shot_text() -> None:
    specification = _clean_specification()
    manifest, candidate = _candidate(
        role=TaskPartitionRole.DEVELOPMENT_EVALUATION,
        specification=specification,
    )
    few_shot = AuthorFewShotExample(
        example_id="author-example-clean-protected-wise",
        components=(ContaminationTextComponent(component_id="prompt", text=specification),),
    )

    report = _check(manifest, candidate, author_few_shots=(few_shot,))

    assert any(
        item.checker_id == "partition-overlap"
        and "author-few-shot:author-example-clean-protected-wise" in item.conflicting_id
        for item in report.findings
    )


def test_reserved_qualification_must_be_disjoint_from_author_few_shot() -> None:
    manifest, candidate = _candidate()
    reserved_text = "Classify whether a normalized token stream is a palindrome under reversal."
    few_shot = AuthorFewShotExample(
        example_id="author-example-qualification-collision",
        task_id="qualification/future-a",
        components=(ContaminationTextComponent(component_id="prompt", text=reserved_text),),
    )
    qualification = PartitionTask(
        task_id="qualification/future-a",
        role=TaskPartitionRole.QUALIFICATION_EVALUATION,
        components=(ContaminationTextComponent(component_id="specification", text=reserved_text),),
    )

    report = _check(
        manifest,
        candidate,
        author_few_shots=(few_shot,),
        partition_entries=(qualification,),
    )

    assert any(
        item.checker_id == "partition-overlap"
        and item.finding_type == "exact-task-id"
        and "qualification/future-a" in item.subject_id
        for item in report.findings
    )
    assert any(
        item.checker_id == "partition-overlap"
        and "author-few-shot:author-example-qualification-collision" in item.conflicting_id
        for item in report.findings
    )


def test_contamination_input_fails_closed_on_missing_mismatched_or_non_utf8_artifacts() -> None:
    specification = _clean_specification()
    artifacts = _artifact_bytes()
    manifest = _manifest(specification=specification, artifacts=artifacts)

    missing = dict(artifacts)
    missing.pop("grader/hidden_tests.py")
    with pytest.raises(PVRLContaminationError, match="every manifest artifact"):
        create_environment_contamination_input(
            manifest,
            task_id="synthetic/a",
            role=TaskPartitionRole.TRAINING_CURRICULUM,
            specification=specification,
            artifact_contents=missing,
        )

    mismatched = dict(artifacts)
    mismatched["grader/hidden_tests.py"] = b"different"
    with pytest.raises(PVRLContaminationError, match="manifest SHA-256"):
        create_environment_contamination_input(
            manifest,
            task_id="synthetic/a",
            role=TaskPartitionRole.TRAINING_CURRICULUM,
            specification=specification,
            artifact_contents=mismatched,
        )

    binary_artifacts = dict(artifacts)
    binary_artifacts["grader/hidden_tests.py"] = b"\xff\xfe"
    binary_manifest = _manifest(specification=specification, artifacts=binary_artifacts)
    with pytest.raises(PVRLContaminationError, match="not UTF-8"):
        create_environment_contamination_input(
            binary_manifest,
            task_id="synthetic/a",
            role=TaskPartitionRole.TRAINING_CURRICULUM,
            specification=specification,
            artifact_contents=binary_artifacts,
        )


def test_protected_example_registry_identity_and_coverage_fail_closed() -> None:
    manifest, candidate = _candidate()
    wrong_identity = replace(_protected_examples()[0], dataset_revision="wrong-revision")

    with pytest.raises(PVRLContaminationError, match="registered dataset identity"):
        check_environment_contamination(
            candidate,
            (wrong_identity,),
            manifest=manifest,
            language="python",
            registry=_registry(),
            author_few_shots=(),
            partition_entries=(),
        )

    with pytest.raises(PVRLContaminationError, match="coverage"):
        check_environment_contamination(
            candidate,
            (),
            manifest=manifest,
            language="python",
            registry=_registry(),
            author_few_shots=(),
            partition_entries=(),
        )


def test_report_tampering_and_completed_manifest_reuse_fail_closed() -> None:
    manifest, candidate = _candidate()
    report = _check(manifest, candidate)

    with pytest.raises(PVRLContaminationError, match="stale evidence_sha256"):
        replace(report, partition_inventory_sha256="9" * 64)

    admitted = admit_environment_contamination(manifest, report)
    with pytest.raises(PVRLContaminationError, match="cannot be overwritten"):
        admit_environment_contamination(admitted, report)

    other_specification = _clean_specification() + "\nDo not count non-ASCII vowels."
    other_manifest, _other_candidate = _candidate(
        task_id="synthetic/other",
        specification=other_specification,
    )
    assert other_manifest.environment_id != report.environment_id
    with pytest.raises(PVRLContaminationError, match="another environment_id"):
        admit_environment_contamination(other_manifest, report)
