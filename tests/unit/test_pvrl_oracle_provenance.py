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
from tiny_qwen_coder.pvrl.oracle_provenance import (
    OracleIndependenceGrade,
    OracleProvenanceError,
    ReferenceProvenanceIdentity,
    create_oracle_provenance_assessment,
    oracle_provenance_assessment_from_json,
    oracle_provenance_assessment_json,
)

A = "a" * 64
B = "b" * 64
C = "c" * 64
D = "d" * 64
E = "e" * 64
F = "f" * 64


def _manifest(
    *,
    oracle_type: str = "programmatic",
    oracle_id: str = "pvrl-contract-generator",
) -> EnvironmentManifest:
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
            producer_type=oracle_type,
            producer_id=oracle_id,
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


def _reference(*, family: str = "qwen") -> ReferenceProvenanceIdentity:
    return ReferenceProvenanceIdentity(
        producer_type="model",
        producer_id="Qwen/Qwen3.8-27B",
        revision="72a217afab8029b39e4af1c7273a829995a3dbaf",
        config_sha256=E,
        output_sha256=F,
        producer_family=family,
    )


def test_programmatic_oracle_assessment_round_trip() -> None:
    assessment = create_oracle_provenance_assessment(
        manifest=_manifest(),
        reference_provenance=_reference(),
        author_family="qwen",
        oracle_family=None,
        independence_grade=OracleIndependenceGrade.DETERMINISTIC_PROGRAMMATIC_CONTRACT,
    )

    payload = oracle_provenance_assessment_json(assessment)
    assert oracle_provenance_assessment_from_json(payload) == assessment
    assert len(assessment.assessment_sha256) == 64
    assert assessment.reference_provenance.producer_id == "Qwen/Qwen3.8-27B"


def test_same_family_generated_oracle_requires_explicit_caveat() -> None:
    manifest = _manifest(oracle_type="model", oracle_id="Qwen/Qwen3.8-27B")

    with pytest.raises(OracleProvenanceError, match="correlation_caveat"):
        create_oracle_provenance_assessment(
            manifest=manifest,
            reference_provenance=_reference(),
            author_family="qwen",
            oracle_family="qwen",
            independence_grade=(OracleIndependenceGrade.SAME_FAMILY_GENERATED_CONTRACT_REFERENCE),
        )


def test_same_family_generated_oracle_cannot_claim_independent() -> None:
    manifest = _manifest(oracle_type="model", oracle_id="Qwen/Qwen3.8-27B")

    with pytest.raises(OracleProvenanceError, match="cannot be labeled independently"):
        create_oracle_provenance_assessment(
            manifest=manifest,
            reference_provenance=_reference(),
            author_family="qwen",
            oracle_family="qwen",
            independence_grade=(OracleIndependenceGrade.INDEPENDENTLY_GENERATED_CONTRACT_REFERENCE),
        )


def test_same_family_generated_oracle_is_explicitly_downgraded() -> None:
    assessment = create_oracle_provenance_assessment(
        manifest=_manifest(oracle_type="model", oracle_id="Qwen/Qwen3.8-27B"),
        reference_provenance=_reference(),
        author_family="qwen",
        oracle_family="qwen",
        independence_grade=OracleIndependenceGrade.SAME_FAMILY_GENERATED_CONTRACT_REFERENCE,
        correlation_caveat=(
            "Environment author and generated oracle share the Qwen model family; "
            "behavioral execution remains authoritative but semantic errors may correlate."
        ),
    )

    assert (
        assessment.independence_grade
        is OracleIndependenceGrade.SAME_FAMILY_GENERATED_CONTRACT_REFERENCE
    )
    assert assessment.correlation_caveat is not None


def test_different_model_family_can_claim_independent_generation() -> None:
    assessment = create_oracle_provenance_assessment(
        manifest=_manifest(oracle_type="model", oracle_id="Other/Verifier-Model"),
        reference_provenance=_reference(),
        author_family="qwen",
        oracle_family="other",
        independence_grade=OracleIndependenceGrade.INDEPENDENTLY_GENERATED_CONTRACT_REFERENCE,
    )

    assert (
        assessment.independence_grade
        is OracleIndependenceGrade.INDEPENDENTLY_GENERATED_CONTRACT_REFERENCE
    )


@pytest.mark.parametrize(
    ("grade", "oracle_type"),
    [
        (OracleIndependenceGrade.EXISTING_HUMAN_SOURCE_TESTS, "programmatic"),
        (OracleIndependenceGrade.DETERMINISTIC_PROGRAMMATIC_CONTRACT, "repository"),
        (OracleIndependenceGrade.REPOSITORY_NATIVE_TESTS, "source"),
    ],
)
def test_independence_grade_rejects_incompatible_oracle_type(
    grade: OracleIndependenceGrade,
    oracle_type: str,
) -> None:
    with pytest.raises(OracleProvenanceError):
        create_oracle_provenance_assessment(
            manifest=_manifest(oracle_type=oracle_type),
            reference_provenance=_reference(),
            author_family="qwen",
            oracle_family=None,
            independence_grade=grade,
        )


def test_model_reference_requires_explicit_family() -> None:
    with pytest.raises(OracleProvenanceError, match="producer_family"):
        ReferenceProvenanceIdentity(
            producer_type="model",
            producer_id="Qwen/Qwen3.8-27B",
            revision="rev",
            config_sha256=A,
            output_sha256=B,
        )


def test_non_model_oracle_cannot_carry_model_family() -> None:
    with pytest.raises(OracleProvenanceError, match="only valid for model-produced"):
        create_oracle_provenance_assessment(
            manifest=_manifest(),
            reference_provenance=_reference(),
            author_family="qwen",
            oracle_family="qwen",
            independence_grade=OracleIndependenceGrade.DETERMINISTIC_PROGRAMMATIC_CONTRACT,
        )


def test_assessment_hash_detects_posthoc_provenance_mutation() -> None:
    assessment = create_oracle_provenance_assessment(
        manifest=_manifest(),
        reference_provenance=_reference(),
        author_family="qwen",
        oracle_family=None,
        independence_grade=OracleIndependenceGrade.DETERMINISTIC_PROGRAMMATIC_CONTRACT,
    )

    with pytest.raises(OracleProvenanceError, match="stale assessment_sha256"):
        replace(assessment, correlation_caveat="mutated after freeze")


def test_parser_rejects_role_confusion() -> None:
    assessment = create_oracle_provenance_assessment(
        manifest=_manifest(),
        reference_provenance=_reference(),
        author_family="qwen",
        oracle_family=None,
        independence_grade=OracleIndependenceGrade.DETERMINISTIC_PROGRAMMATIC_CONTRACT,
    )
    payload = oracle_provenance_assessment_json(assessment).replace(
        '"role": "author"',
        '"role": "oracle"',
        1,
    )

    with pytest.raises(OracleProvenanceError, match="author role"):
        oracle_provenance_assessment_from_json(payload)
