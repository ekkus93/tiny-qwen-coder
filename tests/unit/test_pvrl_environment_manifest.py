from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from tiny_qwen_coder.pvrl.environment_manifest import (
    ArtifactKind,
    ArtifactVisibility,
    ContaminationEvidence,
    ContaminationStatus,
    EnvironmentArtifact,
    EnvironmentManifest,
    EnvironmentManifestError,
    EvidenceStatus,
    NetworkMode,
    NetworkPolicy,
    ProvenanceIdentity,
    ProvenanceRole,
    ReferenceValidationEvidence,
    ResourcePolicy,
    RuntimeIdentity,
    create_environment_manifest,
    environment_manifest_from_json,
    environment_manifest_json,
    environment_manifest_sha256,
    read_environment_manifest,
    write_environment_manifest,
)

A = "a" * 64
B = "b" * 64
C = "c" * 64
D = "d" * 64
E = "e" * 64
F = "f" * 64


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


def test_manifest_round_trip_is_deterministic_and_content_addressed(tmp_path: Path) -> None:
    manifest = _manifest()
    payload = environment_manifest_json(manifest)

    assert environment_manifest_from_json(payload) == manifest
    assert environment_manifest_json(environment_manifest_from_json(payload)) == payload
    assert manifest.environment_id.startswith("pvrl-env-v1-g0001-")
    assert len(environment_manifest_sha256(manifest)) == 64

    path = write_environment_manifest(manifest, tmp_path / "nested" / "environment-manifest.json")
    assert read_environment_manifest(path) == manifest
    assert not (path.parent / ".environment-manifest.json.tmp").exists()


def test_material_mutation_cannot_retain_environment_identity() -> None:
    manifest = _manifest()

    with pytest.raises(EnvironmentManifestError, match="material changed"):
        replace(manifest, specification_sha256=B)


def test_evidence_can_change_without_changing_material_identity() -> None:
    manifest = _manifest()
    validated = replace(
        manifest,
        contamination=ContaminationEvidence(
            status=ContaminationStatus.CLEAN,
            checker_ids=("protected-registry",),
            evidence_sha256=E,
        ),
        reference_validation=ReferenceValidationEvidence(
            status=EvidenceStatus.PASSED,
            validator_id="reference-first-v1",
            evidence_sha256=F,
        ),
    )

    assert validated.environment_id == manifest.environment_id
    assert validated.material_sha256 == manifest.material_sha256
    assert environment_manifest_sha256(validated) != environment_manifest_sha256(manifest)


def test_missing_author_provenance_fails_closed() -> None:
    payload = asdict(_manifest())
    payload.pop("author_provenance")

    with pytest.raises(EnvironmentManifestError, match="author_provenance"):
        environment_manifest_from_json(json.dumps(payload))


def test_missing_artifact_visibility_fails_closed() -> None:
    payload = asdict(_manifest())
    artifacts = payload["artifacts"]
    assert isinstance(artifacts, tuple)
    first = artifacts[0]
    assert isinstance(first, dict)
    first.pop("visibility")

    with pytest.raises(EnvironmentManifestError, match="visibility"):
        environment_manifest_from_json(json.dumps(payload))


def test_visibility_contract_rejects_hidden_workspace() -> None:
    with pytest.raises(EnvironmentManifestError, match="workspace artifacts must be public"):
        EnvironmentArtifact(
            path="workspace/solution.py",
            sha256=A,
            visibility=ArtifactVisibility.HIDDEN,
            kind=ArtifactKind.WORKSPACE,
        )


def test_manifest_requires_both_workspace_and_grader_material() -> None:
    manifest = _manifest()
    only_workspace = tuple(
        item for item in manifest.artifacts if item.kind is ArtifactKind.WORKSPACE
    )

    with pytest.raises(EnvironmentManifestError, match="hidden grader"):
        create_environment_manifest(
            generation=1,
            task_category=manifest.task_category,
            specification_sha256=manifest.specification_sha256,
            artifacts=only_workspace,
            runtime=manifest.runtime,
            author_provenance=manifest.author_provenance,
            oracle_provenance=manifest.oracle_provenance,
            resources=manifest.resources,
            network=manifest.network,
            contamination=manifest.contamination,
            reference_validation=manifest.reference_validation,
        )


def test_not_run_evidence_cannot_claim_completed_identity() -> None:
    with pytest.raises(EnvironmentManifestError, match="not_run reference validation"):
        ReferenceValidationEvidence(
            status=EvidenceStatus.NOT_RUN,
            validator_id="reference-first-v1",
            evidence_sha256=A,
        )

    with pytest.raises(EnvironmentManifestError, match="not_run contamination"):
        ContaminationEvidence(
            status=ContaminationStatus.NOT_RUN,
            checker_ids=("protected-registry",),
            evidence_sha256=A,
        )
