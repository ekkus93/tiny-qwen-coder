"""Contract tests for machine-readable training/evaluation run manifests."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tiny_qwen_coder.reporting import (
    AdapterIdentity,
    BaseModelIdentity,
    DependencyVersions,
    GitMetadata,
    GpuMetadata,
    HostMetadata,
    ManifestError,
    create_run_manifest,
    emit_evaluation_run_manifest,
    emit_training_run_manifest,
    load_base_model_identity,
    manifest_json,
    write_run_manifest,
)

_CANONICAL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
_FAKE_GIT_SHA = "a" * 40


def _dependencies() -> DependencyVersions:
    return DependencyVersions(
        accelerate="1",
        datasets="2",
        numpy="2.4",
        peft="3",
        pyyaml="4",
        tiny_qwen_coder="5",
        torch="6",
        transformers="7",
        trl="8",
    )


def _host() -> HostMetadata:
    return HostMetadata(
        hostname="test-host",
        system="Linux",
        release="test-release",
        machine="x86_64",
        python_version="3.11.0",
        cuda_available=True,
        cuda_runtime="13.0",
        gpus=(
            GpuMetadata(
                index=0,
                name="Test GPU",
                total_memory_bytes=16 * 1024**3,
                compute_capability="8.9",
            ),
        ),
    )


def _base() -> BaseModelIdentity:
    return BaseModelIdentity(
        repository="Qwen/Qwen3.5-4B",
        revision=_CANONICAL_REVISION,
        tokenizer_repository="Qwen/Qwen3.5-4B",
        tokenizer_revision=_CANONICAL_REVISION,
    )


def test_load_base_model_identity_uses_exact_canonical_revision() -> None:
    identity = load_base_model_identity(Path("configs/base/qwen35-4b.yaml"))

    assert identity == _base()


def test_training_manifest_contains_complete_provenance() -> None:
    manifest = create_run_manifest(
        run_kind="training",
        base_model=_base(),
        language="python",
        adapter=AdapterIdentity(family="language", adapter_id="language/python/p0"),
        seed=1729,
        run_id="training-python-test",
        created_at=datetime(2026, 8, 28, 8, 30, tzinfo=UTC),
        git=GitMetadata(sha=_FAKE_GIT_SHA, dirty=False),
        dependencies=_dependencies(),
        host=_host(),
    )

    payload = json.loads(manifest_json(manifest))

    assert payload["schema_version"] == 1
    assert payload["run_id"] == "training-python-test"
    assert payload["run_kind"] == "training"
    assert payload["created_at_utc"] == "2026-08-28T08:30:00.000000Z"
    assert payload["git"] == {"dirty": False, "sha": _FAKE_GIT_SHA}
    assert payload["base_model"]["revision"] == _CANONICAL_REVISION
    assert payload["language"] == "python"
    assert payload["adapter"] == {
        "adapter_id": "language/python/p0",
        "family": "language",
    }
    assert payload["seed"] == 1729
    assert payload["dependencies"]["numpy"] == "2.4"
    assert payload["dependencies"]["transformers"] == "7"
    assert payload["host"]["gpus"][0]["total_memory_bytes"] == 16 * 1024**3


def test_manifest_rejects_seed_outside_project_range() -> None:
    with pytest.raises(ManifestError, match="seed must be between 0 and"):
        create_run_manifest(
            run_kind="training",
            base_model=_base(),
            language="python",
            adapter=AdapterIdentity(family="language", adapter_id="language/python/p0"),
            seed=-1,
            run_id="invalid-seed",
            created_at=datetime(2026, 8, 28, 8, 30, tzinfo=UTC),
            git=GitMetadata(sha=_FAKE_GIT_SHA, dirty=False),
            dependencies=_dependencies(),
            host=_host(),
        )


def test_training_manifest_rejects_missing_adapter_identity() -> None:
    with pytest.raises(ManifestError, match="training manifests require an adapter identity"):
        create_run_manifest(
            run_kind="training",
            base_model=_base(),
            language="python",
            adapter=AdapterIdentity(family=None, adapter_id=None),
            seed=1729,
            run_id="invalid-training",
            created_at=datetime(2026, 8, 28, 8, 30, tzinfo=UTC),
            git=GitMetadata(sha=_FAKE_GIT_SHA, dirty=False),
            dependencies=_dependencies(),
            host=_host(),
        )


def test_base_only_evaluation_manifest_is_explicit() -> None:
    manifest = create_run_manifest(
        run_kind="evaluation",
        base_model=_base(),
        language="rust",
        adapter=AdapterIdentity(family=None, adapter_id=None),
        seed=42,
        run_id="evaluation-rust-base",
        created_at=datetime(2026, 8, 28, 8, 30, tzinfo=UTC),
        git=GitMetadata(sha=_FAKE_GIT_SHA, dirty=True),
        dependencies=_dependencies(),
        host=_host(),
    )

    payload = json.loads(manifest_json(manifest))

    assert payload["run_kind"] == "evaluation"
    assert payload["adapter"] == {"adapter_id": None, "family": None}
    assert payload["git"]["dirty"] is True


def test_manifest_json_is_deterministic_and_write_is_atomic(tmp_path: Path) -> None:
    manifest = create_run_manifest(
        run_kind="evaluation",
        base_model=_base(),
        language="typescript",
        adapter=AdapterIdentity(family="language", adapter_id="language/typescript/test"),
        seed=9,
        run_id="evaluation-typescript-test",
        created_at=datetime(2026, 8, 28, 8, 30, tzinfo=UTC),
        git=GitMetadata(sha=_FAKE_GIT_SHA, dirty=False),
        dependencies=_dependencies(),
        host=_host(),
    )

    first = manifest_json(manifest)
    second = manifest_json(manifest)
    destination = write_run_manifest(manifest, tmp_path / "run")

    assert first == second
    assert destination.name == "run-manifest.json"
    assert destination.read_text(encoding="utf-8") == first
    assert not (destination.parent / ".run-manifest.json.tmp").exists()


def test_training_and_evaluation_emitters_write_machine_readable_manifests(tmp_path: Path) -> None:
    base_config = Path("configs/base/qwen35-4b.yaml")

    training_path = emit_training_run_manifest(
        output_dir=tmp_path / "training",
        base_config=base_config,
        language="python",
        adapter_family="language",
        adapter_id="language/python/test",
        seed=1729,
    )
    evaluation_path = emit_evaluation_run_manifest(
        output_dir=tmp_path / "evaluation",
        base_config=base_config,
        language="python",
        seed=1729,
    )

    training = json.loads(training_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))

    assert training["run_kind"] == "training"
    assert training["adapter"]["adapter_id"] == "language/python/test"
    assert evaluation["run_kind"] == "evaluation"
    assert evaluation["adapter"]["adapter_id"] is None
    assert training["base_model"]["revision"] == _CANONICAL_REVISION
    assert evaluation["base_model"]["revision"] == _CANONICAL_REVISION
    assert len(training["git"]["sha"]) == 40
    assert training["dependencies"]["torch"]
    assert "cuda_available" in training["host"]


def test_manifest_rejects_non_pinned_revision_and_partial_adapter() -> None:
    with pytest.raises(ManifestError, match="40-character Git SHA"):
        BaseModelIdentity(
            repository="Qwen/Qwen3.5-4B",
            revision="main",
            tokenizer_repository="Qwen/Qwen3.5-4B",
            tokenizer_revision=_CANONICAL_REVISION,
        )

    with pytest.raises(ManifestError, match="must be defined together"):
        AdapterIdentity(family="language", adapter_id=None)
