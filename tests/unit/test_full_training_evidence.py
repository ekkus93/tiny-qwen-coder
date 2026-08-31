"""P7-006 regression tests for full-training completion evidence."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from tiny_qwen_coder.training.full_training import (
    _artifact_digests,
    _artifact_set_sha256,
    _require_finite_float,
    _training_metrics,
    _verify_full_training_outputs,
)
from tiny_qwen_coder.training.plan import (
    AdapterTrainingError,
    AdapterTrainingPlan,
    TrainerArtifactPaths,
)

_WORKFLOW = Path(".github/workflows/python-p0-full-training.yml")


def _artifacts(root: Path) -> TrainerArtifactPaths:
    return TrainerArtifactPaths(
        output_dir=root,
        checkpoints=root / "checkpoints",
        adapter=root / "adapter",
        dataset_manifest=root / "dataset-manifest.json",
        training_config=root / "training-config.json",
        training_metrics=root / "training-metrics.jsonl",
        run_manifest=root / "run-manifest.json",
        adapter_manifest=root / "adapter-manifest.json",
    )


def _plan(root: Path) -> AdapterTrainingPlan:
    return cast(AdapterTrainingPlan, SimpleNamespace(artifacts=_artifacts(root)))


def test_full_training_metrics_fail_closed_on_nonfinite_or_zero_values() -> None:
    assert _require_finite_float(1.25, field_name="loss") == 1.25
    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(AdapterTrainingError, match="finite"):
            _require_finite_float(value, field_name="loss")
    with pytest.raises(AdapterTrainingError, match="greater than zero"):
        _require_finite_float(0.0, field_name="throughput", positive=True)


def test_full_training_extracts_finite_persisted_metrics() -> None:
    history = (
        {"loss": 2.0, "step": 1},
        {
            "train_loss": 1.5,
            "train_runtime": 20.0,
            "train_samples_per_second": 1.25,
            "train_steps_per_second": 0.15,
            "step": 2,
        },
        {"eval_loss": 1.25, "step": 2},
    )

    assert _training_metrics(history) == (
        1.5,
        1.25,
        (2.0,),
        20.0,
        1.25,
        0.15,
    )

    with pytest.raises(AdapterTrainingError, match="logged training loss must be finite"):
        _training_metrics(({"loss": math.nan}, {"train_loss": 1.0}, {"eval_loss": 1.0}))
    with pytest.raises(AdapterTrainingError, match="validation loss"):
        _training_metrics(
            (
                {"loss": 1.0},
                {
                    "train_loss": 1.0,
                    "train_runtime": 1.0,
                    "train_samples_per_second": 1.0,
                    "train_steps_per_second": 1.0,
                },
            )
        )


def test_full_training_requires_lora_adapter_and_rejects_merged_model(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    checkpoint = artifacts.checkpoints / "checkpoint-12"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text("{}\n", encoding="utf-8")
    artifacts.adapter.mkdir()
    (artifacts.adapter / "adapter_config.json").write_text(
        '{"peft_type": "LORA"}\n', encoding="utf-8"
    )
    (artifacts.adapter / "adapter_model.safetensors").write_bytes(b"adapter")

    assert _verify_full_training_outputs(_plan(tmp_path), global_steps=12) == checkpoint

    (artifacts.adapter / "model.safetensors").write_bytes(b"merged")
    with pytest.raises(AdapterTrainingError, match="merged/full-model"):
        _verify_full_training_outputs(_plan(tmp_path), global_steps=12)
    (artifacts.adapter / "model.safetensors").unlink()

    (checkpoint / "pytorch_model.bin").write_bytes(b"full-model")
    with pytest.raises(AdapterTrainingError, match="merged/full-model"):
        _verify_full_training_outputs(_plan(tmp_path), global_steps=12)


def test_full_training_artifact_fingerprint_is_deterministic(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    artifacts.adapter.mkdir(parents=True)
    (artifacts.adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    artifacts.run_manifest.write_text("{}\n", encoding="utf-8")

    first = _artifact_digests(tmp_path, (artifacts.adapter, artifacts.run_manifest))
    second = _artifact_digests(tmp_path, (artifacts.run_manifest, artifacts.adapter))

    assert first == second
    assert len(_artifact_set_sha256(first)) == 64
    assert tuple(item.path for item in first) == (
        "adapter/adapter_config.json",
        "run-manifest.json",
    )


def test_full_training_workflow_runs_frozen_p0_and_verifies_evidence() -> None:
    workflow = _WORKFLOW.read_text(encoding="utf-8")

    assert "workflow_dispatch:" in workflow
    assert "- self-hosted\n      - linux\n      - x64" in workflow
    assert "timeout-minutes: 1440" in workflow
    assert "uv sync --frozen --extra qlora" in workflow
    assert "scripts/freeze_python_p0.py" in workflow
    assert "tiny_qwen_coder.training.full_training" in workflow
    assert "configs/train/python/p0.yaml" in workflow
    assert "training-report.json" in workflow
    assert 'report.get("train_records") != 38000' in workflow
    assert 'report.get("validation_records") != 2000' in workflow
    assert 'report.get("global_steps") != 4750' in workflow
    assert "train_samples_per_second" in workflow
    assert "peak_allocated_vram_bytes" in workflow
    assert "peak_reserved_vram_bytes" in workflow
    assert "adapter_model.safetensors" in workflow
    assert "model.safetensors" in workflow
    assert "artifact_set_sha256" in workflow
    assert "source_training_config_sha256" in workflow
    assert "dataset_manifest_sha256" in workflow


def test_workflow_is_valid_yaml_mapping() -> None:
    import yaml

    payload = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert "jobs" in payload


def test_artifact_set_hash_matches_canonical_json_encoding(tmp_path: Path) -> None:
    import hashlib

    path = tmp_path / "evidence.json"
    path.write_text("{}\n", encoding="utf-8")
    artifacts = _artifact_digests(tmp_path, (path,))
    payload = json.dumps(
        [
            {"path": item.path, "size_bytes": item.size_bytes, "sha256": item.sha256}
            for item in artifacts
        ],
        sort_keys=True,
        separators=(",", ":"),
    )

    assert _artifact_set_sha256(artifacts) == hashlib.sha256(payload.encode("utf-8")).hexdigest()
