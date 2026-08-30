from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from tiny_qwen_coder.training.plan import AdapterTrainingError, AdapterTrainingPlan
from tiny_qwen_coder.training.preflight_output import verify_training_output_directory
from tiny_qwen_coder.training.runtime import AdapterTrainingRuntimeOptions
from tiny_qwen_coder.training.smoke import (
    TrainingSmokeConfig,
    _artifact_digests,
    _require_finite_float,
    _validate_smoke_bounds,
    _verify_saved_training_artifacts,
    load_training_smoke_config,
    training_smoke_config_sha256,
)

_SMOKE_CONFIG = Path("configs/train/python/p0_smoke.yaml")
_WORKFLOW = Path(".github/workflows/python-p0-smoke-training.yml")


def test_python_p0_smoke_contract_is_bounded_and_noncanonical() -> None:
    config = load_training_smoke_config(_SMOKE_CONFIG)

    assert config.schema_version == 1
    assert config.training_config == "configs/train/python/p0.yaml"
    assert config.output_dir == "artifacts/train/python/p0-smoke"
    assert config.max_steps == 1
    assert config.train_samples == 8
    assert config.validation_samples == 4
    assert len(training_smoke_config_sha256(config)) == 64


def test_smoke_config_rejects_unknown_fields_and_unbounded_values(tmp_path: Path) -> None:
    unknown = tmp_path / "unknown.yaml"
    unknown.write_text(
        """schema_version: 1
training_config: configs/train/python/p0.yaml
output_dir: artifacts/train/python/p0-smoke
max_steps: 1
train_samples: 8
validation_samples: 4
unexpected: true
""",
        encoding="utf-8",
    )
    with pytest.raises(AdapterTrainingError, match="unknown field"):
        load_training_smoke_config(unknown)

    with pytest.raises(AdapterTrainingError, match="max_steps"):
        TrainingSmokeConfig(
            schema_version=1,
            training_config="configs/train/python/p0.yaml",
            output_dir="artifacts/train/python/p0-smoke",
            max_steps=5,
            train_samples=8,
            validation_samples=4,
        )


def test_smoke_bounds_cover_gradient_accumulation_and_isolate_output() -> None:
    plan = cast(
        AdapterTrainingPlan,
        SimpleNamespace(
            config=SimpleNamespace(
                micro_batch_size=1,
                gradient_accumulation_steps=8,
                output_dir="artifacts/train/python/p0",
            )
        ),
    )
    valid = TrainingSmokeConfig(
        schema_version=1,
        training_config="configs/train/python/p0.yaml",
        output_dir="artifacts/train/python/p0-smoke",
        max_steps=1,
        train_samples=8,
        validation_samples=4,
    )
    _validate_smoke_bounds(valid, plan)

    with pytest.raises(AdapterTrainingError, match="at least 8"):
        _validate_smoke_bounds(
            TrainingSmokeConfig(
                schema_version=1,
                training_config=valid.training_config,
                output_dir=valid.output_dir,
                max_steps=1,
                train_samples=7,
                validation_samples=4,
            ),
            plan,
        )

    with pytest.raises(AdapterTrainingError, match="canonical training output"):
        _validate_smoke_bounds(
            TrainingSmokeConfig(
                schema_version=1,
                training_config=valid.training_config,
                output_dir="artifacts/train/python/p0",
                max_steps=1,
                train_samples=8,
                validation_samples=4,
            ),
            plan,
        )


def test_explicit_smoke_output_uses_same_fail_closed_output_policy(tmp_path: Path) -> None:
    output = tmp_path / "artifacts" / "train" / "python" / "p0-smoke"
    evidence = verify_training_output_directory(output, repo_root=tmp_path)
    assert evidence.output_dir == str(output.resolve())

    output.mkdir(parents=True)
    with pytest.raises(AdapterTrainingError, match="already exists"):
        verify_training_output_directory(output, repo_root=tmp_path)

    with pytest.raises(AdapterTrainingError, match="must be beneath"):
        verify_training_output_directory(tmp_path / "outside", repo_root=tmp_path)


def test_runtime_bounds_must_be_positive() -> None:
    assert AdapterTrainingRuntimeOptions(max_steps=1, train_sample_limit=8).max_steps == 1
    with pytest.raises(AdapterTrainingError, match="max_steps"):
        AdapterTrainingRuntimeOptions(max_steps=0)


def test_smoke_losses_must_be_finite() -> None:
    assert _require_finite_float(1.25, field_name="loss") == 1.25
    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(AdapterTrainingError, match="finite"):
            _require_finite_float(value, field_name="loss")
    with pytest.raises(AdapterTrainingError, match="numeric"):
        _require_finite_float(True, field_name="loss")


def test_smoke_requires_final_checkpoint_and_adapter_weights(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints" / "checkpoint-1"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"checkpoint")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")

    resolved_checkpoint, resolved_adapter = _verify_saved_training_artifacts(
        tmp_path,
        global_steps=1,
    )
    assert resolved_checkpoint == checkpoint
    assert resolved_adapter == adapter
    digests = _artifact_digests(tmp_path, (checkpoint, adapter))
    assert tuple(item.path for item in digests) == (
        "adapter/adapter_config.json",
        "adapter/adapter_model.safetensors",
        "checkpoints/checkpoint-1/adapter_model.safetensors",
    )
    assert all(len(item.sha256) == 64 for item in digests)

    (adapter / "adapter_model.safetensors").unlink()
    with pytest.raises(AdapterTrainingError, match="adapter weights"):
        _verify_saved_training_artifacts(tmp_path, global_steps=1)


def test_gpu_smoke_workflow_uses_self_hosted_cuda_and_frozen_p0_data() -> None:
    workflow = _WORKFLOW.read_text(encoding="utf-8")

    assert "- self-hosted\n      - linux\n      - x64" in workflow
    assert "uv sync --frozen --extra qlora" in workflow
    assert "nvidia-smi" in workflow
    assert "scripts/freeze_python_p0.py" in workflow
    assert "tiny-qwen-coder-train-smoke" in workflow
    assert "configs/train/python/p0_smoke.yaml" in workflow
    assert "checkpoints/checkpoint-1" in workflow
    assert "peak_reserved_vram_bytes" in workflow
