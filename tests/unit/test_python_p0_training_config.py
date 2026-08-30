"""P7-003 regression tests for the canonical Python P0 training config."""

from __future__ import annotations

from pathlib import Path

import pytest

import tiny_qwen_coder.training.plan as training_plan
from tiny_qwen_coder.adapters import load_frozen_selective_lora_target_profile
from tiny_qwen_coder.training import TrainingDatasetIdentity, resolve_adapter_training_plan

_CONFIG_PATH = Path("configs/train/python/p0.yaml")
_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
_TEMPLATE_SHA = "f" * 64


def _compatible_dataset() -> TrainingDatasetIdentity:
    return TrainingDatasetIdentity(
        manifest_id="python-p0",
        language="python",
        tokenizer_repository="Qwen/Qwen3.5-4B",
        tokenizer_revision=_REVISION,
        chat_template_sha256=_TEMPLATE_SHA,
        sha256="a" * 64,
    )


def _plan(monkeypatch: pytest.MonkeyPatch) -> training_plan.AdapterTrainingPlan:
    monkeypatch.setattr(
        training_plan,
        "load_training_dataset_identity",
        lambda _path: _compatible_dataset(),
    )
    return resolve_adapter_training_plan(_CONFIG_PATH)


def test_python_p0_training_config_freezes_p2_and_p7_decisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(monkeypatch)
    config = plan.config
    profile = load_frozen_selective_lora_target_profile()

    assert plan.language == "python"
    assert plan.target.model_revision == _REVISION
    assert plan.target.tokenizer_revision == _REVISION
    assert config.adapter_family == "language"
    assert config.adapter_id == "language/python/p0"
    assert config.dataset_manifest == "data/python/p0/dataset-manifest.json"
    assert config.train_records == "data/python/p0/train.jsonl"
    assert config.validation_records == "data/python/p0/validation.jsonl"
    assert config.output_dir == "artifacts/train/python/p0"
    assert config.seed == 1729

    assert config.training_mode == "qlora_4bit"
    assert config.compute_dtype == "bfloat16"
    assert config.sequence_length == 2048
    assert config.micro_batch_size == 1
    assert config.gradient_accumulation_steps == 8
    assert config.micro_batch_size * config.gradient_accumulation_steps == 8
    assert config.epochs == 1.0
    assert config.learning_rate == 2e-4
    assert config.scheduler == "cosine"
    assert config.warmup_ratio == 0.03
    assert config.gradient_checkpointing is True
    assert config.loss_mode == "assistant_only"

    assert config.lora.rank == profile.measurement_rank == 16
    assert config.lora.alpha == 32
    assert config.lora.dropout == 0.05
    assert config.lora.bias == "none"
    assert config.lora.target_strategy == "selective"
    assert config.lora.target_modules == profile.target_modules
    assert profile.measured_trainable_parameters == 32_464_896

    assert config.quantization is not None
    assert config.quantization.bits == 4
    assert config.quantization.quant_type == "nf4"
    assert config.quantization.double_quant is True
    assert config.quantization.compute_dtype == "bfloat16"


def test_python_p0_training_config_is_captured_by_resolved_run_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(monkeypatch)

    payload = plan.resolved_config_payload()
    config_payload = payload["config"]

    assert isinstance(config_payload, dict)
    assert payload["source_config"] == "configs/train/python/p0.yaml"
    assert len(plan.config_sha256) == 64
    assert config_payload["training_mode"] == "qlora_4bit"
    assert config_payload["micro_batch_size"] == 1
    assert config_payload["gradient_accumulation_steps"] == 8
    assert config_payload["loss_mode"] == "assistant_only"
    lora_payload = config_payload["lora"]
    assert isinstance(lora_payload, dict)
    assert (
        lora_payload["target_modules"]
        == load_frozen_selective_lora_target_profile().target_modules
    )
