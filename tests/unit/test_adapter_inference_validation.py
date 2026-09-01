"""CPU-side contract tests for P7-007 adapter inference validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from torch import nn

from tiny_qwen_coder.identities import BaseModelIdentity
from tiny_qwen_coder.runtime.adapter_validation import (
    AdapterInferenceValidationError,
    GenerationObservation,
    _freeze_inference_parameters,
    validate_adapter_artifacts,
    validate_generation_recovery,
)

_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
_TRAINING_SHA = "02df92a9c2d347b9fb013dc25714fe066c6bcafe"
_CONFIG_SHA = "4b0c742ad3a55f4eaffd4f2283be7291d6434eb89b07c13dc90c2166238a5f46"


def _base() -> BaseModelIdentity:
    return BaseModelIdentity(
        repository="Qwen/Qwen3.5-4B",
        revision=_REVISION,
        tokenizer_repository="Qwen/Qwen3.5-4B",
        tokenizer_revision=_REVISION,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_training_output(root: Path) -> Path:
    adapter = root / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"synthetic-lora-weights")
    (adapter / "chat_template.jinja").write_text("synthetic-chat-template\n", encoding="utf-8")
    template_sha = _sha256(adapter / "chat_template.jinja")
    adapter_config = {
        "base_model_name_or_path": "Qwen/Qwen3.5-4B",
        "bias": "none",
        "inference_mode": True,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "peft_type": "LORA",
        "peft_version": "0.20.0",
        "r": 16,
        "target_modules": ["gate_proj", "q_proj"],
        "task_type": "CAUSAL_LM",
    }
    _write_json(adapter / "adapter_config.json", adapter_config)

    manifest = {
        "schema_version": 1,
        "adapter_id": "language/python/p0",
        "family": "language",
        "language": "python",
        "created_at_utc": "2026-08-31T18:09:16.446466Z",
        "base_model": {"repository": "Qwen/Qwen3.5-4B", "revision": _REVISION},
        "tokenizer": {
            "repository": "Qwen/Qwen3.5-4B",
            "revision": _REVISION,
            "chat_template": {"identifier": "trl_training_template", "sha256": "b" * 64},
        },
        "training": {
            "run_id": "training-python-test",
            "git_sha": _TRAINING_SHA,
            "config_sha256": _CONFIG_SHA,
            "seed": 1729,
            "transformers_version": "5.16.1",
            "peft_version": "0.20.0",
        },
        "datasets": [{"manifest_id": "dataset/python/p0", "sha256": "c" * 64}],
        "lora": {
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
            "bias": "none",
            "target_strategy": "selective",
            "target_modules": [
                "model.language_model.layers.0.mlp.gate_proj",
                "model.language_model.layers.0.self_attn.q_proj",
            ],
            "trainable_parameters": 1024,
        },
        "training_summary": {
            "precision": "bfloat16",
            "sequence_length": 2048,
            "optimizer": {"name": "adamw_torch", "settings": []},
            "scheduler": {"name": "cosine", "settings": []},
            "steps": 4750,
            "epochs": 1.0,
            "peak_vram_bytes": 1,
        },
        "validation_metrics": [],
        "evaluation_artifacts": [],
    }
    _write_json(root / "adapter-manifest.json", manifest)

    training_config = {
        "schema_version": 1,
        "source_config": "configs/train/python/p0.yaml",
        "config_sha256": _CONFIG_SHA,
        "base": {
            "model_repository": "Qwen/Qwen3.5-4B",
            "model_revision": _REVISION,
            "tokenizer_repository": "Qwen/Qwen3.5-4B",
            "tokenizer_revision": _REVISION,
        },
        "dataset": {"chat_template_sha256": template_sha},
    }
    _write_json(root / "training-config.json", training_config)

    run_manifest = {
        "schema_version": 1,
        "run_id": "training-python-test",
        "run_kind": "training",
        "git": {"sha": _TRAINING_SHA, "dirty": False},
        "base_model": {
            "repository": "Qwen/Qwen3.5-4B",
            "revision": _REVISION,
            "tokenizer_repository": "Qwen/Qwen3.5-4B",
            "tokenizer_revision": _REVISION,
        },
        "language": "python",
        "adapter": {"family": "language", "adapter_id": "language/python/p0"},
    }
    _write_json(root / "run-manifest.json", run_manifest)

    weights = adapter / "adapter_model.safetensors"
    config = adapter / "adapter_config.json"
    training_report = {
        "adapter_id": "language/python/p0",
        "language": "python",
        "global_steps": 4750,
        "source_training_config": "configs/train/python/p0.yaml",
        "source_training_config_sha256": _CONFIG_SHA,
        "persisted_artifacts": [
            {
                "path": "adapter/adapter_model.safetensors",
                "size_bytes": weights.stat().st_size,
                "sha256": _sha256(weights),
            },
            {
                "path": "adapter/adapter_config.json",
                "size_bytes": config.stat().st_size,
                "sha256": _sha256(config),
            },
        ],
    }
    _write_json(root / "training-report.json", training_report)
    return root


def _observation(text: str, token_ids: tuple[int, ...]) -> GenerationObservation:
    return GenerationObservation(
        text=text,
        token_ids=token_ids,
        prompt_tokens=10,
        generated_tokens=len(token_ids),
        latency_seconds=0.5,
    )


def test_validated_artifacts_bind_exact_p0_identity_and_weight_hash(tmp_path: Path) -> None:
    root = _write_training_output(tmp_path / "p0")

    verified = validate_adapter_artifacts(root, _base())

    assert verified.manifest.adapter_id == "language/python/p0"
    assert verified.training_git_sha == _TRAINING_SHA
    assert verified.adapter_model_sha256 == _sha256(root / "adapter" / "adapter_model.safetensors")
    assert verified.adapter_model_size_bytes > 0


def test_tampered_adapter_weights_fail_closed_before_model_load(tmp_path: Path) -> None:
    root = _write_training_output(tmp_path / "p0")
    (root / "adapter" / "adapter_model.safetensors").write_bytes(b"tampered")

    with pytest.raises(AdapterInferenceValidationError, match="persisted adapter weight"):
        validate_adapter_artifacts(root, _base())


def test_base_revision_mismatch_fails_closed_before_model_load(tmp_path: Path) -> None:
    root = _write_training_output(tmp_path / "p0")
    incompatible = BaseModelIdentity(
        repository="Qwen/Qwen3.5-4B",
        revision="a" * 40,
        tokenizer_repository="Qwen/Qwen3.5-4B",
        tokenizer_revision="a" * 40,
    )

    with pytest.raises(AdapterInferenceValidationError, match="base_model.revision mismatch"):
        validate_adapter_artifacts(root, incompatible)


def test_peft_config_mismatch_fails_closed(tmp_path: Path) -> None:
    root = _write_training_output(tmp_path / "p0")
    config_path = root / "adapter" / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["r"] = 8
    _write_json(config_path, config)
    report_path = root / "training-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for item in report["persisted_artifacts"]:
        if item["path"] == "adapter/adapter_config.json":
            item["size_bytes"] = config_path.stat().st_size
            item["sha256"] = _sha256(config_path)
    _write_json(report_path, report)

    with pytest.raises(AdapterInferenceValidationError, match="adapter_config.r mismatch"):
        validate_adapter_artifacts(root, _base())


def test_merged_or_full_model_weights_are_rejected(tmp_path: Path) -> None:
    root = _write_training_output(tmp_path / "p0")
    (root / "adapter" / "model.safetensors").write_bytes(b"forbidden")

    with pytest.raises(AdapterInferenceValidationError, match="forbidden merged/full-model"):
        validate_adapter_artifacts(root, _base())


def test_inference_freeze_is_reapplied_after_adapter_reactivation() -> None:
    model = nn.Linear(2, 2)
    assert any(parameter.requires_grad for parameter in model.parameters())

    _freeze_inference_parameters(model)
    assert all(not parameter.requires_grad for parameter in model.parameters())

    for parameter in model.parameters():
        parameter.requires_grad_(True)
    assert any(parameter.requires_grad for parameter in model.parameters())

    _freeze_inference_parameters(model)
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_disable_and_reenable_must_recover_exact_deterministic_outputs() -> None:
    base = _observation("base", (1, 2))
    adapted = _observation("adapted", (3, 4))

    report = validate_generation_recovery(
        prompt_id="smoke",
        prompt="prompt",
        base=base,
        adapter_enabled=adapted,
        adapter_disabled=_observation("base", (1, 2)),
        adapter_reenabled=_observation("adapted", (3, 4)),
    )

    assert report.base_recovered_exactly is True
    assert report.adapter_reenabled_exactly is True
    assert report.adapter_changed_output is True


def test_disable_mismatch_is_not_silently_accepted() -> None:
    with pytest.raises(AdapterInferenceValidationError, match="recover exact base output"):
        validate_generation_recovery(
            prompt_id="smoke",
            prompt="prompt",
            base=_observation("base", (1, 2)),
            adapter_enabled=_observation("adapted", (3, 4)),
            adapter_disabled=_observation("almost-base", (1, 5)),
            adapter_reenabled=_observation("adapted", (3, 4)),
        )


def test_reenable_mismatch_is_not_silently_accepted() -> None:
    with pytest.raises(AdapterInferenceValidationError, match="recover exact adapted output"):
        validate_generation_recovery(
            prompt_id="smoke",
            prompt="prompt",
            base=_observation("base", (1, 2)),
            adapter_enabled=_observation("adapted", (3, 4)),
            adapter_disabled=_observation("base", (1, 2)),
            adapter_reenabled=_observation("changed", (3, 5)),
        )
