from __future__ import annotations

import json

import pytest

from tiny_qwen_coder.model import InspectionTarget
from tiny_qwen_coder.model.smoke_test import (
    ModelSmokeTestError,
    ModelSmokeTestReport,
    SmokeMemoryReport,
    model_smoke_report_json,
    model_smoke_report_text,
)

REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def _target() -> InspectionTarget:
    return InspectionTarget(
        config_id="qwen35-4b",
        model_repository="Qwen/Qwen3.5-4B",
        model_revision=REVISION,
        tokenizer_repository="Qwen/Qwen3.5-4B",
        tokenizer_revision=REVISION,
        model_load_dtype="bfloat16",
    )


def _memory() -> SmokeMemoryReport:
    gib = 1024**3
    return SmokeMemoryReport(
        cuda_total_bytes=16 * gib,
        cuda_free_before_load_bytes=15 * gib,
        cuda_free_after_load_bytes=6 * gib,
        torch_allocated_after_load_bytes=9 * gib,
        torch_reserved_after_load_bytes=9 * gib + 128 * 1024**2,
        load_peak_allocated_bytes=9 * gib + 64 * 1024**2,
        load_peak_reserved_bytes=9 * gib + 128 * 1024**2,
        torch_allocated_before_generation_bytes=9 * gib + 16 * 1024**2,
        torch_reserved_before_generation_bytes=9 * gib + 128 * 1024**2,
        generation_peak_allocated_bytes=9 * gib + 256 * 1024**2,
        generation_peak_reserved_bytes=9 * gib + 512 * 1024**2,
    )


def _report(**overrides: object) -> ModelSmokeTestReport:
    values: dict[str, object] = {
        "schema_version": 1,
        "target": _target(),
        "torch_version": "2.13.0",
        "transformers_version": "5.16.1",
        "model_class": "transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForConditionalGeneration",
        "resolved_model_revision": REVISION,
        "parameter_dtypes": ("torch.bfloat16",),
        "device": "cuda:0",
        "gpu_name": "NVIDIA GeForce RTX 4070 Ti SUPER",
        "gpu_compute_capability": "8.9",
        "seed": 0,
        "system_prompt": "You are concise.",
        "user_prompt": "Reply with OK.",
        "max_new_tokens": 16,
        "input_token_count": 12,
        "generated_token_count": 2,
        "generated_token_ids": (100, 101),
        "generated_text": "OK<|im_end|>",
        "deterministic_repeat": True,
        "memory": _memory(),
    }
    values.update(overrides)
    return ModelSmokeTestReport(**values)  # type: ignore[arg-type]


def test_smoke_report_serializes_deterministically() -> None:
    report = _report()

    first = model_smoke_report_json(report)
    second = model_smoke_report_json(report)

    assert first == second
    payload = json.loads(first)
    assert payload["target"]["model_repository"] == "Qwen/Qwen3.5-4B"
    assert payload["resolved_model_revision"] == REVISION
    assert payload["parameter_dtypes"] == ["torch.bfloat16"]
    assert payload["deterministic_repeat"] is True


def test_smoke_report_text_contains_memory_and_generation_evidence() -> None:
    rendered = model_smoke_report_text(_report())

    assert "Qwen/Qwen3.5-4B" in rendered
    assert "torch.bfloat16" in rendered
    assert "Deterministic repeat: True" in rendered
    assert "PyTorch allocated after load:" in rendered
    assert "Generation peak allocated:" in rendered


def test_smoke_report_rejects_non_bf16_parameters() -> None:
    with pytest.raises(ModelSmokeTestError, match="floating model parameter"):
        _report(parameter_dtypes=("torch.bfloat16", "torch.float32"))


def test_smoke_report_rejects_nondeterministic_repeat() -> None:
    with pytest.raises(ModelSmokeTestError, match="not deterministic"):
        _report(deterministic_repeat=False)


def test_smoke_memory_rejects_peak_below_baseline() -> None:
    gib = 1024**3
    with pytest.raises(ModelSmokeTestError, match="generation peak allocated"):
        SmokeMemoryReport(
            cuda_total_bytes=16 * gib,
            cuda_free_before_load_bytes=15 * gib,
            cuda_free_after_load_bytes=6 * gib,
            torch_allocated_after_load_bytes=9 * gib,
            torch_reserved_after_load_bytes=10 * gib,
            load_peak_allocated_bytes=9 * gib,
            load_peak_reserved_bytes=10 * gib,
            torch_allocated_before_generation_bytes=9 * gib,
            torch_reserved_before_generation_bytes=10 * gib,
            generation_peak_allocated_bytes=8 * gib,
            generation_peak_reserved_bytes=10 * gib,
        )
