from __future__ import annotations

import json

import pytest

from tiny_qwen_coder.model import InspectionTarget
from tiny_qwen_coder.training.memory_preflight import (
    CudaMemorySnapshot,
    QuantizationSpec,
    TrainingMemoryPreflightError,
    TrainingMemoryPreflightReport,
    required_safety_headroom_bytes,
    training_memory_report_json,
)

_TARGET_MODULES = (
    "down_proj",
    "gate_proj",
    "in_proj_a",
    "in_proj_b",
    "in_proj_qkv",
    "in_proj_z",
    "k_proj",
    "o_proj",
    "out_proj",
    "q_proj",
    "up_proj",
    "v_proj",
)
_TOTAL_BYTES = 16_688_218_112


def _target() -> InspectionTarget:
    return InspectionTarget(
        config_id="qwen35-4b",
        model_repository="Qwen/Qwen3.5-4B",
        model_revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        tokenizer_repository="Qwen/Qwen3.5-4B",
        tokenizer_revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        model_load_dtype="bfloat16",
    )


def _snapshot(
    *, allocated: int, reserved: int, peak_allocated: int, peak_reserved: int
) -> CudaMemorySnapshot:
    return CudaMemorySnapshot(
        allocated_bytes=allocated,
        reserved_bytes=reserved,
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
        free_bytes=max(0, _TOTAL_BYTES - reserved),
        total_bytes=_TOTAL_BYTES,
    )


def _report(
    *, comfortable: bool = True, headroom: int = 2_027_028_480
) -> TrainingMemoryPreflightReport:
    peak_allocated = 13_569_101_824
    peak_reserved = _TOTAL_BYTES - headroom
    attached = _snapshot(
        allocated=4_705_453_056,
        reserved=4_743_757_824,
        peak_allocated=4_705_453_056,
        peak_reserved=4_743_757_824,
    )
    forward = _snapshot(
        allocated=9_500_643_840,
        reserved=10_081_009_664,
        peak_allocated=9_521_615_360,
        peak_reserved=10_081_009_664,
    )
    backward = _snapshot(
        allocated=6_978_602_496,
        reserved=min(14_424_211_456, peak_reserved),
        peak_allocated=peak_allocated,
        peak_reserved=min(14_424_211_456, peak_reserved),
    )
    optimizer = _snapshot(
        allocated=7_108_462_080,
        reserved=peak_reserved,
        peak_allocated=peak_allocated,
        peak_reserved=peak_reserved,
    )
    return TrainingMemoryPreflightReport(
        schema_version=1,
        target=_target(),
        mode="qlora_4bit",
        quantization=QuantizationSpec(),
        sequence_length=2048,
        micro_batch_size=1,
        gradient_checkpointing=True,
        rank=16,
        alpha=32,
        dropout=0.05,
        target_modules=_TARGET_MODULES,
        trainable_parameters=32_464_896,
        four_bit_module_count=594,
        loss=13.949459075927734,
        forward_seconds=2.15,
        backward_seconds=2.80,
        optimizer_seconds=0.08,
        attached=attached,
        after_forward=forward,
        after_backward=backward,
        after_optimizer=optimizer,
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
        safety_headroom_bytes=headroom,
        required_headroom_bytes=required_safety_headroom_bytes(_TOTAL_BYTES),
        comfortable=comfortable,
        bitsandbytes_version="0.50.2",
        torch_version="2.13.0+cu130",
        gpu_name="NVIDIA GeForce RTX 4070 Ti SUPER",
        compute_capability="8.9",
    )


def test_reference_gpu_uses_ten_percent_headroom_threshold() -> None:
    assert required_safety_headroom_bytes(_TOTAL_BYTES) == 1_668_821_812


def test_measured_qlora_report_is_comfortable_and_deterministic() -> None:
    report = _report()
    assert report.comfortable
    assert report.safety_headroom_bytes > report.required_headroom_bytes
    assert json.loads(training_memory_report_json(report))["quantization"] == {
        "bits": 4,
        "compute_dtype": "bfloat16",
        "double_quant": True,
        "quant_type": "nf4",
    }
    assert training_memory_report_json(report) == training_memory_report_json(report)


def test_report_rejects_insufficient_headroom() -> None:
    with pytest.raises(TrainingMemoryPreflightError, match="lacks safe VRAM headroom"):
        _report(comfortable=False, headroom=1_000_000_000)


def test_quantization_spec_is_frozen_to_nf4_double_quant_bf16() -> None:
    with pytest.raises(TrainingMemoryPreflightError, match="must use NF4"):
        QuantizationSpec(quant_type="fp4")
    with pytest.raises(TrainingMemoryPreflightError, match="enable double quantization"):
        QuantizationSpec(double_quant=False)
    with pytest.raises(TrainingMemoryPreflightError, match="must be bfloat16"):
        QuantizationSpec(compute_dtype="float16")
