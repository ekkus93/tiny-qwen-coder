"""Language-neutral LoRA training services."""

from __future__ import annotations

from typing import NoReturn

from tiny_qwen_coder.training.loss_masking import (
    ChatLossMaskReport,
    ChatMessage,
    LossMaskingError,
    LossMaskSpan,
    LossMaskToken,
    TokenizedLossExample,
    build_chat_loss_mask_report,
    canonical_probe_messages,
    completion_only_fallback,
    loss_mask_report_json,
    loss_mask_report_text,
    validate_loss_masking_main,
)
from tiny_qwen_coder.training.memory_preflight import (
    CudaMemorySnapshot,
    QuantizationSpec,
    TrainingMemoryPreflightError,
    TrainingMemoryPreflightReport,
    required_safety_headroom_bytes,
    run_canonical_qlora_memory_preflight,
    training_memory_preflight_main,
    training_memory_report_json,
    training_memory_report_text,
)


def train_adapter() -> NoReturn:
    """Run generic adapter training once P7-002 implements it."""
    raise SystemExit("Adapter training is scaffolded; implementation is tracked by P7-002.")


__all__ = [
    "ChatLossMaskReport",
    "ChatMessage",
    "CudaMemorySnapshot",
    "LossMaskSpan",
    "LossMaskToken",
    "LossMaskingError",
    "QuantizationSpec",
    "TokenizedLossExample",
    "TrainingMemoryPreflightError",
    "TrainingMemoryPreflightReport",
    "build_chat_loss_mask_report",
    "canonical_probe_messages",
    "completion_only_fallback",
    "loss_mask_report_json",
    "loss_mask_report_text",
    "required_safety_headroom_bytes",
    "run_canonical_qlora_memory_preflight",
    "train_adapter",
    "training_memory_preflight_main",
    "training_memory_report_json",
    "training_memory_report_text",
    "validate_loss_masking_main",
]
