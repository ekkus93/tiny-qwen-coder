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


def train_adapter() -> NoReturn:
    """Run generic adapter training once P7-002 implements it."""
    raise SystemExit("Adapter training is scaffolded; implementation is tracked by P7-002.")


__all__ = [
    "ChatLossMaskReport",
    "ChatMessage",
    "LossMaskSpan",
    "LossMaskToken",
    "LossMaskingError",
    "TokenizedLossExample",
    "build_chat_loss_mask_report",
    "canonical_probe_messages",
    "completion_only_fallback",
    "loss_mask_report_json",
    "loss_mask_report_text",
    "train_adapter",
    "validate_loss_masking_main",
]
