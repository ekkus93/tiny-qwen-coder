"""Chat-loss masking checks for adapter-training preflight."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, cast

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from tiny_qwen_coder.training.loss_masking import (
    build_chat_loss_mask_report,
    canonical_probe_messages,
    completion_only_fallback,
)
from tiny_qwen_coder.training.plan import AdapterTrainingError, AdapterTrainingPlan


@dataclass(frozen=True, slots=True)
class LossPreflightEvidence:
    """Token-level proof that configured SFT loss excludes prompt tokens."""

    loss_mode: str
    strategy: str
    checkpoint_chat_template_sha256: str
    loss_token_count: int
    ignored_token_count: int


def _load_tokenizer(plan: AdapterTrainingPlan) -> PreTrainedTokenizerBase:
    factory = cast(Any, AutoTokenizer)
    tokenizer: object = factory.from_pretrained(
        plan.target.tokenizer_repository,
        revision=plan.target.tokenizer_revision,
    )
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise AdapterTrainingError("canonical tokenizer factory returned an unsupported tokenizer")
    return tokenizer


def _checkpoint_template_sha256(tokenizer: PreTrainedTokenizerBase) -> str:
    template = tokenizer.chat_template
    if not isinstance(template, str) or not template:
        raise AdapterTrainingError("canonical tokenizer does not expose a chat template")
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def verify_training_loss_mask(
    plan: AdapterTrainingPlan,
    *,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> LossPreflightEvidence:
    """Prove the configured assistant/completion loss boundary before model loading."""

    selected = tokenizer or _load_tokenizer(plan)
    template_sha = _checkpoint_template_sha256(selected)
    if template_sha != plan.dataset.chat_template_sha256:
        raise AdapterTrainingError(
            "canonical tokenizer chat template does not match frozen dataset manifest"
        )

    messages = canonical_probe_messages()
    if plan.config.loss_mode == "assistant_only":
        report = build_chat_loss_mask_report(selected, plan.target, messages)
        if report.strategy != "trl_assistant_only":
            raise AdapterTrainingError(
                "assistant-only training requires a proven TRL assistant-only generation mask"
            )
        return LossPreflightEvidence(
            loss_mode=plan.config.loss_mode,
            strategy=report.strategy,
            checkpoint_chat_template_sha256=report.checkpoint_chat_template_sha256,
            loss_token_count=report.loss_token_count,
            ignored_token_count=report.ignored_token_count,
        )

    example = completion_only_fallback(selected, messages)
    loss_count = sum(example.loss_mask)
    return LossPreflightEvidence(
        loss_mode=plan.config.loss_mode,
        strategy=example.strategy,
        checkpoint_chat_template_sha256=template_sha,
        loss_token_count=loss_count,
        ignored_token_count=len(example.loss_mask) - loss_count,
    )
