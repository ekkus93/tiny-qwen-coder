"""Frozen base/LoRA architecture checks for adapter-training preflight."""

from __future__ import annotations

from dataclasses import dataclass

from tiny_qwen_coder.adapters import load_frozen_selective_lora_target_profile
from tiny_qwen_coder.training.plan import AdapterTrainingError, AdapterTrainingPlan


@dataclass(frozen=True, slots=True)
class TargetPreflightEvidence:
    """Revision-bound proof for the selected LoRA architecture."""

    base_repository: str
    base_revision: str
    profile_id: str
    profile_sha256: str
    rank: int
    target_modules: tuple[str, ...]
    measured_rank: int
    measured_trainable_parameters: int


def verify_frozen_lora_targets(plan: AdapterTrainingPlan) -> TargetPreflightEvidence:
    """Require the exact P7-001 target profile for the immutable canonical base."""

    profile = load_frozen_selective_lora_target_profile()
    if plan.target.model_repository != profile.base_repository:
        raise AdapterTrainingError("training base repository does not match frozen LoRA profile")
    if plan.target.model_revision != profile.base_revision:
        raise AdapterTrainingError("training base revision does not match frozen LoRA profile")
    if plan.config.lora.target_strategy != "selective":
        raise AdapterTrainingError("canonical training preflight requires selective LoRA targeting")
    if plan.config.lora.target_modules != profile.target_modules:
        raise AdapterTrainingError("configured LoRA targets do not match frozen P7-001 profile")
    if plan.config.lora.rank != profile.measurement_rank:
        raise AdapterTrainingError(
            f"configured LoRA rank {plan.config.lora.rank} does not match measured "
            f"P2-008/P7-001 rank {profile.measurement_rank}"
        )
    return TargetPreflightEvidence(
        base_repository=profile.base_repository,
        base_revision=profile.base_revision,
        profile_id=profile.profile_id,
        profile_sha256=profile.source_sha256,
        rank=plan.config.lora.rank,
        target_modules=profile.target_modules,
        measured_rank=profile.measurement_rank,
        measured_trainable_parameters=profile.measured_trainable_parameters,
    )
