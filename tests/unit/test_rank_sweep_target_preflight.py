"""Regression coverage for P9-001 candidate ranks in training preflight."""

from __future__ import annotations

from pathlib import Path

import pytest

import tiny_qwen_coder.training.plan as training_plan
from tiny_qwen_coder.training.plan import TrainingDatasetIdentity
from tiny_qwen_coder.training.preflight_targets import verify_frozen_lora_targets

_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def _compatible_dataset() -> TrainingDatasetIdentity:
    return TrainingDatasetIdentity(
        manifest_id="dataset/python/p0",
        language="python",
        tokenizer_repository="Qwen/Qwen3.5-4B",
        tokenizer_revision=_REVISION,
        chat_template_sha256="f" * 64,
        sha256="a" * 64,
    )


@pytest.mark.parametrize(
    ("rank", "config_path"),
    (
        (8, Path("configs/train/python/p9_rank_r8.yaml")),
        (16, Path("configs/train/python/p0.yaml")),
        (32, Path("configs/train/python/p9_rank_r32.yaml")),
        (64, Path("configs/train/python/p9_rank_r64.yaml")),
    ),
)
def test_target_preflight_preserves_profile_while_allowing_candidate_rank(
    monkeypatch: pytest.MonkeyPatch,
    rank: int,
    config_path: Path,
) -> None:
    monkeypatch.setattr(
        training_plan,
        "load_training_dataset_identity",
        lambda _path: _compatible_dataset(),
    )
    plan = training_plan.resolve_adapter_training_plan(config_path)

    evidence = verify_frozen_lora_targets(plan)

    assert evidence.rank == rank
    assert evidence.measured_rank == 16
    assert evidence.measured_trainable_parameters == 32_464_896
    assert evidence.target_modules == plan.config.lora.target_modules
