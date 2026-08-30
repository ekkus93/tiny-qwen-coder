"""P7-001 tests for the frozen selective language-LoRA target profile."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypeAlias

import pytest

from tiny_qwen_coder.adapters import (
    SelectiveLoraTargetProfileError,
    discover_peft_targets,
    load_frozen_selective_lora_target_profile,
    load_selective_lora_target_profile,
    require_measured_trainable_parameters,
    require_profile_matches_discovery,
)
from tiny_qwen_coder.model import InspectionTarget, LinearModuleRecord

_Component: TypeAlias = Literal[
    "text_backbone",
    "vision_encoder",
    "multimodal_projector",
    "other",
]
_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
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


def _target() -> InspectionTarget:
    return InspectionTarget(
        config_id="qwen35-4b",
        model_repository="Qwen/Qwen3.5-4B",
        model_revision=_REVISION,
        tokenizer_repository="Qwen/Qwen3.5-4B",
        tokenizer_revision=_REVISION,
        model_load_dtype="bfloat16",
    )


def _linear(
    name: str,
    component: _Component = "text_backbone",
) -> LinearModuleRecord:
    return LinearModuleRecord(
        name=name,
        component=component,
        class_name="torch.nn.modules.linear.Linear",
        in_features=2560,
        out_features=2560,
        has_bias=False,
    )


def _observed_architecture() -> tuple[LinearModuleRecord, ...]:
    modules: list[LinearModuleRecord] = []
    for leaf_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        modules.append(_linear(f"model.language_model.layers.3.self_attn.{leaf_name}"))
    for leaf_name in ("gate_proj", "up_proj", "down_proj"):
        modules.append(_linear(f"model.language_model.layers.0.mlp.{leaf_name}"))
    for leaf_name in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"):
        modules.append(_linear(f"model.language_model.layers.0.linear_attn.{leaf_name}"))
    modules.extend(
        (
            _linear("lm_head"),
            _linear("model.visual.blocks.0.attn.qkv", "vision_encoder"),
            _linear("model.visual.merger.linear_fc1", "multimodal_projector"),
        )
    )
    return tuple(modules)


def test_frozen_profile_records_exact_architecture_and_p2_measurement() -> None:
    profile = load_frozen_selective_lora_target_profile()

    assert profile.profile_id == "qwen35-4b-selective-lora-v1"
    assert profile.base_repository == "Qwen/Qwen3.5-4B"
    assert profile.base_revision == _REVISION
    assert profile.strategy == "selective"
    assert profile.target_modules == _TARGET_MODULES
    assert profile.measurement_source_task == "P2-008"
    assert profile.measurement_rank == 16
    assert profile.measured_trainable_parameters == 32_464_896
    assert (
        profile.source_sha256
        == "edc61481737903c729eb6671bee846879004b91ba0644175beb9fe5e0be05dc6"
    )


def test_frozen_profile_matches_p2_architecture_discovery() -> None:
    profile = load_frozen_selective_lora_target_profile()
    discovery = discover_peft_targets(_observed_architecture(), _target())

    require_profile_matches_discovery(profile, discovery)


def test_architecture_target_drift_fails_closed() -> None:
    profile = load_frozen_selective_lora_target_profile()
    drifted = tuple(
        module for module in _observed_architecture() if not module.name.endswith(".in_proj_b")
    )
    discovery = discover_peft_targets(drifted, _target())

    with pytest.raises(SelectiveLoraTargetProfileError, match="selective target modules"):
        require_profile_matches_discovery(profile, discovery)


def test_measured_rank_and_trainable_parameter_count_are_frozen() -> None:
    profile = load_frozen_selective_lora_target_profile()

    require_measured_trainable_parameters(
        profile,
        rank=16,
        trainable_parameters=32_464_896,
    )
    with pytest.raises(SelectiveLoraTargetProfileError, match="LoRA rank"):
        require_measured_trainable_parameters(
            profile,
            rank=8,
            trainable_parameters=32_464_896,
        )
    with pytest.raises(SelectiveLoraTargetProfileError, match="trainable parameter count"):
        require_measured_trainable_parameters(
            profile,
            rank=16,
            trainable_parameters=32_000_000,
        )


def test_frozen_profile_rejects_unreviewed_file_drift(tmp_path: Path) -> None:
    source = Path("configs/base/qwen35-4b-selective-lora-v1.yaml").read_text(encoding="utf-8")
    drifted = tmp_path / "drifted.yaml"
    drifted.write_text(source.replace("rank: 16", "rank: 8"), encoding="utf-8")

    with pytest.raises(SelectiveLoraTargetProfileError, match="fingerprint mismatch"):
        load_frozen_selective_lora_target_profile(drifted)


def test_generic_loader_rejects_target_group_mismatch(tmp_path: Path) -> None:
    source = Path("configs/base/qwen35-4b-selective-lora-v1.yaml").read_text(encoding="utf-8")
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(
        source.replace("  - v_proj\n\nmeasurement:", "  - v_proj\n  - extra_proj\n\nmeasurement:"),
        encoding="utf-8",
    )

    with pytest.raises(SelectiveLoraTargetProfileError, match="union"):
        load_selective_lora_target_profile(invalid)
