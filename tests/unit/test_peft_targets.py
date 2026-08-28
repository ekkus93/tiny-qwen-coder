from __future__ import annotations

import json
from typing import Literal, TypeAlias

import pytest

from tiny_qwen_coder.adapters import (
    TargetDiscoveryError,
    discover_peft_targets,
    peft_target_report_json,
    peft_target_report_text,
)
from tiny_qwen_coder.model import InspectionTarget, LinearModuleRecord

_Component: TypeAlias = Literal[
    "text_backbone",
    "vision_encoder",
    "multimodal_projector",
    "other",
]


def _target() -> InspectionTarget:
    return InspectionTarget(
        config_id="qwen35-4b",
        model_repository="Qwen/Qwen3.5-4B",
        model_revision="a" * 40,
        tokenizer_repository="Qwen/Qwen3.5-4B",
        tokenizer_revision="a" * 40,
        model_load_dtype="bfloat16",
    )


def _linear(
    name: str,
    component: _Component = "text_backbone",
    *,
    in_features: int = 2560,
    out_features: int = 2560,
) -> LinearModuleRecord:
    return LinearModuleRecord(
        name=name,
        component=component,
        class_name="torch.nn.modules.linear.Linear",
        in_features=in_features,
        out_features=out_features,
        has_bias=False,
    )


def _observed_fixture() -> tuple[LinearModuleRecord, ...]:
    modules: list[LinearModuleRecord] = []
    for leaf_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        modules.append(_linear(f"model.language_model.layers.3.self_attn.{leaf_name}"))
    for leaf_name in ("gate_proj", "up_proj", "down_proj"):
        modules.append(_linear(f"model.language_model.layers.0.mlp.{leaf_name}"))
    for leaf_name in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"):
        modules.append(_linear(f"model.language_model.layers.0.linear_attn.{leaf_name}"))
    modules.extend(
        [
            _linear("lm_head"),
            _linear("model.visual.blocks.0.attn.qkv", "vision_encoder"),
            _linear("model.visual.merger.linear_fc1", "multimodal_projector"),
        ]
    )
    return tuple(modules)


def test_discovers_observed_attention_mlp_and_deltanet_targets() -> None:
    report = discover_peft_targets(_observed_fixture(), _target())
    summaries = {summary.category: summary for summary in report.categories}

    assert summaries["full_attention"].leaf_names == ("k_proj", "o_proj", "q_proj", "v_proj")
    assert summaries["full_attention"].module_count == 4
    assert summaries["mlp"].leaf_names == ("down_proj", "gate_proj", "up_proj")
    assert summaries["mlp"].module_count == 3
    assert summaries["gated_deltanet"].leaf_names == (
        "in_proj_a",
        "in_proj_b",
        "in_proj_qkv",
        "in_proj_z",
        "out_proj",
    )
    assert summaries["gated_deltanet"].module_count == 5
    assert report.selective_matched_module_count == 12
    assert report.unclassified_text_module_count == 0


def test_selective_candidate_is_derived_from_observed_selected_leaf_names() -> None:
    modules = tuple(
        module
        for module in _observed_fixture()
        if not module.name.endswith(".in_proj_b") and not module.name.endswith(".v_proj")
    )

    report = discover_peft_targets(modules, _target())

    assert "in_proj_b" not in report.selective_target_modules
    assert "v_proj" not in report.selective_target_modules
    assert report.selective_target_modules == tuple(
        sorted({module.leaf_name for module in report.modules if module.selected_by_default})
    )


def test_vision_projector_and_output_head_are_excluded_by_default() -> None:
    report = discover_peft_targets(_observed_fixture(), _target())
    records = {module.name: module for module in report.modules}

    assert records["lm_head"].category == "language_output"
    assert records["lm_head"].selected_by_default is False
    assert records["model.visual.blocks.0.attn.qkv"].category == "vision_encoder"
    assert records["model.visual.blocks.0.attn.qkv"].selected_by_default is False
    assert records["model.visual.merger.linear_fc1"].category == "multimodal_projector"
    assert records["model.visual.merger.linear_fc1"].selected_by_default is False
    assert report.excluded_module_count == 3


def test_unclassified_text_linear_fails_closed() -> None:
    modules = _observed_fixture() + (_linear("model.language_model.layers.0.new_projection"),)

    with pytest.raises(TargetDiscoveryError, match="new_projection"):
        discover_peft_targets(modules, _target())


def test_incomplete_discovery_can_be_reported_without_freezing_candidate() -> None:
    modules = _observed_fixture() + (_linear("model.language_model.layers.0.new_projection"),)

    report = discover_peft_targets(modules, _target(), require_complete=False)

    assert report.unclassified_text_module_count == 1
    unknown = [module for module in report.modules if module.category == "unclassified_text"]
    assert [module.name for module in unknown] == ["model.language_model.layers.0.new_projection"]
    assert unknown[0].selected_by_default is False


def test_text_and_json_reports_share_candidate_and_revision() -> None:
    report = discover_peft_targets(_observed_fixture(), _target())

    payload = json.loads(peft_target_report_json(report))
    text = peft_target_report_text(report)

    assert payload["model_revision"] == "a" * 40
    assert payload["selective_target_modules"] == list(report.selective_target_modules)
    assert "Qwen/Qwen3.5-4B@" + "a" * 40 in text
    assert "Selective PEFT target module names" in text
    assert "gated_deltanet: 5 modules" in text
