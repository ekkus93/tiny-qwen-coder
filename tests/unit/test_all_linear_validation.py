from __future__ import annotations

import json

import pytest

from tiny_qwen_coder.adapters import (
    AllLinearValidationError,
    all_linear_report_json,
    all_linear_report_text,
    build_all_linear_validation_report,
)
from tiny_qwen_coder.adapters.targets import (
    PeftTargetDiscoveryReport,
    TargetCategory,
    TargetCategorySummary,
    TargetModuleRecord,
)
from tiny_qwen_coder.model import InspectionTarget


def _target() -> InspectionTarget:
    return InspectionTarget(
        config_id="qwen35-4b",
        model_repository="Qwen/Qwen3.5-4B",
        model_revision="a" * 40,
        tokenizer_repository="Qwen/Qwen3.5-4B",
        tokenizer_revision="a" * 40,
        model_load_dtype="bfloat16",
    )


def _module(
    name: str,
    *,
    leaf_name: str,
    category: TargetCategory,
    component: str,
    selected: bool,
    in_features: int = 4,
    out_features: int = 6,
) -> TargetModuleRecord:
    return TargetModuleRecord(
        name=name,
        leaf_name=leaf_name,
        category=category,
        component=component,
        in_features=in_features,
        out_features=out_features,
        has_bias=False,
        selected_by_default=selected,
    )


def _discovery() -> PeftTargetDiscoveryReport:
    modules = (
        _module(
            "lm_head",
            leaf_name="lm_head",
            category="language_output",
            component="text_backbone",
            selected=False,
            out_features=8,
        ),
        _module(
            "model.language_model.layers.0.mlp.gate_proj",
            leaf_name="gate_proj",
            category="mlp",
            component="text_backbone",
            selected=True,
        ),
        _module(
            "model.language_model.layers.0.self_attn.q_proj",
            leaf_name="q_proj",
            category="full_attention",
            component="text_backbone",
            selected=True,
        ),
        _module(
            "model.visual.blocks.0.attn.qkv",
            leaf_name="qkv",
            category="vision_encoder",
            component="vision_encoder",
            selected=False,
        ),
        _module(
            "model.visual.merger.linear_fc1",
            leaf_name="linear_fc1",
            category="multimodal_projector",
            component="multimodal_projector",
            selected=False,
        ),
    )
    categories = (
        TargetCategorySummary("full_attention", 1, ("q_proj",), True),
        TargetCategorySummary("mlp", 1, ("gate_proj",), True),
        TargetCategorySummary("gated_deltanet", 0, (), True),
        TargetCategorySummary("language_output", 1, ("lm_head",), False),
        TargetCategorySummary("vision_encoder", 1, ("qkv",), False),
        TargetCategorySummary("multimodal_projector", 1, ("linear_fc1",), False),
        TargetCategorySummary("unclassified_text", 0, (), False),
        TargetCategorySummary("other", 0, (), False),
    )
    return PeftTargetDiscoveryReport(
        schema_version=1,
        model_repository="Qwen/Qwen3.5-4B",
        model_revision="a" * 40,
        linear_module_count=len(modules),
        selective_matched_module_count=2,
        excluded_module_count=3,
        unclassified_text_module_count=0,
        selective_target_modules=("gate_proj", "q_proj"),
        categories=categories,
        modules=modules,
    )


def test_literal_all_linear_report_exposes_multimodal_scope_difference() -> None:
    discovery = _discovery()
    matched = tuple(module.name for module in discovery.modules if module.name != "lm_head")

    report = build_all_linear_validation_report(
        discovery,
        _target(),
        matched_module_names=matched,
        trainable_parameter_count=999,
        peft_version="0.20.0",
        rank=16,
    )

    assert report.matched_module_count == 4
    assert report.selective_overlap_count == 2
    assert report.extra_vs_selective_count == 2
    assert report.missing_vs_selective_count == 0
    assert report.literal_all_linear_is_language_only is False
    assert report.language_scoped_all_linear_is_selective_equivalent is True
    assert report.selective_expected_trainable_parameter_count == 2 * 16 * (4 + 6)
    by_category = {item.category: item.matched_module_count for item in report.categories}
    assert by_category["full_attention"] == 1
    assert by_category["mlp"] == 1
    assert by_category["vision_encoder"] == 1
    assert by_category["multimodal_projector"] == 1
    assert by_category["language_output"] == 0


def test_missing_selective_module_fails_closed() -> None:
    discovery = _discovery()

    with pytest.raises(AllLinearValidationError, match="unexpectedly missed selective"):
        build_all_linear_validation_report(
            discovery,
            _target(),
            matched_module_names=("model.language_model.layers.0.self_attn.q_proj",),
            trainable_parameter_count=100,
            peft_version="0.20.0",
        )


def test_unknown_peft_match_fails_closed() -> None:
    discovery = _discovery()
    matched = [module.name for module in discovery.modules if module.selected_by_default]
    matched.append("model.unknown.proj")

    with pytest.raises(AllLinearValidationError, match="absent from the inspected"):
        build_all_linear_validation_report(
            discovery,
            _target(),
            matched_module_names=matched,
            trainable_parameter_count=100,
            peft_version="0.20.0",
        )


def test_json_and_text_reports_are_deterministic_and_comparable() -> None:
    discovery = _discovery()
    matched = tuple(module.name for module in discovery.modules if module.name != "lm_head")
    report = build_all_linear_validation_report(
        discovery,
        _target(),
        matched_module_names=matched,
        trainable_parameter_count=999,
        peft_version="0.20.0",
    )

    payload = json.loads(all_linear_report_json(report))
    text = all_linear_report_text(report)

    assert payload["target_modules"] == "all-linear"
    assert payload["extra_vs_selective_count"] == 2
    assert "Literal all-linear language-only safe: False" in text
    assert "Language-scoped all-linear equals selective candidate: True" in text
