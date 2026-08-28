"""Tests for fail-closed adapter/base compatibility validation."""

from __future__ import annotations

import json

import pytest

from tiny_qwen_coder.adapters import (
    AdapterCompatibilityError,
    AdapterCompatibilityTarget,
    AdapterManifest,
    CompatibleLinearModule,
    PeftTargetDiscoveryReport,
    TargetCategorySummary,
    TargetModuleRecord,
    build_compatibility_target,
    compatibility_report_json,
    parse_adapter_manifest,
    require_adapter_compatible,
    validate_adapter_compatibility,
)
from tiny_qwen_coder.model import (
    ComponentSummary,
    InspectionTarget,
    LinearModuleRecord,
    ModelInspectionReport,
    ModelMetadata,
    TokenizerMetadata,
)

_BASE_REVISION = "a" * 40
_TOKENIZER_REVISION = "a" * 40
_TEMPLATE_HASH = "b" * 64


def _manifest_mapping() -> dict[str, object]:
    return {
        "schema_version": 1,
        "adapter_id": "language/python/p0-test",
        "family": "language",
        "language": "python",
        "created_at_utc": "2026-08-28T10:30:00Z",
        "base_model": {
            "repository": "Qwen/Qwen3.5-4B",
            "revision": _BASE_REVISION,
        },
        "tokenizer": {
            "repository": "Qwen/Qwen3.5-4B",
            "revision": _TOKENIZER_REVISION,
            "chat_template": {
                "identifier": "qwen35-4b-pinned-checkpoint",
                "sha256": _TEMPLATE_HASH,
            },
        },
        "training": {
            "run_id": "training-python-test",
            "git_sha": "c" * 40,
            "config_sha256": "d" * 64,
            "seed": 42,
            "transformers_version": "5.16.1",
            "peft_version": "0.20.0",
        },
        "datasets": [{"manifest_id": "dataset/python/test", "sha256": "e" * 64}],
        "lora": {
            "rank": 2,
            "alpha": 4,
            "dropout": 0.05,
            "bias": "none",
            "target_strategy": "selective",
            "target_modules": [
                "model.language_model.layers.0.mlp.gate_proj",
                "model.language_model.layers.0.self_attn.q_proj",
            ],
            "trainable_parameters": 88,
        },
        "training_summary": {
            "precision": "bfloat16",
            "sequence_length": 2048,
            "optimizer": {"name": "adamw_torch", "settings": []},
            "scheduler": {"name": "cosine", "settings": []},
            "steps": 10,
            "epochs": 1.0,
            "peak_vram_bytes": 0,
        },
        "validation_metrics": [],
        "evaluation_artifacts": [],
    }


def _target(*, template_hash: str | None = _TEMPLATE_HASH) -> AdapterCompatibilityTarget:
    return AdapterCompatibilityTarget(
        base_repository="Qwen/Qwen3.5-4B",
        base_revision=_BASE_REVISION,
        tokenizer_repository="Qwen/Qwen3.5-4B",
        tokenizer_revision=_TOKENIZER_REVISION,
        chat_template_identifier="qwen35-4b-pinned-checkpoint",
        chat_template_sha256=template_hash,
        linear_modules=(
            CompatibleLinearModule(
                name="model.language_model.layers.0.mlp.gate_proj",
                in_features=8,
                out_features=12,
                has_bias=False,
                language_lora_allowed=True,
            ),
            CompatibleLinearModule(
                name="model.language_model.layers.0.self_attn.q_proj",
                in_features=16,
                out_features=8,
                has_bias=False,
                language_lora_allowed=True,
            ),
            CompatibleLinearModule(
                name="model.visual.blocks.0.attn.qkv",
                in_features=8,
                out_features=8,
                has_bias=False,
                language_lora_allowed=False,
            ),
        ),
    )


def _manifest() -> AdapterManifest:
    return parse_adapter_manifest(_manifest_mapping())


def test_compatible_manifest_passes_and_recomputes_lora_parameters() -> None:
    report = require_adapter_compatible(_manifest(), _target())

    assert report.compatible is True
    assert report.issues == ()
    assert report.expected_lora_trainable_parameters == 88


def test_exact_base_revision_mismatch_fails_closed() -> None:
    target = _target()
    incompatible_target = AdapterCompatibilityTarget(
        base_repository=target.base_repository,
        base_revision="f" * 40,
        tokenizer_repository=target.tokenizer_repository,
        tokenizer_revision=target.tokenizer_revision,
        chat_template_identifier=target.chat_template_identifier,
        chat_template_sha256=target.chat_template_sha256,
        linear_modules=target.linear_modules,
    )

    report = validate_adapter_compatibility(_manifest(), incompatible_target)

    assert report.compatible is False
    assert [issue.code for issue in report.issues] == ["base_revision_mismatch"]
    with pytest.raises(AdapterCompatibilityError, match="base_revision_mismatch"):
        require_adapter_compatible(_manifest(), incompatible_target)


def test_base_repository_mismatch_is_rejected() -> None:
    target = _target()
    incompatible_target = AdapterCompatibilityTarget(
        base_repository="Qwen/Qwen3.5-4B-Base",
        base_revision=target.base_revision,
        tokenizer_repository=target.tokenizer_repository,
        tokenizer_revision=target.tokenizer_revision,
        chat_template_identifier=target.chat_template_identifier,
        chat_template_sha256=target.chat_template_sha256,
        linear_modules=target.linear_modules,
    )

    report = validate_adapter_compatibility(_manifest(), incompatible_target)

    assert [issue.code for issue in report.issues] == ["base_repository_mismatch"]


def test_tokenizer_repository_and_revision_are_compared_exactly() -> None:
    target = _target()
    incompatible_target = AdapterCompatibilityTarget(
        base_repository=target.base_repository,
        base_revision=target.base_revision,
        tokenizer_repository="Qwen/other-tokenizer",
        tokenizer_revision="f" * 40,
        chat_template_identifier=target.chat_template_identifier,
        chat_template_sha256=target.chat_template_sha256,
        linear_modules=target.linear_modules,
    )

    report = validate_adapter_compatibility(_manifest(), incompatible_target)

    assert [issue.code for issue in report.issues] == [
        "tokenizer_repository_mismatch",
        "tokenizer_revision_mismatch",
    ]


def test_chat_template_identifier_and_hash_are_compared_when_available() -> None:
    target = _target()
    incompatible_target = AdapterCompatibilityTarget(
        base_repository=target.base_repository,
        base_revision=target.base_revision,
        tokenizer_repository=target.tokenizer_repository,
        tokenizer_revision=target.tokenizer_revision,
        chat_template_identifier="different-template",
        chat_template_sha256="f" * 64,
        linear_modules=target.linear_modules,
    )

    report = validate_adapter_compatibility(_manifest(), incompatible_target)

    assert [issue.code for issue in report.issues] == [
        "chat_template_identifier_mismatch",
        "chat_template_hash_mismatch",
    ]


def test_manifest_hash_requires_target_hash_evidence() -> None:
    report = validate_adapter_compatibility(_manifest(), _target(template_hash=None))

    assert [issue.code for issue in report.issues] == ["chat_template_hash_unverifiable"]


def test_manifest_without_optional_template_hash_can_use_exact_tokenizer_identity() -> None:
    raw = _manifest_mapping()
    tokenizer = raw["tokenizer"]
    assert isinstance(tokenizer, dict)
    template = tokenizer["chat_template"]
    assert isinstance(template, dict)
    template["sha256"] = None

    report = validate_adapter_compatibility(parse_adapter_manifest(raw), _target())

    assert report.compatible is True


def test_missing_resolved_lora_target_is_rejected() -> None:
    raw = _manifest_mapping()
    lora = raw["lora"]
    assert isinstance(lora, dict)
    targets = lora["target_modules"]
    assert isinstance(targets, list)
    targets.append("model.language_model.layers.99.self_attn.q_proj")

    report = validate_adapter_compatibility(parse_adapter_manifest(raw), _target())

    assert [issue.code for issue in report.issues] == ["lora_target_module_missing"]
    assert report.expected_lora_trainable_parameters is None


def test_language_adapter_rejects_observed_but_out_of_scope_module() -> None:
    raw = _manifest_mapping()
    lora = raw["lora"]
    assert isinstance(lora, dict)
    targets = lora["target_modules"]
    assert isinstance(targets, list)
    targets.append("model.visual.blocks.0.attn.qkv")
    lora["trainable_parameters"] = 120

    report = validate_adapter_compatibility(parse_adapter_manifest(raw), _target())

    assert [issue.code for issue in report.issues] == ["lora_target_scope_mismatch"]
    assert report.expected_lora_trainable_parameters == 120


def test_bias_none_trainable_parameter_count_is_checked_against_geometry() -> None:
    raw = _manifest_mapping()
    lora = raw["lora"]
    assert isinstance(lora, dict)
    lora["trainable_parameters"] = 89

    report = validate_adapter_compatibility(parse_adapter_manifest(raw), _target())

    assert report.expected_lora_trainable_parameters == 88
    assert [issue.code for issue in report.issues] == ["lora_trainable_parameter_count_mismatch"]


def test_multiple_incompatibilities_have_deterministic_order_and_json() -> None:
    raw = _manifest_mapping()
    base = raw["base_model"]
    tokenizer = raw["tokenizer"]
    lora = raw["lora"]
    assert isinstance(base, dict)
    assert isinstance(tokenizer, dict)
    assert isinstance(lora, dict)
    base["repository"] = "wrong/base"
    base["revision"] = "f" * 40
    tokenizer["repository"] = "wrong/tokenizer"
    tokenizer["revision"] = "e" * 40
    lora["trainable_parameters"] = 89
    manifest = parse_adapter_manifest(raw)

    first = validate_adapter_compatibility(manifest, _target())
    second = validate_adapter_compatibility(manifest, _target())

    assert [issue.code for issue in first.issues] == [
        "base_repository_mismatch",
        "base_revision_mismatch",
        "tokenizer_repository_mismatch",
        "tokenizer_revision_mismatch",
        "lora_trainable_parameter_count_mismatch",
    ]
    assert compatibility_report_json(first) == compatibility_report_json(second)
    assert json.loads(compatibility_report_json(first))["compatible"] is False


def test_build_target_uses_inspection_and_discovery_evidence() -> None:
    inspection_target = InspectionTarget(
        config_id="qwen35-4b",
        model_repository="Qwen/Qwen3.5-4B",
        model_revision=_BASE_REVISION,
        tokenizer_repository="Qwen/Qwen3.5-4B",
        tokenizer_revision=_TOKENIZER_REVISION,
        model_load_dtype="bfloat16",
    )
    linear_modules = (
        LinearModuleRecord(
            name="model.language_model.layers.0.mlp.gate_proj",
            component="text_backbone",
            class_name="torch.nn.Linear",
            in_features=8,
            out_features=12,
            has_bias=False,
        ),
        LinearModuleRecord(
            name="model.language_model.layers.0.self_attn.q_proj",
            component="text_backbone",
            class_name="torch.nn.Linear",
            in_features=16,
            out_features=8,
            has_bias=False,
        ),
    )
    inspection = ModelInspectionReport(
        schema_version=1,
        target=inspection_target,
        model=ModelMetadata(
            model_class="Qwen3_5ForConditionalGeneration",
            config_class="Qwen3_5Config",
            model_type="qwen3_5",
            architectures=("Qwen3_5ForConditionalGeneration",),
            resolved_revision=_BASE_REVISION,
            text_model_type="qwen3_5_text",
            vision_model_type="qwen3_5_vision",
            text_layer_types=("full_attention",),
        ),
        total_parameters=100,
        trainable_parameters=0,
        components=(
            ComponentSummary("text_backbone", 100, 2),
            ComponentSummary("vision_encoder", 0, 0),
            ComponentSummary("multimodal_projector", 0, 0),
            ComponentSummary("other", 0, 0),
        ),
        linear_modules=linear_modules,
        tokenizer=TokenizerMetadata(
            tokenizer_class="Qwen2TokenizerFast",
            vocab_size=100,
            tokenizer_length=100,
            model_max_length=2048,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=2,
            padding_side="right",
            truncation_side="right",
            chat_template_present=True,
            chat_template_sha256=_TEMPLATE_HASH,
            chat_template_length=1000,
        ),
    )
    discovery_modules = (
        TargetModuleRecord(
            name=linear_modules[0].name,
            leaf_name="gate_proj",
            category="mlp",
            component="text_backbone",
            in_features=8,
            out_features=12,
            has_bias=False,
            selected_by_default=True,
        ),
        TargetModuleRecord(
            name=linear_modules[1].name,
            leaf_name="q_proj",
            category="full_attention",
            component="text_backbone",
            in_features=16,
            out_features=8,
            has_bias=False,
            selected_by_default=True,
        ),
    )
    discovery = PeftTargetDiscoveryReport(
        schema_version=1,
        model_repository=inspection_target.model_repository,
        model_revision=inspection_target.model_revision,
        linear_module_count=2,
        selective_matched_module_count=2,
        excluded_module_count=0,
        unclassified_text_module_count=0,
        selective_target_modules=("gate_proj", "q_proj"),
        categories=(
            TargetCategorySummary("full_attention", 1, ("q_proj",), True),
            TargetCategorySummary("mlp", 1, ("gate_proj",), True),
            TargetCategorySummary("gated_deltanet", 0, (), True),
            TargetCategorySummary("language_output", 0, (), False),
            TargetCategorySummary("vision_encoder", 0, (), False),
            TargetCategorySummary("multimodal_projector", 0, (), False),
            TargetCategorySummary("unclassified_text", 0, (), False),
            TargetCategorySummary("other", 0, (), False),
        ),
        modules=discovery_modules,
    )

    target = build_compatibility_target(
        inspection,
        discovery,
        chat_template_identifier="qwen35-4b-pinned-checkpoint",
    )

    assert target.base_revision == _BASE_REVISION
    assert target.chat_template_sha256 == _TEMPLATE_HASH
    assert all(module.language_lora_allowed for module in target.linear_modules)
    assert tuple(module.name for module in target.linear_modules) == tuple(
        sorted(module.name for module in discovery_modules)
    )
