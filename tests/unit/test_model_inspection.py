# mypy: disable-error-code="union-attr,index"
from __future__ import annotations

import json
from types import SimpleNamespace

from torch import nn

from tiny_qwen_coder.model import (
    InspectionError,
    InspectionTarget,
    build_inspection_report,
    inspect_tokenizer,
    inspection_report_json,
    inspection_report_text,
)


class _FakeTokenizer:
    vocab_size = 8
    model_max_length = 128
    bos_token_id = None
    eos_token_id = 6
    pad_token_id = 7
    padding_side = "left"
    truncation_side = "right"
    chat_template = "{{ messages }}"

    def __len__(self) -> int:
        return 10


class _CanonicalShapeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList([nn.Linear(4, 4, bias=False)])
        self.model.visual = nn.Module()
        self.model.visual.encoder = nn.Linear(4, 4, bias=False)
        self.model.visual.merger = nn.Linear(4, 2, bias=True)
        self.lm_head = nn.Linear(4, 8, bias=False)
        self.config = SimpleNamespace(
            model_type="qwen3_5",
            architectures=["Qwen3_5ForConditionalGeneration"],
            _commit_hash="a" * 40,
            text_config=SimpleNamespace(
                model_type="qwen3_5_text",
                layer_types=["linear_attention", "full_attention"],
            ),
            vision_config=SimpleNamespace(model_type="qwen3_5"),
        )


def _target() -> InspectionTarget:
    return InspectionTarget(
        config_id="qwen35-4b",
        model_repository="Qwen/Qwen3.5-4B",
        model_revision="a" * 40,
        tokenizer_repository="Qwen/Qwen3.5-4B",
        tokenizer_revision="a" * 40,
        model_load_dtype="bfloat16",
    )


def test_report_distinguishes_components_and_linear_hierarchy() -> None:
    report = build_inspection_report(_CanonicalShapeModel(), _FakeTokenizer(), _target())

    assert report.total_parameters == 16 + 16 + 10 + 32
    assert [component.parameter_count for component in report.components] == [48, 16, 10, 0]
    assert [component.linear_module_count for component in report.components] == [2, 1, 1, 0]
    assert [(module.name, module.component) for module in report.linear_modules] == [
        ("lm_head", "text_backbone"),
        ("model.language_model.layers.0", "text_backbone"),
        ("model.visual.encoder", "vision_encoder"),
        ("model.visual.merger", "multimodal_projector"),
    ]
    assert report.model.text_layer_types == ("linear_attention", "full_attention")


def test_tokenizer_metadata_hashes_chat_template_without_embedding_it() -> None:
    metadata = inspect_tokenizer(_FakeTokenizer())

    assert metadata.tokenizer_length == 10
    assert metadata.chat_template_present is True
    assert metadata.chat_template_length == len(_FakeTokenizer.chat_template)
    assert metadata.chat_template_sha256 is not None


def test_json_and_text_reports_expose_same_canonical_identity() -> None:
    report = build_inspection_report(_CanonicalShapeModel(), _FakeTokenizer(), _target())

    payload = json.loads(inspection_report_json(report))
    text = inspection_report_text(report)

    assert payload["target"]["model_revision"] == "a" * 40
    assert payload["model"]["model_class"].endswith("._CanonicalShapeModel")
    assert "Qwen/Qwen3.5-4B@" + "a" * 40 in text
    assert "multimodal_projector" in text
    assert "LoRA-relevant Linear module hierarchy" in text


def test_unexpected_upstream_revision_fails_closed() -> None:
    model = _CanonicalShapeModel()
    model.config._commit_hash = "b" * 40

    try:
        build_inspection_report(model, _FakeTokenizer(), _target())
    except InspectionError as exc:
        assert "unexpected upstream revision" in str(exc)
    else:
        raise AssertionError("expected revision mismatch to fail")


def test_missing_canonical_component_root_fails_closed() -> None:
    model = _CanonicalShapeModel()
    del model.model.visual.merger

    try:
        build_inspection_report(model, _FakeTokenizer(), _target())
    except InspectionError as exc:
        assert "model.visual.merger" in str(exc)
    else:
        raise AssertionError("expected missing canonical module root to fail")


def test_report_counts_unique_parameters_when_weights_are_tied() -> None:
    model = _CanonicalShapeModel()
    model.lm_head.weight = model.model.language_model.layers[0].weight

    report = build_inspection_report(model, _FakeTokenizer(), _target())

    unique_count = sum(parameter.numel() for parameter in model.parameters())
    assert report.total_parameters == unique_count
    assert sum(component.parameter_count for component in report.components) == unique_count


def test_target_rejects_floating_revision() -> None:
    try:
        InspectionTarget(
            config_id="qwen35-4b",
            model_repository="Qwen/Qwen3.5-4B",
            model_revision="main",
            tokenizer_repository="Qwen/Qwen3.5-4B",
            tokenizer_revision="main",
            model_load_dtype="bfloat16",
        )
    except InspectionError as exc:
        assert "immutable 40-character Git SHA" in str(exc)
    else:
        raise AssertionError("expected floating revision to fail")
