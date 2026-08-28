"""Tests for assistant/completion-only chat loss masking."""

from __future__ import annotations

from typing import cast

import pytest
from transformers import PreTrainedTokenizerBase

import tiny_qwen_coder.training.loss_masking as loss_masking
from tiny_qwen_coder.model import InspectionTarget
from tiny_qwen_coder.training import (
    LossMaskingError,
    TokenizedLossExample,
    build_chat_loss_mask_report,
    canonical_probe_messages,
    completion_only_fallback,
    loss_mask_report_json,
)

_RAW_TEMPLATE = "raw-template-without-generation-markers"
_TRAINING_TEMPLATE = "{% generation %}assistant{% endgeneration %}"


class FakeTokenizer:
    """Minimal deterministic tokenizer surface used by masking unit tests."""

    chat_template = _RAW_TEMPLATE

    def __init__(self, *, preserve_prefix: bool = True) -> None:
        self.preserve_prefix = preserve_prefix

    def apply_chat_template(
        self,
        conversation: object,
        *,
        tokenize: bool,
        return_dict: bool = False,
        return_assistant_tokens_mask: bool = False,
        add_generation_prompt: bool = False,
        chat_template: str | None = None,
    ) -> object:
        del conversation
        if not tokenize:
            raise AssertionError("unit fake only supports tokenized chat templates")
        if return_dict:
            mask = [0, 0, 1, 1] if chat_template == _TRAINING_TEMPLATE else [0, 0, 0, 0]
            result: dict[str, object] = {"input_ids": [10, 11, 12, 13]}
            if return_assistant_tokens_mask:
                result["assistant_masks"] = mask
            return result
        if add_generation_prompt:
            return [10, 11] if self.preserve_prefix else [99, 11]
        return [10, 11, 12, 13]

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        pieces = {10: "<system/user>", 11: "<assistant-prefix>", 12: "answer", 13: "<eot>"}
        return "".join(pieces[token_id] for token_id in token_ids)


def _target() -> InspectionTarget:
    return InspectionTarget(
        config_id="qwen35-4b",
        model_repository="Qwen/Qwen3.5-4B",
        model_revision="a" * 40,
        tokenizer_repository="Qwen/Qwen3.5-4B",
        tokenizer_revision="a" * 40,
        model_load_dtype="bfloat16",
    )


def _tokenizer(*, preserve_prefix: bool = True) -> PreTrainedTokenizerBase:
    return cast(PreTrainedTokenizerBase, FakeTokenizer(preserve_prefix=preserve_prefix))


def test_trl_training_template_selects_only_assistant_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        loss_masking, "get_training_chat_template", lambda tokenizer: _TRAINING_TEMPLATE
    )
    monkeypatch.setattr(
        loss_masking, "is_chat_template_stop_token_trained", lambda *args, **kwargs: True
    )

    report = build_chat_loss_mask_report(_tokenizer(), _target(), canonical_probe_messages())

    assert report.checkpoint_has_generation_markers is False
    assert report.checkpoint_assistant_token_count == 0
    assert report.trl_training_template_has_generation_markers is True
    assert report.trl_assistant_token_count == 2
    assert report.trl_stop_token_trained is True
    assert report.strategy == "trl_assistant_only"
    assert [token.receives_loss for token in report.tokens] == [False, False, True, True]
    assert report.loss_text == "answer<eot>"
    assert [(span.start, span.end) for span in report.loss_spans] == [(2, 4)]


def test_completion_only_fallback_marks_only_final_assistant_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unsupported_template(tokenizer: object) -> None:
        del tokenizer
        raise ValueError("unsupported")

    monkeypatch.setattr(loss_masking, "get_training_chat_template", unsupported_template)

    report = build_chat_loss_mask_report(_tokenizer(), _target(), canonical_probe_messages())

    assert report.strategy == "completion_only_fallback"
    assert [token.receives_loss for token in report.tokens] == [False, False, True, True]
    assert report.trl_assistant_token_count == 0
    assert report.trl_stop_token_trained is False


def test_completion_only_fallback_requires_prefix_preservation() -> None:
    with pytest.raises(LossMaskingError, match="stable prompt/completion token boundary"):
        completion_only_fallback(_tokenizer(preserve_prefix=False), canonical_probe_messages())


def test_tokenized_loss_example_converts_mask_to_ignore_index_labels() -> None:
    example = TokenizedLossExample(
        input_ids=(10, 11, 12, 13),
        loss_mask=(False, False, True, True),
        strategy="completion_only_fallback",
    )

    assert example.labels() == (-100, -100, 12, 13)
    assert example.labels(ignore_index=-1) == (-1, -1, 12, 13)


def test_loss_mask_report_json_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        loss_masking, "get_training_chat_template", lambda tokenizer: _TRAINING_TEMPLATE
    )
    monkeypatch.setattr(
        loss_masking, "is_chat_template_stop_token_trained", lambda *args, **kwargs: True
    )

    first = build_chat_loss_mask_report(_tokenizer(), _target(), canonical_probe_messages())
    second = build_chat_loss_mask_report(_tokenizer(), _target(), canonical_probe_messages())

    assert loss_mask_report_json(first) == loss_mask_report_json(second)


def test_canonical_probe_contains_system_user_and_assistant_turns() -> None:
    messages = canonical_probe_messages()

    assert tuple(message.role for message in messages) == ("system", "user", "assistant")
    assert "return x + 1" in messages[-1].content.lower()
