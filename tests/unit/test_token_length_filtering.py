from __future__ import annotations

from typing import cast

import pytest

from tiny_qwen_coder.data import (
    LengthFilterConfig,
    LengthFilteringError,
    LengthRejectionReason,
    LicenseMetadata,
    NormalizedTrainingRecord,
    SourceProvenance,
    TokenLengthCount,
    TrainingMessage,
    TruncationPolicy,
    filter_by_token_length,
    single_turn_messages,
    token_length_distribution,
    tokenize_training_record,
)
from tiny_qwen_coder.model import InspectionTarget

_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


class FakeTokenizer:
    chat_template = "fixture canonical template"

    def __init__(self, *, revision: str = _REVISION) -> None:
        self.init_kwargs: dict[str, object] = {"_commit_hash": revision}
        self.calls: list[dict[str, object]] = []

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        truncation: bool,
        return_dict: bool,
        chat_template: str,
    ) -> list[int]:
        self.calls.append(
            {
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "truncation": truncation,
                "return_dict": return_dict,
                "chat_template": chat_template,
            }
        )
        token_count = 2 + sum(len(message["content"].split()) for message in conversation)
        return list(range(token_count))


def _target() -> InspectionTarget:
    return InspectionTarget(
        config_id="qwen35-4b",
        model_repository="Qwen/Qwen3.5-4B",
        model_revision=_REVISION,
        tokenizer_repository="Qwen/Qwen3.5-4B",
        tokenizer_revision=_REVISION,
        model_load_dtype="bfloat16",
    )


def _record(
    token_count: int,
    *,
    language: str = "python",
    record_id: str = "row-1",
) -> NormalizedTrainingRecord:
    if token_count < 4:
        raise ValueError("fixture token_count must be at least 4")
    assistant_words = token_count - 3
    return NormalizedTrainingRecord(
        schema_version=1,
        language=language,
        messages=single_turn_messages(
            system=None,
            user="prompt",
            assistant=" ".join("answer" for _ in range(assistant_words)),
        ),
        provenance=SourceProvenance(
            source_id="fixture/source",
            revision=_REVISION,
            license=LicenseMetadata(name="Apache-2.0"),
            split="train",
            record_id=record_id,
        ),
        validation=None,
    )


def test_default_policy_is_explicit_rejection_at_2048_tokens() -> None:
    config = LengthFilterConfig()

    assert config.min_tokens == 1
    assert config.max_tokens == 2048
    assert config.truncation_policy is TruncationPolicy.REJECT


def test_length_filter_config_rejects_invalid_bounds_and_unknown_policy() -> None:
    with pytest.raises(LengthFilteringError, match="min_tokens"):
        LengthFilterConfig(min_tokens=0)
    with pytest.raises(LengthFilteringError, match="max_tokens"):
        LengthFilterConfig(min_tokens=10, max_tokens=9)
    with pytest.raises(LengthFilteringError, match="unsupported truncation policy"):
        LengthFilterConfig(truncation_policy=cast(TruncationPolicy, "truncate_right"))


def test_tokenization_uses_full_explicit_chat_template_without_truncation() -> None:
    tokenizer = FakeTokenizer()
    record = _record(8)

    input_ids = tokenize_training_record(tokenizer, record)

    assert len(input_ids) == 8
    assert tokenizer.calls == [
        {
            "tokenize": True,
            "add_generation_prompt": False,
            "truncation": False,
            "return_dict": False,
            "chat_template": tokenizer.chat_template,
        }
    ]


def test_min_and_max_bounds_reject_full_measured_lengths() -> None:
    tokenizer = FakeTokenizer()
    records = (
        _record(4, record_id="short"),
        _record(6, record_id="accepted"),
        _record(10, record_id="long"),
    )

    report = filter_by_token_length(
        records,
        tokenizer,
        _target(),
        config=LengthFilterConfig(min_tokens=5, max_tokens=8),
    )

    assert report.total_records == 3
    assert report.accepted_count == 1
    assert report.rejected_count == 2
    assert report.accepted_records[0].provenance.record_id == "accepted"
    assert report.accepted_lengths[0].token_count == 6
    assert tuple((item.token_count, item.reason) for item in report.rejected_records) == (
        (4, LengthRejectionReason.TOO_SHORT),
        (10, LengthRejectionReason.TOO_LONG),
    )
    assert all(call["truncation"] is False for call in tokenizer.calls)


def test_overlength_record_is_rejected_not_silently_truncated() -> None:
    tokenizer = FakeTokenizer()

    report = filter_by_token_length(
        (_record(25, record_id="long"),),
        tokenizer,
        _target(),
        config=LengthFilterConfig(max_tokens=8),
    )

    assert report.accepted_count == 0
    rejected = report.rejected_records[0]
    assert rejected.token_count == 25
    assert rejected.reason is LengthRejectionReason.TOO_LONG
    assert tokenizer.calls[0]["truncation"] is False


def test_distribution_records_exact_histogram_and_nearest_rank_percentiles() -> None:
    distribution = token_length_distribution((4, 4, 6, 10, 20))

    assert distribution.count == 5
    assert distribution.minimum == 4
    assert distribution.maximum == 20
    assert distribution.mean == pytest.approx(8.8)
    assert distribution.p50 == 6
    assert distribution.p90 == 20
    assert distribution.p95 == 20
    assert distribution.p99 == 20
    assert distribution.histogram == (
        TokenLengthCount(token_count=4, record_count=2),
        TokenLengthCount(token_count=6, record_count=1),
        TokenLengthCount(token_count=10, record_count=1),
        TokenLengthCount(token_count=20, record_count=1),
    )


def test_report_records_input_and_accepted_distributions() -> None:
    report = filter_by_token_length(
        (_record(4), _record(6), _record(10)),
        FakeTokenizer(),
        _target(),
        config=LengthFilterConfig(min_tokens=5, max_tokens=8),
    )

    assert report.input_distribution.count == 3
    assert report.input_distribution.minimum == 4
    assert report.input_distribution.maximum == 10
    assert report.accepted_distribution.count == 1
    assert report.accepted_distribution.minimum == 6
    assert report.accepted_distribution.maximum == 6
    assert tuple((item.reason, item.count) for item in report.rejection_counts) == (
        (LengthRejectionReason.TOO_SHORT, 1),
        (LengthRejectionReason.TOO_LONG, 1),
    )


def test_empty_input_has_empty_distributions() -> None:
    report = filter_by_token_length((), FakeTokenizer(), _target())

    assert report.total_records == 0
    assert report.input_distribution.count == 0
    assert report.input_distribution.minimum is None
    assert report.input_distribution.histogram == ()
    assert report.accepted_distribution == report.input_distribution


def test_filter_is_programming_language_independent() -> None:
    records = tuple(
        _record(6, language=language, record_id=language)
        for language in ("python", "typescript", "rust")
    )

    report = filter_by_token_length(records, FakeTokenizer(), _target())

    assert report.accepted_count == 3
    assert tuple(record.language for record in report.accepted_records) == (
        "python",
        "typescript",
        "rust",
    )


def test_loaded_tokenizer_revision_mismatch_fails_closed() -> None:
    tokenizer = FakeTokenizer(revision="0123456789abcdef0123456789abcdef01234567")

    with pytest.raises(LengthFilteringError, match="unexpected upstream revision"):
        filter_by_token_length((_record(6),), tokenizer, _target())


def test_missing_chat_template_fails_instead_of_falling_back() -> None:
    tokenizer = FakeTokenizer()
    tokenizer.chat_template = ""

    with pytest.raises(LengthFilteringError, match="non-empty chat template"):
        filter_by_token_length((_record(6),), tokenizer, _target())


def test_non_integer_tokenization_output_fails_closed() -> None:
    class BadTokenizer:
        chat_template = "fixture canonical template"

        def apply_chat_template(
            self,
            conversation: list[dict[str, str]],
            *,
            tokenize: bool,
            add_generation_prompt: bool,
            truncation: bool,
            return_dict: bool,
            chat_template: str,
        ) -> list[str]:
            del (
                conversation,
                tokenize,
                add_generation_prompt,
                truncation,
                return_dict,
                chat_template,
            )
            return ["bad"]

    with pytest.raises(LengthFilteringError, match="must be an integer"):
        tokenize_training_record(BadTokenizer(), _record(6))


def test_multi_turn_record_length_is_measured_after_chat_rendering() -> None:
    record = NormalizedTrainingRecord(
        schema_version=1,
        language="python",
        messages=(
            TrainingMessage(role="system", content="system words"),
            TrainingMessage(role="user", content="first prompt"),
            TrainingMessage(role="assistant", content="first answer"),
            TrainingMessage(role="user", content="second prompt"),
            TrainingMessage(role="assistant", content="second answer"),
        ),
        provenance=SourceProvenance(
            source_id="fixture/source",
            revision=_REVISION,
            license=LicenseMetadata(name="Apache-2.0"),
            record_id="multi-turn",
        ),
        validation=None,
    )

    report = filter_by_token_length((record,), FakeTokenizer(), _target())

    assert report.accepted_lengths[0].token_count == 12
