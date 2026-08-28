from __future__ import annotations

import pytest

from tiny_qwen_coder.data import (
    ContentRejectionReason,
    LicenseMetadata,
    NormalizedTrainingRecord,
    RequiredContentFilterReport,
    SourceProvenance,
    TrainingMessage,
    ValidationMetadata,
    ValidationResult,
    filter_required_content,
    normalize_training_text,
    single_turn_messages,
)


def _record(
    *,
    language: str = "python",
    messages: tuple[TrainingMessage, ...] | None = None,
    record_id: str = "row-1",
) -> NormalizedTrainingRecord:
    return NormalizedTrainingRecord(
        schema_version=1,
        messages=(
            messages
            if messages is not None
            else single_turn_messages(
                system="You are a coding assistant.",
                user="Return one.",
                assistant="return 1",
            )
        ),
        language=language,
        provenance=SourceProvenance(
            source_id="example/source",
            revision="0123456789abcdef0123456789abcdef01234567",
            license=LicenseMetadata(name="Apache-2.0"),
            split="train",
            record_id=record_id,
        ),
        validation=ValidationMetadata(
            results=(ValidationResult(validator_id="generic.fixture", passed=True),)
        ),
    )


def _reason_counts(
    report: RequiredContentFilterReport,
) -> dict[ContentRejectionReason, int]:
    return {item.reason: item.count for item in report.reason_counts}


def test_normalize_training_text_is_conservative() -> None:
    text = "\ufeff  first\r\nsecond\rthird  \n"

    assert normalize_training_text(text) == "  first\nsecond\nthird  \n"


def test_filter_accepts_and_normalizes_without_changing_metadata() -> None:
    record = _record(
        messages=single_turn_messages(
            system="\ufeffSystem\r\nline",
            user="  prompt\r",
            assistant="answer\r\n  ",
        )
    )

    report = filter_required_content((record,))

    assert report.total_records == 1
    assert report.accepted_count == 1
    assert report.rejected_count == 0
    accepted = report.accepted_records[0]
    assert tuple(message.content for message in accepted.messages) == (
        "System\nline",
        "  prompt\n",
        "answer\n  ",
    )
    assert accepted.provenance is record.provenance
    assert accepted.validation is record.validation


def test_empty_prompt_is_rejected() -> None:
    record = _record(messages=single_turn_messages(system=None, user=" \t\n", assistant="ok"))

    report = filter_required_content((record,))

    assert report.accepted_count == 0
    assert report.rejected_records[0].reasons == (ContentRejectionReason.EMPTY_PROMPT,)
    assert _reason_counts(report)[ContentRejectionReason.EMPTY_PROMPT] == 1


def test_empty_response_is_rejected() -> None:
    record = _record(messages=single_turn_messages(system=None, user="question", assistant="\r\n"))

    report = filter_required_content((record,))

    assert report.accepted_count == 0
    assert report.rejected_records[0].reasons == (ContentRejectionReason.EMPTY_RESPONSE,)
    assert _reason_counts(report)[ContentRejectionReason.EMPTY_RESPONSE] == 1


def test_multiple_rejection_reasons_are_accounted_per_record() -> None:
    record = _record(messages=single_turn_messages(system=None, user="", assistant=""))

    report = filter_required_content((record,))

    assert report.rejected_count == 1
    assert report.rejected_records[0].reasons == (
        ContentRejectionReason.EMPTY_PROMPT,
        ContentRejectionReason.EMPTY_RESPONSE,
    )
    counts = _reason_counts(report)
    assert counts[ContentRejectionReason.EMPTY_PROMPT] == 1
    assert counts[ContentRejectionReason.EMPTY_RESPONSE] == 1


@pytest.mark.parametrize(
    "messages",
    [
        (),
        (TrainingMessage(role="system", content="system"),),
        (TrainingMessage(role="assistant", content="answer"),),
        (
            TrainingMessage(role="user", content="one"),
            TrainingMessage(role="user", content="two"),
            TrainingMessage(role="assistant", content="answer"),
        ),
        (
            TrainingMessage(role="user", content="question"),
            TrainingMessage(role="system", content="late system"),
            TrainingMessage(role="assistant", content="answer"),
        ),
        (TrainingMessage(role="user", content="question"),),
    ],
)
def test_malformed_conversation_shapes_are_rejected(
    messages: tuple[TrainingMessage, ...],
) -> None:
    report = filter_required_content((_record(messages=messages),))

    assert ContentRejectionReason.MALFORMED_RECORD in report.rejected_records[0].reasons


def test_valid_multi_turn_conversation_is_accepted() -> None:
    messages = (
        TrainingMessage(role="system", content="system"),
        TrainingMessage(role="user", content="first"),
        TrainingMessage(role="assistant", content="one"),
        TrainingMessage(role="user", content="second"),
        TrainingMessage(role="assistant", content="two"),
    )

    report = filter_required_content((_record(messages=messages),))

    assert report.accepted_count == 1
    assert report.rejected_count == 0


def test_invalid_utf8_scalar_text_is_rejected_without_repair() -> None:
    record = _record(
        messages=single_turn_messages(system=None, user="bad\ud800text", assistant="ok")
    )

    report = filter_required_content((record,))

    assert report.accepted_count == 0
    assert report.rejected_records[0].reasons[0] is ContentRejectionReason.INVALID_TEXT_ENCODING
    assert _reason_counts(report)[ContentRejectionReason.INVALID_TEXT_ENCODING] == 1


def test_filter_is_language_independent() -> None:
    records = tuple(
        _record(language=language, record_id=language)
        for language in ("python", "rust", "typescript")
    )

    report = filter_required_content(records)

    assert report.accepted_count == 3
    assert tuple(record.language for record in report.accepted_records) == (
        "python",
        "rust",
        "typescript",
    )
    assert all(item.count == 0 for item in report.reason_counts)


def test_rejection_accounting_is_stable_and_retains_compact_provenance() -> None:
    records = (
        _record(record_id="accepted"),
        _record(
            record_id="blank-prompt",
            messages=single_turn_messages(system=None, user="", assistant="ok"),
        ),
        _record(
            language="rust",
            record_id="malformed",
            messages=(TrainingMessage(role="assistant", content="answer"),),
        ),
    )

    report = filter_required_content(records)

    assert report.total_records == 3
    assert report.accepted_count == 1
    assert report.rejected_count == 2
    assert tuple(item.reason for item in report.reason_counts) == tuple(ContentRejectionReason)
    assert tuple(item.input_index for item in report.rejected_records) == (1, 2)
    assert report.rejected_records[0].source_id == "example/source"
    assert report.rejected_records[0].source_record_id == "blank-prompt"
    assert report.rejected_records[1].language == "rust"
