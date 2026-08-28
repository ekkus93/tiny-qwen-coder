from __future__ import annotations

import pytest

from tiny_qwen_coder.data import (
    DeduplicationError,
    DuplicateReason,
    ExactDeduplicationReport,
    LicenseMetadata,
    NormalizedTrainingRecord,
    SourceProvenance,
    TrainingMessage,
    deduplicate_exact_records,
    deterministic_train_validation_split,
    normalized_record_fingerprint,
    single_turn_messages,
    source_record_identity,
)

_REVISION_A = "0123456789abcdef0123456789abcdef01234567"
_REVISION_B = "89abcdef0123456789abcdef0123456789abcdef"


def _record(
    *,
    user: str = "Return one.",
    assistant: str = "return 1",
    system: str | None = "You are a coding assistant.",
    language: str = "python",
    source_id: str = "fixture/source",
    revision: str = _REVISION_A,
    split: str | None = "train",
    record_id: str | None = "row-1",
    messages: tuple[TrainingMessage, ...] | None = None,
) -> NormalizedTrainingRecord:
    return NormalizedTrainingRecord(
        schema_version=1,
        language=language,
        messages=(
            messages
            if messages is not None
            else single_turn_messages(system=system, user=user, assistant=assistant)
        ),
        provenance=SourceProvenance(
            source_id=source_id,
            revision=revision,
            license=LicenseMetadata(name="Apache-2.0"),
            split=split,
            record_id=record_id,
        ),
        validation=None,
    )


def _reason_counts(report: ExactDeduplicationReport) -> dict[DuplicateReason, int]:
    return {item.reason: item.count for item in report.reason_counts}


def test_fingerprint_normalizes_only_p3_004_safe_text_variants() -> None:
    left = _record(user="\ufeffline one\r\nline two", assistant="answer\r")
    right = _record(user="line one\nline two", assistant="answer\n", record_id="row-2")

    assert normalized_record_fingerprint(left) == normalized_record_fingerprint(right)


def test_semantic_whitespace_remains_part_of_exact_identity() -> None:
    left = _record(user="return x", assistant="return 1")
    right = _record(user=" return x", assistant="return 1", record_id="row-2")

    assert normalized_record_fingerprint(left) != normalized_record_fingerprint(right)


def test_multi_turn_prompt_and_response_hashes_are_separate() -> None:
    base_messages = (
        TrainingMessage(role="system", content="system"),
        TrainingMessage(role="user", content="first"),
        TrainingMessage(role="assistant", content="one"),
        TrainingMessage(role="user", content="second"),
        TrainingMessage(role="assistant", content="two"),
    )
    changed_history = (
        *base_messages[:2],
        TrainingMessage(role="assistant", content="ONE"),
        *base_messages[3:],
    )
    changed_response = (*base_messages[:-1], TrainingMessage(role="assistant", content="TWO"))

    base = normalized_record_fingerprint(_record(messages=base_messages))
    history = normalized_record_fingerprint(_record(messages=changed_history, record_id="row-2"))
    response = normalized_record_fingerprint(_record(messages=changed_response, record_id="row-3"))

    assert base.prompt_sha256 != history.prompt_sha256
    assert base.response_sha256 == history.response_sha256
    assert base.prompt_sha256 == response.prompt_sha256
    assert base.response_sha256 != response.response_sha256
    assert all(
        len(value) == 64 for value in (base.prompt_sha256, base.response_sha256, base.record_sha256)
    )


def test_exact_content_duplicate_across_sources_keeps_first_occurrence() -> None:
    records = (
        _record(source_id="source/a", record_id="a-1"),
        _record(source_id="source/b", record_id="b-1"),
    )

    report = deduplicate_exact_records(records)

    assert report.total_records == 2
    assert report.unique_count == 1
    assert report.duplicate_count == 1
    assert report.unique_records[0].provenance.source_id == "source/a"
    duplicate = report.duplicate_records[0]
    assert duplicate.reasons == (DuplicateReason.EXACT_CONTENT,)
    assert duplicate.content_duplicate_of_input_index == 0
    assert duplicate.source_duplicate_of_input_index is None
    assert _reason_counts(report) == {
        DuplicateReason.EXACT_CONTENT: 1,
        DuplicateReason.SOURCE_IDENTITY: 0,
    }


def test_repeated_source_identity_with_same_content_is_counted_both_ways() -> None:
    record = _record()

    report = deduplicate_exact_records((record, record))

    duplicate = report.duplicate_records[0]
    assert duplicate.reasons == (
        DuplicateReason.EXACT_CONTENT,
        DuplicateReason.SOURCE_IDENTITY,
    )
    assert duplicate.content_duplicate_of_input_index == 0
    assert duplicate.source_duplicate_of_input_index == 0
    assert _reason_counts(report) == {
        DuplicateReason.EXACT_CONTENT: 1,
        DuplicateReason.SOURCE_IDENTITY: 1,
    }


def test_reused_source_identity_with_different_content_fails_closed() -> None:
    records = (
        _record(assistant="return 1"),
        _record(assistant="return 2"),
    )

    with pytest.raises(DeduplicationError, match="source-record identity conflict"):
        deduplicate_exact_records(records)


def test_source_identity_is_remembered_even_when_first_occurrence_is_content_duplicate() -> None:
    records = (
        _record(source_id="source/a", record_id="a-1"),
        _record(source_id="source/b", record_id="b-1"),
        _record(source_id="source/b", record_id="b-1", assistant="different"),
    )

    with pytest.raises(DeduplicationError, match="input indexes 1 and 2"):
        deduplicate_exact_records(records)


def test_different_source_record_ids_allow_distinct_content() -> None:
    records = (
        _record(record_id="row-1", assistant="one"),
        _record(record_id="row-2", assistant="two"),
    )

    report = deduplicate_exact_records(records)

    assert report.unique_count == 2
    assert report.duplicate_count == 0


def test_source_identity_includes_revision_and_split() -> None:
    base = _record(record_id="same", revision=_REVISION_A, split="train")
    changed_revision = _record(record_id="same", revision=_REVISION_B, split="train")
    changed_split = _record(record_id="same", revision=_REVISION_A, split="validation")

    assert source_record_identity(base) != source_record_identity(changed_revision)
    assert source_record_identity(base) != source_record_identity(changed_split)


def test_missing_source_record_id_disables_only_source_identity_matching() -> None:
    records = (
        _record(record_id=None, assistant="one"),
        _record(record_id=None, assistant="two"),
    )

    report = deduplicate_exact_records(records)

    assert source_record_identity(records[0]) is None
    assert report.unique_count == 2
    assert _reason_counts(report)[DuplicateReason.SOURCE_IDENTITY] == 0


def test_exact_model_visible_content_is_deduplicated_independently_of_language_metadata() -> None:
    records = (
        _record(language="python", source_id="source/python", record_id="p-1"),
        _record(language="rust", source_id="source/rust", record_id="r-1"),
    )

    report = deduplicate_exact_records(records)

    assert report.unique_count == 1
    assert report.unique_records[0].language == "python"
    assert report.duplicate_records[0].language == "rust"


def test_unique_output_is_normalized_and_input_order_is_authoritative() -> None:
    records = (
        _record(user="\ufefffirst\r\nline", assistant="one\r", record_id="first"),
        _record(user="second", assistant="two", record_id="second"),
        _record(user="first\nline", assistant="one\n", source_id="other", record_id="duplicate"),
    )

    report = deduplicate_exact_records(records)

    assert tuple(record.provenance.record_id for record in report.unique_records) == (
        "first",
        "second",
    )
    assert report.unique_records[0].messages[1].content == "first\nline"
    assert report.unique_records[0].messages[-1].content == "one\n"
    assert report.duplicate_records[0].content_duplicate_of_input_index == 0


def test_empty_input_has_complete_zero_statistics() -> None:
    report = deduplicate_exact_records(())

    assert report.total_records == 0
    assert report.unique_count == 0
    assert report.duplicate_count == 0
    assert report.reason_counts == tuple(
        type(item)(reason=item.reason, count=0) for item in report.reason_counts
    )
    assert tuple(item.reason for item in report.reason_counts) == tuple(DuplicateReason)


def test_malformed_record_is_rejected_instead_of_hashing_ambiguous_content() -> None:
    record = _record(
        messages=(
            TrainingMessage(role="user", content="question"),
            TrainingMessage(role="user", content="still prompt"),
        )
    )

    with pytest.raises(DeduplicationError, match="must end with an assistant response"):
        deduplicate_exact_records((record,))


def test_deduplication_before_split_prevents_exact_duplicates_crossing_partitions() -> None:
    records = (
        _record(user="alpha", assistant="A", source_id="source/a", record_id="1"),
        _record(user="beta", assistant="B", source_id="source/a", record_id="2"),
        _record(user="alpha", assistant="A", source_id="source/b", record_id="3"),
        _record(user="gamma", assistant="C", source_id="source/a", record_id="4"),
    )
    report = deduplicate_exact_records(records)
    split = deterministic_train_validation_split(
        report.unique_records,
        validation_fraction=1 / 3,
        seed=20260828,
    )

    train_hashes = {normalized_record_fingerprint(record).record_sha256 for record in split.train}
    validation_hashes = {
        normalized_record_fingerprint(record).record_sha256 for record in split.validation
    }

    assert report.unique_count == 3
    assert train_hashes.isdisjoint(validation_hashes)
    assert train_hashes | validation_hashes == {
        fingerprint.record_sha256 for fingerprint in report.unique_fingerprints
    }
