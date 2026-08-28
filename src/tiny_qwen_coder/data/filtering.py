"""Generic required-content filtering for normalized training records."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import StrEnum

from tiny_qwen_coder.data.records import NormalizedTrainingRecord, TrainingMessage


class ContentRejectionReason(StrEnum):
    """Stable language-neutral reasons for rejecting a normalized record."""

    INVALID_TEXT_ENCODING = "invalid_text_encoding"
    MALFORMED_RECORD = "malformed_record"
    EMPTY_PROMPT = "empty_prompt"
    EMPTY_RESPONSE = "empty_response"


@dataclass(frozen=True, slots=True)
class RejectionReasonCount:
    """Count of rejected records exhibiting one rejection reason."""

    reason: ContentRejectionReason
    count: int

    def __post_init__(self) -> None:
        if self.count < 0:
            raise ValueError("rejection-reason count must not be negative")


@dataclass(frozen=True, slots=True)
class RejectedTrainingRecord:
    """Compact provenance for one rejected input record."""

    input_index: int
    language: str
    source_id: str
    source_record_id: str | None
    reasons: tuple[ContentRejectionReason, ...]

    def __post_init__(self) -> None:
        if self.input_index < 0:
            raise ValueError("rejected-record input index must not be negative")
        if not self.reasons:
            raise ValueError("rejected record must contain at least one reason")
        if len(self.reasons) != len(set(self.reasons)):
            raise ValueError("rejected-record reasons must be unique")


@dataclass(frozen=True, slots=True)
class RequiredContentFilterReport:
    """Accepted records plus deterministic rejection accounting."""

    total_records: int
    accepted_records: tuple[NormalizedTrainingRecord, ...]
    rejected_records: tuple[RejectedTrainingRecord, ...]
    reason_counts: tuple[RejectionReasonCount, ...]

    def __post_init__(self) -> None:
        if self.total_records < 0:
            raise ValueError("total record count must not be negative")
        if self.total_records != len(self.accepted_records) + len(self.rejected_records):
            raise ValueError("filter report total does not match accepted plus rejected records")
        reasons = tuple(item.reason for item in self.reason_counts)
        if reasons != tuple(ContentRejectionReason):
            raise ValueError("reason counts must contain every rejection reason in stable order")

    @property
    def accepted_count(self) -> int:
        """Return the number of accepted records."""

        return len(self.accepted_records)

    @property
    def rejected_count(self) -> int:
        """Return the number of rejected records."""

        return len(self.rejected_records)


def normalize_training_text(text: str) -> str:
    """Apply only semantics-preserving text normalization used by P3-004.

    The input must be strictly UTF-8 encodable. A leading Unicode BOM is
    removed and CRLF/CR line endings are converted to LF. No whitespace
    trimming or Unicode normalization is performed because either could change
    source-code semantics.
    """

    text.encode("utf-8", errors="strict")
    return text.removeprefix("\ufeff").replace("\r\n", "\n").replace("\r", "\n")


def normalize_record_text(record: NormalizedTrainingRecord) -> NormalizedTrainingRecord:
    """Return a record whose message text has safe canonical line endings."""

    messages = tuple(
        TrainingMessage(role=message.role, content=normalize_training_text(message.content))
        for message in record.messages
    )
    return replace(record, messages=messages)


def _conversation_is_well_formed(messages: tuple[TrainingMessage, ...]) -> bool:
    if not messages:
        return False

    start = 1 if messages[0].role == "system" else 0
    if start >= len(messages):
        return False

    expected_role = "user"
    for message in messages[start:]:
        if message.role != expected_role:
            return False
        expected_role = "assistant" if expected_role == "user" else "user"

    return messages[-1].role == "assistant"


def _content_rejection_reasons(
    record: NormalizedTrainingRecord,
) -> tuple[ContentRejectionReason, ...]:
    reasons: list[ContentRejectionReason] = []

    if not _conversation_is_well_formed(record.messages):
        reasons.append(ContentRejectionReason.MALFORMED_RECORD)
    if any(message.role == "user" and not message.content.strip() for message in record.messages):
        reasons.append(ContentRejectionReason.EMPTY_PROMPT)
    if any(
        message.role == "assistant" and not message.content.strip() for message in record.messages
    ):
        reasons.append(ContentRejectionReason.EMPTY_RESPONSE)

    return tuple(reasons)


def filter_required_content(
    records: Iterable[NormalizedTrainingRecord],
) -> RequiredContentFilterReport:
    """Normalize and filter records without consulting programming-language rules.

    A rejected record may exhibit more than one reason. ``reason_counts``
    therefore counts records per reason and is not required to sum to the
    number of rejected records.
    """

    accepted: list[NormalizedTrainingRecord] = []
    rejected: list[RejectedTrainingRecord] = []
    counts: Counter[ContentRejectionReason] = Counter()
    total = 0

    for input_index, record in enumerate(records):
        total += 1
        reasons: list[ContentRejectionReason] = []
        try:
            normalized = normalize_record_text(record)
        except UnicodeEncodeError:
            normalized = record
            reasons.append(ContentRejectionReason.INVALID_TEXT_ENCODING)

        reasons.extend(_content_rejection_reasons(normalized))
        unique_reasons = tuple(dict.fromkeys(reasons))

        if unique_reasons:
            rejected.append(
                RejectedTrainingRecord(
                    input_index=input_index,
                    language=record.language,
                    source_id=record.provenance.source_id,
                    source_record_id=record.provenance.record_id,
                    reasons=unique_reasons,
                )
            )
            counts.update(unique_reasons)
        else:
            accepted.append(normalized)

    reason_counts = tuple(
        RejectionReasonCount(reason=reason, count=counts[reason])
        for reason in ContentRejectionReason
    )
    return RequiredContentFilterReport(
        total_records=total,
        accepted_records=tuple(accepted),
        rejected_records=tuple(rejected),
        reason_counts=reason_counts,
    )
