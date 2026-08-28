"""Exact content and source-record deduplication for normalized training data."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from tiny_qwen_coder.data.filtering import normalize_record_text
from tiny_qwen_coder.data.records import NormalizedTrainingRecord, TrainingMessage

_SCHEMA_VERSION = 1


class DeduplicationError(ValueError):
    """Raised when exact deduplication cannot proceed safely."""


class DuplicateReason(StrEnum):
    """Stable reasons why an input record was removed as a duplicate."""

    EXACT_CONTENT = "exact_content"
    SOURCE_IDENTITY = "source_identity"


@dataclass(frozen=True, slots=True)
class RecordContentFingerprint:
    """SHA-256 identities for the normalized prompt, response, and complete pair."""

    prompt_sha256: str
    response_sha256: str
    record_sha256: str

    def __post_init__(self) -> None:
        for field_name, value in (
            ("prompt_sha256", self.prompt_sha256),
            ("response_sha256", self.response_sha256),
            ("record_sha256", self.record_sha256),
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise DeduplicationError(f"{field_name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class SourceRecordIdentity:
    """Exact upstream record key when an upstream record ID is available."""

    source_id: str
    revision: str
    split: str | None
    record_id: str


@dataclass(frozen=True, slots=True)
class DuplicateTrainingRecord:
    """Compact evidence explaining why one input record was removed."""

    input_index: int
    language: str
    source_id: str
    source_record_id: str | None
    fingerprint: RecordContentFingerprint
    reasons: tuple[DuplicateReason, ...]
    content_duplicate_of_input_index: int | None
    source_duplicate_of_input_index: int | None

    def __post_init__(self) -> None:
        if self.input_index < 0:
            raise DeduplicationError("duplicate input_index must not be negative")
        if not self.reasons:
            raise DeduplicationError("duplicate record must contain at least one reason")
        if len(self.reasons) != len(set(self.reasons)):
            raise DeduplicationError("duplicate reasons must be unique")
        if tuple(reason for reason in DuplicateReason if reason in self.reasons) != self.reasons:
            raise DeduplicationError("duplicate reasons must use stable enum order")
        if DuplicateReason.EXACT_CONTENT in self.reasons:
            if self.content_duplicate_of_input_index is None:
                raise DeduplicationError("exact-content duplicate must reference its first record")
        elif self.content_duplicate_of_input_index is not None:
            raise DeduplicationError("non-content duplicate must not reference content provenance")
        if DuplicateReason.SOURCE_IDENTITY in self.reasons:
            if self.source_duplicate_of_input_index is None:
                raise DeduplicationError("source duplicate must reference its first source record")
        elif self.source_duplicate_of_input_index is not None:
            raise DeduplicationError("non-source duplicate must not reference source provenance")
        for duplicate_index in (
            self.content_duplicate_of_input_index,
            self.source_duplicate_of_input_index,
        ):
            if duplicate_index is not None and not 0 <= duplicate_index < self.input_index:
                raise DeduplicationError(
                    "duplicate references must point to an earlier input record"
                )


@dataclass(frozen=True, slots=True)
class DuplicateReasonCount:
    """Number of removed records exhibiting one duplicate reason."""

    reason: DuplicateReason
    count: int

    def __post_init__(self) -> None:
        if self.count < 0:
            raise DeduplicationError("duplicate reason count must not be negative")


@dataclass(frozen=True, slots=True)
class ExactDeduplicationReport:
    """Unique normalized records plus deterministic duplicate accounting."""

    schema_version: int
    total_records: int
    unique_records: tuple[NormalizedTrainingRecord, ...]
    unique_fingerprints: tuple[RecordContentFingerprint, ...]
    duplicate_records: tuple[DuplicateTrainingRecord, ...]
    reason_counts: tuple[DuplicateReasonCount, ...]

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise DeduplicationError(
                f"unsupported deduplication schema_version {self.schema_version}; "
                f"expected {_SCHEMA_VERSION}"
            )
        if self.total_records < 0:
            raise DeduplicationError("total_records must not be negative")
        if self.total_records != len(self.unique_records) + len(self.duplicate_records):
            raise DeduplicationError("total_records must equal unique plus duplicate counts")
        if len(self.unique_records) != len(self.unique_fingerprints):
            raise DeduplicationError("unique records and fingerprints must align")
        record_hashes = tuple(item.record_sha256 for item in self.unique_fingerprints)
        if len(record_hashes) != len(set(record_hashes)):
            raise DeduplicationError("unique records must have unique content fingerprints")
        if tuple(item.reason for item in self.reason_counts) != tuple(DuplicateReason):
            raise DeduplicationError("reason_counts must contain every reason in stable order")

    @property
    def unique_count(self) -> int:
        """Return the number of retained records."""

        return len(self.unique_records)

    @property
    def duplicate_count(self) -> int:
        """Return the number of removed records."""

        return len(self.duplicate_records)


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8", errors="strict")
    return hashlib.sha256(encoded).hexdigest()


def _message_payload(messages: tuple[TrainingMessage, ...]) -> list[dict[str, str]]:
    return [{"role": message.role, "content": message.content} for message in messages]


def normalized_record_fingerprint(record: NormalizedTrainingRecord) -> RecordContentFingerprint:
    """Hash the P3-004-normalized prompt history and final assistant response.

    The prompt history is every message except the final assistant response. This
    preserves system messages and earlier assistant turns in multi-turn examples.
    P3-004's conservative text normalization is applied before hashing, so BOM and
    line-ending variants compare equal while all other whitespace remains semantic.
    """

    try:
        normalized = normalize_record_text(record)
    except UnicodeEncodeError as error:
        raise DeduplicationError(
            "record contains text that is not UTF-8 encodable; run P3-004 filtering first"
        ) from error
    if len(normalized.messages) < 2 or normalized.messages[-1].role != "assistant":
        raise DeduplicationError(
            "record must end with an assistant response; run P3-004 filtering first"
        )

    prompt_payload = {
        "schema_version": _SCHEMA_VERSION,
        "messages": _message_payload(normalized.messages[:-1]),
    }
    response_payload = {
        "schema_version": _SCHEMA_VERSION,
        "role": "assistant",
        "content": normalized.messages[-1].content,
    }
    prompt_sha256 = _canonical_json_sha256(prompt_payload)
    response_sha256 = _canonical_json_sha256(response_payload)
    record_sha256 = _canonical_json_sha256(
        {
            "schema_version": _SCHEMA_VERSION,
            "prompt_sha256": prompt_sha256,
            "response_sha256": response_sha256,
        }
    )
    return RecordContentFingerprint(
        prompt_sha256=prompt_sha256,
        response_sha256=response_sha256,
        record_sha256=record_sha256,
    )


def source_record_identity(record: NormalizedTrainingRecord) -> SourceRecordIdentity | None:
    """Return the exact immutable upstream record key, if one was supplied."""

    provenance = record.provenance
    if provenance.record_id is None:
        return None
    return SourceRecordIdentity(
        source_id=provenance.source_id,
        revision=provenance.revision,
        split=provenance.split,
        record_id=provenance.record_id,
    )


def deduplicate_exact_records(
    records: Iterable[NormalizedTrainingRecord],
) -> ExactDeduplicationReport:
    """Retain first occurrences and remove exact normalized duplicates deterministically.

    Input order is authoritative: the earliest occurrence wins. Exact content is
    deduplicated across source identities. Repeating an exact source-record key with
    different normalized content is an integrity conflict and fails closed.
    """

    unique_records: list[NormalizedTrainingRecord] = []
    unique_fingerprints: list[RecordContentFingerprint] = []
    duplicate_records: list[DuplicateTrainingRecord] = []
    reason_counts: Counter[DuplicateReason] = Counter()

    content_seen: dict[str, tuple[int, tuple[TrainingMessage, ...]]] = {}
    source_seen: dict[SourceRecordIdentity, tuple[int, RecordContentFingerprint]] = {}
    total = 0

    for input_index, record in enumerate(records):
        total += 1
        try:
            normalized = normalize_record_text(record)
        except UnicodeEncodeError as error:
            raise DeduplicationError(
                f"record at input index {input_index} is not UTF-8 encodable; "
                "run P3-004 filtering first"
            ) from error
        fingerprint = normalized_record_fingerprint(normalized)
        source_identity = source_record_identity(normalized)

        content_match = content_seen.get(fingerprint.record_sha256)
        if content_match is not None and content_match[1] != normalized.messages:
            raise DeduplicationError(
                "SHA-256 content collision detected between normalized training records"
            )
        content_duplicate_of = content_match[0] if content_match is not None else None

        source_duplicate_of: int | None = None
        source_match: tuple[int, RecordContentFingerprint] | None = None
        if source_identity is not None:
            source_match = source_seen.get(source_identity)
            if source_match is not None:
                source_duplicate_of, source_fingerprint = source_match
                if source_fingerprint != fingerprint:
                    raise DeduplicationError(
                        "source-record identity conflict: "
                        f"{source_identity.source_id}@{source_identity.revision} "
                        f"split={source_identity.split!r} record_id={source_identity.record_id!r} "
                        f"appears with different content at input indexes "
                        f"{source_duplicate_of} and {input_index}"
                    )
            else:
                source_seen[source_identity] = (input_index, fingerprint)

        reasons = tuple(
            reason
            for reason, matched in (
                (DuplicateReason.EXACT_CONTENT, content_duplicate_of is not None),
                (DuplicateReason.SOURCE_IDENTITY, source_duplicate_of is not None),
            )
            if matched
        )
        if reasons:
            duplicate_records.append(
                DuplicateTrainingRecord(
                    input_index=input_index,
                    language=normalized.language,
                    source_id=normalized.provenance.source_id,
                    source_record_id=normalized.provenance.record_id,
                    fingerprint=fingerprint,
                    reasons=reasons,
                    content_duplicate_of_input_index=content_duplicate_of,
                    source_duplicate_of_input_index=source_duplicate_of,
                )
            )
            reason_counts.update(reasons)
            continue

        content_seen[fingerprint.record_sha256] = (input_index, normalized.messages)
        unique_records.append(normalized)
        unique_fingerprints.append(fingerprint)

    counts = tuple(
        DuplicateReasonCount(reason=reason, count=reason_counts[reason])
        for reason in DuplicateReason
    )
    return ExactDeduplicationReport(
        schema_version=_SCHEMA_VERSION,
        total_records=total,
        unique_records=tuple(unique_records),
        unique_fingerprints=tuple(unique_fingerprints),
        duplicate_records=tuple(duplicate_records),
        reason_counts=counts,
    )
