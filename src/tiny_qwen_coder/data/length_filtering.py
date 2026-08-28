"""Tokenizer-aware length filtering for normalized training records."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from tiny_qwen_coder.data.records import NormalizedTrainingRecord
from tiny_qwen_coder.model import InspectionTarget, load_inspection_target

_SCHEMA_VERSION = 1
_DEFAULT_BASE_CONFIG = Path("configs/base/qwen35-4b.yaml")
_DEFAULT_MIN_TOKENS = 1
_DEFAULT_MAX_TOKENS = 2048


class LengthFilteringError(ValueError):
    """Raised when token-length filtering cannot proceed safely."""


class TruncationPolicy(StrEnum):
    """Explicit policies allowed by the generic length filter."""

    REJECT = "reject"


class LengthRejectionReason(StrEnum):
    """Stable reasons for rejecting a record after full tokenization."""

    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"


@dataclass(frozen=True, slots=True)
class LengthFilterConfig:
    """Bounds and explicit truncation policy for token-length filtering."""

    min_tokens: int = _DEFAULT_MIN_TOKENS
    max_tokens: int = _DEFAULT_MAX_TOKENS
    truncation_policy: TruncationPolicy = TruncationPolicy.REJECT

    def __post_init__(self) -> None:
        if self.min_tokens < 1:
            raise LengthFilteringError("min_tokens must be at least 1")
        if self.max_tokens < self.min_tokens:
            raise LengthFilteringError("max_tokens must be greater than or equal to min_tokens")
        if self.truncation_policy != TruncationPolicy.REJECT:
            raise LengthFilteringError(
                "unsupported truncation policy; P3-005 permits only explicit overlength rejection"
            )


@dataclass(frozen=True, slots=True)
class TokenLengthCount:
    """Exact histogram entry for one observed token length."""

    token_count: int
    record_count: int

    def __post_init__(self) -> None:
        if self.token_count < 0:
            raise LengthFilteringError("histogram token_count must not be negative")
        if self.record_count <= 0:
            raise LengthFilteringError("histogram record_count must be greater than zero")


@dataclass(frozen=True, slots=True)
class TokenLengthDistribution:
    """Deterministic summary of a collection of measured token lengths."""

    count: int
    minimum: int | None
    maximum: int | None
    mean: float | None
    p50: int | None
    p90: int | None
    p95: int | None
    p99: int | None
    histogram: tuple[TokenLengthCount, ...]

    def __post_init__(self) -> None:
        histogram_count = sum(item.record_count for item in self.histogram)
        if histogram_count != self.count:
            raise LengthFilteringError("distribution count does not match histogram counts")
        if tuple(item.token_count for item in self.histogram) != tuple(
            sorted(item.token_count for item in self.histogram)
        ):
            raise LengthFilteringError("distribution histogram must be sorted by token_count")
        if self.count == 0:
            if any(
                value is not None
                for value in (
                    self.minimum,
                    self.maximum,
                    self.mean,
                    self.p50,
                    self.p90,
                    self.p95,
                    self.p99,
                )
            ):
                raise LengthFilteringError("empty distributions must use null summary statistics")
            if self.histogram:
                raise LengthFilteringError("empty distributions must have an empty histogram")
            return
        if None in (
            self.minimum,
            self.maximum,
            self.mean,
            self.p50,
            self.p90,
            self.p95,
            self.p99,
        ):
            raise LengthFilteringError("non-empty distributions require all summary statistics")
        if not self.histogram:
            raise LengthFilteringError("non-empty distributions require histogram entries")
        if self.minimum != self.histogram[0].token_count:
            raise LengthFilteringError("distribution minimum is inconsistent with histogram")
        if self.maximum != self.histogram[-1].token_count:
            raise LengthFilteringError("distribution maximum is inconsistent with histogram")


@dataclass(frozen=True, slots=True)
class AcceptedTokenLength:
    """Measured full token length for one accepted input record."""

    input_index: int
    token_count: int

    def __post_init__(self) -> None:
        if self.input_index < 0:
            raise LengthFilteringError("accepted input_index must not be negative")
        if self.token_count <= 0:
            raise LengthFilteringError("accepted token_count must be greater than zero")


@dataclass(frozen=True, slots=True)
class LengthRejectedTrainingRecord:
    """Compact provenance and measured length for one rejected input record."""

    input_index: int
    language: str
    source_id: str
    source_record_id: str | None
    token_count: int
    reason: LengthRejectionReason

    def __post_init__(self) -> None:
        if self.input_index < 0:
            raise LengthFilteringError("rejected input_index must not be negative")
        if self.token_count <= 0:
            raise LengthFilteringError("rejected token_count must be greater than zero")


@dataclass(frozen=True, slots=True)
class LengthRejectionCount:
    """Count of records rejected for one length-bound reason."""

    reason: LengthRejectionReason
    count: int

    def __post_init__(self) -> None:
        if self.count < 0:
            raise LengthFilteringError("length rejection count must not be negative")


@dataclass(frozen=True, slots=True)
class TokenLengthFilterReport:
    """Accepted records and auditable full-token-length filtering evidence."""

    schema_version: int
    target: InspectionTarget
    tokenizer_class: str
    chat_template_sha256: str
    config: LengthFilterConfig
    total_records: int
    accepted_records: tuple[NormalizedTrainingRecord, ...]
    accepted_lengths: tuple[AcceptedTokenLength, ...]
    rejected_records: tuple[LengthRejectedTrainingRecord, ...]
    rejection_counts: tuple[LengthRejectionCount, ...]
    input_distribution: TokenLengthDistribution
    accepted_distribution: TokenLengthDistribution

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise LengthFilteringError(
                f"unsupported token-length report schema_version {self.schema_version}; "
                f"expected {_SCHEMA_VERSION}"
            )
        if self.total_records < 0:
            raise LengthFilteringError("total_records must not be negative")
        if self.total_records != len(self.accepted_records) + len(self.rejected_records):
            raise LengthFilteringError(
                "total_records must equal accepted plus rejected record counts"
            )
        if len(self.accepted_records) != len(self.accepted_lengths):
            raise LengthFilteringError("accepted records and accepted lengths must align")
        if self.input_distribution.count != self.total_records:
            raise LengthFilteringError("input distribution count must equal total_records")
        if self.accepted_distribution.count != len(self.accepted_records):
            raise LengthFilteringError(
                "accepted distribution count must equal accepted record count"
            )
        if tuple(item.reason for item in self.rejection_counts) != tuple(LengthRejectionReason):
            raise LengthFilteringError(
                "rejection_counts must contain every length reason in stable order"
            )
        if not self.chat_template_sha256 or len(self.chat_template_sha256) != 64:
            raise LengthFilteringError("chat template SHA-256 must be a 64-character digest")

    @property
    def accepted_count(self) -> int:
        """Return the number of accepted records."""

        return len(self.accepted_records)

    @property
    def rejected_count(self) -> int:
        """Return the number of rejected records."""

        return len(self.rejected_records)


def _require_chat_template(tokenizer: object) -> str:
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template:
        raise LengthFilteringError("tokenizer must expose a non-empty chat template")
    return template


def _int_sequence(value: object, *, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise LengthFilteringError(f"{field_name} must be an unbatched integer sequence")
    result: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise LengthFilteringError(f"{field_name}[{index}] must be an integer")
        result.append(item)
    if not result:
        raise LengthFilteringError("chat-template tokenization returned no tokens")
    return tuple(result)


def _message_dicts(record: NormalizedTrainingRecord) -> list[dict[str, str]]:
    return [{"role": message.role, "content": message.content} for message in record.messages]


def tokenize_training_record(
    tokenizer: object,
    record: NormalizedTrainingRecord,
) -> tuple[int, ...]:
    """Render and fully tokenize one record without applying tokenizer truncation."""

    template = _require_chat_template(tokenizer)
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise LengthFilteringError("tokenizer does not provide apply_chat_template")
    encoded = apply_chat_template(
        _message_dicts(record),
        tokenize=True,
        add_generation_prompt=False,
        truncation=False,
        return_dict=False,
        chat_template=template,
    )
    return _int_sequence(encoded, field_name="input_ids")


def _nearest_rank(lengths: Sequence[int], percentile: float) -> int:
    ordered = sorted(lengths)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def token_length_distribution(lengths: Iterable[int]) -> TokenLengthDistribution:
    """Build exact histogram and deterministic nearest-rank percentiles."""

    values = tuple(lengths)
    if any(value < 0 for value in values):
        raise LengthFilteringError("token lengths must not be negative")
    if not values:
        return TokenLengthDistribution(
            count=0,
            minimum=None,
            maximum=None,
            mean=None,
            p50=None,
            p90=None,
            p95=None,
            p99=None,
            histogram=(),
        )
    counts = Counter(values)
    histogram = tuple(
        TokenLengthCount(token_count=token_count, record_count=counts[token_count])
        for token_count in sorted(counts)
    )
    return TokenLengthDistribution(
        count=len(values),
        minimum=min(values),
        maximum=max(values),
        mean=sum(values) / len(values),
        p50=_nearest_rank(values, 0.50),
        p90=_nearest_rank(values, 0.90),
        p95=_nearest_rank(values, 0.95),
        p99=_nearest_rank(values, 0.99),
        histogram=histogram,
    )


def _chat_template_sha256(tokenizer: object) -> str:
    template = _require_chat_template(tokenizer)
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def _validate_resolved_tokenizer_revision(tokenizer: object, target: InspectionTarget) -> None:
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if not isinstance(init_kwargs, Mapping):
        return
    resolved_revision = init_kwargs.get("_commit_hash")
    if resolved_revision is not None and resolved_revision != target.tokenizer_revision:
        raise LengthFilteringError(
            "loaded tokenizer resolved to an unexpected upstream revision: "
            f"{resolved_revision!r} != {target.tokenizer_revision!r}"
        )


def filter_by_token_length(
    records: Iterable[NormalizedTrainingRecord],
    tokenizer: object,
    target: InspectionTarget,
    *,
    config: LengthFilterConfig | None = None,
) -> TokenLengthFilterReport:
    """Filter records by full canonical chat-template token length.

    Tokenization always runs with ``truncation=False``. The canonical P3-005
    policy is explicit rejection for overlength examples, so every reported
    token count is the full untruncated example length.
    """

    selected_config = config or LengthFilterConfig()
    _validate_resolved_tokenizer_revision(tokenizer, target)
    accepted: list[NormalizedTrainingRecord] = []
    accepted_lengths: list[AcceptedTokenLength] = []
    rejected: list[LengthRejectedTrainingRecord] = []
    rejection_counter: Counter[LengthRejectionReason] = Counter()
    all_lengths: list[int] = []

    for input_index, record in enumerate(records):
        token_count = len(tokenize_training_record(tokenizer, record))
        all_lengths.append(token_count)
        reason: LengthRejectionReason | None = None
        if token_count < selected_config.min_tokens:
            reason = LengthRejectionReason.TOO_SHORT
        elif token_count > selected_config.max_tokens:
            reason = LengthRejectionReason.TOO_LONG

        if reason is None:
            accepted.append(record)
            accepted_lengths.append(
                AcceptedTokenLength(input_index=input_index, token_count=token_count)
            )
            continue

        rejected.append(
            LengthRejectedTrainingRecord(
                input_index=input_index,
                language=record.language,
                source_id=record.provenance.source_id,
                source_record_id=record.provenance.record_id,
                token_count=token_count,
                reason=reason,
            )
        )
        rejection_counter[reason] += 1

    rejection_counts = tuple(
        LengthRejectionCount(reason=reason, count=rejection_counter[reason])
        for reason in LengthRejectionReason
    )
    accepted_token_counts = tuple(item.token_count for item in accepted_lengths)
    return TokenLengthFilterReport(
        schema_version=_SCHEMA_VERSION,
        target=target,
        tokenizer_class=type(tokenizer).__name__,
        chat_template_sha256=_chat_template_sha256(tokenizer),
        config=selected_config,
        total_records=len(all_lengths),
        accepted_records=tuple(accepted),
        accepted_lengths=tuple(accepted_lengths),
        rejected_records=tuple(rejected),
        rejection_counts=rejection_counts,
        input_distribution=token_length_distribution(all_lengths),
        accepted_distribution=token_length_distribution(accepted_token_counts),
    )


def load_canonical_tokenizer(
    target: InspectionTarget,
    *,
    local_files_only: bool = False,
) -> object:
    """Load the exact tokenizer revision named by the canonical base target."""

    from transformers import AutoTokenizer

    tokenizer: object = AutoTokenizer.from_pretrained(
        target.tokenizer_repository,
        revision=target.tokenizer_revision,
        local_files_only=local_files_only,
    )
    _require_chat_template(tokenizer)
    _validate_resolved_tokenizer_revision(tokenizer, target)
    return tokenizer


def filter_with_canonical_tokenizer(
    records: Iterable[NormalizedTrainingRecord],
    *,
    config: LengthFilterConfig | None = None,
    base_config: Path = _DEFAULT_BASE_CONFIG,
    local_files_only: bool = False,
) -> TokenLengthFilterReport:
    """Load the pinned canonical Qwen tokenizer and filter normalized records."""

    target = load_inspection_target(base_config)
    tokenizer = load_canonical_tokenizer(target, local_files_only=local_files_only)
    return filter_by_token_length(records, tokenizer, target, config=config)
