"""Language-neutral dataset preparation services."""

from __future__ import annotations

from typing import NoReturn

from tiny_qwen_coder.data.deduplication import (
    DeduplicationError,
    DuplicateReason,
    DuplicateReasonCount,
    DuplicateTrainingRecord,
    ExactDeduplicationReport,
    RecordContentFingerprint,
    SourceRecordIdentity,
    deduplicate_exact_records,
    normalized_record_fingerprint,
    source_record_identity,
)
from tiny_qwen_coder.data.filtering import (
    ContentRejectionReason,
    RejectedTrainingRecord,
    RejectionReasonCount,
    RequiredContentFilterReport,
    filter_required_content,
    normalize_record_text,
    normalize_training_text,
)
from tiny_qwen_coder.data.length_filtering import (
    AcceptedTokenLength,
    LengthFilterConfig,
    LengthFilteringError,
    LengthRejectedTrainingRecord,
    LengthRejectionCount,
    LengthRejectionReason,
    TokenLengthCount,
    TokenLengthDistribution,
    TokenLengthFilterReport,
    TruncationPolicy,
    filter_by_token_length,
    filter_with_canonical_tokenizer,
    load_canonical_tokenizer,
    token_length_distribution,
    tokenize_training_record,
)
from tiny_qwen_coder.data.records import (
    LicenseMetadata,
    MessageRole,
    NormalizedTrainingRecord,
    SourceProvenance,
    TrainingMessage,
    ValidationMetadata,
    ValidationResult,
    single_turn_messages,
)
from tiny_qwen_coder.data.splitting import (
    TrainValidationSplit,
    deterministic_train_validation_split,
)

__all__ = [
    "AcceptedTokenLength",
    "ContentRejectionReason",
    "DeduplicationError",
    "DuplicateReason",
    "DuplicateReasonCount",
    "DuplicateTrainingRecord",
    "ExactDeduplicationReport",
    "LengthFilterConfig",
    "LengthFilteringError",
    "LengthRejectedTrainingRecord",
    "LengthRejectionCount",
    "LengthRejectionReason",
    "LicenseMetadata",
    "MessageRole",
    "NormalizedTrainingRecord",
    "RecordContentFingerprint",
    "RejectedTrainingRecord",
    "RejectionReasonCount",
    "RequiredContentFilterReport",
    "SourceProvenance",
    "SourceRecordIdentity",
    "TokenLengthCount",
    "TokenLengthDistribution",
    "TokenLengthFilterReport",
    "TrainingMessage",
    "TrainValidationSplit",
    "TruncationPolicy",
    "ValidationMetadata",
    "ValidationResult",
    "deduplicate_exact_records",
    "deterministic_train_validation_split",
    "filter_by_token_length",
    "filter_required_content",
    "filter_with_canonical_tokenizer",
    "load_canonical_tokenizer",
    "normalize_record_text",
    "normalize_training_text",
    "normalized_record_fingerprint",
    "prepare_data",
    "single_turn_messages",
    "source_record_identity",
    "token_length_distribution",
    "tokenize_training_record",
]


def prepare_data() -> NoReturn:
    """Run generic data preparation once the Phase 3 pipeline is implemented."""
    raise SystemExit("Data preparation is scaffolded; implementation is tracked by Phase 3.")
