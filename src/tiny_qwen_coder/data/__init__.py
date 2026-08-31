"""Language-neutral dataset preparation services."""

from __future__ import annotations

from typing import NoReturn

# Keep this package initializer limited to leaf data services. ``data.pipeline``
# depends on ``reporting.dataset_manifest``, which imports these leaf modules;
# re-exporting the pipeline here creates a circular package initialization path.

from tiny_qwen_coder.data.deduplication import (
    DeduplicationError,
    DuplicateReason,
    DuplicateReasonCount,
    DuplicateTrainingRecord,
    ExactDeduplicationReport,
    RecordContentFingerprint,
    SourceRecordIdentity,
    deduplicate_exact_records,
    normalized_prompt_sha256,
    normalized_record_fingerprint,
    normalized_response_sha256,
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
from tiny_qwen_coder.data.loading import (
    TrainingRecordLoadingError,
    load_normalized_training_records_jsonl,
    parse_normalized_training_record,
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
    DatasetPartition,
    DatasetSplitMembership,
    DatasetSplittingError,
    DeduplicatedDatasetSplit,
    LinkedPromptGroup,
    TrainValidationSplit,
    deterministic_train_validation_split,
    split_deduplicated_records,
)

__all__ = [
    "AcceptedTokenLength",
    "ContentRejectionReason",
    "DatasetPartition",
    "DatasetSplitMembership",
    "DatasetSplittingError",
    "DeduplicatedDatasetSplit",
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
    "LinkedPromptGroup",
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
    "TrainingRecordLoadingError",
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
    "load_normalized_training_records_jsonl",
    "normalize_record_text",
    "normalize_training_text",
    "normalized_prompt_sha256",
    "normalized_record_fingerprint",
    "normalized_response_sha256",
    "prepare_data",
    "parse_normalized_training_record",
    "single_turn_messages",
    "source_record_identity",
    "split_deduplicated_records",
    "token_length_distribution",
    "tokenize_training_record",
]


def prepare_data() -> NoReturn:
    """Remain a CLI scaffold until concrete language/source adapters exist."""
    raise SystemExit(
        "Dataset pipeline core is implemented; source loading is tracked by language pipeline tasks."
    )
