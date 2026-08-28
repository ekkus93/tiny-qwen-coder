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
from tiny_qwen_coder.data.pipeline import (
    DatasetPipelineError,
    DatasetPipelineResult,
    LanguageRecordValidator,
    ValidatorResolver,
    apply_language_validators,
    resolve_language_validator,
    run_dataset_pipeline,
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
    "DatasetPipelineError",
    "DatasetPipelineResult",
    "DatasetSplitMembership",
    "DatasetSplittingError",
    "DeduplicatedDatasetSplit",
    "DeduplicationError",
    "DuplicateReason",
    "DuplicateReasonCount",
    "DuplicateTrainingRecord",
    "ExactDeduplicationReport",
    "LanguageRecordValidator",
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
    "TrainValidationSplit",
    "TruncationPolicy",
    "ValidationMetadata",
    "ValidationResult",
    "ValidatorResolver",
    "apply_language_validators",
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
    "resolve_language_validator",
    "run_dataset_pipeline",
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
