"""Language-neutral dataset preparation services."""

from __future__ import annotations

from typing import NoReturn

from tiny_qwen_coder.data.filtering import (
    ContentRejectionReason,
    RejectedTrainingRecord,
    RejectionReasonCount,
    RequiredContentFilterReport,
    filter_required_content,
    normalize_record_text,
    normalize_training_text,
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
    "ContentRejectionReason",
    "LicenseMetadata",
    "MessageRole",
    "NormalizedTrainingRecord",
    "RejectedTrainingRecord",
    "RejectionReasonCount",
    "RequiredContentFilterReport",
    "SourceProvenance",
    "TrainingMessage",
    "TrainValidationSplit",
    "ValidationMetadata",
    "ValidationResult",
    "deterministic_train_validation_split",
    "filter_required_content",
    "normalize_record_text",
    "normalize_training_text",
    "prepare_data",
    "single_turn_messages",
]


def prepare_data() -> NoReturn:
    """Run generic data preparation once the Phase 3 pipeline is implemented."""
    raise SystemExit("Data preparation is scaffolded; implementation is tracked by Phase 3.")
