"""Deterministic dataset ordering and train/validation splitting."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, TypeVar

from tiny_qwen_coder.data.deduplication import (
    ExactDeduplicationReport,
    RecordContentFingerprint,
)
from tiny_qwen_coder.data.records import NormalizedTrainingRecord
from tiny_qwen_coder.reproducibility import validate_seed

T = TypeVar("T")
_SCHEMA_VERSION = 1


class DatasetSplittingError(ValueError):
    """Raised when a leakage-safe deterministic dataset split cannot be produced."""


class DatasetPartition(StrEnum):
    """Stable names for the two prepared-dataset partitions."""

    TRAIN = "train"
    VALIDATION = "validation"


@dataclass(frozen=True, slots=True)
class TrainValidationSplit(Generic[T]):
    """Immutable deterministic train/validation split."""

    train: tuple[T, ...]
    validation: tuple[T, ...]


@dataclass(frozen=True, slots=True)
class LinkedPromptGroup:
    """Unique records that share one normalized prompt fingerprint."""

    prompt_sha256: str
    unique_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.prompt_sha256) != 64:
            raise DatasetSplittingError("linked-group prompt SHA-256 must contain 64 characters")
        if not self.unique_indices:
            raise DatasetSplittingError("linked prompt group must contain at least one record")
        if any(index < 0 for index in self.unique_indices):
            raise DatasetSplittingError("linked prompt group indices must not be negative")
        if tuple(sorted(self.unique_indices)) != self.unique_indices:
            raise DatasetSplittingError("linked prompt group indices must be sorted")
        if len(self.unique_indices) != len(set(self.unique_indices)):
            raise DatasetSplittingError("linked prompt group indices must be unique")


@dataclass(frozen=True, slots=True)
class DatasetSplitMembership:
    """Auditable split assignment for one deduplicated record."""

    unique_index: int
    partition: DatasetPartition
    record_sha256: str
    prompt_sha256: str
    source_id: str
    source_record_id: str | None

    def __post_init__(self) -> None:
        if self.unique_index < 0:
            raise DatasetSplittingError("membership unique_index must not be negative")
        for field_name, value in (
            ("record_sha256", self.record_sha256),
            ("prompt_sha256", self.prompt_sha256),
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise DatasetSplittingError(f"{field_name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class DeduplicatedDatasetSplit:
    """Leakage-safe deterministic split of one exact-deduplication report."""

    schema_version: int
    seed: int
    requested_validation_fraction: float
    target_validation_records: int
    linked_prompt_group_count: int
    train_records: tuple[NormalizedTrainingRecord, ...]
    validation_records: tuple[NormalizedTrainingRecord, ...]
    train_fingerprints: tuple[RecordContentFingerprint, ...]
    validation_fingerprints: tuple[RecordContentFingerprint, ...]
    memberships: tuple[DatasetSplitMembership, ...]

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise DatasetSplittingError(
                f"unsupported split schema_version {self.schema_version}; expected {_SCHEMA_VERSION}"
            )
        validate_seed(self.seed)
        if not 0.0 < self.requested_validation_fraction < 1.0:
            raise DatasetSplittingError(
                "requested_validation_fraction must be greater than 0 and less than 1"
            )
        total = len(self.train_records) + len(self.validation_records)
        if total < 2:
            raise DatasetSplittingError("dataset split requires at least two records")
        if not self.train_records or not self.validation_records:
            raise DatasetSplittingError("both train and validation partitions must be non-empty")
        if len(self.train_records) != len(self.train_fingerprints):
            raise DatasetSplittingError("train records and fingerprints must align")
        if len(self.validation_records) != len(self.validation_fingerprints):
            raise DatasetSplittingError("validation records and fingerprints must align")
        if len(self.memberships) != total:
            raise DatasetSplittingError("membership count must equal total split record count")
        if self.target_validation_records < 1 or self.target_validation_records >= total:
            raise DatasetSplittingError(
                "target validation count must leave both partitions non-empty"
            )
        if self.linked_prompt_group_count < 2:
            raise DatasetSplittingError("at least two linked prompt groups are required")

        train_record_hashes = {item.record_sha256 for item in self.train_fingerprints}
        validation_record_hashes = {item.record_sha256 for item in self.validation_fingerprints}
        if train_record_hashes & validation_record_hashes:
            raise DatasetSplittingError("exact content fingerprints must not cross partitions")

        train_prompt_hashes = {item.prompt_sha256 for item in self.train_fingerprints}
        validation_prompt_hashes = {item.prompt_sha256 for item in self.validation_fingerprints}
        if train_prompt_hashes & validation_prompt_hashes:
            raise DatasetSplittingError("linked prompt fingerprints must not cross partitions")

        membership_indices = tuple(item.unique_index for item in self.memberships)
        if membership_indices != tuple(range(total)):
            raise DatasetSplittingError("memberships must be ordered by every unique record index")
        membership_record_hashes = {item.record_sha256 for item in self.memberships}
        if membership_record_hashes != train_record_hashes | validation_record_hashes:
            raise DatasetSplittingError("memberships must cover every split content fingerprint")

    @property
    def total_records(self) -> int:
        """Return the number of deduplicated records assigned to either partition."""

        return len(self.train_records) + len(self.validation_records)

    @property
    def actual_validation_fraction(self) -> float:
        """Return the achieved validation fraction after whole-group assignment."""

        return len(self.validation_records) / self.total_records


def deterministic_train_validation_split(
    items: Sequence[T],
    *,
    validation_fraction: float,
    seed: int,
) -> TrainValidationSplit[T]:
    """Shuffle and split ``items`` reproducibly using an isolated seeded RNG.

    This generic P1 helper remains available for callers without dataset linkage.
    P3-007 dataset preparation should use :func:`split_deduplicated_records`, which
    keeps normalized-prompt-linked records together.
    """

    validate_seed(seed)
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be greater than 0 and less than 1")
    if len(items) < 2:
        raise ValueError("at least two items are required for a train/validation split")

    indices = list(range(len(items)))
    random.Random(seed).shuffle(indices)

    validation_count = round(len(indices) * validation_fraction)
    validation_count = max(1, min(len(indices) - 1, validation_count))

    validation = tuple(items[index] for index in indices[:validation_count])
    train = tuple(items[index] for index in indices[validation_count:])
    return TrainValidationSplit(train=train, validation=validation)


def _linked_prompt_groups(
    fingerprints: Sequence[RecordContentFingerprint],
) -> tuple[LinkedPromptGroup, ...]:
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for unique_index, fingerprint in enumerate(fingerprints):
        grouped_indices[fingerprint.prompt_sha256].append(unique_index)
    return tuple(
        LinkedPromptGroup(
            prompt_sha256=prompt_sha256,
            unique_indices=tuple(grouped_indices[prompt_sha256]),
        )
        for prompt_sha256 in sorted(grouped_indices)
    )


def _target_validation_count(total_records: int, validation_fraction: float) -> int:
    if not 0.0 < validation_fraction < 1.0:
        raise DatasetSplittingError("validation_fraction must be greater than 0 and less than 1")
    if total_records < 2:
        raise DatasetSplittingError("at least two deduplicated records are required")
    target = round(total_records * validation_fraction)
    return max(1, min(total_records - 1, target))


def _choose_validation_group_count(
    shuffled_groups: Sequence[LinkedPromptGroup],
    target_records: int,
) -> int:
    """Choose the shuffled non-empty prefix closest to the record-count target."""

    if len(shuffled_groups) < 2:
        raise DatasetSplittingError(
            "cannot create leakage-safe train/validation split from one linked prompt group"
        )

    cumulative_records = 0
    best_group_count = 1
    best_key: tuple[int, bool, int, int] | None = None
    for group_count in range(1, len(shuffled_groups)):
        cumulative_records += len(shuffled_groups[group_count - 1].unique_indices)
        key = (
            abs(cumulative_records - target_records),
            cumulative_records > target_records,
            cumulative_records,
            group_count,
        )
        if best_key is None or key < best_key:
            best_key = key
            best_group_count = group_count
    return best_group_count


def split_deduplicated_records(
    deduplication: ExactDeduplicationReport,
    *,
    validation_fraction: float,
    seed: int,
) -> DeduplicatedDatasetSplit:
    """Split an exact-deduplicated corpus reproducibly without prompt-link leakage.

    P3-006's report is required as input, making deduplication structurally precede
    splitting. Records sharing a normalized prompt SHA-256 are treated as one
    indivisible linkage group even when their assistant responses differ. Groups are
    sorted by prompt fingerprint, shuffled with an isolated RNG seeded from the
    configured dataset seed, and assigned as whole units.
    """

    validate_seed(seed)
    records = deduplication.unique_records
    fingerprints = deduplication.unique_fingerprints
    if len(records) != len(fingerprints):
        raise DatasetSplittingError("deduplicated records and fingerprints must align")

    target_validation_records = _target_validation_count(len(records), validation_fraction)
    groups = list(_linked_prompt_groups(fingerprints))
    if len(groups) < 2:
        raise DatasetSplittingError(
            "cannot create leakage-safe train/validation split from one linked prompt group"
        )
    random.Random(seed).shuffle(groups)

    validation_group_count = _choose_validation_group_count(groups, target_validation_records)
    validation_groups = groups[:validation_group_count]
    train_groups = groups[validation_group_count:]

    validation_indices = tuple(
        unique_index for group in validation_groups for unique_index in group.unique_indices
    )
    train_indices = tuple(
        unique_index for group in train_groups for unique_index in group.unique_indices
    )

    validation_index_set = set(validation_indices)
    memberships = tuple(
        DatasetSplitMembership(
            unique_index=unique_index,
            partition=(
                DatasetPartition.VALIDATION
                if unique_index in validation_index_set
                else DatasetPartition.TRAIN
            ),
            record_sha256=fingerprints[unique_index].record_sha256,
            prompt_sha256=fingerprints[unique_index].prompt_sha256,
            source_id=records[unique_index].provenance.source_id,
            source_record_id=records[unique_index].provenance.record_id,
        )
        for unique_index in range(len(records))
    )

    return DeduplicatedDatasetSplit(
        schema_version=_SCHEMA_VERSION,
        seed=seed,
        requested_validation_fraction=validation_fraction,
        target_validation_records=target_validation_records,
        linked_prompt_group_count=len(groups),
        train_records=tuple(records[index] for index in train_indices),
        validation_records=tuple(records[index] for index in validation_indices),
        train_fingerprints=tuple(fingerprints[index] for index in train_indices),
        validation_fingerprints=tuple(fingerprints[index] for index in validation_indices),
        memberships=memberships,
    )
