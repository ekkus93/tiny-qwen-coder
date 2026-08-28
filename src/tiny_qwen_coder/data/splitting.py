"""Deterministic dataset ordering and train/validation splitting."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from tiny_qwen_coder.reproducibility import validate_seed

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class TrainValidationSplit(Generic[T]):
    """Immutable deterministic train/validation split."""

    train: tuple[T, ...]
    validation: tuple[T, ...]


def deterministic_train_validation_split(
    items: Sequence[T],
    *,
    validation_fraction: float,
    seed: int,
) -> TrainValidationSplit[T]:
    """Shuffle and split ``items`` reproducibly using an isolated seeded RNG.

    The local RNG means the result does not depend on unrelated calls to the process-global
    ``random`` module. Both partitions are returned in their deterministic shuffled order.
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
