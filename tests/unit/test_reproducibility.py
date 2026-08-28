"""Tests for deterministic project-wide seeding and dataset splitting."""

import os
import random

import numpy as np
import pytest
import torch

from tiny_qwen_coder.config import ConfigError, DataPreparationConfig
from tiny_qwen_coder.data import deterministic_train_validation_split
from tiny_qwen_coder.reproducibility import (
    SeedError,
    make_torch_generator,
    seed_everything,
    seed_torch_worker,
    validate_seed,
)


def _draw_random_values() -> tuple[float, float, tuple[float, ...]]:
    return (
        random.random(),
        float(np.random.random()),
        tuple(torch.rand(4).tolist()),
    )


def test_seed_everything_repeats_python_numpy_and_torch_cpu_sequences() -> None:
    first_settings = seed_everything(1729)
    first = _draw_random_values()

    second_settings = seed_everything(1729)
    second = _draw_random_values()

    assert first == second
    assert first_settings == second_settings
    assert first_settings.deterministic_algorithms is True
    assert first_settings.cudnn_benchmark is False
    assert first_settings.cudnn_deterministic is True
    assert os.environ["PYTHONHASHSEED"] == "1729"
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_different_seed_changes_random_sequences() -> None:
    seed_everything(100)
    first = _draw_random_values()
    seed_everything(101)
    second = _draw_random_values()

    assert first != second


def test_seed_validation_uses_common_numpy_compatible_range() -> None:
    validate_seed(0)
    validate_seed(2**32 - 1)

    for invalid in (-1, 2**32, True):
        with pytest.raises(SeedError):
            validate_seed(invalid)


def test_config_rejects_seed_outside_project_range() -> None:
    with pytest.raises(ConfigError, match="seed must be between 0 and"):
        DataPreparationConfig(
            schema_version=1,
            language="python",
            source_configs=("configs/data/python/source.yaml",),
            output_dir="data/python/p0",
            seed=-1,
            validation_fraction=0.05,
            min_tokens=0,
            max_tokens=2048,
            truncation_policy="reject",
            deduplicate=True,
        )


def test_torch_worker_seed_repeats_python_and_numpy_sequences() -> None:
    torch.manual_seed(31415)
    seed_torch_worker(0)
    first = (random.random(), float(np.random.random()))

    torch.manual_seed(31415)
    seed_torch_worker(0)
    second = (random.random(), float(np.random.random()))

    assert first == second


def test_seeded_torch_generator_repeats_sampler_sequence() -> None:
    first = torch.randperm(20, generator=make_torch_generator(42))
    second = torch.randperm(20, generator=make_torch_generator(42))

    assert torch.equal(first, second)


def test_deterministic_split_is_repeatable_and_isolated_from_global_rng() -> None:
    records = tuple(f"record-{index}" for index in range(100))

    random.seed(999)
    first = deterministic_train_validation_split(records, validation_fraction=0.05, seed=1729)
    for _ in range(100):
        random.random()
    second = deterministic_train_validation_split(records, validation_fraction=0.05, seed=1729)

    assert first == second
    assert len(first.train) == 95
    assert len(first.validation) == 5
    assert set(first.train).isdisjoint(first.validation)
    assert set(first.train) | set(first.validation) == set(records)


def test_deterministic_split_changes_assignment_with_seed() -> None:
    records = tuple(range(100))

    first = deterministic_train_validation_split(records, validation_fraction=0.05, seed=1)
    second = deterministic_train_validation_split(records, validation_fraction=0.05, seed=2)

    assert first != second


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_seed_everything_repeats_cuda_sequence() -> None:
    seed_everything(1729)
    first = torch.rand(8, device="cuda").cpu()
    seed_everything(1729)
    second = torch.rand(8, device="cuda").cpu()

    assert torch.equal(first, second)
