"""Tests for P3-007 deterministic leakage-safe dataset splitting."""

import random

import pytest

from tiny_qwen_coder.config import DataPreparationConfig
from tiny_qwen_coder.data import (
    DatasetPartition,
    DatasetSplittingError,
    LicenseMetadata,
    NormalizedTrainingRecord,
    SourceProvenance,
    deduplicate_exact_records,
    single_turn_messages,
    split_deduplicated_records,
)


def _record(
    *,
    prompt: str,
    response: str,
    record_id: str,
    source_id: str = "synthetic/p3-007",
) -> NormalizedTrainingRecord:
    return NormalizedTrainingRecord(
        schema_version=1,
        language="python",
        messages=single_turn_messages(
            system="You are a coding assistant.",
            user=prompt,
            assistant=response,
        ),
        provenance=SourceProvenance(
            source_id=source_id,
            revision="fixture-r1",
            split="upstream",
            record_id=record_id,
            license=LicenseMetadata(name="synthetic-test-fixture"),
        ),
    )


def _corpus(count: int = 20) -> tuple[NormalizedTrainingRecord, ...]:
    return tuple(
        _record(
            prompt=f"Return the integer {index}.",
            response=f"return {index}",
            record_id=f"record-{index}",
        )
        for index in range(count)
    )


def test_same_inputs_config_and_seed_produce_same_membership() -> None:
    deduplication = deduplicate_exact_records(_corpus())
    config = DataPreparationConfig(
        schema_version=1,
        language="python",
        source_configs=("configs/data/python/source.yaml",),
        output_dir="data/python/p0",
        seed=1729,
        validation_fraction=0.20,
        min_tokens=1,
        max_tokens=2048,
        truncation_policy="reject",
        deduplicate=True,
    )

    first = split_deduplicated_records(
        deduplication,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
    )
    second = split_deduplicated_records(
        deduplication,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
    )

    assert first == second
    assert len(first.validation_records) == 4
    assert len(first.train_records) == 16
    assert first.actual_validation_fraction == 0.20


def test_split_is_isolated_from_process_global_random_state() -> None:
    deduplication = deduplicate_exact_records(_corpus())

    random.seed(999)
    first = split_deduplicated_records(deduplication, validation_fraction=0.25, seed=42)
    for _ in range(200):
        random.random()
    second = split_deduplicated_records(deduplication, validation_fraction=0.25, seed=42)

    assert first == second


def test_different_seed_changes_group_assignment() -> None:
    deduplication = deduplicate_exact_records(_corpus(30))

    first = split_deduplicated_records(deduplication, validation_fraction=0.20, seed=1)
    second = split_deduplicated_records(deduplication, validation_fraction=0.20, seed=2)

    first_validation = {item.record_sha256 for item in first.validation_fingerprints}
    second_validation = {item.record_sha256 for item in second.validation_fingerprints}
    assert first_validation != second_validation


def test_split_consumes_deduplicated_records_only() -> None:
    original = _record(prompt="Return one.", response="return 1", record_id="one-a")
    duplicate = _record(
        prompt="Return one.",
        response="return 1",
        record_id="one-b",
        source_id="synthetic/other-source",
    )
    second = _record(prompt="Return two.", response="return 2", record_id="two")
    report = deduplicate_exact_records((original, duplicate, second))

    split = split_deduplicated_records(report, validation_fraction=0.50, seed=7)

    assert report.unique_count == 2
    assert split.total_records == 2
    assert len(split.memberships) == 2


def test_records_with_same_prompt_and_different_responses_stay_together() -> None:
    first_variant = _record(
        prompt="Implement add(a, b).",
        response="def add(a, b):\n    return a + b",
        record_id="add-a",
    )
    second_variant = _record(
        prompt="Implement add(a, b).",
        response="def add(a, b):\n    result = a + b\n    return result",
        record_id="add-b",
    )
    others = tuple(
        _record(
            prompt=f"Implement task {index}.",
            response=f"return {index}",
            record_id=f"other-{index}",
        )
        for index in range(8)
    )
    deduplication = deduplicate_exact_records((first_variant, second_variant, *others))

    split = split_deduplicated_records(deduplication, validation_fraction=0.30, seed=123)

    memberships = {item.unique_index: item.partition for item in split.memberships}
    assert memberships[0] == memberships[1]
    assert deduplication.unique_fingerprints[0].prompt_sha256 == (
        deduplication.unique_fingerprints[1].prompt_sha256
    )


def test_prompt_fingerprints_are_disjoint_across_partitions() -> None:
    linked = (
        _record(prompt="Shared prompt", response="answer one", record_id="linked-1"),
        _record(prompt="Shared prompt", response="answer two", record_id="linked-2"),
    )
    deduplication = deduplicate_exact_records((*linked, *_corpus(12)))

    split = split_deduplicated_records(deduplication, validation_fraction=0.25, seed=31415)

    train_prompts = {item.prompt_sha256 for item in split.train_fingerprints}
    validation_prompts = {item.prompt_sha256 for item in split.validation_fingerprints}
    assert train_prompts.isdisjoint(validation_prompts)


def test_exact_record_fingerprints_are_disjoint_across_partitions() -> None:
    deduplication = deduplicate_exact_records(_corpus())

    split = split_deduplicated_records(deduplication, validation_fraction=0.20, seed=1729)

    train_records = {item.record_sha256 for item in split.train_fingerprints}
    validation_records = {item.record_sha256 for item in split.validation_fingerprints}
    assert train_records.isdisjoint(validation_records)
    assert train_records | validation_records == {
        item.record_sha256 for item in deduplication.unique_fingerprints
    }


def test_memberships_cover_every_unique_record_in_original_index_order() -> None:
    deduplication = deduplicate_exact_records(_corpus(10))

    split = split_deduplicated_records(deduplication, validation_fraction=0.30, seed=50)

    assert tuple(item.unique_index for item in split.memberships) == tuple(range(10))
    assert {item.partition for item in split.memberships} == {
        DatasetPartition.TRAIN,
        DatasetPartition.VALIDATION,
    }
    assert all(item.source_id == "synthetic/p3-007" for item in split.memberships)


def test_whole_group_assignment_reports_achieved_fraction_when_exact_target_is_impossible() -> None:
    linked = tuple(
        _record(
            prompt="One linked task",
            response=f"variant {index}",
            record_id=f"linked-{index}",
        )
        for index in range(4)
    )
    singleton = _record(prompt="Independent task", response="answer", record_id="singleton")
    deduplication = deduplicate_exact_records((*linked, singleton))

    split = split_deduplicated_records(deduplication, validation_fraction=0.50, seed=0)

    assert split.target_validation_records == 2
    assert split.actual_validation_fraction in {0.20, 0.80}
    assert split.actual_validation_fraction != 0.50
    assert split.linked_prompt_group_count == 2


def test_single_linked_prompt_group_fails_closed() -> None:
    records = tuple(
        _record(
            prompt="Same task",
            response=f"response {index}",
            record_id=f"variant-{index}",
        )
        for index in range(3)
    )
    deduplication = deduplicate_exact_records(records)

    with pytest.raises(DatasetSplittingError, match="one linked prompt group"):
        split_deduplicated_records(deduplication, validation_fraction=0.20, seed=1)


@pytest.mark.parametrize("validation_fraction", [0.0, 1.0, -0.1, 1.1])
def test_invalid_validation_fraction_is_rejected(validation_fraction: float) -> None:
    deduplication = deduplicate_exact_records(_corpus())

    with pytest.raises(DatasetSplittingError, match="validation_fraction"):
        split_deduplicated_records(
            deduplication,
            validation_fraction=validation_fraction,
            seed=1729,
        )
