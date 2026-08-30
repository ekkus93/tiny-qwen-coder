from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from tiny_qwen_coder.data import (
    LicenseMetadata,
    NormalizedTrainingRecord,
    SourceProvenance,
    TrainingMessage,
    TrainingRecordLoadingError,
    ValidationMetadata,
    ValidationResult,
    load_normalized_training_records_jsonl,
    parse_normalized_training_record,
)


def _record(*, language: str = "rust") -> NormalizedTrainingRecord:
    return NormalizedTrainingRecord(
        schema_version=1,
        messages=(
            TrainingMessage(role="system", content="Write safe code."),
            TrainingMessage(role="user", content="Implement add."),
            TrainingMessage(role="assistant", content="fn add(a: i32, b: i32) -> i32 { a + b }"),
        ),
        language=language,
        provenance=SourceProvenance(
            source_id="fixture",
            revision="revision-1",
            license=LicenseMetadata(name="MIT", url="https://example.invalid/license"),
            split="train",
            record_id="row-1",
            source_metadata=(("origin", "unit-test"),),
        ),
        validation=ValidationMetadata(
            results=(ValidationResult(validator_id="rust.syntax", passed=True),)
        ),
    )


def test_parse_normalized_training_record_round_trips_language_neutral_schema() -> None:
    expected = _record()

    actual = parse_normalized_training_record(asdict(expected))

    assert actual == expected


def test_jsonl_loader_preserves_records_and_enforces_language(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    expected = _record()
    path.write_text(json.dumps(asdict(expected), sort_keys=True) + "\n", encoding="utf-8")

    assert load_normalized_training_records_jsonl(path, expected_language="rust") == (expected,)

    with pytest.raises(TrainingRecordLoadingError, match="configured language"):
        load_normalized_training_records_jsonl(path, expected_language="python")


def test_jsonl_loader_rejects_blank_lines_and_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    payload = asdict(_record())
    payload["unexpected"] = True
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(TrainingRecordLoadingError, match="unknown field"):
        load_normalized_training_records_jsonl(path)

    path.write_text(json.dumps(asdict(_record())) + "\n\n", encoding="utf-8")
    with pytest.raises(TrainingRecordLoadingError, match="must not be blank"):
        load_normalized_training_records_jsonl(path)
