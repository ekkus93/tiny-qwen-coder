"""Importable fixture hooks used by the generic dataset-pipeline tests."""

from __future__ import annotations

from tiny_qwen_coder.data.records import NormalizedTrainingRecord, ValidationResult


def validate_fixture(record: NormalizedTrainingRecord) -> ValidationResult:
    """Return deterministic evidence proving the plugin hook saw normalized text."""

    text_is_normalized = all(
        "\r" not in message.content and not message.content.startswith("\ufeff")
        for message in record.messages
    )
    assistant = next(
        message for message in reversed(record.messages) if message.role == "assistant"
    )
    contains_invalid_marker = "INVALID" in assistant.content
    passed = text_is_normalized and not contains_invalid_marker
    detail = None if passed else "fixture validator rejected non-normalized or INVALID content"
    return ValidationResult(validator_id="fixture.syntax", passed=passed, detail=detail)


def execute_fixture() -> None:
    """Satisfy the fixture language spec's declared executor reference."""

    raise NotImplementedError("execution is outside the P3-009 dataset-pipeline fixture")
