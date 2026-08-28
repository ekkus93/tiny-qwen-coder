from __future__ import annotations

from collections.abc import Mapping

import pytest

from tiny_qwen_coder.data import (
    LicenseMetadata,
    NormalizedTrainingRecord,
    SourceProvenance,
    TrainingMessage,
    ValidationMetadata,
    ValidationResult,
    single_turn_messages,
)


def _provenance() -> SourceProvenance:
    return SourceProvenance(
        source_id="example/coding-instruct",
        revision="0123456789abcdef0123456789abcdef01234567",
        license=LicenseMetadata(
            name="Apache-2.0",
            url="https://www.apache.org/licenses/LICENSE-2.0",
        ),
        split="train",
        record_id="row-17",
        url="https://example.invalid/datasets/coding-instruct",
    )


def _normalize_instruction_row(row: Mapping[str, str]) -> NormalizedTrainingRecord:
    return NormalizedTrainingRecord(
        schema_version=1,
        messages=single_turn_messages(
            system=row["system"],
            user=row["instruction"],
            assistant=row["output"],
        ),
        language="python",
        provenance=_provenance(),
    )


def _normalize_message_row(row: Mapping[str, object]) -> NormalizedTrainingRecord:
    raw_messages = row["messages"]
    assert isinstance(raw_messages, list)
    messages = tuple(
        TrainingMessage(role=message["role"], content=message["content"])
        for message in raw_messages
        if isinstance(message, dict)
    )
    return NormalizedTrainingRecord(
        schema_version=1,
        messages=messages,
        language="python",
        provenance=_provenance(),
    )


def test_single_turn_messages_use_system_user_assistant_representation() -> None:
    messages = single_turn_messages(
        system="You are a Python coding assistant.",
        user="Return the square of x.",
        assistant="def square(x):\n    return x * x",
    )

    assert tuple(message.role for message in messages) == ("system", "user", "assistant")
    assert messages[1].content == "Return the square of x."


def test_multiple_upstream_shapes_normalize_to_same_internal_record() -> None:
    instruction_row = {
        "system": "You are a Python coding assistant.",
        "instruction": "Return the square of x.",
        "output": "def square(x):\n    return x * x",
    }
    message_row: dict[str, object] = {
        "messages": [
            {"role": "system", "content": "You are a Python coding assistant."},
            {"role": "user", "content": "Return the square of x."},
            {"role": "assistant", "content": "def square(x):\n    return x * x"},
        ]
    }

    assert _normalize_instruction_row(instruction_row) == _normalize_message_row(message_row)


def test_record_preserves_source_revision_license_and_optional_validation() -> None:
    validation = ValidationMetadata(
        results=(
            ValidationResult(validator_id="python.syntax", passed=True),
            ValidationResult(
                validator_id="generic.required-content",
                passed=True,
                detail="prompt and response present",
            ),
        )
    )
    record = NormalizedTrainingRecord(
        schema_version=1,
        messages=single_turn_messages(system=None, user="print 1", assistant="print(1)"),
        language="python",
        provenance=_provenance(),
        validation=validation,
    )

    assert record.provenance.source_id == "example/coding-instruct"
    assert record.provenance.revision == "0123456789abcdef0123456789abcdef01234567"
    assert record.provenance.license.name == "Apache-2.0"
    assert record.validation is not None
    assert record.validation.passed is True


def test_validation_metadata_reports_failed_validator() -> None:
    metadata = ValidationMetadata(
        results=(
            ValidationResult(validator_id="python.syntax", passed=False, detail="SyntaxError"),
        )
    )

    assert metadata.passed is False


def test_validation_metadata_rejects_duplicate_validator_ids() -> None:
    with pytest.raises(ValueError, match="must not repeat validator IDs"):
        ValidationMetadata(
            results=(
                ValidationResult(validator_id="python.syntax", passed=True),
                ValidationResult(validator_id="python.syntax", passed=False),
            )
        )


def test_training_message_rejects_unknown_role() -> None:
    with pytest.raises(ValueError, match="unsupported training message role"):
        TrainingMessage(role="tool", content="output")  # type: ignore[arg-type]


def test_record_rejects_invalid_language_id() -> None:
    with pytest.raises(ValueError, match="stable lowercase language ID"):
        NormalizedTrainingRecord(
            schema_version=1,
            messages=single_turn_messages(system=None, user="u", assistant="a"),
            language="Python",
            provenance=_provenance(),
        )


def test_record_rejects_unknown_schema_version() -> None:
    with pytest.raises(ValueError, match="unsupported normalized training-record schema version"):
        NormalizedTrainingRecord(
            schema_version=2,
            messages=single_turn_messages(system=None, user="u", assistant="a"),
            language="python",
            provenance=_provenance(),
        )


def test_empty_content_is_representable_for_p3_004_filtering() -> None:
    record = NormalizedTrainingRecord(
        schema_version=1,
        messages=single_turn_messages(system=None, user="", assistant=""),
        language="python",
        provenance=_provenance(),
    )

    assert record.messages[0].content == ""
    assert record.messages[1].content == ""


def test_provenance_requires_source_identity_revision_and_license_name() -> None:
    with pytest.raises(ValueError, match="source id must not be empty"):
        SourceProvenance(
            source_id=" ",
            revision="revision",
            license=LicenseMetadata(name="MIT"),
        )
    with pytest.raises(ValueError, match="source revision must not be empty"):
        SourceProvenance(
            source_id="example/source",
            revision=" ",
            license=LicenseMetadata(name="MIT"),
        )
    with pytest.raises(ValueError, match="license name must not be empty"):
        LicenseMetadata(name=" ")
