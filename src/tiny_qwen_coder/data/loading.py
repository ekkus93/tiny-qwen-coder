"""Strict loading for normalized training-record JSONL artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from tiny_qwen_coder.data.records import (
    LicenseMetadata,
    MessageRole,
    NormalizedTrainingRecord,
    SourceProvenance,
    TrainingMessage,
    ValidationMetadata,
    ValidationResult,
)


class TrainingRecordLoadingError(ValueError):
    """Raised when normalized training-record JSONL cannot be loaded safely."""


def _mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TrainingRecordLoadingError(f"{context} must be a mapping")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TrainingRecordLoadingError(f"{context} keys must be strings")
        result[key] = item
    return result


def _validate_keys(
    mapping: Mapping[str, object],
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
    context: str,
) -> None:
    allowed = required | optional
    unknown = sorted(set(mapping) - allowed)
    missing = sorted(required - set(mapping))
    if unknown:
        raise TrainingRecordLoadingError(
            f"{context} contains unknown field(s): {', '.join(unknown)}"
        )
    if missing:
        raise TrainingRecordLoadingError(
            f"{context} is missing required field(s): {', '.join(missing)}"
        )


def _string(mapping: Mapping[str, object], key: str, *, context: str) -> str:
    value = mapping[key]
    if not isinstance(value, str):
        raise TrainingRecordLoadingError(f"{context}.{key} must be a string")
    return value


def _optional_string(mapping: Mapping[str, object], key: str, *, context: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TrainingRecordLoadingError(f"{context}.{key} must be a string or null")
    return value


def _integer(mapping: Mapping[str, object], key: str, *, context: str) -> int:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TrainingRecordLoadingError(f"{context}.{key} must be an integer")
    return value


def _boolean(mapping: Mapping[str, object], key: str, *, context: str) -> bool:
    value = mapping[key]
    if not isinstance(value, bool):
        raise TrainingRecordLoadingError(f"{context}.{key} must be a boolean")
    return value


def _parse_license(value: object, *, context: str) -> LicenseMetadata:
    mapping = _mapping(value, context=context)
    _validate_keys(
        mapping,
        required=frozenset({"name"}),
        optional=frozenset({"url", "attribution"}),
        context=context,
    )
    return LicenseMetadata(
        name=_string(mapping, "name", context=context),
        url=_optional_string(mapping, "url", context=context),
        attribution=_optional_string(mapping, "attribution", context=context),
    )


def _parse_source_metadata(value: object, *, context: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise TrainingRecordLoadingError(f"{context} must be a sequence")
    output: list[tuple[str, str]] = []
    for index, item in enumerate(value):
        item_context = f"{context}[{index}]"
        if not isinstance(item, Sequence) or isinstance(item, str | bytes) or len(item) != 2:
            raise TrainingRecordLoadingError(f"{item_context} must contain exactly two strings")
        key, item_value = item
        if not isinstance(key, str) or not isinstance(item_value, str):
            raise TrainingRecordLoadingError(f"{item_context} must contain exactly two strings")
        output.append((key, item_value))
    return tuple(output)


def _parse_provenance(value: object, *, context: str) -> SourceProvenance:
    mapping = _mapping(value, context=context)
    _validate_keys(
        mapping,
        required=frozenset({"source_id", "revision", "license"}),
        optional=frozenset({"split", "record_id", "url", "source_metadata"}),
        context=context,
    )
    source_metadata = mapping.get("source_metadata", ())
    return SourceProvenance(
        source_id=_string(mapping, "source_id", context=context),
        revision=_string(mapping, "revision", context=context),
        license=_parse_license(mapping["license"], context=f"{context}.license"),
        split=_optional_string(mapping, "split", context=context),
        record_id=_optional_string(mapping, "record_id", context=context),
        url=_optional_string(mapping, "url", context=context),
        source_metadata=_parse_source_metadata(
            source_metadata, context=f"{context}.source_metadata"
        ),
    )


def _parse_messages(value: object, *, context: str) -> tuple[TrainingMessage, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise TrainingRecordLoadingError(f"{context} must be a sequence")
    messages: list[TrainingMessage] = []
    for index, item in enumerate(value):
        item_context = f"{context}[{index}]"
        mapping = _mapping(item, context=item_context)
        _validate_keys(mapping, required=frozenset({"role", "content"}), context=item_context)
        role = _string(mapping, "role", context=item_context)
        if role not in {"system", "user", "assistant"}:
            raise TrainingRecordLoadingError(f"{item_context}.role has unsupported value {role!r}")
        messages.append(
            TrainingMessage(
                role=cast(MessageRole, role),
                content=_string(mapping, "content", context=item_context),
            )
        )
    return tuple(messages)


def _parse_validation(value: object, *, context: str) -> ValidationMetadata | None:
    if value is None:
        return None
    mapping = _mapping(value, context=context)
    _validate_keys(mapping, required=frozenset({"results"}), context=context)
    raw_results = mapping["results"]
    if not isinstance(raw_results, Sequence) or isinstance(raw_results, str | bytes):
        raise TrainingRecordLoadingError(f"{context}.results must be a sequence")
    results: list[ValidationResult] = []
    for index, item in enumerate(raw_results):
        item_context = f"{context}.results[{index}]"
        result_mapping = _mapping(item, context=item_context)
        _validate_keys(
            result_mapping,
            required=frozenset({"validator_id", "passed"}),
            optional=frozenset({"detail"}),
            context=item_context,
        )
        results.append(
            ValidationResult(
                validator_id=_string(result_mapping, "validator_id", context=item_context),
                passed=_boolean(result_mapping, "passed", context=item_context),
                detail=_optional_string(result_mapping, "detail", context=item_context),
            )
        )
    return ValidationMetadata(results=tuple(results))


def parse_normalized_training_record(value: object) -> NormalizedTrainingRecord:
    """Parse one serialized :class:`NormalizedTrainingRecord` mapping strictly."""

    context = "training record"
    mapping = _mapping(value, context=context)
    _validate_keys(
        mapping,
        required=frozenset({"schema_version", "messages", "language", "provenance"}),
        optional=frozenset({"validation"}),
        context=context,
    )
    try:
        return NormalizedTrainingRecord(
            schema_version=_integer(mapping, "schema_version", context=context),
            messages=_parse_messages(mapping["messages"], context=f"{context}.messages"),
            language=_string(mapping, "language", context=context),
            provenance=_parse_provenance(mapping["provenance"], context=f"{context}.provenance"),
            validation=_parse_validation(
                mapping.get("validation"), context=f"{context}.validation"
            ),
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, TrainingRecordLoadingError):
            raise
        raise TrainingRecordLoadingError(str(exc)) from exc


def load_normalized_training_records_jsonl(
    path: Path,
    *,
    expected_language: str | None = None,
) -> tuple[NormalizedTrainingRecord, ...]:
    """Load normalized training records from deterministic UTF-8 JSONL."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise TrainingRecordLoadingError(f"could not read training records {path}: {exc}") from exc

    records: list[NormalizedTrainingRecord] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise TrainingRecordLoadingError(f"{path}:{line_number} must not be blank")
        try:
            raw: object = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TrainingRecordLoadingError(
                f"invalid JSON at {path}:{line_number}: {exc}"
            ) from exc
        record = parse_normalized_training_record(raw)
        if expected_language is not None and record.language != expected_language:
            raise TrainingRecordLoadingError(
                f"{path}:{line_number} language {record.language!r} does not match "
                f"configured language {expected_language!r}"
            )
        records.append(record)

    if not records:
        raise TrainingRecordLoadingError(
            f"training records {path} must contain at least one record"
        )
    return tuple(records)
