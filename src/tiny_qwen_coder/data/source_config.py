"""Strict source configuration for pinned Hugging Face training datasets."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from tiny_qwen_coder.data.records import LicenseMetadata

_COMPONENT_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]*$")
_LANGUAGE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")
_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_IMPORT_REFERENCE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_.]*$")


class DatasetSourceConfigError(ValueError):
    """Raised when one dataset-source configuration is malformed."""


def _require_non_empty(value: str, *, field_name: str) -> None:
    if not value.strip():
        raise DatasetSourceConfigError(f"{field_name} must not be empty")


@dataclass(frozen=True, slots=True)
class HuggingFaceDatasetRef:
    """Immutable identity for one Hugging Face dataset split."""

    repository: str
    revision: str
    split: str

    def __post_init__(self) -> None:
        _require_non_empty(self.repository, field_name="dataset repository")
        if "/" not in self.repository:
            raise DatasetSourceConfigError("dataset repository must use 'owner/name' form")
        if not _REVISION_PATTERN.fullmatch(self.revision):
            raise DatasetSourceConfigError("dataset revision must be a lowercase 40-character SHA")
        _require_non_empty(self.split, field_name="dataset split")


@dataclass(frozen=True, slots=True)
class SourceSelection:
    """One explicit equality filter evaluated against an upstream dataset row."""

    field: str
    equals: str

    def __post_init__(self) -> None:
        _require_non_empty(self.field, field_name="selection field")
        if any(not segment for segment in self.field.split(".")):
            raise DatasetSourceConfigError("selection field must be a dotted non-empty path")
        _require_non_empty(self.equals, field_name="selection equals")


@dataclass(frozen=True, slots=True)
class DatasetSourceConfig:
    """Pinned source configuration shared by language-specific dataset adapters."""

    schema_version: int
    id: str
    language: str
    adapter: str
    dataset: HuggingFaceDatasetRef
    license: LicenseMetadata
    selection: SourceSelection

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise DatasetSourceConfigError("unsupported dataset source schema version")
        if not _COMPONENT_ID_PATTERN.fullmatch(self.id):
            raise DatasetSourceConfigError("source id must be a stable lowercase component ID")
        if not _LANGUAGE_ID_PATTERN.fullmatch(self.language):
            raise DatasetSourceConfigError("source language must be a stable lowercase language ID")
        if not _IMPORT_REFERENCE_PATTERN.fullmatch(self.adapter):
            raise DatasetSourceConfigError(
                "source adapter must use 'package.module:attribute' syntax"
            )


def _strict_mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise DatasetSourceConfigError(f"{context} must be a mapping")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise DatasetSourceConfigError(f"{context} keys must be strings")
        result[key] = item
    return result


def _validate_keys(
    mapping: Mapping[str, object],
    *,
    required: frozenset[str],
    context: str,
) -> None:
    unknown = sorted(set(mapping) - required)
    missing = sorted(required - set(mapping))
    if unknown:
        raise DatasetSourceConfigError(f"{context} contains unknown field(s): {', '.join(unknown)}")
    if missing:
        raise DatasetSourceConfigError(
            f"{context} is missing required field(s): {', '.join(missing)}"
        )


def _expect_str(mapping: Mapping[str, object], key: str, *, context: str) -> str:
    value = mapping[key]
    if not isinstance(value, str) or not value.strip():
        raise DatasetSourceConfigError(f"{context}.{key} must be a non-empty string")
    return value


def _expect_int(mapping: Mapping[str, object], key: str, *, context: str) -> int:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise DatasetSourceConfigError(f"{context}.{key} must be an integer")
    return value


def _parse_dataset(value: object) -> HuggingFaceDatasetRef:
    context = "source.dataset"
    mapping = _strict_mapping(value, context=context)
    _validate_keys(
        mapping,
        required=frozenset({"repository", "revision", "split"}),
        context=context,
    )
    return HuggingFaceDatasetRef(
        repository=_expect_str(mapping, "repository", context=context),
        revision=_expect_str(mapping, "revision", context=context),
        split=_expect_str(mapping, "split", context=context),
    )


def _parse_license(value: object) -> LicenseMetadata:
    context = "source.license"
    mapping = _strict_mapping(value, context=context)
    _validate_keys(mapping, required=frozenset({"name", "url"}), context=context)
    try:
        return LicenseMetadata(
            name=_expect_str(mapping, "name", context=context),
            url=_expect_str(mapping, "url", context=context),
        )
    except (TypeError, ValueError) as exc:
        raise DatasetSourceConfigError(str(exc)) from exc


def _parse_selection(value: object) -> SourceSelection:
    context = "source.selection"
    mapping = _strict_mapping(value, context=context)
    _validate_keys(mapping, required=frozenset({"field", "equals"}), context=context)
    return SourceSelection(
        field=_expect_str(mapping, "field", context=context),
        equals=_expect_str(mapping, "equals", context=context),
    )


def parse_dataset_source_config(value: object) -> DatasetSourceConfig:
    """Parse one strict pinned source configuration."""

    context = "source"
    mapping = _strict_mapping(value, context=context)
    _validate_keys(
        mapping,
        required=frozenset(
            {"schema_version", "id", "language", "adapter", "dataset", "license", "selection"}
        ),
        context=context,
    )
    return DatasetSourceConfig(
        schema_version=_expect_int(mapping, "schema_version", context=context),
        id=_expect_str(mapping, "id", context=context),
        language=_expect_str(mapping, "language", context=context),
        adapter=_expect_str(mapping, "adapter", context=context),
        dataset=_parse_dataset(mapping["dataset"]),
        license=_parse_license(mapping["license"]),
        selection=_parse_selection(mapping["selection"]),
    )


def load_dataset_source_config(path: Path) -> DatasetSourceConfig:
    """Load one pinned dataset source config from YAML."""

    try:
        loaded: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DatasetSourceConfigError(f"could not read source config {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise DatasetSourceConfigError(f"invalid YAML in {path}: {exc}") from exc
    return parse_dataset_source_config(loaded)
