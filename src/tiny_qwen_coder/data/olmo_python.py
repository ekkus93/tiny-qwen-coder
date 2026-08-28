"""OLMo StarCoder Python-instruct adapter for the normalized training schema."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import cast

from tiny_qwen_coder.data.records import (
    NormalizedTrainingRecord,
    SourceProvenance,
    single_turn_messages,
)
from tiny_qwen_coder.data.source_config import DatasetSourceConfig
from tiny_qwen_coder.languages.schema import LanguageConfig

_OLMO_SOURCE_ID = "olmo-starcoder-python-instruct"
_OLMO_REPOSITORY = "OLMo-Coding/starcoder-python-instruct"
_OLMO_SELECTION_FIELD = "metadata.extension"
_OLMO_SELECTION_VALUE = "python3"

DatasetRow = Mapping[str, object]
DatasetRowsLoader = Callable[..., Iterable[DatasetRow]]


class OlmoPythonLoaderError(ValueError):
    """Raised when the pinned OLMo source cannot be normalized safely."""


def _load_huggingface_rows(
    repository: str,
    *,
    revision: str,
    split: str,
    streaming: bool,
) -> Iterable[DatasetRow]:
    from datasets import load_dataset

    loaded = load_dataset(
        repository,
        revision=revision,
        split=split,
        streaming=streaming,
    )
    return cast(Iterable[DatasetRow], loaded)


def _expect_mapping(value: object, *, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise OlmoPythonLoaderError(f"{field_name} must be a mapping")
    for key in value:
        if not isinstance(key, str):
            raise OlmoPythonLoaderError(f"{field_name} keys must be strings")
    return cast(Mapping[str, object], value)


def _expect_str(
    mapping: Mapping[str, object],
    key: str,
    *,
    field_name: str,
    allow_empty: bool = False,
) -> str:
    value = mapping.get(key)
    if not isinstance(value, str):
        raise OlmoPythonLoaderError(f"{field_name} must be a string")
    if not allow_empty and not value.strip():
        raise OlmoPythonLoaderError(f"{field_name} must not be empty")
    return value


def _selection_value(row: Mapping[str, object], field: str) -> object:
    value: object = row
    for segment in field.split("."):
        mapping = _expect_mapping(value, field_name=f"selection parent for {field!r}")
        if segment not in mapping:
            raise OlmoPythonLoaderError(f"selection field {field!r} is missing")
        value = mapping[segment]
    return value


def _validate_source_contract(source: DatasetSourceConfig, language: LanguageConfig) -> None:
    if source.id != _OLMO_SOURCE_ID:
        raise OlmoPythonLoaderError(f"expected source id {_OLMO_SOURCE_ID!r}; got {source.id!r}")
    if source.language != language.id:
        raise OlmoPythonLoaderError(
            f"source language {source.language!r} does not match language config {language.id!r}"
        )
    if source.dataset.repository != _OLMO_REPOSITORY:
        raise OlmoPythonLoaderError(
            f"expected dataset repository {_OLMO_REPOSITORY!r}; got {source.dataset.repository!r}"
        )
    if source.selection.field != _OLMO_SELECTION_FIELD:
        raise OlmoPythonLoaderError(f"OLMo Python selection must use {_OLMO_SELECTION_FIELD!r}")
    if source.selection.equals != _OLMO_SELECTION_VALUE:
        raise OlmoPythonLoaderError(
            f"OLMo Python selection must retain {_OLMO_SELECTION_VALUE!r} rows"
        )


def _source_metadata(
    row: Mapping[str, object],
    metadata: Mapping[str, object],
) -> tuple[tuple[str, str], ...]:
    values = {
        "added": _expect_str(row, "added", field_name="row.added"),
        "created": _expect_str(row, "created", field_name="row.created"),
        "metadata.extension": _expect_str(
            metadata,
            "extension",
            field_name="row.metadata.extension",
        ),
        "metadata.max_stars_count": _expect_str(
            metadata,
            "max_stars_count",
            field_name="row.metadata.max_stars_count",
        ),
        "metadata.max_stars_repo_name": _expect_str(
            metadata,
            "max_stars_repo_name",
            field_name="row.metadata.max_stars_repo_name",
        ),
        "metadata.max_stars_repo_path": _expect_str(
            metadata,
            "max_stars_repo_path",
            field_name="row.metadata.max_stars_repo_path",
        ),
        "metadata.provenance": _expect_str(
            metadata,
            "provenance",
            field_name="row.metadata.provenance",
        ),
        "source": _expect_str(row, "source", field_name="row.source"),
    }
    return tuple(sorted(values.items()))


def _normalize_row(
    row: Mapping[str, object],
    *,
    source: DatasetSourceConfig,
    language: LanguageConfig,
) -> NormalizedTrainingRecord:
    metadata = _expect_mapping(row.get("metadata"), field_name="row.metadata")
    instruction = _expect_str(
        row,
        "instruction",
        field_name="row.instruction",
        allow_empty=True,
    )
    response = _expect_str(row, "text", field_name="row.text", allow_empty=True)
    record_id = _expect_str(row, "id", field_name="row.id")
    return NormalizedTrainingRecord(
        schema_version=1,
        messages=single_turn_messages(
            system=language.system_prompt.text,
            user=instruction,
            assistant=response,
        ),
        language=language.id,
        provenance=SourceProvenance(
            source_id=source.dataset.repository,
            revision=source.dataset.revision,
            license=source.license,
            split=source.dataset.split,
            record_id=record_id,
            url=(
                "https://huggingface.co/datasets/"
                f"{source.dataset.repository}/tree/{source.dataset.revision}"
            ),
            source_metadata=_source_metadata(row, metadata),
        ),
    )


def load_olmo_python_instruct(
    source: DatasetSourceConfig,
    language: LanguageConfig,
    *,
    max_records: int | None = None,
    dataset_loader: DatasetRowsLoader = _load_huggingface_rows,
) -> tuple[NormalizedTrainingRecord, ...]:
    """Load pinned OLMo rows and retain only metadata-declared Python 3 examples."""

    _validate_source_contract(source, language)
    if max_records is not None and max_records <= 0:
        raise OlmoPythonLoaderError("max_records must be greater than zero when provided")

    rows = dataset_loader(
        source.dataset.repository,
        revision=source.dataset.revision,
        split=source.dataset.split,
        streaming=True,
    )
    records: list[NormalizedTrainingRecord] = []
    for raw_row in rows:
        row = _expect_mapping(raw_row, field_name="dataset row")
        if _selection_value(row, source.selection.field) != source.selection.equals:
            continue
        records.append(_normalize_row(row, source=source, language=language))
        if max_records is not None and len(records) >= max_records:
            break
    return tuple(records)
