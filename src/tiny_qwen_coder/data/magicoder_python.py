"""Magicoder OSS-Instruct Python adapter for the normalized training schema."""

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

_MAGICODER_SOURCE_ID = "magicoder-oss-instruct-75k"
_MAGICODER_REPOSITORY = "ise-uiuc/Magicoder-OSS-Instruct-75K"
_MAGICODER_SELECTION_FIELD = "lang"
_MAGICODER_SELECTION_VALUE = "python"

DatasetRow = Mapping[str, object]
DatasetRowsLoader = Callable[..., Iterable[DatasetRow]]


class MagicoderPythonLoaderError(ValueError):
    """Raised when the pinned Magicoder source cannot be normalized safely."""


def _load_huggingface_rows(
    repository: str,
    *,
    revision: str,
    split: str,
    streaming: bool,
) -> Iterable[DatasetRow]:
    from datasets import load_dataset  # type: ignore[import-untyped]

    loaded = load_dataset(
        repository,
        revision=revision,
        split=split,
        streaming=streaming,
    )
    return cast(Iterable[DatasetRow], loaded)


def _expect_mapping(value: object, *, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise MagicoderPythonLoaderError(f"{field_name} must be a mapping")
    for key in value:
        if not isinstance(key, str):
            raise MagicoderPythonLoaderError(f"{field_name} keys must be strings")
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
        raise MagicoderPythonLoaderError(f"{field_name} must be a string")
    if not allow_empty and not value.strip():
        raise MagicoderPythonLoaderError(f"{field_name} must not be empty")
    return value


def _expect_int(mapping: Mapping[str, object], key: str, *, field_name: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise MagicoderPythonLoaderError(f"{field_name} must be an integer")
    return value


def _validate_source_contract(source: DatasetSourceConfig, language: LanguageConfig) -> None:
    if source.id != _MAGICODER_SOURCE_ID:
        raise MagicoderPythonLoaderError(
            f"expected source id {_MAGICODER_SOURCE_ID!r}; got {source.id!r}"
        )
    if source.language != language.id:
        raise MagicoderPythonLoaderError(
            f"source language {source.language!r} does not match language config {language.id!r}"
        )
    if source.dataset.repository != _MAGICODER_REPOSITORY:
        raise MagicoderPythonLoaderError(
            f"expected dataset repository {_MAGICODER_REPOSITORY!r}; "
            f"got {source.dataset.repository!r}"
        )
    if source.selection.field != _MAGICODER_SELECTION_FIELD:
        raise MagicoderPythonLoaderError(
            f"Magicoder Python selection must use {_MAGICODER_SELECTION_FIELD!r}"
        )
    if source.selection.equals != _MAGICODER_SELECTION_VALUE:
        raise MagicoderPythonLoaderError(
            f"Magicoder Python selection must retain {_MAGICODER_SELECTION_VALUE!r} rows"
        )


def _source_metadata(row: Mapping[str, object]) -> tuple[tuple[str, str], ...]:
    seed = _expect_str(row, "seed", field_name="row.seed", allow_empty=True)
    values = {
        "index": str(_expect_int(row, "index", field_name="row.index")),
        "lang": _expect_str(row, "lang", field_name="row.lang"),
        "openai_fingerprint": _expect_str(
            row,
            "openai_fingerprint",
            field_name="row.openai_fingerprint",
        ),
        "raw_index": str(_expect_int(row, "raw_index", field_name="row.raw_index")),
    }
    if seed.strip():
        values["seed"] = seed
    return tuple(sorted(values.items()))


def _normalize_row(
    row: Mapping[str, object],
    *,
    source: DatasetSourceConfig,
    language: LanguageConfig,
) -> NormalizedTrainingRecord:
    problem = _expect_str(row, "problem", field_name="row.problem", allow_empty=True)
    solution = _expect_str(row, "solution", field_name="row.solution", allow_empty=True)
    raw_index = _expect_int(row, "raw_index", field_name="row.raw_index")
    return NormalizedTrainingRecord(
        schema_version=1,
        messages=single_turn_messages(
            system=language.system_prompt.text,
            user=problem,
            assistant=solution,
        ),
        language=language.id,
        provenance=SourceProvenance(
            source_id=source.dataset.repository,
            revision=source.dataset.revision,
            license=source.license,
            split=source.dataset.split,
            record_id=str(raw_index),
            url=(
                "https://huggingface.co/datasets/"
                f"{source.dataset.repository}/tree/{source.dataset.revision}"
            ),
            source_metadata=_source_metadata(row),
        ),
    )


def iter_magicoder_python(
    source: DatasetSourceConfig,
    language: LanguageConfig,
    *,
    dataset_loader: DatasetRowsLoader = _load_huggingface_rows,
) -> Iterable[NormalizedTrainingRecord]:
    """Stream normalized rows from the pinned Magicoder source in upstream order."""

    _validate_source_contract(source, language)
    rows = dataset_loader(
        source.dataset.repository,
        revision=source.dataset.revision,
        split=source.dataset.split,
        streaming=True,
    )
    for raw_row in rows:
        row = _expect_mapping(raw_row, field_name="dataset row")
        selected_language = _expect_str(row, source.selection.field, field_name="row.lang")
        if selected_language != source.selection.equals:
            continue
        yield _normalize_row(row, source=source, language=language)


def load_magicoder_python(
    source: DatasetSourceConfig,
    language: LanguageConfig,
    *,
    max_records: int | None = None,
    dataset_loader: DatasetRowsLoader = _load_huggingface_rows,
) -> tuple[NormalizedTrainingRecord, ...]:
    """Load a bounded tuple of normalized Magicoder Python records."""

    if max_records is not None and max_records <= 0:
        raise MagicoderPythonLoaderError("max_records must be greater than zero when provided")

    records: list[NormalizedTrainingRecord] = []
    for record in iter_magicoder_python(
        source,
        language,
        dataset_loader=dataset_loader,
    ):
        records.append(record)
        if max_records is not None and len(records) >= max_records:
            break
    return tuple(records)
