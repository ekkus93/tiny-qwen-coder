from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import pytest

from tiny_qwen_coder.data.olmo_python import OlmoPythonLoaderError, load_olmo_python_instruct
from tiny_qwen_coder.data.source_config import (
    DatasetSourceConfigError,
    load_dataset_source_config,
)
from tiny_qwen_coder.languages import load_language_config
from tiny_qwen_coder.languages.python import load_python_plugin

_SOURCE_PATH = Path("configs/data/python/olmo_starcoder_python_instruct.yaml")
_LANGUAGE_PATH = Path("configs/languages/python.yaml")
_REVISION = "5bcafbc00100ec7cf1e6e5a9e353dc2f4eaad9fc"


def _row(*, record_id: str, extension: str, instruction: str, text: str) -> dict[str, object]:
    return {
        "instruction": instruction,
        "text": text,
        "id": record_id,
        "metadata": {
            "extension": extension,
            "max_stars_count": "17",
            "max_stars_repo_name": "example/project",
            "max_stars_repo_path": f"src/{record_id}.py",
            "provenance": f"train-00001-of-00059.jsonl.gz:{record_id}",
        },
        "added": "2023-09-08T23:13:54.429Z",
        "created": "2023-09-08T23:13:54.429Z",
        "source": "starcoder",
    }


class RecordingDatasetLoader:
    def __init__(self, rows: Iterable[Mapping[str, object]]) -> None:
        self.rows = tuple(rows)
        self.calls: list[tuple[str, str, str, bool]] = []

    def __call__(
        self,
        repository: str,
        *,
        revision: str,
        split: str,
        streaming: bool,
    ) -> Iterable[Mapping[str, object]]:
        self.calls.append((repository, revision, split, streaming))
        return self.rows


def test_olmo_source_config_pins_upstream_identity_and_python3_selection() -> None:
    source = load_dataset_source_config(_SOURCE_PATH)

    assert source.id == "olmo-starcoder-python-instruct"
    assert source.language == "python"
    assert source.adapter == "tiny_qwen_coder.data.olmo_python:load_olmo_python_instruct"
    assert source.dataset.repository == "OLMo-Coding/starcoder-python-instruct"
    assert source.dataset.revision == _REVISION
    assert source.dataset.split == "train"
    assert source.license.name == "Apache-2.0"
    assert source.selection.field == "metadata.extension"
    assert source.selection.equals == "python3"


def test_source_config_rejects_unknown_fields_and_floating_revision(tmp_path: Path) -> None:
    original = _SOURCE_PATH.read_text(encoding="utf-8")
    unknown = tmp_path / "unknown.yaml"
    unknown.write_text(original + "surprise: forbidden\n", encoding="utf-8")
    with pytest.raises(DatasetSourceConfigError, match=r"unknown field\(s\): surprise"):
        load_dataset_source_config(unknown)

    floating = tmp_path / "floating.yaml"
    floating.write_text(original.replace(_REVISION, "main"), encoding="utf-8")
    with pytest.raises(DatasetSourceConfigError, match="40-character SHA"):
        load_dataset_source_config(floating)


def test_python_plugin_registers_olmo_adapter_from_source_config() -> None:
    source = load_dataset_source_config(_SOURCE_PATH)
    plugin = load_python_plugin()

    assert plugin.spec.id == "python"
    assert plugin.spec.data_adapters[0].id == source.id
    assert plugin.spec.data_adapters[0].import_ref == source.adapter


def test_loader_uses_pinned_revision_and_filters_by_metadata_extension() -> None:
    source = load_dataset_source_config(_SOURCE_PATH)
    language = load_language_config(_LANGUAGE_PATH)
    loader = RecordingDatasetLoader(
        (
            _row(
                record_id="python2",
                extension="python2",
                instruction="Write Python 3 code despite misleading prompt text.",
                text="print 'legacy'",
            ),
            _row(
                record_id="python3",
                extension="python3",
                instruction="Implement the requested function.",
                text="def answer() -> int:\n    return 42\n",
            ),
        )
    )

    records = load_olmo_python_instruct(source, language, dataset_loader=loader)

    assert loader.calls == [("OLMo-Coding/starcoder-python-instruct", _REVISION, "train", True)]
    assert len(records) == 1
    record = records[0]
    assert record.language == "python"
    assert tuple(message.role for message in record.messages) == ("system", "user", "assistant")
    assert record.messages[0].content == language.system_prompt.text
    assert record.messages[1].content == "Implement the requested function."
    assert record.messages[2].content == "def answer() -> int:\n    return 42\n"


def test_loader_preserves_dataset_and_original_source_provenance() -> None:
    source = load_dataset_source_config(_SOURCE_PATH)
    language = load_language_config(_LANGUAGE_PATH)
    loader = RecordingDatasetLoader(
        (
            _row(
                record_id="9922669",
                extension="python3",
                instruction="Create a migration.",
                text="from django.db import migrations\n",
            ),
        )
    )

    (record,) = load_olmo_python_instruct(source, language, dataset_loader=loader)

    assert record.provenance.source_id == "OLMo-Coding/starcoder-python-instruct"
    assert record.provenance.revision == _REVISION
    assert record.provenance.split == "train"
    assert record.provenance.record_id == "9922669"
    assert record.provenance.license.name == "Apache-2.0"
    assert record.provenance.source_metadata == (
        ("added", "2023-09-08T23:13:54.429Z"),
        ("created", "2023-09-08T23:13:54.429Z"),
        ("metadata.extension", "python3"),
        ("metadata.max_stars_count", "17"),
        ("metadata.max_stars_repo_name", "example/project"),
        ("metadata.max_stars_repo_path", "src/9922669.py"),
        ("metadata.provenance", "train-00001-of-00059.jsonl.gz:9922669"),
        ("source", "starcoder"),
    )


def test_loader_retains_empty_content_for_generic_required_content_filtering() -> None:
    source = load_dataset_source_config(_SOURCE_PATH)
    language = load_language_config(_LANGUAGE_PATH)
    loader = RecordingDatasetLoader(
        (_row(record_id="empty", extension="python3", instruction="", text=""),)
    )

    (record,) = load_olmo_python_instruct(source, language, dataset_loader=loader)

    assert record.messages[1].content == ""
    assert record.messages[2].content == ""


def test_loader_max_records_is_deterministic_after_python3_filtering() -> None:
    source = load_dataset_source_config(_SOURCE_PATH)
    language = load_language_config(_LANGUAGE_PATH)
    loader = RecordingDatasetLoader(
        (
            _row(record_id="p2", extension="python2", instruction="old", text="old"),
            _row(record_id="first", extension="python3", instruction="one", text="one"),
            _row(record_id="second", extension="python3", instruction="two", text="two"),
        )
    )

    records = load_olmo_python_instruct(
        source,
        language,
        max_records=1,
        dataset_loader=loader,
    )

    assert tuple(record.provenance.record_id for record in records) == ("first",)


def test_loader_fails_closed_on_wrong_source_contract() -> None:
    source = load_dataset_source_config(_SOURCE_PATH)
    language = load_language_config(_LANGUAGE_PATH)
    wrong = source.__class__(
        schema_version=source.schema_version,
        id=source.id,
        language=source.language,
        adapter=source.adapter,
        dataset=source.dataset,
        license=source.license,
        selection=source.selection.__class__(field="metadata.extension", equals="python2"),
    )

    with pytest.raises(OlmoPythonLoaderError, match="must retain 'python3'"):
        load_olmo_python_instruct(wrong, language, dataset_loader=RecordingDatasetLoader(()))
