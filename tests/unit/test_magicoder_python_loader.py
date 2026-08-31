from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import pytest

from tiny_qwen_coder.data.magicoder_python import (
    MagicoderPythonLoaderError,
    load_magicoder_python,
)
from tiny_qwen_coder.data.source_config import load_dataset_source_config
from tiny_qwen_coder.languages import load_language_config
from tiny_qwen_coder.languages.python import load_python_plugin

_SOURCE_PATH = Path("configs/data/python/magicoder_oss_instruct_75k.yaml")
_LANGUAGE_PATH = Path("configs/languages/python.yaml")
_REVISION = "5f839b1f368a76b161028bb9edff055db34022b2"


def _row(
    *,
    lang: str,
    raw_index: int,
    index: int,
    problem: str,
    solution: str,
) -> dict[str, object]:
    return {
        "lang": lang,
        "raw_index": raw_index,
        "index": index,
        "seed": f"seed-{raw_index}",
        "openai_fingerprint": "fp_eeff13170a",
        "problem": problem,
        "solution": solution,
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


def test_magicoder_source_config_pins_identity_license_and_python_selection() -> None:
    source = load_dataset_source_config(_SOURCE_PATH)

    assert source.id == "magicoder-oss-instruct-75k"
    assert source.language == "python"
    assert source.adapter == "tiny_qwen_coder.data.magicoder_python:load_magicoder_python"
    assert source.dataset.repository == "ise-uiuc/Magicoder-OSS-Instruct-75K"
    assert source.dataset.revision == _REVISION
    assert source.dataset.split == "train"
    assert source.license.name == "MIT"
    assert source.selection.field == "lang"
    assert source.selection.equals == "python"


def test_python_plugin_registers_olmo_then_magicoder_from_source_configs() -> None:
    plugin = load_python_plugin()

    assert tuple(adapter.id for adapter in plugin.spec.data_adapters) == (
        "olmo-starcoder-python-instruct",
        "magicoder-oss-instruct-75k",
    )
    assert plugin.spec.data_adapters[1].import_ref == (
        "tiny_qwen_coder.data.magicoder_python:load_magicoder_python"
    )


def test_loader_uses_pinned_revision_and_filters_by_lang() -> None:
    source = load_dataset_source_config(_SOURCE_PATH)
    language = load_language_config(_LANGUAGE_PATH)
    loader = RecordingDatasetLoader(
        (
            _row(
                lang="cpp",
                raw_index=101533,
                index=4626,
                problem="Write C++ code.",
                solution="```cpp\nint main() {}\n```",
            ),
            _row(
                lang="python",
                raw_index=52597,
                index=12220,
                problem="Validate a domain against allowed hosts.",
                solution=(
                    "```python\ndef validate_domain(domain, allowed):\n"
                    "    return domain in allowed\n```"
                ),
            ),
        )
    )

    records = load_magicoder_python(source, language, dataset_loader=loader)

    assert loader.calls == [("ise-uiuc/Magicoder-OSS-Instruct-75K", _REVISION, "train", True)]
    assert len(records) == 1
    record = records[0]
    assert record.language == "python"
    assert tuple(message.role for message in record.messages) == ("system", "user", "assistant")
    assert record.messages[0].content == language.system_prompt.text
    assert record.messages[1].content == "Validate a domain against allowed hosts."
    assert record.messages[2].content.startswith("```python")


def test_loader_preserves_magicoder_source_provenance() -> None:
    source = load_dataset_source_config(_SOURCE_PATH)
    language = load_language_config(_LANGUAGE_PATH)
    loader = RecordingDatasetLoader(
        (
            _row(
                lang="python",
                raw_index=142176,
                index=38750,
                problem="Implement a logging decorator.",
                solution="```python\ndef logged(fn):\n    return fn\n```",
            ),
        )
    )

    (record,) = load_magicoder_python(source, language, dataset_loader=loader)

    assert record.provenance.source_id == "ise-uiuc/Magicoder-OSS-Instruct-75K"
    assert record.provenance.revision == _REVISION
    assert record.provenance.split == "train"
    assert record.provenance.record_id == "142176"
    assert record.provenance.license.name == "MIT"
    assert record.provenance.source_metadata == (
        ("index", "38750"),
        ("lang", "python"),
        ("openai_fingerprint", "fp_eeff13170a"),
        ("raw_index", "142176"),
        ("seed", "seed-142176"),
    )


def test_loader_omits_empty_auxiliary_seed_metadata() -> None:
    source = load_dataset_source_config(_SOURCE_PATH)
    language = load_language_config(_LANGUAGE_PATH)
    row = _row(
        lang="python",
        raw_index=7,
        index=3,
        problem="Return one.",
        solution="return 1",
    )
    row["seed"] = "   "
    loader = RecordingDatasetLoader((row,))

    (record,) = load_magicoder_python(source, language, dataset_loader=loader)

    assert record.provenance.source_metadata == (
        ("index", "3"),
        ("lang", "python"),
        ("openai_fingerprint", "fp_eeff13170a"),
        ("raw_index", "7"),
    )


def test_loader_retains_empty_content_for_generic_required_content_filtering() -> None:
    source = load_dataset_source_config(_SOURCE_PATH)
    language = load_language_config(_LANGUAGE_PATH)
    loader = RecordingDatasetLoader(
        (_row(lang="python", raw_index=1, index=1, problem="", solution=""),)
    )

    (record,) = load_magicoder_python(source, language, dataset_loader=loader)

    assert record.messages[1].content == ""
    assert record.messages[2].content == ""


def test_loader_max_records_is_deterministic_after_python_filtering() -> None:
    source = load_dataset_source_config(_SOURCE_PATH)
    language = load_language_config(_LANGUAGE_PATH)
    loader = RecordingDatasetLoader(
        (
            _row(lang="java", raw_index=1, index=1, problem="java", solution="java"),
            _row(lang="python", raw_index=2, index=2, problem="one", solution="one"),
            _row(lang="python", raw_index=3, index=3, problem="two", solution="two"),
        )
    )

    records = load_magicoder_python(
        source,
        language,
        max_records=1,
        dataset_loader=loader,
    )

    assert tuple(record.provenance.record_id for record in records) == ("2",)


def test_loader_fails_closed_on_wrong_source_contract_and_bad_row_types() -> None:
    source = load_dataset_source_config(_SOURCE_PATH)
    language = load_language_config(_LANGUAGE_PATH)
    wrong = source.__class__(
        schema_version=source.schema_version,
        id=source.id,
        language=source.language,
        adapter=source.adapter,
        dataset=source.dataset,
        license=source.license,
        selection=source.selection.__class__(field="lang", equals="cpp"),
    )
    with pytest.raises(MagicoderPythonLoaderError, match="must retain 'python'"):
        load_magicoder_python(wrong, language, dataset_loader=RecordingDatasetLoader(()))

    invalid = RecordingDatasetLoader(
        (
            {
                "lang": "python",
                "raw_index": "not-an-int",
                "index": 1,
                "seed": "seed",
                "openai_fingerprint": "fp_eeff13170a",
                "problem": "problem",
                "solution": "solution",
            },
        )
    )
    with pytest.raises(MagicoderPythonLoaderError, match="row.raw_index must be an integer"):
        load_magicoder_python(source, language, dataset_loader=invalid)
