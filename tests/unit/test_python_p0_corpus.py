from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from tiny_qwen_coder.data.python_corpus import (
    PythonP0CorpusConfig,
    PythonP0CorpusError,
    PythonP0RejectionStage,
    PythonP0SourceBudget,
    build_python_p0_corpus,
    load_python_p0_config,
    parse_python_p0_config,
)
from tiny_qwen_coder.data.python_corpus_io import (
    python_p0_summary_json,
    write_python_p0_jsonl,
)
from tiny_qwen_coder.data.records import (
    NormalizedTrainingRecord,
    SourceProvenance,
    single_turn_messages,
)
from tiny_qwen_coder.data.source_config import DatasetSourceConfig, load_dataset_source_config
from tiny_qwen_coder.languages.python import load_python_plugin
from tiny_qwen_coder.model.inspection import InspectionTarget

_CONFIG_PATH = Path("configs/data/python/p0.yaml")
_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
_OLMO_ID = "olmo-starcoder-python-instruct"
_MAGICODER_ID = "magicoder-oss-instruct-75k"


class FakeTokenizer:
    chat_template = "fixture canonical chat template"

    def __init__(self) -> None:
        self.init_kwargs: dict[str, object] = {"_commit_hash": _REVISION}

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        truncation: bool,
        return_dict: bool,
        chat_template: str,
    ) -> list[int]:
        assert tokenize is True
        assert add_generation_prompt is False
        assert truncation is False
        assert return_dict is False
        assert chat_template == self.chat_template
        token_count = 2 + sum(len(message["content"].split()) for message in conversation)
        return list(range(token_count))


def _target() -> InspectionTarget:
    return InspectionTarget(
        config_id="qwen35-4b",
        model_repository="Qwen/Qwen3.5-4B",
        model_revision=_REVISION,
        tokenizer_repository="Qwen/Qwen3.5-4B",
        tokenizer_revision=_REVISION,
        model_load_dtype="bfloat16",
    )


def _small_config(*, fill: str | None = _OLMO_ID) -> PythonP0CorpusConfig:
    return PythonP0CorpusConfig(
        schema_version=1,
        id="python-p0",
        language="python",
        target_total=5,
        min_tokens=1,
        max_tokens=50,
        sources=(
            PythonP0SourceBudget(
                id=_OLMO_ID,
                source_config="configs/data/python/olmo_starcoder_python_instruct.yaml",
                target_accepted=3,
            ),
            PythonP0SourceBudget(
                id=_MAGICODER_ID,
                source_config="configs/data/python/magicoder_oss_instruct_75k.yaml",
                target_accepted=2,
            ),
        ),
        fill_shortfall_from=fill,
        output_jsonl="data/python/p0/accepted.jsonl",
    )


def _source_configs() -> dict[str, DatasetSourceConfig]:
    return {
        _OLMO_ID: load_dataset_source_config(
            Path("configs/data/python/olmo_starcoder_python_instruct.yaml")
        ),
        _MAGICODER_ID: load_dataset_source_config(
            Path("configs/data/python/magicoder_oss_instruct_75k.yaml")
        ),
    }


def _record(
    source: DatasetSourceConfig,
    *,
    record_id: str,
    user: str,
    assistant: str,
) -> NormalizedTrainingRecord:
    plugin = load_python_plugin()
    return NormalizedTrainingRecord(
        schema_version=1,
        language="python",
        messages=single_turn_messages(
            system=plugin.spec.config.system_prompt.text,
            user=user,
            assistant=assistant,
        ),
        provenance=SourceProvenance(
            source_id=source.dataset.repository,
            revision=source.dataset.revision,
            license=source.license,
            split=source.dataset.split,
            record_id=record_id,
        ),
    )


def test_canonical_python_p0_config_freezes_30k_10k_composition_and_fill() -> None:
    config = load_python_p0_config(_CONFIG_PATH)

    assert config.id == "python-p0"
    assert config.target_total == 40_000
    assert config.max_tokens == 2_048
    assert tuple((source.id, source.target_accepted) for source in config.sources) == (
        (_OLMO_ID, 30_000),
        (_MAGICODER_ID, 10_000),
    )
    assert config.fill_shortfall_from == _OLMO_ID


def test_python_p0_config_is_strict_and_targets_must_sum() -> None:
    bad_sum = {
        "schema_version": 1,
        "id": "python-p0",
        "language": "python",
        "target_total": 5,
        "min_tokens": 1,
        "max_tokens": 2048,
        "sources": [
            {"id": _OLMO_ID, "source_config": "olmo.yaml", "target_accepted": 3},
            {"id": _MAGICODER_ID, "source_config": "magic.yaml", "target_accepted": 1},
        ],
        "fill_shortfall_from": _OLMO_ID,
        "output_jsonl": "data/python/p0/accepted.jsonl",
    }
    with pytest.raises(PythonP0CorpusError, match="sum exactly"):
        parse_python_p0_config(bad_sum)

    unknown = {
        "schema_version": 1,
        "id": "python-p0",
        "language": "python",
        "target_total": 5,
        "min_tokens": 1,
        "max_tokens": 2048,
        "sources": [
            {"id": _OLMO_ID, "source_config": "olmo.yaml", "target_accepted": 3},
            {"id": _MAGICODER_ID, "source_config": "magic.yaml", "target_accepted": 2},
        ],
        "fill_shortfall_from": _OLMO_ID,
        "output_jsonl": "data/python/p0/accepted.jsonl",
        "surprise": True,
    }
    with pytest.raises(PythonP0CorpusError, match=r"unknown field\(s\): surprise"):
        parse_python_p0_config(unknown)


def test_builder_applies_filters_dedup_and_explicit_olmo_fill() -> None:
    plugin = load_python_plugin()
    sources = _source_configs()
    magic_valid = _record(
        sources[_MAGICODER_ID],
        record_id="m-valid",
        user="Return one.",
        assistant="def answer():\n    return 1\n",
    )
    streams = {
        _MAGICODER_ID: lambda _source, _language: iter(
            (
                magic_valid,
                _record(
                    sources[_MAGICODER_ID],
                    record_id="m-invalid",
                    user="Broken function.",
                    assistant="def broken(:\n    pass\n",
                ),
                _record(
                    sources[_MAGICODER_ID],
                    record_id="m-long",
                    user="Explain.",
                    assistant=" ".join(["word"] * 100),
                ),
                _record(
                    sources[_MAGICODER_ID],
                    record_id="m-empty",
                    user="   ",
                    assistant="def ok():\n    return True\n",
                ),
            )
        ),
        _OLMO_ID: lambda _source, _language: iter(
            (
                replace(
                    magic_valid,
                    provenance=replace(
                        magic_valid.provenance,
                        source_id=sources[_OLMO_ID].dataset.repository,
                        revision=sources[_OLMO_ID].dataset.revision,
                        license=sources[_OLMO_ID].license,
                        record_id="o-duplicate",
                    ),
                ),
                _record(
                    sources[_OLMO_ID],
                    record_id="o-1",
                    user="Return two.",
                    assistant="def two():\n    return 2\n",
                ),
                _record(
                    sources[_OLMO_ID],
                    record_id="o-2",
                    user="Return three.",
                    assistant="def three():\n    return 3\n",
                ),
                _record(
                    sources[_OLMO_ID],
                    record_id="o-3",
                    user="Return four.",
                    assistant="def four():\n    return 4\n",
                ),
                _record(
                    sources[_OLMO_ID],
                    record_id="o-4",
                    user="Return five.",
                    assistant="def five():\n    return 5\n",
                ),
            )
        ),
    }

    result = build_python_p0_corpus(
        _small_config(),
        plugin=plugin,
        tokenizer=FakeTokenizer(),
        target=_target(),
        source_configs=sources,
        stream_factories=streams,
    )

    assert result.accepted_total == 5
    assert result.shortfall == 0
    assert result.fill_accepted == 1
    olmo_stats, magic_stats = result.source_stats
    assert olmo_stats.requested_accepted == 4
    assert olmo_stats.accepted == 4
    assert olmo_stats.duplicate_rejected == 1
    assert magic_stats.requested_accepted == 2
    assert magic_stats.scanned == 4
    assert magic_stats.accepted == 1
    assert magic_stats.content_rejected == 1
    assert magic_stats.validation_rejected == 1
    assert magic_stats.length_rejected == 1

    rejection_keys = {(item.stage, item.reason) for item in result.rejection_counts}
    assert (PythonP0RejectionStage.CONTENT, "empty_prompt") in rejection_keys
    assert (PythonP0RejectionStage.DUPLICATE, "exact_content") in rejection_keys
    assert (PythonP0RejectionStage.LENGTH, "too_long") in rejection_keys
    assert any(
        stage is PythonP0RejectionStage.VALIDATION and "reason=syntax_error" in reason
        for stage, reason in rejection_keys
    )
    for record in result.accepted_records:
        assert record.validation is not None
        assert record.validation.passed


def test_builder_does_not_fill_shortfall_without_explicit_policy() -> None:
    plugin = load_python_plugin()
    sources = _source_configs()
    streams = {
        _MAGICODER_ID: lambda _source, _language: iter(
            (
                _record(
                    sources[_MAGICODER_ID],
                    record_id="m-1",
                    user="One.",
                    assistant="def one():\n    return 1\n",
                ),
            )
        ),
        _OLMO_ID: lambda _source, _language: iter(
            tuple(
                _record(
                    sources[_OLMO_ID],
                    record_id=f"o-{index}",
                    user=f"Value {index}.",
                    assistant=f"value_{index} = {index}\n",
                )
                for index in range(10)
            )
        ),
    }

    result = build_python_p0_corpus(
        _small_config(fill=None),
        plugin=plugin,
        tokenizer=FakeTokenizer(),
        target=_target(),
        source_configs=sources,
        stream_factories=streams,
    )

    assert result.accepted_total == 4
    assert result.shortfall == 1
    assert result.fill_accepted == 0
    olmo_stats, magic_stats = result.source_stats
    assert olmo_stats.requested_accepted == 3
    assert magic_stats.requested_accepted == 2


def test_summary_and_jsonl_are_deterministic_and_report_measured_counts(tmp_path: Path) -> None:
    plugin = load_python_plugin()
    sources = _source_configs()
    config = PythonP0CorpusConfig(
        schema_version=1,
        id="python-p0",
        language="python",
        target_total=2,
        min_tokens=1,
        max_tokens=50,
        sources=(
            PythonP0SourceBudget(
                id=_OLMO_ID,
                source_config="configs/data/python/olmo_starcoder_python_instruct.yaml",
                target_accepted=1,
            ),
            PythonP0SourceBudget(
                id=_MAGICODER_ID,
                source_config="configs/data/python/magicoder_oss_instruct_75k.yaml",
                target_accepted=1,
            ),
        ),
        fill_shortfall_from=_OLMO_ID,
        output_jsonl="data/python/p0/accepted.jsonl",
    )
    streams = {
        _MAGICODER_ID: lambda _source, _language: iter(
            (_record(sources[_MAGICODER_ID], record_id="m", user="M.", assistant="m = 1\n"),)
        ),
        _OLMO_ID: lambda _source, _language: iter(
            (_record(sources[_OLMO_ID], record_id="o", user="O.", assistant="o = 2\n"),)
        ),
    }
    result = build_python_p0_corpus(
        config,
        plugin=plugin,
        tokenizer=FakeTokenizer(),
        target=_target(),
        source_configs=sources,
        stream_factories=streams,
    )

    summary = python_p0_summary_json(result)
    assert summary == python_p0_summary_json(result)
    payload = json.loads(summary)
    assert payload["target_total"] == 2
    assert payload["accepted_total"] == 2
    assert payload["shortfall"] == 0

    output = write_python_p0_jsonl(result, tmp_path / "accepted.jsonl")
    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["provenance"]["record_id"] for line in lines] == ["o", "m"]
