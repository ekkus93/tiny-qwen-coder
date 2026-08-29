from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tiny_qwen_coder.data.python_corpus import (
    PythonP0CorpusConfig,
    PythonP0CorpusError,
    PythonP0RejectionStage,
    PythonP0SourceBudget,
    build_python_p0_corpus,
)
from tiny_qwen_coder.data.records import (
    NormalizedTrainingRecord,
    SourceProvenance,
    single_turn_messages,
)
from tiny_qwen_coder.data.source_config import DatasetSourceConfig, load_dataset_source_config
from tiny_qwen_coder.languages.python import load_python_plugin
from tiny_qwen_coder.model.inspection import InspectionTarget

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


def _source_configs() -> dict[str, DatasetSourceConfig]:
    return {
        _OLMO_ID: load_dataset_source_config(
            Path("configs/data/python/olmo_starcoder_python_instruct.yaml")
        ),
        _MAGICODER_ID: load_dataset_source_config(
            Path("configs/data/python/magicoder_oss_instruct_75k.yaml")
        ),
    }


def _config(*, fill: str | None = _OLMO_ID) -> PythonP0CorpusConfig:
    return PythonP0CorpusConfig(
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
        fill_shortfall_from=fill,
        output_jsonl="data/python/p0/accepted.jsonl",
    )


def _record(
    source: DatasetSourceConfig,
    *,
    record_id: str | None,
    user: str,
    assistant: str,
    source_id: str | None = None,
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
            source_id=source_id or source.dataset.repository,
            revision=source.dataset.revision,
            license=source.license,
            split=source.dataset.split,
            record_id=record_id,
        ),
    )


def test_shortfall_fill_must_use_primary_configured_source() -> None:
    with pytest.raises(PythonP0CorpusError, match=r"primary \(first\) configured source"):
        _config(fill=_MAGICODER_ID)


def test_builder_requires_plugin_adapter_registration_match() -> None:
    sources = _source_configs()
    sources[_MAGICODER_ID] = replace(
        sources[_MAGICODER_ID],
        adapter="tiny_qwen_coder.data.magicoder_python:iter_magicoder_python",
    )
    streams = {
        _MAGICODER_ID: lambda _source, _language: iter(()),
        _OLMO_ID: lambda _source, _language: iter(()),
    }

    with pytest.raises(PythonP0CorpusError, match="plugin adapter registration"):
        build_python_p0_corpus(
            _config(),
            plugin=load_python_plugin(),
            tokenizer=FakeTokenizer(),
            target=_target(),
            source_configs=sources,
            stream_factories=streams,
        )


@pytest.mark.parametrize(
    ("record_id", "source_id", "expected"),
    (
        ("mismatch", "wrong/repository", "mismatched provenance source_id"),
        (None, None, "without record_id"),
    ),
)
def test_builder_rejects_unpinned_source_provenance(
    record_id: str | None,
    source_id: str | None,
    expected: str,
) -> None:
    sources = _source_configs()
    bad = _record(
        sources[_MAGICODER_ID],
        record_id=record_id,
        user="Return one.",
        assistant="def answer():\n    return 1\n",
        source_id=source_id,
    )
    streams = {
        _MAGICODER_ID: lambda _source, _language: iter((bad,)),
        _OLMO_ID: lambda _source, _language: iter(()),
    }

    with pytest.raises(PythonP0CorpusError, match=expected):
        build_python_p0_corpus(
            _config(),
            plugin=load_python_plugin(),
            tokenizer=FakeTokenizer(),
            target=_target(),
            source_configs=sources,
            stream_factories=streams,
        )


def test_validation_rejection_counts_use_stable_reason_without_syntax_location() -> None:
    sources = _source_configs()
    streams = {
        _MAGICODER_ID: lambda _source, _language: iter(
            (
                _record(
                    sources[_MAGICODER_ID],
                    record_id="m-bad",
                    user="Broken function.",
                    assistant="def broken(value)\n    return value\n",
                ),
                _record(
                    sources[_MAGICODER_ID],
                    record_id="m-good",
                    user="Return one.",
                    assistant="def answer():\n    return 1\n",
                ),
            )
        ),
        _OLMO_ID: lambda _source, _language: iter(
            (
                _record(
                    sources[_OLMO_ID],
                    record_id="o-good",
                    user="Return two.",
                    assistant="def answer_two():\n    return 2\n",
                ),
            )
        ),
    }

    result = build_python_p0_corpus(
        _config(),
        plugin=load_python_plugin(),
        tokenizer=FakeTokenizer(),
        target=_target(),
        source_configs=sources,
        stream_factories=streams,
    )

    validation_rejections = tuple(
        item for item in result.rejection_counts if item.stage is PythonP0RejectionStage.VALIDATION
    )
    assert len(validation_rejections) == 1
    assert validation_rejections[0].reason == "python.quality:reason=syntax_error"
    assert validation_rejections[0].count == 1
