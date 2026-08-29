from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from tiny_qwen_coder.data.python_corpus import (
    PythonP0CorpusConfig,
    PythonP0CorpusResult,
    PythonP0SourceBudget,
    build_python_p0_corpus,
)
from tiny_qwen_coder.data.python_corpus_io import freeze_python_p0_result
from tiny_qwen_coder.data.python_p0_manifest import (
    PythonP0DatasetManifest,
    PythonP0ManifestError,
    create_python_p0_dataset_manifest,
    python_p0_dataset_manifest_json,
    python_p0_dataset_manifest_sha256,
    split_python_p0_corpus,
)
from tiny_qwen_coder.data.records import (
    NormalizedTrainingRecord,
    SourceProvenance,
    single_turn_messages,
)
from tiny_qwen_coder.data.source_config import DatasetSourceConfig, load_dataset_source_config
from tiny_qwen_coder.data.splitting import DeduplicatedDatasetSplit
from tiny_qwen_coder.evaluation.contamination import EXACT_PROMPT_CHECK_ID
from tiny_qwen_coder.languages.python import load_python_plugin
from tiny_qwen_coder.model.inspection import InspectionTarget
from tiny_qwen_coder.reporting.dataset_manifest import (
    ContaminationFinding,
    ContaminationStatus,
    ContaminationSummary,
)
from tiny_qwen_coder.reporting.manifest import GitMetadata

_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
_GIT_SHA = "a" * 40
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


def _config(*, olmo_target: int = 2, target_total: int = 4) -> PythonP0CorpusConfig:
    return PythonP0CorpusConfig(
        schema_version=1,
        id="python-p0",
        language="python",
        target_total=target_total,
        min_tokens=1,
        max_tokens=50,
        sources=(
            PythonP0SourceBudget(
                id=_OLMO_ID,
                source_config="configs/data/python/olmo_starcoder_python_instruct.yaml",
                target_accepted=olmo_target,
            ),
            PythonP0SourceBudget(
                id=_MAGICODER_ID,
                source_config="configs/data/python/magicoder_oss_instruct_75k.yaml",
                target_accepted=2,
            ),
        ),
        fill_shortfall_from=_OLMO_ID,
        output_jsonl="data/python/p0/accepted.jsonl",
        seed=17,
        validation_fraction=0.25,
    )


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


def _build_result(
    *, shortfall: bool = False
) -> tuple[PythonP0CorpusResult, dict[str, DatasetSourceConfig]]:
    sources = _source_configs()
    magic_shared = _record(
        sources[_MAGICODER_ID],
        record_id="m-shared",
        user="Shared task.",
        assistant="def magic_shared():\n    return 1\n",
    )
    streams = {
        _MAGICODER_ID: lambda _source, _language: iter(
            (
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
                magic_shared,
                _record(
                    sources[_MAGICODER_ID],
                    record_id="m-unique",
                    user="Magic unique.",
                    assistant="magic_unique = 4\n",
                ),
            )
        ),
        _OLMO_ID: lambda _source, _language: iter(
            (
                replace(
                    magic_shared,
                    provenance=replace(
                        magic_shared.provenance,
                        source_id=sources[_OLMO_ID].dataset.repository,
                        revision=sources[_OLMO_ID].dataset.revision,
                        license=sources[_OLMO_ID].license,
                        record_id="o-duplicate",
                    ),
                ),
                _record(
                    sources[_OLMO_ID],
                    record_id="o-shared",
                    user="Shared task.",
                    assistant="def olmo_shared():\n    return 2\n",
                ),
                _record(
                    sources[_OLMO_ID],
                    record_id="o-unique",
                    user="OLMo unique.",
                    assistant="olmo_unique = 3\n",
                ),
            )
        ),
    }
    config = _config(olmo_target=3, target_total=5) if shortfall else _config()
    result = build_python_p0_corpus(
        config,
        plugin=load_python_plugin(),
        tokenizer=FakeTokenizer(),
        target=_target(),
        source_configs=sources,
        stream_factories=streams,
    )
    return result, sources


def _manifest() -> tuple[
    PythonP0CorpusResult,
    dict[str, DatasetSourceConfig],
    DeduplicatedDatasetSplit,
    PythonP0DatasetManifest,
]:
    result, sources = _build_result()
    split = split_python_p0_corpus(result)
    manifest = create_python_p0_dataset_manifest(
        result,
        source_configs=sources,
        split=split,
        git=GitMetadata(sha=_GIT_SHA, dirty=False),
    )
    return result, sources, split, manifest


def test_manifest_captures_measured_p0_composition_split_tokens_and_sources() -> None:
    result, _, split, manifest = _manifest()

    assert manifest.manifest_id == "dataset/python/p0"
    assert manifest.corpus_id == "python-p0"
    assert manifest.seed == 17
    assert manifest.config.validation_fraction == 0.25
    assert manifest.counts.scanned_records == 8
    assert manifest.counts.content_rejected == 1
    assert manifest.counts.validation_rejected == 1
    assert manifest.counts.length_rejected == 1
    assert manifest.counts.duplicate_rejected == 1
    assert manifest.counts.accepted_records == result.accepted_total == 4
    assert manifest.counts.train_records + manifest.counts.validation_records == 4

    assert tuple(source.config.id for source in manifest.sources) == (_OLMO_ID, _MAGICODER_ID)
    assert manifest.sources[0].config.dataset.repository == "OLMo-Coding/starcoder-python-instruct"
    assert manifest.sources[0].config.license.name == "Apache-2.0"
    assert manifest.sources[1].config.dataset.repository == "ise-uiuc/Magicoder-OSS-Instruct-75K"
    assert manifest.sources[1].config.license.name == "MIT"
    assert manifest.sources[0].stats.scanned == 3
    assert manifest.sources[1].stats.scanned == 5

    assert manifest.tokenizer.repository == "Qwen/Qwen3.5-4B"
    assert manifest.tokenizer.revision == _REVISION
    assert manifest.tokenizer.measured_distribution.count == 6
    assert manifest.tokenizer.accepted_distribution.count == 4
    assert len(manifest.memberships) == 4
    assert manifest.split.requested_validation_fraction == 0.25
    assert manifest.contamination.status is ContaminationStatus.NOT_RUN

    shared_memberships = [
        item
        for item in split.memberships
        if item.prompt_sha256
        == next(
            member.prompt_sha256
            for member in split.memberships
            if member.source_record_id == "o-shared"
        )
    ]
    assert {item.source_record_id for item in shared_memberships} == {"o-shared", "m-shared"}
    assert len({item.partition for item in shared_memberships}) == 1

    for digest in (
        manifest.config_sha256,
        manifest.checksums.accepted_audit_sha256,
        manifest.checksums.accepted_content_sha256,
        manifest.checksums.train_content_sha256,
        manifest.checksums.validation_content_sha256,
        manifest.checksums.split_membership_sha256,
        manifest.checksums.composition_sha256,
    ):
        assert len(digest) == 64


def test_manifest_serialization_is_deterministic_and_contains_no_training_text() -> None:
    _, _, _, first = _manifest()
    _, _, _, second = _manifest()

    assert first == second
    rendered = python_p0_dataset_manifest_json(first)
    assert rendered == python_p0_dataset_manifest_json(second)
    assert python_p0_dataset_manifest_sha256(first) == python_p0_dataset_manifest_sha256(second)
    assert "Shared task." not in rendered
    assert "magic_unique = 4" not in rendered
    assert "OLMo-Coding/starcoder-python-instruct" in rendered
    assert "5bcafbc00100ec7cf1e6e5a9e353dc2f4eaad9fc" in rendered


def test_manifest_preserves_detailed_filter_and_dedup_reasons() -> None:
    _, _, _, manifest = _manifest()
    reasons = {(item.stage.value, item.reason): item.count for item in manifest.rejection_counts}

    assert reasons[("content", "empty_prompt")] == 1
    assert reasons[("length", "too_long")] == 1
    assert reasons[("duplicate", "exact_content")] == 1
    assert any(
        stage == "validation" and reason == "python.quality:reason=syntax_error"
        for stage, reason in reasons
    )


def test_freeze_rejects_shortfall_before_writing_artifacts(tmp_path: Path) -> None:
    result, sources = _build_result(shortfall=True)

    assert result.shortfall == 1
    with pytest.raises(PythonP0ManifestError, match="shortfall"):
        freeze_python_p0_result(
            result,
            source_configs=sources,
            output_dir=tmp_path,
            git=GitMetadata(sha=_GIT_SHA, dirty=False),
        )
    assert list(tmp_path.iterdir()) == []


def test_manifest_rejects_dirty_preparation_tree() -> None:
    result, sources = _build_result()
    split = split_python_p0_corpus(result)

    with pytest.raises(PythonP0ManifestError, match="clean Git tree"):
        create_python_p0_dataset_manifest(
            result,
            source_configs=sources,
            split=split,
            git=GitMetadata(sha=_GIT_SHA, dirty=True),
        )


def test_manifest_rejects_source_config_drift() -> None:
    result, sources = _build_result()
    sources[_OLMO_ID] = replace(
        sources[_OLMO_ID],
        dataset=replace(sources[_OLMO_ID].dataset, revision="b" * 40),
    )
    split = split_python_p0_corpus(result)

    with pytest.raises(PythonP0ManifestError, match="accepted count"):
        create_python_p0_dataset_manifest(
            result,
            source_configs=sources,
            split=split,
            git=GitMetadata(sha=_GIT_SHA, dirty=False),
        )


def test_manifest_rejects_contamination_finding_outside_frozen_corpus() -> None:
    result, sources = _build_result()
    split = split_python_p0_corpus(result)
    contamination = ContaminationSummary(
        status=ContaminationStatus.FINDINGS,
        check_ids=(EXACT_PROMPT_CHECK_ID,),
        findings=(
            ContaminationFinding(
                checker_id=EXACT_PROMPT_CHECK_ID,
                protected_dataset_id="humaneval",
                finding_type="exact_prompt_match",
                training_record_sha256="f" * 64,
            ),
        ),
    )

    with pytest.raises(PythonP0ManifestError, match="outside the frozen P0 corpus"):
        create_python_p0_dataset_manifest(
            result,
            source_configs=sources,
            split=split,
            contamination=contamination,
            git=GitMetadata(sha=_GIT_SHA, dirty=False),
        )


def test_freeze_writes_deterministic_split_manifest_and_checksum(tmp_path: Path) -> None:
    result, sources = _build_result()
    artifacts = freeze_python_p0_result(
        result,
        source_configs=sources,
        output_dir=tmp_path,
        git=GitMetadata(sha=_GIT_SHA, dirty=False),
    )

    assert artifacts.accepted_path == tmp_path / "accepted.jsonl"
    assert artifacts.train_path == tmp_path / "train.jsonl"
    assert artifacts.validation_path == tmp_path / "validation.jsonl"
    assert artifacts.manifest_path == tmp_path / "dataset-manifest.json"
    assert artifacts.manifest_checksum_path == tmp_path / "dataset-manifest.sha256"
    assert artifacts.composition_path == tmp_path / "composition.json"

    accepted_lines = artifacts.accepted_path.read_text(encoding="utf-8").splitlines()
    train_lines = artifacts.train_path.read_text(encoding="utf-8").splitlines()
    validation_lines = artifacts.validation_path.read_text(encoding="utf-8").splitlines()
    assert len(accepted_lines) == 4
    assert len(train_lines) + len(validation_lines) == 4
    assert all(json.loads(line)["language"] == "python" for line in accepted_lines)

    manifest_text = artifacts.manifest_path.read_text(encoding="utf-8")
    assert manifest_text == python_p0_dataset_manifest_json(artifacts.manifest)
    checksum_text = artifacts.manifest_checksum_path.read_text(encoding="ascii")
    assert checksum_text == f"{artifacts.manifest_sha256}  dataset-manifest.json\n"
    assert not tuple(tmp_path.glob(".*.tmp"))
