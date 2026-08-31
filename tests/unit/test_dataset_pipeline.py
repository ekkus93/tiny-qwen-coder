from __future__ import annotations

from dataclasses import replace

import pytest

from tiny_qwen_coder.config import DataPreparationConfig
from tiny_qwen_coder.data import (
    LengthRejectionReason,
    LicenseMetadata,
    NormalizedTrainingRecord,
    SourceProvenance,
    ValidationResult,
    single_turn_messages,
)
from tiny_qwen_coder.data.pipeline import (
    DatasetPipelineError,
    DatasetPipelineResult,
    LanguageRecordValidator,
    run_dataset_pipeline,
)
from tiny_qwen_coder.languages import (
    ConfigReferences,
    LanguageComponentRef,
    LanguageConfig,
    LanguageHookReferences,
    LanguageSpec,
    RepositoryDetectionSignals,
    StaticLanguagePlugin,
    SystemPromptSpec,
)
from tiny_qwen_coder.model import InspectionTarget
from tiny_qwen_coder.reporting import GitMetadata, dataset_manifest_json

_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
_GIT_SHA = "b" * 40


class FakeTokenizer:
    chat_template = "fixture canonical template"

    def __init__(self) -> None:
        self.init_kwargs: dict[str, object] = {"_commit_hash": _REVISION}
        self.calls: list[dict[str, object]] = []

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
        self.calls.append(
            {
                "conversation": conversation,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "truncation": truncation,
                "return_dict": return_dict,
                "chat_template": chat_template,
            }
        )
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


def _config() -> DataPreparationConfig:
    return DataPreparationConfig(
        schema_version=1,
        language="python",
        source_configs=("configs/data/python/fixture.yaml",),
        output_dir="data/python/fixture",
        seed=17,
        validation_fraction=0.33,
        min_tokens=1,
        max_tokens=8,
        truncation_policy="reject",
        deduplicate=True,
    )


def _plugin() -> StaticLanguagePlugin:
    validator_ref = "tests.fixtures.pipeline_hooks:validate_fixture"
    executor_ref = "tests.fixtures.pipeline_hooks:execute_fixture"
    config = LanguageConfig(
        schema_version=1,
        id="python",
        aliases=("py",),
        extensions=(".py",),
        repository_detection=RepositoryDetectionSignals(files=("pyproject.toml",)),
        system_prompt=SystemPromptSpec(version="fixture-v1", text="Write Python code."),
        config_refs=ConfigReferences(
            data_sources=("configs/data/python/fixture.yaml",),
            evaluation=("configs/eval/python/fixture.yaml",),
        ),
        hooks=LanguageHookReferences(validator=validator_ref, executor=executor_ref),
    )
    return StaticLanguagePlugin(
        LanguageSpec(
            config=config,
            execution_hook=LanguageComponentRef(id="default", import_ref=executor_ref),
            validators=(LanguageComponentRef(id="fixture.syntax", import_ref=validator_ref),),
        )
    )


def _record(
    *,
    user: str,
    assistant: str,
    source_id: str,
    record_id: str,
) -> NormalizedTrainingRecord:
    license_name = "Apache-2.0" if source_id.endswith("a") else "MIT"
    return NormalizedTrainingRecord(
        schema_version=1,
        language="python",
        messages=single_turn_messages(system=None, user=user, assistant=assistant),
        provenance=SourceProvenance(
            source_id=source_id,
            revision=_REVISION,
            license=LicenseMetadata(name=license_name),
            split="train",
            record_id=record_id,
        ),
        validation=None,
    )


def _records() -> tuple[NormalizedTrainingRecord, ...]:
    return (
        _record(
            user="\ufeffalpha\r\nprompt",
            assistant="one",
            source_id="fixture/source-a",
            record_id="a-1",
        ),
        _record(
            user="alpha\nprompt",
            assistant="one",
            source_id="fixture/source-a",
            record_id="a-2",
        ),
        _record(
            user="alpha\nprompt",
            assistant="alternate",
            source_id="fixture/source-a",
            record_id="a-3",
        ),
        _record(
            user="beta",
            assistant="two",
            source_id="fixture/source-a",
            record_id="a-4",
        ),
        _record(
            user="gamma",
            assistant="INVALID answer",
            source_id="fixture/source-b",
            record_id="b-1",
        ),
        _record(
            user="   ",
            assistant="empty prompt rejection",
            source_id="fixture/source-b",
            record_id="b-2",
        ),
        _record(
            user="overlength",
            assistant="one two three four five six seven eight nine ten",
            source_id="fixture/source-b",
            record_id="b-3",
        ),
        _record(
            user="delta",
            assistant="four",
            source_id="fixture/source-b",
            record_id="b-4",
        ),
    )


def _run() -> DatasetPipelineResult:
    return run_dataset_pipeline(
        _records(),
        config=_config(),
        plugin=_plugin(),
        tokenizer=FakeTokenizer(),
        target=_target(),
        git=GitMetadata(sha=_GIT_SHA, dirty=False),
    )


def test_cpu_fixture_exercises_complete_generic_pipeline() -> None:
    result = _run()

    assert result.content_filter.total_records == 8
    assert result.content_filter.accepted_count == 7
    assert result.content_filter.rejected_count == 1
    normalized_first = result.content_filter.accepted_records[0]
    assert normalized_first.messages[0].content == "alpha\nprompt"

    assert len(result.language_validated_records) == 7
    validation_by_record_id = {
        record.provenance.record_id: record.validation
        for record in result.language_validated_records
    }
    assert all(metadata is not None for metadata in validation_by_record_id.values())
    a1_validation = validation_by_record_id["a-1"]
    b1_validation = validation_by_record_id["b-1"]
    assert a1_validation is not None
    assert b1_validation is not None
    assert a1_validation.results[0].passed is True
    assert b1_validation.results[0].passed is False

    assert result.length_filter.total_records == 7
    assert result.length_filter.accepted_count == 6
    assert result.length_filter.rejected_count == 1
    assert result.length_filter.rejected_records[0].source_record_id == "b-3"
    assert result.length_filter.rejected_records[0].reason is LengthRejectionReason.TOO_LONG

    assert result.deduplication.total_records == 6
    assert result.deduplication.unique_count == 5
    assert result.deduplication.duplicate_count == 1
    assert result.deduplication.duplicate_records[0].source_record_id == "a-2"

    assert result.split.total_records == 5
    train_prompts = {item.prompt_sha256 for item in result.split.train_fingerprints}
    validation_prompts = {item.prompt_sha256 for item in result.split.validation_fingerprints}
    assert train_prompts.isdisjoint(validation_prompts)
    partition_by_record_id = {
        membership.source_record_id: membership.partition for membership in result.split.memberships
    }
    assert partition_by_record_id["a-1"] == partition_by_record_id["a-3"]

    assert result.manifest.counts.input_records == 8
    assert result.manifest.counts.content_rejected == 1
    assert result.manifest.counts.length_rejected == 1
    assert result.manifest.counts.duplicates_removed == 1
    assert result.manifest.counts.deduplicated_unique == 5
    assert result.manifest.contamination.status == "not_run"


def test_pipeline_is_deterministic_for_same_fixture_config_and_seed() -> None:
    first = _run()
    second = _run()

    assert first.split.memberships == second.split.memberships
    assert first.manifest.checksums == second.manifest.checksums
    assert dataset_manifest_json(first.manifest) == dataset_manifest_json(second.manifest)


def test_manifest_and_tokenizer_calls_never_require_silent_truncation() -> None:
    tokenizer = FakeTokenizer()
    result = run_dataset_pipeline(
        _records(),
        config=_config(),
        plugin=_plugin(),
        tokenizer=tokenizer,
        target=_target(),
        git=GitMetadata(sha=_GIT_SHA, dirty=False),
    )

    assert tokenizer.calls
    assert all(call["truncation"] is False for call in tokenizer.calls)
    manifest_json = dataset_manifest_json(result.manifest)
    assert "alpha\\nprompt" not in manifest_json
    assert "INVALID answer" not in manifest_json


def test_pipeline_rejects_language_plugin_mismatch_before_processing() -> None:
    config = replace(_config(), language="rust")

    with pytest.raises(DatasetPipelineError, match="does not match plugin"):
        run_dataset_pipeline(
            _records(),
            config=config,
            plugin=_plugin(),
            tokenizer=FakeTokenizer(),
            target=_target(),
            git=GitMetadata(sha=_GIT_SHA, dirty=False),
        )


def test_pipeline_rejects_unsupported_noncanonical_data_policies() -> None:
    with pytest.raises(DatasetPipelineError, match="min_tokens >= 1"):
        run_dataset_pipeline(
            _records(),
            config=replace(_config(), min_tokens=0),
            plugin=_plugin(),
            tokenizer=FakeTokenizer(),
            target=_target(),
            git=GitMetadata(sha=_GIT_SHA, dirty=False),
        )

    with pytest.raises(DatasetPipelineError, match="overlength rejection"):
        run_dataset_pipeline(
            _records(),
            config=replace(_config(), truncation_policy="truncate"),
            plugin=_plugin(),
            tokenizer=FakeTokenizer(),
            target=_target(),
            git=GitMetadata(sha=_GIT_SHA, dirty=False),
        )

    with pytest.raises(DatasetPipelineError, match="deduplicate=true"):
        run_dataset_pipeline(
            _records(),
            config=replace(_config(), deduplicate=False),
            plugin=_plugin(),
            tokenizer=FakeTokenizer(),
            target=_target(),
            git=GitMetadata(sha=_GIT_SHA, dirty=False),
        )


def test_pipeline_rejects_validator_result_identity_drift() -> None:
    def bad_resolver(_component: LanguageComponentRef) -> LanguageRecordValidator:
        def validate(_record: NormalizedTrainingRecord) -> ValidationResult:
            return ValidationResult(validator_id="wrong.validator", passed=True)

        return validate

    with pytest.raises(DatasetPipelineError, match="mismatched validator_id"):
        run_dataset_pipeline(
            _records(),
            config=_config(),
            plugin=_plugin(),
            tokenizer=FakeTokenizer(),
            target=_target(),
            validator_resolver=bad_resolver,
            git=GitMetadata(sha=_GIT_SHA, dirty=False),
        )
