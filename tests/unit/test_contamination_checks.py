"""P4-002 protected-benchmark contamination checks."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tiny_qwen_coder.config import DataPreparationConfig
from tiny_qwen_coder.data.deduplication import normalized_record_fingerprint
from tiny_qwen_coder.data.pipeline import DatasetPipelineError, run_dataset_pipeline
from tiny_qwen_coder.data.records import (
    LicenseMetadata,
    NormalizedTrainingRecord,
    SourceProvenance,
    TrainingMessage,
    single_turn_messages,
)
from tiny_qwen_coder.evaluation import (
    EXACT_PROMPT_CHECK_ID,
    EXACT_SOLUTION_CHECK_ID,
    HIGH_OVERLAP_CHECK_ID,
    ContaminationCheckError,
    HighOverlapConfig,
    ProtectedBenchmark,
    ProtectedBenchmarkExample,
    ProtectedBenchmarkRegistry,
    check_training_contamination,
)
from tiny_qwen_coder.languages import (
    ConfigReferences,
    LanguageComponentRef,
    LanguageConfig,
    LanguageHookReferences,
    LanguageSpec,
    ProtectedBenchmarkRef,
    RepositoryDetectionSignals,
    StaticLanguagePlugin,
    SystemPromptSpec,
)
from tiny_qwen_coder.model.inspection import InspectionTarget
from tiny_qwen_coder.reporting import ContaminationStatus, ContaminationSummary, GitMetadata

_REVISION = "a" * 40
_MODEL_REVISION = "b" * 40


class FakeTokenizer:
    chat_template = "fixture canonical template"

    def __init__(self) -> None:
        self.init_kwargs: dict[str, object] = {"_commit_hash": _MODEL_REVISION}

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
        del tokenize, add_generation_prompt, truncation, return_dict, chat_template
        token_count = 2 + sum(len(message["content"].split()) for message in conversation)
        return list(range(token_count))


def _plugin(*benchmark_ids: str) -> StaticLanguagePlugin:
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
            evaluation=("configs/eval/python/benchmarks.yaml",),
        ),
        hooks=LanguageHookReferences(validator=validator_ref, executor=executor_ref),
    )
    return StaticLanguagePlugin(
        LanguageSpec(
            config=config,
            execution_hook=LanguageComponentRef(id="default", import_ref=executor_ref),
            protected_benchmarks=tuple(
                ProtectedBenchmarkRef(id=benchmark_id) for benchmark_id in benchmark_ids
            ),
        )
    )


def _registry(*benchmark_ids: str) -> tuple[StaticLanguagePlugin, ProtectedBenchmarkRegistry]:
    plugin = _plugin(*benchmark_ids)
    registry = ProtectedBenchmarkRegistry()
    registry.register_language(
        plugin,
        tuple(
            ProtectedBenchmark(
                language="python",
                id=benchmark_id,
                dataset_id=f"fixtures/{benchmark_id}",
                dataset_revision=_REVISION,
                source_configs=(f"configs/eval/python/{benchmark_id}.yaml",),
            )
            for benchmark_id in benchmark_ids
        ),
    )
    return plugin, registry


def _protected(
    *,
    benchmark_id: str,
    record_id: str,
    prompt: str,
    solution: str | None,
) -> ProtectedBenchmarkExample:
    return ProtectedBenchmarkExample(
        language="python",
        benchmark_id=benchmark_id,
        dataset_id=f"fixtures/{benchmark_id}",
        dataset_revision=_REVISION,
        record_id=record_id,
        prompt_messages=(TrainingMessage(role="user", content=prompt),),
        solution=solution,
    )


def _record(*, record_id: str, prompt: str, response: str) -> NormalizedTrainingRecord:
    return NormalizedTrainingRecord(
        schema_version=1,
        language="python",
        messages=single_turn_messages(system=None, user=prompt, assistant=response),
        provenance=SourceProvenance(
            source_id="fixtures/train",
            revision=_REVISION,
            license=LicenseMetadata(name="MIT"),
            split="train",
            record_id=record_id,
        ),
    )


def _config() -> DataPreparationConfig:
    return DataPreparationConfig(
        schema_version=1,
        language="python",
        source_configs=("configs/data/python/fixture.yaml",),
        output_dir="data/python/fixture",
        seed=17,
        validation_fraction=0.34,
        min_tokens=1,
        max_tokens=128,
        truncation_policy="reject",
        deduplicate=True,
    )


def _target() -> InspectionTarget:
    return InspectionTarget(
        config_id="qwen35-4b",
        model_repository="Qwen/Qwen3.5-4B",
        model_revision=_MODEL_REVISION,
        tokenizer_repository="Qwen/Qwen3.5-4B",
        tokenizer_revision=_MODEL_REVISION,
        model_load_dtype="bfloat16",
    )


def test_detects_exact_normalized_prompt_copy_and_exact_solution_copy() -> None:
    _plugin_value, registry = _registry("holdout")
    protected = (
        _protected(
            benchmark_id="holdout",
            record_id="p1",
            prompt="\ufeffWrite a function\r\nthat adds two integers.",
            solution="def add(a, b):\n    return a + b\n",
        ),
    )
    prompt_copy = _record(
        record_id="train-prompt-copy",
        prompt="Write a function\nthat adds two integers.",
        response="different implementation",
    )
    solution_copy = _record(
        record_id="train-solution-copy",
        prompt="Completely different task",
        response="def add(a, b):\r\n    return a + b\r\n",
    )

    summary = check_training_contamination(
        (prompt_copy, solution_copy),
        protected,
        language="python",
        registry=registry,
    )

    assert summary.status is ContaminationStatus.FINDINGS
    assert summary.check_ids == (
        EXACT_PROMPT_CHECK_ID,
        EXACT_SOLUTION_CHECK_ID,
        HIGH_OVERLAP_CHECK_ID,
    )
    assert {(item.finding_type, item.protected_record_id) for item in summary.findings} == {
        ("exact_prompt_match", "p1"),
        ("exact_solution_match", "p1"),
    }
    hashes = {
        normalized_record_fingerprint(item).record_sha256 for item in (prompt_copy, solution_copy)
    }
    assert {item.training_record_sha256 for item in summary.findings} == hashes


def test_reports_suspicious_high_prompt_and_solution_overlap_without_exact_duplicates() -> None:
    _plugin_value, registry = _registry("holdout")
    protected_prompt = (
        "Implement a function that receives a list of integer values and returns the first "
        "duplicate value while preserving the original scan order and returning none when "
        "every element is unique."
    )
    protected_solution = (
        "def first_duplicate(values):\n"
        "    seen = set()\n"
        "    for value in values:\n"
        "        if value in seen:\n"
        "            return value\n"
        "        seen.add(value)\n"
        "    return None\n"
    )
    protected = (
        _protected(
            benchmark_id="holdout",
            record_id="p1",
            prompt=protected_prompt,
            solution=protected_solution,
        ),
    )
    near_copy = _record(
        record_id="near-copy",
        prompt=f"For this exercise only, {protected_prompt} Please provide Python code.",
        response=protected_solution + "\n# copied with a trailing comment\n",
    )

    summary = check_training_contamination(
        (near_copy,),
        protected,
        language="python",
        registry=registry,
        overlap=HighOverlapConfig(threshold=0.8, shingle_size=5, min_tokens=16),
    )

    assert summary.status is ContaminationStatus.FINDINGS
    finding_types = {item.finding_type for item in summary.findings}
    assert finding_types == {"high_prompt_overlap", "high_solution_overlap"}
    assert all(item.checker_id == HIGH_OVERLAP_CHECK_ID for item in summary.findings)
    assert all(item.score is not None and item.score >= 0.8 for item in summary.findings)


def test_clean_summary_requires_complete_registered_benchmark_coverage() -> None:
    _plugin_value, registry = _registry("bench-a", "bench-b")
    only_a = (
        _protected(
            benchmark_id="bench-a",
            record_id="a1",
            prompt="A protected prompt with enough meaningful words for deterministic checking.",
            solution=None,
        ),
    )

    with pytest.raises(ContaminationCheckError, match="coverage.*missing=.*bench-b"):
        check_training_contamination(
            (_record(record_id="train", prompt="unrelated prompt", response="unrelated answer"),),
            only_a,
            language="python",
            registry=registry,
        )


def test_protected_examples_must_match_registered_dataset_identity() -> None:
    _plugin_value, registry = _registry("holdout")
    example = replace(
        _protected(
            benchmark_id="holdout",
            record_id="p1",
            prompt="Protected prompt",
            solution=None,
        ),
        dataset_revision="c" * 40,
    )

    with pytest.raises(ContaminationCheckError, match="does not match registered dataset identity"):
        check_training_contamination(
            (_record(record_id="train", prompt="unrelated", response="answer"),),
            (example,),
            language="python",
            registry=registry,
        )


def test_clean_result_is_deterministic_and_does_not_claim_solution_check_when_unavailable() -> None:
    _plugin_value, registry = _registry("holdout")
    protected = (
        _protected(
            benchmark_id="holdout",
            record_id="p1",
            prompt="This protected prompt is intentionally distinct from all training examples.",
            solution=None,
        ),
    )
    training = (
        _record(record_id="t2", prompt="second unrelated prompt", response="second response"),
        _record(record_id="t1", prompt="first unrelated prompt", response="first response"),
    )

    first = check_training_contamination(
        training,
        protected,
        language="python",
        registry=registry,
    )
    second = check_training_contamination(
        training,
        protected,
        language="python",
        registry=registry,
    )

    assert first == second
    assert first.status is ContaminationStatus.CLEAN
    assert first.check_ids == (EXACT_PROMPT_CHECK_ID, HIGH_OVERLAP_CHECK_ID)
    assert first.findings == ()


def test_pipeline_manifest_detects_injected_protected_benchmark_copy() -> None:
    plugin, registry = _registry("holdout")
    protected = (
        _protected(
            benchmark_id="holdout",
            record_id="p1",
            prompt="Return the sum of two integers.",
            solution="def add(a, b): return a + b",
        ),
    )
    records = (
        _record(
            record_id="copy",
            prompt="Return the sum of two integers.",
            response="different answer",
        ),
        _record(record_id="safe-1", prompt="Return the square of x.", response="return x * x"),
        _record(record_id="safe-2", prompt="Return the cube of x.", response="return x * x * x"),
    )

    result = run_dataset_pipeline(
        records,
        config=_config(),
        plugin=plugin,
        tokenizer=FakeTokenizer(),
        target=_target(),
        protected_benchmarks=registry,
        protected_examples=protected,
        git=GitMetadata(sha="d" * 40, dirty=False),
    )

    assert result.manifest.contamination.status is ContaminationStatus.FINDINGS
    assert len(result.manifest.contamination.findings) == 1
    finding = result.manifest.contamination.findings[0]
    assert finding.finding_type == "exact_prompt_match"
    assert finding.protected_dataset_id == "holdout"
    assert finding.protected_record_id == "p1"
    assert finding.training_record_sha256 in {
        membership.record_sha256 for membership in result.manifest.memberships
    }


def test_pipeline_rejects_ambiguous_contamination_configuration_before_processing() -> None:
    plugin, registry = _registry("holdout")
    protected = (
        _protected(
            benchmark_id="holdout",
            record_id="p1",
            prompt="Protected prompt",
            solution=None,
        ),
    )
    external = ContaminationSummary(
        status=ContaminationStatus.CLEAN,
        check_ids=("custom_check",),
        findings=(),
    )

    with pytest.raises(DatasetPipelineError, match="either external contamination evidence"):
        run_dataset_pipeline(
            (),
            config=_config(),
            plugin=plugin,
            tokenizer=object(),
            target=_target(),
            contamination=external,
            protected_benchmarks=registry,
            protected_examples=protected,
        )

    with pytest.raises(DatasetPipelineError, match="contamination_overlap requires"):
        run_dataset_pipeline(
            (),
            config=_config(),
            plugin=plugin,
            tokenizer=object(),
            target=_target(),
            protected_benchmarks=registry,
            contamination_overlap=HighOverlapConfig(),
        )
