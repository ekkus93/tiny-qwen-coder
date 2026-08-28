"""Pipeline-boundary tests for protected benchmark SFT access control."""

from __future__ import annotations

from typing import cast

import pytest

from tiny_qwen_coder.config import DataPreparationConfig
from tiny_qwen_coder.data.pipeline import run_dataset_pipeline
from tiny_qwen_coder.evaluation import (
    ProtectedBenchmark,
    ProtectedBenchmarkRegistrationError,
    ProtectedBenchmarkRegistry,
    ProtectedBenchmarkTrainingSelectionError,
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


def _protected_plugin() -> StaticLanguagePlugin:
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
            evaluation=("configs/eval/python/holdout.yaml",),
        ),
        hooks=LanguageHookReferences(validator=validator_ref, executor=executor_ref),
    )
    return StaticLanguagePlugin(
        LanguageSpec(
            config=config,
            execution_hook=LanguageComponentRef(id="default", import_ref=executor_ref),
            protected_benchmarks=(ProtectedBenchmarkRef(id="holdout"),),
        )
    )


def _data_config(source_config: str) -> DataPreparationConfig:
    return DataPreparationConfig(
        schema_version=1,
        language="python",
        source_configs=(source_config,),
        output_dir="data/python/fixture",
        seed=17,
        validation_fraction=0.1,
        min_tokens=1,
        max_tokens=8,
        truncation_policy="reject",
        deduplicate=True,
    )


def _unused_target() -> InspectionTarget:
    """Return a typed sentinel; protected checks fail before target access."""

    return cast(InspectionTarget, object())


def test_pipeline_requires_registry_for_declared_protected_benchmarks() -> None:
    with pytest.raises(ProtectedBenchmarkRegistrationError, match="registry mismatch"):
        run_dataset_pipeline(
            (),
            config=_data_config("configs/data/python/fixture.yaml"),
            plugin=_protected_plugin(),
            tokenizer=object(),
            target=_unused_target(),
        )


def test_pipeline_rejects_protected_benchmark_before_record_processing() -> None:
    plugin = _protected_plugin()
    registry = ProtectedBenchmarkRegistry()
    registry.register_language(
        plugin,
        (
            ProtectedBenchmark(
                language="python",
                id="holdout",
                dataset_id="fixtures/python-holdout",
                dataset_revision="a" * 40,
                source_configs=("configs/eval/python/holdout.yaml",),
            ),
        ),
    )

    with pytest.raises(ProtectedBenchmarkTrainingSelectionError, match="python/holdout"):
        run_dataset_pipeline(
            (),
            config=_data_config("configs/eval/python/holdout.yaml"),
            plugin=plugin,
            tokenizer=object(),
            target=_unused_target(),
            protected_benchmarks=registry,
        )
