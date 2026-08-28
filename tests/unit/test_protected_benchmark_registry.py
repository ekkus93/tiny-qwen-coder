"""Tests for protected evaluation-dataset registration and SFT access control."""

from __future__ import annotations

import pytest

from tiny_qwen_coder.config import DataPreparationConfig
from tiny_qwen_coder.evaluation import (
    ProtectedBenchmark,
    ProtectedBenchmarkRegistrationError,
    ProtectedBenchmarkRegistry,
    ProtectedBenchmarkTrainingSelectionError,
    UnknownProtectedBenchmarkError,
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


def _plugin(
    language: str,
    *benchmark_ids: str,
) -> StaticLanguagePlugin:
    validator_ref = f"tiny_qwen_coder.languages.{language}:validate"
    executor_ref = f"tiny_qwen_coder.languages.{language}:execute"
    config = LanguageConfig(
        schema_version=1,
        id=language,
        aliases=(),
        extensions=(f".{language}",),
        repository_detection=RepositoryDetectionSignals(files=(f"{language}.project",)),
        system_prompt=SystemPromptSpec(version="v1", text=f"Write {language} code."),
        config_refs=ConfigReferences(
            data_sources=(f"configs/data/{language}/source.yaml",),
            evaluation=(f"configs/eval/{language}/benchmarks.yaml",),
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


def _benchmark(
    language: str,
    benchmark_id: str,
    *,
    dataset_id: str | None = None,
) -> ProtectedBenchmark:
    return ProtectedBenchmark(
        language=language,
        id=benchmark_id,
        dataset_id=dataset_id or f"fixtures/{language}-{benchmark_id}",
        dataset_revision="a" * 40,
        source_configs=(f"configs/eval/{language}/{benchmark_id}.yaml",),
    )


def _data_config(language: str, *source_configs: str) -> DataPreparationConfig:
    return DataPreparationConfig(
        schema_version=1,
        language=language,
        source_configs=source_configs,
        output_dir=f"data/{language}/p0",
        seed=17,
        validation_fraction=0.1,
        min_tokens=1,
        max_tokens=2048,
        truncation_policy="reject",
        deduplicate=True,
    )


def test_registers_protected_datasets_per_language_deterministically() -> None:
    python = _plugin("python", "humaneval", "mbpp")
    rust = _plugin("rust", "rust-eval")
    registry = ProtectedBenchmarkRegistry()

    registry.register_language(
        rust,
        (_benchmark("rust", "rust-eval", dataset_id="fixtures/rust-eval"),),
    )
    registry.register_language(
        python,
        (
            _benchmark("python", "mbpp", dataset_id="google-research-datasets/mbpp"),
            _benchmark("python", "humaneval", dataset_id="openai/openai_humaneval"),
        ),
    )

    assert registry.list_languages() == ("python", "rust")
    assert tuple(item.qualified_id for item in registry.list_benchmarks()) == (
        "python/humaneval",
        "python/mbpp",
        "rust/rust-eval",
    )
    assert registry.resolve("python", "humaneval").dataset_id == "openai/openai_humaneval"


def test_registration_requires_exact_plugin_declarations_and_is_atomic() -> None:
    plugin = _plugin("python", "humaneval", "mbpp")
    registry = ProtectedBenchmarkRegistry()

    with pytest.raises(ProtectedBenchmarkRegistrationError, match="do not match registrations"):
        registry.register_language(plugin, (_benchmark("python", "humaneval"),))

    assert registry.list_languages() == ()
    assert registry.list_benchmarks() == ()


def test_registration_rejects_cross_language_and_selector_collisions_atomically() -> None:
    python = _plugin("python", "humaneval")
    rust = _plugin("rust", "rust-eval")
    registry = ProtectedBenchmarkRegistry()
    registry.register_language(python, (_benchmark("python", "humaneval"),))

    with pytest.raises(ProtectedBenchmarkRegistrationError, match="cannot be registered"):
        registry.register_language(rust, (_benchmark("python", "rust-eval"),))

    conflicting = ProtectedBenchmark(
        language="rust",
        id="rust-eval",
        dataset_id="fixtures/rust-rust-eval",
        dataset_revision="a" * 40,
        source_configs=("configs/eval/python/humaneval.yaml",),
    )
    with pytest.raises(ProtectedBenchmarkRegistrationError, match="already owned"):
        registry.register_language(rust, (conflicting,))

    assert registry.list_languages() == ("python",)


def test_normal_sft_sources_are_allowed() -> None:
    plugin = _plugin("python", "humaneval", "mbpp")
    registry = ProtectedBenchmarkRegistry()
    registry.register_language(
        plugin,
        (_benchmark("python", "humaneval"), _benchmark("python", "mbpp")),
    )

    registry.assert_plugin_registration_matches(plugin)
    registry.assert_sft_config_allowed(
        _data_config(
            "python",
            "configs/data/python/olmo-coding.yaml",
            "configs/data/python/magicoder.yaml",
        )
    )


def test_sft_rejects_protected_source_config_with_clear_identity() -> None:
    plugin = _plugin("python", "humaneval", "mbpp")
    registry = ProtectedBenchmarkRegistry()
    registry.register_language(
        plugin,
        (_benchmark("python", "humaneval"), _benchmark("python", "mbpp")),
    )

    with pytest.raises(
        ProtectedBenchmarkTrainingSelectionError,
        match=r"evaluation-only.*'configs/eval/python/humaneval.yaml' -> 'python/humaneval'",
    ):
        registry.assert_sft_config_allowed(
            _data_config("python", "configs/eval/python/humaneval.yaml")
        )


def test_sft_rejects_direct_upstream_dataset_id_and_cross_language_selection() -> None:
    python = _plugin("python", "humaneval")
    registry = ProtectedBenchmarkRegistry()
    registry.register_language(
        python,
        (
            _benchmark(
                "python",
                "humaneval",
                dataset_id="openai/openai_humaneval",
            ),
        ),
    )

    with pytest.raises(ProtectedBenchmarkTrainingSelectionError, match="python/humaneval"):
        registry.assert_sft_config_allowed(_data_config("rust", "openai/openai_humaneval"))


def test_declared_protection_cannot_be_silently_omitted_from_registry() -> None:
    plugin = _plugin("python", "humaneval")

    with pytest.raises(ProtectedBenchmarkRegistrationError, match="registry mismatch"):
        ProtectedBenchmarkRegistry().assert_plugin_registration_matches(plugin)


def test_unknown_benchmark_failure_lists_registered_ids_deterministically() -> None:
    plugin = _plugin("python", "humaneval", "mbpp")
    registry = ProtectedBenchmarkRegistry()
    registry.register_language(
        plugin,
        (_benchmark("python", "mbpp"), _benchmark("python", "humaneval")),
    )

    with pytest.raises(
        UnknownProtectedBenchmarkError,
        match="registered benchmarks: python/humaneval, python/mbpp",
    ):
        registry.resolve("python", "unknown")
