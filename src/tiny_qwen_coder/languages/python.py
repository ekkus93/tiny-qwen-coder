"""Python language hooks used by the declarative Python plugin config."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import NoReturn

from tiny_qwen_coder.data.records import NormalizedTrainingRecord, ValidationResult
from tiny_qwen_coder.evaluation.protected_benchmarks import (
    ProtectedBenchmark,
    ProtectedBenchmarkRegistry,
    load_protected_benchmark_config,
)
from tiny_qwen_coder.languages.loading import PRIMARY_VALIDATOR_ID, load_language_plugin
from tiny_qwen_coder.languages.python_quality import PYTHON_QUALITY_VALIDATOR_ID
from tiny_qwen_coder.languages.spec import (
    LanguageComponentRef,
    ProtectedBenchmarkRef,
    StaticLanguagePlugin,
)

_PYTHON_CONFIG_PATH = Path("configs/languages/python.yaml")
_OLMO_SOURCE_CONFIG_PATH = Path("configs/data/python/olmo_starcoder_python_instruct.yaml")
_MAGICODER_SOURCE_CONFIG_PATH = Path("configs/data/python/magicoder_oss_instruct_75k.yaml")
_PYTHON_QUALITY_IMPORT_REF = "tiny_qwen_coder.languages.python_quality:validate_python_quality"


def validate_python_record(record: NormalizedTrainingRecord) -> ValidationResult:
    """Verify that a normalized record is labeled for the Python plugin."""

    if record.language == "python":
        return ValidationResult(validator_id=PRIMARY_VALIDATOR_ID, passed=True)
    return ValidationResult(
        validator_id=PRIMARY_VALIDATOR_ID,
        passed=False,
        detail=f"expected record language 'python'; got {record.language!r}",
    )


def execute_python() -> NoReturn:
    """Fail clearly until Phase 6 wires Python evaluation to the constrained harness."""

    raise NotImplementedError("Python execution is implemented by the Phase 6 evaluators")


def _load_python_protected_benchmarks(
    evaluation_configs: tuple[str, ...],
) -> tuple[ProtectedBenchmark, ...]:
    benchmarks = tuple(
        load_protected_benchmark_config(Path(config_path))
        for config_path in evaluation_configs
    )
    for benchmark in benchmarks:
        if benchmark.language != "python":
            raise ValueError(
                f"protected benchmark {benchmark.qualified_id!r} does not belong to Python"
            )
    return benchmarks


def load_python_plugin(
    config_path: Path = _PYTHON_CONFIG_PATH,
    olmo_source_config_path: Path = _OLMO_SOURCE_CONFIG_PATH,
    magicoder_source_config_path: Path = _MAGICODER_SOURCE_CONFIG_PATH,
) -> StaticLanguagePlugin:
    """Load the concrete Python plugin with source adapters and quality validation."""

    from tiny_qwen_coder.data.source_config import load_dataset_source_config

    sources = (
        load_dataset_source_config(olmo_source_config_path),
        load_dataset_source_config(magicoder_source_config_path),
    )
    base_plugin = load_language_plugin(config_path)
    config = base_plugin.spec.config
    for source in sources:
        if source.language != config.id:
            raise ValueError(
                f"source language {source.language!r} does not match Python config {config.id!r}"
            )
    adapters = tuple(
        LanguageComponentRef(id=source.id, import_ref=source.adapter) for source in sources
    )
    protected_benchmarks = _load_python_protected_benchmarks(config.config_refs.evaluation)
    quality_validator = LanguageComponentRef(
        id=PYTHON_QUALITY_VALIDATOR_ID,
        import_ref=_PYTHON_QUALITY_IMPORT_REF,
    )
    return StaticLanguagePlugin(
        replace(
            base_plugin.spec,
            data_adapters=adapters,
            validators=base_plugin.spec.validators + (quality_validator,),
            protected_benchmarks=tuple(
                ProtectedBenchmarkRef(id=benchmark.id) for benchmark in protected_benchmarks
            ),
        )
    )


def load_python_protected_benchmark_registry(
    config_path: Path = _PYTHON_CONFIG_PATH,
    olmo_source_config_path: Path = _OLMO_SOURCE_CONFIG_PATH,
    magicoder_source_config_path: Path = _MAGICODER_SOURCE_CONFIG_PATH,
) -> ProtectedBenchmarkRegistry:
    """Load Python's canonical evaluation-only benchmark registry."""

    plugin = load_python_plugin(
        config_path=config_path,
        olmo_source_config_path=olmo_source_config_path,
        magicoder_source_config_path=magicoder_source_config_path,
    )
    benchmarks = _load_python_protected_benchmarks(plugin.spec.config.config_refs.evaluation)
    registry = ProtectedBenchmarkRegistry()
    registry.register_language(plugin, benchmarks)
    return registry
