from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from tiny_qwen_coder.data import apply_language_validators
from tiny_qwen_coder.data.records import (
    LicenseMetadata,
    NormalizedTrainingRecord,
    SourceProvenance,
    ValidationResult,
    single_turn_messages,
)
from tiny_qwen_coder.languages import (
    PRIMARY_VALIDATOR_ID,
    LanguageConfigError,
    LanguageRegistry,
    load_language_config,
    load_language_plugin,
)

_CONFIG_PATH = Path("configs/languages/python.yaml")


def _record(*, language: str, assistant: str = "print('ok')") -> NormalizedTrainingRecord:
    return NormalizedTrainingRecord(
        schema_version=1,
        language=language,
        messages=single_turn_messages(system=None, user="Write a program.", assistant=assistant),
        provenance=SourceProvenance(
            source_id="fixture/python",
            revision="a" * 40,
            license=LicenseMetadata(name="MIT"),
            record_id="fixture-1",
        ),
    )


def _resolve(import_ref: str) -> object:
    module_name, attribute = import_ref.split(":", maxsplit=1)
    return cast(object, getattr(importlib.import_module(module_name), attribute))


def test_python_config_is_concrete_versioned_and_python3_specific() -> None:
    config = load_language_config(_CONFIG_PATH)

    assert config.id == "python"
    assert config.aliases == ("py",)
    assert config.extensions == (".py",)
    assert "pyproject.toml" in config.repository_detection.files
    assert "uv.lock" in config.repository_detection.files
    assert "**/*.py" in config.repository_detection.globs
    assert config.system_prompt.version == "python-v1"
    assert "Python 3" in config.system_prompt.text
    assert config.hooks.validator.endswith(":validate_python_record")
    assert config.hooks.executor.endswith(":execute_python")


def test_python_config_references_planned_data_and_evaluation_configs() -> None:
    config = load_language_config(_CONFIG_PATH)

    assert config.config_refs.data_sources == (
        "configs/data/python/olmo_starcoder_python_instruct.yaml",
        "configs/data/python/magicoder_oss_instruct_75k.yaml",
    )
    assert config.config_refs.evaluation == ("configs/eval/python.yaml",)


def test_language_config_loader_fails_closed_on_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "python.yaml"
    path.write_text(
        _CONFIG_PATH.read_text(encoding="utf-8") + "surprise: forbidden\n",
        encoding="utf-8",
    )

    with pytest.raises(LanguageConfigError, match=r"unknown field\(s\): surprise"):
        load_language_config(path)


def test_registry_selects_python_entirely_from_loaded_plugin_config() -> None:
    plugin = load_language_plugin(_CONFIG_PATH)
    registry = LanguageRegistry((plugin,))

    assert registry.resolve("python") is plugin
    assert registry.resolve("py") is plugin
    assert registry.list_ids() == ("python",)
    assert plugin.spec.config == load_language_config(_CONFIG_PATH)
    assert plugin.spec.validators[0].id == PRIMARY_VALIDATOR_ID
    assert plugin.spec.validators[0].import_ref == plugin.spec.config.hooks.validator
    assert plugin.spec.execution_hook.import_ref == plugin.spec.config.hooks.executor
    assert plugin.spec.data_adapters == ()
    assert plugin.spec.protected_benchmarks == ()
    assert plugin.spec.evaluation_hooks == ()


def test_python_hook_references_resolve_to_callables() -> None:
    config = load_language_config(_CONFIG_PATH)

    assert callable(_resolve(config.hooks.validator))
    assert callable(_resolve(config.hooks.executor))


def test_python_validator_records_language_identity_without_doing_p5_004_syntax_work() -> None:
    config = load_language_config(_CONFIG_PATH)
    validator = cast(
        Callable[[NormalizedTrainingRecord], ValidationResult],
        _resolve(config.hooks.validator),
    )

    python_result = validator(_record(language="python", assistant="not valid Python !!!"))
    assert python_result.validator_id == PRIMARY_VALIDATOR_ID
    assert python_result.passed is True
    assert python_result.detail is None

    rust_result = validator(_record(language="rust"))
    assert rust_result.validator_id == PRIMARY_VALIDATOR_ID
    assert rust_result.passed is False
    assert rust_result.detail == "expected record language 'python'; got 'rust'"


def test_generic_pipeline_applies_python_validator_after_registry_selection() -> None:
    registry = LanguageRegistry((load_language_plugin(_CONFIG_PATH),))
    selected = registry.resolve("py")

    validated = apply_language_validators((_record(language="python"),), selected)

    assert len(validated) == 1
    assert validated[0].validation is not None
    assert validated[0].validation.results == (
        ValidationResult(validator_id=PRIMARY_VALIDATOR_ID, passed=True),
    )
