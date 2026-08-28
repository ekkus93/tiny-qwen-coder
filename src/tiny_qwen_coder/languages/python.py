"""Python language hooks used by the declarative Python plugin config."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import NoReturn

from tiny_qwen_coder.data.records import NormalizedTrainingRecord, ValidationResult
from tiny_qwen_coder.languages.loading import PRIMARY_VALIDATOR_ID, load_language_plugin
from tiny_qwen_coder.languages.python_quality import PYTHON_QUALITY_VALIDATOR_ID
from tiny_qwen_coder.languages.spec import LanguageComponentRef, StaticLanguagePlugin

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
    quality_validator = LanguageComponentRef(
        id=PYTHON_QUALITY_VALIDATOR_ID,
        import_ref=_PYTHON_QUALITY_IMPORT_REF,
    )
    return StaticLanguagePlugin(
        replace(
            base_plugin.spec,
            data_adapters=adapters,
            validators=base_plugin.spec.validators + (quality_validator,),
        )
    )
