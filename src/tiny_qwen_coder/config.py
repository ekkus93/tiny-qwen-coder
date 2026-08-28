"""Strict, deterministic configuration schemas for Tiny Qwen Coder.

The configuration layer is intentionally CPU-only. It validates complete
configuration documents before model loading, dataset downloads, training, or
evaluation begins.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, TypeAlias, cast

import yaml

from tiny_qwen_coder.reproducibility import SeedError, validate_seed

_SCHEMA_VERSION = 1
_LANGUAGE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")

TruncationPolicy: TypeAlias = Literal["reject", "truncate"]
TrainingMode: TypeAlias = Literal["bf16_lora", "qlora_4bit"]
ComputeDtype: TypeAlias = Literal["bfloat16"]
LoraBias: TypeAlias = Literal["none", "all", "lora_only"]
TargetStrategy: TypeAlias = Literal["selective", "all_linear"]
LossMode: TypeAlias = Literal["assistant_only", "completion_only"]
AdapterSelectionMode: TypeAlias = Literal["explicit", "base_only", "auto"]
CompatibilityPolicy: TypeAlias = Literal["strict"]


class ConfigError(ValueError):
    """Raised when configuration is malformed or violates the schema contract."""


def _require_schema_version(value: int) -> None:
    if value != _SCHEMA_VERSION:
        raise ConfigError(f"unsupported schema_version {value}; expected {_SCHEMA_VERSION}")


def _require_language_id(value: str) -> None:
    if not _LANGUAGE_ID_PATTERN.fullmatch(value):
        raise ConfigError("language must match ^[a-z][a-z0-9_-]*$")


def _require_non_empty(value: str, *, field_name: str) -> None:
    if not value.strip():
        raise ConfigError(f"{field_name} must not be empty")


def _require_seed(seed: int) -> None:
    try:
        validate_seed(seed)
    except SeedError as exc:
        raise ConfigError(str(exc)) from exc


def _require_positive(value: int | float, *, field_name: str) -> None:
    if value <= 0:
        raise ConfigError(f"{field_name} must be greater than zero")


def _strict_mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{context} must be a mapping")

    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ConfigError(f"{context} keys must be strings")
        result[key] = item
    return result


def _validate_keys(
    mapping: Mapping[str, object],
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
    context: str,
) -> None:
    allowed = required | optional
    unknown = sorted(set(mapping) - allowed)
    missing = sorted(required - set(mapping))
    if unknown:
        raise ConfigError(f"{context} contains unknown field(s): {', '.join(unknown)}")
    if missing:
        raise ConfigError(f"{context} is missing required field(s): {', '.join(missing)}")


def _expect_str(mapping: Mapping[str, object], key: str, *, context: str) -> str:
    value = mapping[key]
    if not isinstance(value, str):
        raise ConfigError(f"{context}.{key} must be a string")
    _require_non_empty(value, field_name=f"{context}.{key}")
    return value


def _expect_optional_str(mapping: Mapping[str, object], key: str, *, context: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError(f"{context}.{key} must be a string or null")
    _require_non_empty(value, field_name=f"{context}.{key}")
    return value


def _expect_bool(mapping: Mapping[str, object], key: str, *, context: str) -> bool:
    value = mapping[key]
    if not isinstance(value, bool):
        raise ConfigError(f"{context}.{key} must be a boolean")
    return value


def _expect_int(mapping: Mapping[str, object], key: str, *, context: str) -> int:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{context}.{key} must be an integer")
    return value


def _expect_float(mapping: Mapping[str, object], key: str, *, context: str) -> float:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{context}.{key} must be a number")
    return float(value)


def _expect_str_tuple(mapping: Mapping[str, object], key: str, *, context: str) -> tuple[str, ...]:
    value = mapping[key]
    if not isinstance(value, list):
        raise ConfigError(f"{context}.{key} must be a YAML sequence")

    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ConfigError(f"{context}.{key}[{index}] must be a non-empty string")
        result.append(item)

    if len(result) != len(set(result)):
        raise ConfigError(f"{context}.{key} must not contain duplicates")
    return tuple(result)


def _expect_choice(
    mapping: Mapping[str, object],
    key: str,
    *,
    choices: frozenset[str],
    context: str,
) -> str:
    value = _expect_str(mapping, key, context=context)
    if value not in choices:
        expected = ", ".join(sorted(choices))
        raise ConfigError(f"{context}.{key} must be one of: {expected}")
    return value


@dataclass(frozen=True, slots=True)
class DataPreparationConfig:
    """Language-neutral dataset preparation configuration."""

    schema_version: int
    language: str
    source_configs: tuple[str, ...]
    output_dir: str
    seed: int
    validation_fraction: float
    min_tokens: int
    max_tokens: int
    truncation_policy: TruncationPolicy
    deduplicate: bool

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        _require_language_id(self.language)
        _require_seed(self.seed)
        if not self.source_configs:
            raise ConfigError("source_configs must contain at least one reference")
        _require_non_empty(self.output_dir, field_name="output_dir")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ConfigError("validation_fraction must be greater than 0 and less than 1")
        if self.min_tokens < 0:
            raise ConfigError("min_tokens must be non-negative")
        _require_positive(self.max_tokens, field_name="max_tokens")
        if self.min_tokens > self.max_tokens:
            raise ConfigError("min_tokens must not exceed max_tokens")


@dataclass(frozen=True, slots=True)
class LoraConfig:
    """LoRA-specific hyperparameters and target strategy."""

    rank: int
    alpha: int
    dropout: float
    bias: LoraBias
    target_strategy: TargetStrategy
    target_modules: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_positive(self.rank, field_name="lora.rank")
        _require_positive(self.alpha, field_name="lora.alpha")
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError("lora.dropout must be greater than or equal to 0 and less than 1")
        if self.target_strategy == "selective" and not self.target_modules:
            raise ConfigError("selective LoRA targeting requires target_modules")
        if self.target_strategy == "all_linear" and self.target_modules:
            raise ConfigError("all_linear LoRA targeting must not list target_modules")


@dataclass(frozen=True, slots=True)
class QuantizationConfig:
    """4-bit QLoRA quantization contract."""

    bits: int
    quant_type: Literal["nf4", "fp4"]
    double_quant: bool
    compute_dtype: ComputeDtype

    def __post_init__(self) -> None:
        if self.bits != 4:
            raise ConfigError("QLoRA quantization.bits must be 4")


@dataclass(frozen=True, slots=True)
class LoraTrainingConfig:
    """Language-neutral LoRA/QLoRA training configuration."""

    schema_version: int
    base_config: str
    language: str
    dataset_manifest: str
    output_dir: str
    seed: int
    training_mode: TrainingMode
    compute_dtype: ComputeDtype
    sequence_length: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    epochs: float
    learning_rate: float
    scheduler: str
    warmup_ratio: float
    gradient_checkpointing: bool
    loss_mode: LossMode
    lora: LoraConfig
    quantization: QuantizationConfig | None

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        _require_language_id(self.language)
        _require_seed(self.seed)
        for field_name, value in (
            ("base_config", self.base_config),
            ("dataset_manifest", self.dataset_manifest),
            ("output_dir", self.output_dir),
            ("scheduler", self.scheduler),
        ):
            _require_non_empty(value, field_name=field_name)

        _require_positive(self.sequence_length, field_name="sequence_length")
        _require_positive(self.micro_batch_size, field_name="micro_batch_size")
        _require_positive(
            self.gradient_accumulation_steps,
            field_name="gradient_accumulation_steps",
        )
        _require_positive(self.epochs, field_name="epochs")
        _require_positive(self.learning_rate, field_name="learning_rate")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ConfigError("warmup_ratio must be greater than or equal to 0 and less than 1")

        if self.training_mode == "bf16_lora" and self.quantization is not None:
            raise ConfigError("bf16_lora must not define quantization")
        if self.training_mode == "qlora_4bit" and self.quantization is None:
            raise ConfigError("qlora_4bit requires quantization")


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Frozen generation settings shared by base/adapter evaluations."""

    temperature: float
    top_p: float
    top_k: int
    max_new_tokens: int
    prompt_version: str

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ConfigError("generation.temperature must be non-negative")
        if not 0.0 < self.top_p <= 1.0:
            raise ConfigError("generation.top_p must be greater than 0 and at most 1")
        if self.top_k < 0:
            raise ConfigError("generation.top_k must be non-negative")
        _require_positive(self.max_new_tokens, field_name="generation.max_new_tokens")
        _require_non_empty(self.prompt_version, field_name="generation.prompt_version")


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    """Bounded execution settings used by evaluation."""

    timeout_seconds: float
    network_enabled: bool

    def __post_init__(self) -> None:
        _require_positive(self.timeout_seconds, field_name="execution.timeout_seconds")


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Language-neutral evaluation configuration."""

    schema_version: int
    base_config: str
    language: str
    adapter_id: str | None
    suites: tuple[str, ...]
    output_dir: str
    seed: int
    generation: GenerationConfig
    execution: ExecutionConfig

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        _require_language_id(self.language)
        _require_seed(self.seed)
        _require_non_empty(self.base_config, field_name="base_config")
        _require_non_empty(self.output_dir, field_name="output_dir")
        if self.adapter_id is not None:
            _require_non_empty(self.adapter_id, field_name="adapter_id")
        if not self.suites:
            raise ConfigError("suites must contain at least one evaluation suite")


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Runtime and adapter-selection configuration."""

    schema_version: int
    base_config: str
    selection_mode: AdapterSelectionMode
    adapter_id: str | None
    preload_adapters: tuple[str, ...]
    adapter_search_paths: tuple[str, ...]
    compatibility_policy: CompatibilityPolicy
    allow_auto_detection: bool

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        _require_non_empty(self.base_config, field_name="base_config")
        if not self.adapter_search_paths:
            raise ConfigError("adapter_search_paths must contain at least one path")

        if self.selection_mode == "explicit":
            if self.adapter_id is None:
                raise ConfigError("explicit adapter selection requires adapter_id")
        elif self.selection_mode == "base_only":
            if self.adapter_id is not None:
                raise ConfigError("base_only adapter selection must not define adapter_id")
        elif not self.allow_auto_detection:
            raise ConfigError("auto adapter selection requires allow_auto_detection=true")


ConfigModel: TypeAlias = (
    DataPreparationConfig | LoraTrainingConfig | EvaluationConfig | RuntimeConfig
)


def _parse_lora(value: object) -> LoraConfig:
    context = "training.lora"
    mapping = _strict_mapping(value, context=context)
    _validate_keys(
        mapping,
        required=frozenset(
            {"rank", "alpha", "dropout", "bias", "target_strategy", "target_modules"}
        ),
        context=context,
    )
    return LoraConfig(
        rank=_expect_int(mapping, "rank", context=context),
        alpha=_expect_int(mapping, "alpha", context=context),
        dropout=_expect_float(mapping, "dropout", context=context),
        bias=cast(
            LoraBias,
            _expect_choice(
                mapping,
                "bias",
                choices=frozenset({"none", "all", "lora_only"}),
                context=context,
            ),
        ),
        target_strategy=cast(
            TargetStrategy,
            _expect_choice(
                mapping,
                "target_strategy",
                choices=frozenset({"selective", "all_linear"}),
                context=context,
            ),
        ),
        target_modules=_expect_str_tuple(mapping, "target_modules", context=context),
    )


def _parse_quantization(value: object) -> QuantizationConfig:
    context = "training.quantization"
    mapping = _strict_mapping(value, context=context)
    _validate_keys(
        mapping,
        required=frozenset({"bits", "quant_type", "double_quant", "compute_dtype"}),
        context=context,
    )
    return QuantizationConfig(
        bits=_expect_int(mapping, "bits", context=context),
        quant_type=cast(
            Literal["nf4", "fp4"],
            _expect_choice(
                mapping,
                "quant_type",
                choices=frozenset({"nf4", "fp4"}),
                context=context,
            ),
        ),
        double_quant=_expect_bool(mapping, "double_quant", context=context),
        compute_dtype=cast(
            ComputeDtype,
            _expect_choice(
                mapping,
                "compute_dtype",
                choices=frozenset({"bfloat16"}),
                context=context,
            ),
        ),
    )


def parse_data_preparation_config(value: object) -> DataPreparationConfig:
    """Parse and validate one data-preparation config mapping."""

    context = "data"
    mapping = _strict_mapping(value, context=context)
    _validate_keys(
        mapping,
        required=frozenset(
            {
                "schema_version",
                "language",
                "source_configs",
                "output_dir",
                "seed",
                "validation_fraction",
                "min_tokens",
                "max_tokens",
                "truncation_policy",
                "deduplicate",
            }
        ),
        context=context,
    )
    return DataPreparationConfig(
        schema_version=_expect_int(mapping, "schema_version", context=context),
        language=_expect_str(mapping, "language", context=context),
        source_configs=_expect_str_tuple(mapping, "source_configs", context=context),
        output_dir=_expect_str(mapping, "output_dir", context=context),
        seed=_expect_int(mapping, "seed", context=context),
        validation_fraction=_expect_float(mapping, "validation_fraction", context=context),
        min_tokens=_expect_int(mapping, "min_tokens", context=context),
        max_tokens=_expect_int(mapping, "max_tokens", context=context),
        truncation_policy=cast(
            TruncationPolicy,
            _expect_choice(
                mapping,
                "truncation_policy",
                choices=frozenset({"reject", "truncate"}),
                context=context,
            ),
        ),
        deduplicate=_expect_bool(mapping, "deduplicate", context=context),
    )


def parse_lora_training_config(value: object) -> LoraTrainingConfig:
    """Parse and validate one LoRA training config mapping."""

    context = "training"
    mapping = _strict_mapping(value, context=context)
    _validate_keys(
        mapping,
        required=frozenset(
            {
                "schema_version",
                "base_config",
                "language",
                "dataset_manifest",
                "output_dir",
                "seed",
                "training_mode",
                "compute_dtype",
                "sequence_length",
                "micro_batch_size",
                "gradient_accumulation_steps",
                "epochs",
                "learning_rate",
                "scheduler",
                "warmup_ratio",
                "gradient_checkpointing",
                "loss_mode",
                "lora",
            }
        ),
        optional=frozenset({"quantization"}),
        context=context,
    )

    quantization_value = mapping.get("quantization")
    return LoraTrainingConfig(
        schema_version=_expect_int(mapping, "schema_version", context=context),
        base_config=_expect_str(mapping, "base_config", context=context),
        language=_expect_str(mapping, "language", context=context),
        dataset_manifest=_expect_str(mapping, "dataset_manifest", context=context),
        output_dir=_expect_str(mapping, "output_dir", context=context),
        seed=_expect_int(mapping, "seed", context=context),
        training_mode=cast(
            TrainingMode,
            _expect_choice(
                mapping,
                "training_mode",
                choices=frozenset({"bf16_lora", "qlora_4bit"}),
                context=context,
            ),
        ),
        compute_dtype=cast(
            ComputeDtype,
            _expect_choice(
                mapping,
                "compute_dtype",
                choices=frozenset({"bfloat16"}),
                context=context,
            ),
        ),
        sequence_length=_expect_int(mapping, "sequence_length", context=context),
        micro_batch_size=_expect_int(mapping, "micro_batch_size", context=context),
        gradient_accumulation_steps=_expect_int(
            mapping,
            "gradient_accumulation_steps",
            context=context,
        ),
        epochs=_expect_float(mapping, "epochs", context=context),
        learning_rate=_expect_float(mapping, "learning_rate", context=context),
        scheduler=_expect_str(mapping, "scheduler", context=context),
        warmup_ratio=_expect_float(mapping, "warmup_ratio", context=context),
        gradient_checkpointing=_expect_bool(
            mapping,
            "gradient_checkpointing",
            context=context,
        ),
        loss_mode=cast(
            LossMode,
            _expect_choice(
                mapping,
                "loss_mode",
                choices=frozenset({"assistant_only", "completion_only"}),
                context=context,
            ),
        ),
        lora=_parse_lora(mapping["lora"]),
        quantization=(
            None if quantization_value is None else _parse_quantization(quantization_value)
        ),
    )


def _parse_generation(value: object) -> GenerationConfig:
    context = "evaluation.generation"
    mapping = _strict_mapping(value, context=context)
    _validate_keys(
        mapping,
        required=frozenset({"temperature", "top_p", "top_k", "max_new_tokens", "prompt_version"}),
        context=context,
    )
    return GenerationConfig(
        temperature=_expect_float(mapping, "temperature", context=context),
        top_p=_expect_float(mapping, "top_p", context=context),
        top_k=_expect_int(mapping, "top_k", context=context),
        max_new_tokens=_expect_int(mapping, "max_new_tokens", context=context),
        prompt_version=_expect_str(mapping, "prompt_version", context=context),
    )


def _parse_execution(value: object) -> ExecutionConfig:
    context = "evaluation.execution"
    mapping = _strict_mapping(value, context=context)
    _validate_keys(
        mapping,
        required=frozenset({"timeout_seconds", "network_enabled"}),
        context=context,
    )
    return ExecutionConfig(
        timeout_seconds=_expect_float(mapping, "timeout_seconds", context=context),
        network_enabled=_expect_bool(mapping, "network_enabled", context=context),
    )


def parse_evaluation_config(value: object) -> EvaluationConfig:
    """Parse and validate one evaluation config mapping."""

    context = "evaluation"
    mapping = _strict_mapping(value, context=context)
    _validate_keys(
        mapping,
        required=frozenset(
            {
                "schema_version",
                "base_config",
                "language",
                "suites",
                "output_dir",
                "seed",
                "generation",
                "execution",
            }
        ),
        optional=frozenset({"adapter_id"}),
        context=context,
    )
    return EvaluationConfig(
        schema_version=_expect_int(mapping, "schema_version", context=context),
        base_config=_expect_str(mapping, "base_config", context=context),
        language=_expect_str(mapping, "language", context=context),
        adapter_id=_expect_optional_str(mapping, "adapter_id", context=context),
        suites=_expect_str_tuple(mapping, "suites", context=context),
        output_dir=_expect_str(mapping, "output_dir", context=context),
        seed=_expect_int(mapping, "seed", context=context),
        generation=_parse_generation(mapping["generation"]),
        execution=_parse_execution(mapping["execution"]),
    )


def parse_runtime_config(value: object) -> RuntimeConfig:
    """Parse and validate one runtime/adapter-selection config mapping."""

    context = "runtime"
    mapping = _strict_mapping(value, context=context)
    _validate_keys(
        mapping,
        required=frozenset(
            {
                "schema_version",
                "base_config",
                "selection_mode",
                "preload_adapters",
                "adapter_search_paths",
                "compatibility_policy",
                "allow_auto_detection",
            }
        ),
        optional=frozenset({"adapter_id"}),
        context=context,
    )
    return RuntimeConfig(
        schema_version=_expect_int(mapping, "schema_version", context=context),
        base_config=_expect_str(mapping, "base_config", context=context),
        selection_mode=cast(
            AdapterSelectionMode,
            _expect_choice(
                mapping,
                "selection_mode",
                choices=frozenset({"explicit", "base_only", "auto"}),
                context=context,
            ),
        ),
        adapter_id=_expect_optional_str(mapping, "adapter_id", context=context),
        preload_adapters=_expect_str_tuple(mapping, "preload_adapters", context=context),
        adapter_search_paths=_expect_str_tuple(mapping, "adapter_search_paths", context=context),
        compatibility_policy=cast(
            CompatibilityPolicy,
            _expect_choice(
                mapping,
                "compatibility_policy",
                choices=frozenset({"strict"}),
                context=context,
            ),
        ),
        allow_auto_detection=_expect_bool(mapping, "allow_auto_detection", context=context),
    )


def load_yaml_mapping(path: Path) -> dict[str, object]:
    """Load one YAML document and require a string-keyed mapping root."""

    try:
        loaded: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"could not read config {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    return _strict_mapping(loaded, context=str(path))


def load_data_preparation_config(path: Path) -> DataPreparationConfig:
    """Load a data-preparation config from YAML."""

    return parse_data_preparation_config(load_yaml_mapping(path))


def load_lora_training_config(path: Path) -> LoraTrainingConfig:
    """Load a LoRA training config from YAML."""

    return parse_lora_training_config(load_yaml_mapping(path))


def load_evaluation_config(path: Path) -> EvaluationConfig:
    """Load an evaluation config from YAML."""

    return parse_evaluation_config(load_yaml_mapping(path))


def load_runtime_config(path: Path) -> RuntimeConfig:
    """Load a runtime/adapter-selection config from YAML."""

    return parse_runtime_config(load_yaml_mapping(path))


def canonical_config_json(config: ConfigModel) -> str:
    """Return a deterministic JSON representation suitable for hashing/comparison."""

    return json.dumps(asdict(config), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
