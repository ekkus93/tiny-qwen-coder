"""Strict, language-neutral schema for portable LoRA adapter manifests.

Adapter manifests describe the trained adapter artifact itself. They are distinct
from per-run manifests: the adapter contract keeps the exact compatibility,
training, dataset, target-module, and evaluation provenance needed after the
original training process has finished.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypeAlias, cast

import yaml

from tiny_qwen_coder.reproducibility import SeedError, validate_seed

AdapterBias: TypeAlias = Literal["none", "all", "lora_only"]
AdapterTargetStrategy: TypeAlias = Literal["selective", "all_linear"]
ScalarSettingValue: TypeAlias = str | int | float | bool

_SCHEMA_VERSION = 1
_GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_LANGUAGE_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")
_ADAPTER_ID_COMPONENT_PATTERN = re.compile(r"^[a-z][a-z0-9._-]*$")


class AdapterManifestError(ValueError):
    """Raised when an adapter manifest violates the portable artifact contract."""


def _require_non_empty(value: str, *, field_name: str) -> None:
    if not value.strip():
        raise AdapterManifestError(f"{field_name} must not be empty")


def _require_git_sha(value: str, *, field_name: str) -> None:
    if not _GIT_SHA_PATTERN.fullmatch(value):
        raise AdapterManifestError(f"{field_name} must be a lowercase 40-character Git SHA")


def _require_sha256(value: str, *, field_name: str) -> None:
    if not _SHA256_PATTERN.fullmatch(value):
        raise AdapterManifestError(f"{field_name} must be a lowercase 64-character SHA-256")


def _require_language(value: str) -> None:
    if not _LANGUAGE_PATTERN.fullmatch(value):
        raise AdapterManifestError("language must match ^[a-z][a-z0-9_-]*$")


def _require_positive(value: int | float, *, field_name: str) -> None:
    if value <= 0:
        raise AdapterManifestError(f"{field_name} must be greater than zero")


def _require_non_negative(value: int | float, *, field_name: str) -> None:
    if value < 0:
        raise AdapterManifestError(f"{field_name} must be non-negative")


def _require_utc_timestamp(value: str, *, field_name: str) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AdapterManifestError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise AdapterManifestError(f"{field_name} must be UTC")


def _validate_adapter_id(adapter_id: str, *, family: str, language: str) -> None:
    _require_non_empty(adapter_id, field_name="adapter_id")
    parts = adapter_id.split("/")
    if len(parts) < 3:
        raise AdapterManifestError(
            "adapter_id must contain at least family/language/experiment components"
        )
    if any(not _ADAPTER_ID_COMPONENT_PATTERN.fullmatch(part) for part in parts):
        raise AdapterManifestError("adapter_id components must match ^[a-z][a-z0-9._-]*$")
    if parts[0] != family or parts[1] != language:
        raise AdapterManifestError("adapter_id must begin with family/language")


def _strict_mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise AdapterManifestError(f"{context} must be a mapping")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise AdapterManifestError(f"{context} keys must be strings")
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
        raise AdapterManifestError(f"{context} contains unknown field(s): {', '.join(unknown)}")
    if missing:
        raise AdapterManifestError(f"{context} is missing required field(s): {', '.join(missing)}")


def _expect_str(mapping: Mapping[str, object], key: str, *, context: str) -> str:
    value = mapping[key]
    if not isinstance(value, str):
        raise AdapterManifestError(f"{context}.{key} must be a string")
    _require_non_empty(value, field_name=f"{context}.{key}")
    return value


def _expect_optional_str(mapping: Mapping[str, object], key: str, *, context: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise AdapterManifestError(f"{context}.{key} must be a string or null")
    _require_non_empty(value, field_name=f"{context}.{key}")
    return value


def _expect_int(mapping: Mapping[str, object], key: str, *, context: str) -> int:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise AdapterManifestError(f"{context}.{key} must be an integer")
    return value


def _expect_number(mapping: Mapping[str, object], key: str, *, context: str) -> float:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AdapterManifestError(f"{context}.{key} must be a number")
    return float(value)


def _expect_str_tuple(mapping: Mapping[str, object], key: str, *, context: str) -> tuple[str, ...]:
    value = mapping[key]
    if not isinstance(value, list):
        raise AdapterManifestError(f"{context}.{key} must be a YAML sequence")
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise AdapterManifestError(f"{context}.{key}[{index}] must be a non-empty string")
        result.append(item)
    if len(result) != len(set(result)):
        raise AdapterManifestError(f"{context}.{key} must not contain duplicates")
    return tuple(result)


def _expect_scalar_setting(value: object, *, context: str) -> ScalarSettingValue:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, str) and value.strip():
        return value
    raise AdapterManifestError(f"{context} must be a non-empty scalar value")


@dataclass(frozen=True, slots=True)
class AdapterBaseIdentity:
    """Exact base-model identity to which the adapter is revision-bound."""

    repository: str
    revision: str

    def __post_init__(self) -> None:
        _require_non_empty(self.repository, field_name="base_model.repository")
        _require_git_sha(self.revision, field_name="base_model.revision")


@dataclass(frozen=True, slots=True)
class ChatTemplateIdentity:
    """Chat-template identity used while training the adapter."""

    identifier: str
    sha256: str | None

    def __post_init__(self) -> None:
        _require_non_empty(self.identifier, field_name="tokenizer.chat_template.identifier")
        if self.sha256 is not None:
            _require_sha256(self.sha256, field_name="tokenizer.chat_template.sha256")


@dataclass(frozen=True, slots=True)
class AdapterTokenizerIdentity:
    """Exact tokenizer and chat-template identity used by the adapter."""

    repository: str
    revision: str
    chat_template: ChatTemplateIdentity

    def __post_init__(self) -> None:
        _require_non_empty(self.repository, field_name="tokenizer.repository")
        _require_git_sha(self.revision, field_name="tokenizer.revision")


@dataclass(frozen=True, slots=True)
class DatasetManifestReference:
    """One immutable dataset manifest consumed by adapter training."""

    manifest_id: str
    sha256: str

    def __post_init__(self) -> None:
        _require_non_empty(self.manifest_id, field_name="datasets[].manifest_id")
        _require_sha256(self.sha256, field_name="datasets[].sha256")


@dataclass(frozen=True, slots=True)
class AdapterTrainingProvenance:
    """Source and dependency identity of the training run that produced an adapter."""

    run_id: str
    git_sha: str
    config_sha256: str
    seed: int
    transformers_version: str
    peft_version: str

    def __post_init__(self) -> None:
        _require_non_empty(self.run_id, field_name="training.run_id")
        _require_git_sha(self.git_sha, field_name="training.git_sha")
        _require_sha256(self.config_sha256, field_name="training.config_sha256")
        try:
            validate_seed(self.seed)
        except SeedError as exc:
            raise AdapterManifestError(str(exc)) from exc
        _require_non_empty(
            self.transformers_version,
            field_name="training.transformers_version",
        )
        _require_non_empty(self.peft_version, field_name="training.peft_version")


@dataclass(frozen=True, slots=True)
class AdapterLoraMetadata:
    """Resolved LoRA structure and hyperparameters stored with the adapter artifact."""

    rank: int
    alpha: int
    dropout: float
    bias: AdapterBias
    target_strategy: AdapterTargetStrategy
    target_modules: tuple[str, ...]
    trainable_parameters: int

    def __post_init__(self) -> None:
        _require_positive(self.rank, field_name="lora.rank")
        _require_positive(self.alpha, field_name="lora.alpha")
        if not 0.0 <= self.dropout < 1.0:
            raise AdapterManifestError(
                "lora.dropout must be greater than or equal to 0 and less than 1"
            )
        if not self.target_modules:
            raise AdapterManifestError(
                "lora.target_modules must record the resolved trained module names"
            )
        if len(self.target_modules) != len(set(self.target_modules)):
            raise AdapterManifestError("lora.target_modules must not contain duplicates")
        for index, module_name in enumerate(self.target_modules):
            _require_non_empty(module_name, field_name=f"lora.target_modules[{index}]")
        _require_positive(
            self.trainable_parameters,
            field_name="lora.trainable_parameters",
        )


@dataclass(frozen=True, slots=True)
class TrainingSetting:
    """One scalar optimizer or scheduler setting."""

    name: str
    value: ScalarSettingValue

    def __post_init__(self) -> None:
        _require_non_empty(self.name, field_name="training_summary.*.settings[].name")
        _expect_scalar_setting(
            self.value,
            context="training_summary.*.settings[].value",
        )


@dataclass(frozen=True, slots=True)
class TrainingComponentSettings:
    """Named optimizer/scheduler plus extensible scalar settings."""

    name: str
    settings: tuple[TrainingSetting, ...]

    def __post_init__(self) -> None:
        _require_non_empty(self.name, field_name="training_summary.*.name")
        names = tuple(setting.name for setting in self.settings)
        if len(names) != len(set(names)):
            raise AdapterManifestError("training component setting names must be unique")


@dataclass(frozen=True, slots=True)
class AdapterTrainingSummary:
    """Training settings needed to reproduce and compare completed adapter runs."""

    precision: str
    sequence_length: int
    optimizer: TrainingComponentSettings
    scheduler: TrainingComponentSettings
    steps: int
    epochs: float
    peak_vram_bytes: int

    def __post_init__(self) -> None:
        _require_non_empty(self.precision, field_name="training_summary.precision")
        _require_positive(
            self.sequence_length,
            field_name="training_summary.sequence_length",
        )
        _require_positive(self.steps, field_name="training_summary.steps")
        _require_positive(self.epochs, field_name="training_summary.epochs")
        _require_non_negative(
            self.peak_vram_bytes,
            field_name="training_summary.peak_vram_bytes",
        )


@dataclass(frozen=True, slots=True)
class ValidationMetric:
    """Portable numeric validation metric recorded before or during promotion."""

    name: str
    value: float
    split: str | None
    unit: str | None

    def __post_init__(self) -> None:
        _require_non_empty(self.name, field_name="validation_metrics[].name")
        if self.split is not None:
            _require_non_empty(self.split, field_name="validation_metrics[].split")
        if self.unit is not None:
            _require_non_empty(self.unit, field_name="validation_metrics[].unit")


@dataclass(frozen=True, slots=True)
class AdapterManifest:
    """Complete portable manifest stored alongside one trained LoRA adapter."""

    schema_version: int
    adapter_id: str
    family: str
    language: str
    created_at_utc: str
    base_model: AdapterBaseIdentity
    tokenizer: AdapterTokenizerIdentity
    training: AdapterTrainingProvenance
    datasets: tuple[DatasetManifestReference, ...]
    lora: AdapterLoraMetadata
    training_summary: AdapterTrainingSummary
    validation_metrics: tuple[ValidationMetric, ...]
    evaluation_artifacts: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise AdapterManifestError(
                f"unsupported adapter manifest schema_version {self.schema_version}; "
                f"expected {_SCHEMA_VERSION}"
            )
        if not _ADAPTER_ID_COMPONENT_PATTERN.fullmatch(self.family):
            raise AdapterManifestError("family must match ^[a-z][a-z0-9._-]*$")
        _require_language(self.language)
        _validate_adapter_id(
            self.adapter_id,
            family=self.family,
            language=self.language,
        )
        _require_utc_timestamp(self.created_at_utc, field_name="created_at_utc")
        if not self.datasets:
            raise AdapterManifestError("datasets must contain at least one manifest reference")
        dataset_ids = tuple(item.manifest_id for item in self.datasets)
        if len(dataset_ids) != len(set(dataset_ids)):
            raise AdapterManifestError("dataset manifest IDs must be unique")
        metric_names = tuple(metric.name for metric in self.validation_metrics)
        if len(metric_names) != len(set(metric_names)):
            raise AdapterManifestError("validation metric names must be unique")
        if len(self.evaluation_artifacts) != len(set(self.evaluation_artifacts)):
            raise AdapterManifestError("evaluation_artifacts must not contain duplicates")
        for index, artifact in enumerate(self.evaluation_artifacts):
            _require_non_empty(artifact, field_name=f"evaluation_artifacts[{index}]")


def _parse_base_model(value: object) -> AdapterBaseIdentity:
    context = "adapter_manifest.base_model"
    mapping = _strict_mapping(value, context=context)
    _validate_keys(
        mapping,
        required=frozenset({"repository", "revision"}),
        context=context,
    )
    return AdapterBaseIdentity(
        repository=_expect_str(mapping, "repository", context=context),
        revision=_expect_str(mapping, "revision", context=context),
    )


def _parse_chat_template(value: object) -> ChatTemplateIdentity:
    context = "adapter_manifest.tokenizer.chat_template"
    mapping = _strict_mapping(value, context=context)
    _validate_keys(
        mapping,
        required=frozenset({"identifier"}),
        optional=frozenset({"sha256"}),
        context=context,
    )
    return ChatTemplateIdentity(
        identifier=_expect_str(mapping, "identifier", context=context),
        sha256=_expect_optional_str(mapping, "sha256", context=context),
    )


def _parse_tokenizer(value: object) -> AdapterTokenizerIdentity:
    context = "adapter_manifest.tokenizer"
    mapping = _strict_mapping(value, context=context)
    _validate_keys(
        mapping,
        required=frozenset({"repository", "revision", "chat_template"}),
        context=context,
    )
    return AdapterTokenizerIdentity(
        repository=_expect_str(mapping, "repository", context=context),
        revision=_expect_str(mapping, "revision", context=context),
        chat_template=_parse_chat_template(mapping["chat_template"]),
    )


def _parse_training(value: object) -> AdapterTrainingProvenance:
    context = "adapter_manifest.training"
    mapping = _strict_mapping(value, context=context)
    _validate_keys(
        mapping,
        required=frozenset(
            {
                "run_id",
                "git_sha",
                "config_sha256",
                "seed",
                "transformers_version",
                "peft_version",
            }
        ),
        context=context,
    )
    return AdapterTrainingProvenance(
        run_id=_expect_str(mapping, "run_id", context=context),
        git_sha=_expect_str(mapping, "git_sha", context=context),
        config_sha256=_expect_str(mapping, "config_sha256", context=context),
        seed=_expect_int(mapping, "seed", context=context),
        transformers_version=_expect_str(
            mapping,
            "transformers_version",
            context=context,
        ),
        peft_version=_expect_str(mapping, "peft_version", context=context),
    )


def _parse_datasets(value: object) -> tuple[DatasetManifestReference, ...]:
    context = "adapter_manifest.datasets"
    if not isinstance(value, list):
        raise AdapterManifestError(f"{context} must be a YAML sequence")
    result: list[DatasetManifestReference] = []
    for index, item in enumerate(value):
        item_context = f"{context}[{index}]"
        mapping = _strict_mapping(item, context=item_context)
        _validate_keys(
            mapping,
            required=frozenset({"manifest_id", "sha256"}),
            context=item_context,
        )
        result.append(
            DatasetManifestReference(
                manifest_id=_expect_str(mapping, "manifest_id", context=item_context),
                sha256=_expect_str(mapping, "sha256", context=item_context),
            )
        )
    return tuple(result)


def _parse_lora(value: object) -> AdapterLoraMetadata:
    context = "adapter_manifest.lora"
    mapping = _strict_mapping(value, context=context)
    _validate_keys(
        mapping,
        required=frozenset(
            {
                "rank",
                "alpha",
                "dropout",
                "bias",
                "target_strategy",
                "target_modules",
                "trainable_parameters",
            }
        ),
        context=context,
    )
    bias = _expect_str(mapping, "bias", context=context)
    if bias not in {"none", "all", "lora_only"}:
        raise AdapterManifestError("adapter_manifest.lora.bias has an unsupported value")
    target_strategy = _expect_str(mapping, "target_strategy", context=context)
    if target_strategy not in {"selective", "all_linear"}:
        raise AdapterManifestError("adapter_manifest.lora.target_strategy has an unsupported value")
    return AdapterLoraMetadata(
        rank=_expect_int(mapping, "rank", context=context),
        alpha=_expect_int(mapping, "alpha", context=context),
        dropout=_expect_number(mapping, "dropout", context=context),
        bias=cast(AdapterBias, bias),
        target_strategy=cast(AdapterTargetStrategy, target_strategy),
        target_modules=_expect_str_tuple(mapping, "target_modules", context=context),
        trainable_parameters=_expect_int(
            mapping,
            "trainable_parameters",
            context=context,
        ),
    )


def _parse_settings(value: object, *, context: str) -> tuple[TrainingSetting, ...]:
    if not isinstance(value, list):
        raise AdapterManifestError(f"{context} must be a YAML sequence")
    result: list[TrainingSetting] = []
    for index, item in enumerate(value):
        item_context = f"{context}[{index}]"
        mapping = _strict_mapping(item, context=item_context)
        _validate_keys(
            mapping,
            required=frozenset({"name", "value"}),
            context=item_context,
        )
        result.append(
            TrainingSetting(
                name=_expect_str(mapping, "name", context=item_context),
                value=_expect_scalar_setting(
                    mapping["value"],
                    context=f"{item_context}.value",
                ),
            )
        )
    return tuple(result)


def _parse_component_settings(value: object, *, context: str) -> TrainingComponentSettings:
    mapping = _strict_mapping(value, context=context)
    _validate_keys(
        mapping,
        required=frozenset({"name", "settings"}),
        context=context,
    )
    return TrainingComponentSettings(
        name=_expect_str(mapping, "name", context=context),
        settings=_parse_settings(mapping["settings"], context=f"{context}.settings"),
    )


def _parse_training_summary(value: object) -> AdapterTrainingSummary:
    context = "adapter_manifest.training_summary"
    mapping = _strict_mapping(value, context=context)
    _validate_keys(
        mapping,
        required=frozenset(
            {
                "precision",
                "sequence_length",
                "optimizer",
                "scheduler",
                "steps",
                "epochs",
                "peak_vram_bytes",
            }
        ),
        context=context,
    )
    return AdapterTrainingSummary(
        precision=_expect_str(mapping, "precision", context=context),
        sequence_length=_expect_int(mapping, "sequence_length", context=context),
        optimizer=_parse_component_settings(
            mapping["optimizer"],
            context=f"{context}.optimizer",
        ),
        scheduler=_parse_component_settings(
            mapping["scheduler"],
            context=f"{context}.scheduler",
        ),
        steps=_expect_int(mapping, "steps", context=context),
        epochs=_expect_number(mapping, "epochs", context=context),
        peak_vram_bytes=_expect_int(mapping, "peak_vram_bytes", context=context),
    )


def _parse_validation_metrics(value: object) -> tuple[ValidationMetric, ...]:
    context = "adapter_manifest.validation_metrics"
    if not isinstance(value, list):
        raise AdapterManifestError(f"{context} must be a YAML sequence")
    result: list[ValidationMetric] = []
    for index, item in enumerate(value):
        item_context = f"{context}[{index}]"
        mapping = _strict_mapping(item, context=item_context)
        _validate_keys(
            mapping,
            required=frozenset({"name", "value"}),
            optional=frozenset({"split", "unit"}),
            context=item_context,
        )
        result.append(
            ValidationMetric(
                name=_expect_str(mapping, "name", context=item_context),
                value=_expect_number(mapping, "value", context=item_context),
                split=_expect_optional_str(mapping, "split", context=item_context),
                unit=_expect_optional_str(mapping, "unit", context=item_context),
            )
        )
    return tuple(result)


def parse_adapter_manifest(value: object) -> AdapterManifest:
    """Parse one complete adapter manifest with strict unknown-field handling."""

    context = "adapter_manifest"
    mapping = _strict_mapping(value, context=context)
    _validate_keys(
        mapping,
        required=frozenset(
            {
                "schema_version",
                "adapter_id",
                "family",
                "language",
                "created_at_utc",
                "base_model",
                "tokenizer",
                "training",
                "datasets",
                "lora",
                "training_summary",
                "validation_metrics",
                "evaluation_artifacts",
            }
        ),
        context=context,
    )
    schema_version = _expect_int(mapping, "schema_version", context=context)
    return AdapterManifest(
        schema_version=schema_version,
        adapter_id=_expect_str(mapping, "adapter_id", context=context),
        family=_expect_str(mapping, "family", context=context),
        language=_expect_str(mapping, "language", context=context),
        created_at_utc=_expect_str(mapping, "created_at_utc", context=context),
        base_model=_parse_base_model(mapping["base_model"]),
        tokenizer=_parse_tokenizer(mapping["tokenizer"]),
        training=_parse_training(mapping["training"]),
        datasets=_parse_datasets(mapping["datasets"]),
        lora=_parse_lora(mapping["lora"]),
        training_summary=_parse_training_summary(mapping["training_summary"]),
        validation_metrics=_parse_validation_metrics(mapping["validation_metrics"]),
        evaluation_artifacts=_expect_str_tuple(
            mapping,
            "evaluation_artifacts",
            context=context,
        ),
    )


def load_adapter_manifest(path: Path) -> AdapterManifest:
    """Load one strict adapter manifest from YAML."""

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise AdapterManifestError(f"could not read adapter manifest {path}") from exc
    return parse_adapter_manifest(raw)


def adapter_manifest_json(manifest: AdapterManifest) -> str:
    """Serialize an adapter manifest deterministically for hashing/reporting."""

    return json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n"
