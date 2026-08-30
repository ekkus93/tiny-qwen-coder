"""CPU-resolvable, language-neutral adapter training plans."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeAlias

import yaml

from tiny_qwen_coder.config import (
    ConfigError,
    LoraTrainingConfig,
    LossMode,
    canonical_config_json,
    load_yaml_mapping,
    parse_lora_training_config,
)
from tiny_qwen_coder.data import NormalizedTrainingRecord
from tiny_qwen_coder.languages import LanguageRegistry, load_language_plugin
from tiny_qwen_coder.model import InspectionTarget, load_inspection_target

TrainingRow: TypeAlias = dict[str, object]

_ADAPTER_COMPONENT_PATTERN = re.compile(r"^[a-z][a-z0-9._-]*$")


@dataclass(frozen=True, slots=True)
class AdapterTrainingConfig(LoraTrainingConfig):
    """Generic orchestration fields layered onto the reusable LoRA schema."""

    adapter_family: str
    adapter_id: str
    train_records: str
    validation_records: str

    def __post_init__(self) -> None:
        LoraTrainingConfig.__post_init__(self)
        if not _ADAPTER_COMPONENT_PATTERN.fullmatch(self.adapter_family):
            raise ConfigError("adapter_family must match ^[a-z][a-z0-9._-]*$")
        parts = self.adapter_id.split("/")
        if len(parts) < 3 or any(not _ADAPTER_COMPONENT_PATTERN.fullmatch(part) for part in parts):
            raise ConfigError(
                "adapter_id must contain lowercase family/language/experiment components"
            )
        if parts[:2] != [self.adapter_family, self.language]:
            raise ConfigError("adapter_id must begin with adapter_family/language")
        if not self.train_records.strip() or not self.validation_records.strip():
            raise ConfigError("train_records and validation_records must not be empty")


def _load_adapter_training_config(path: Path) -> AdapterTrainingConfig:
    payload = load_yaml_mapping(path)
    extra_names = ("adapter_family", "adapter_id", "train_records", "validation_records")
    extras: dict[str, str] = {}
    for name in extra_names:
        value = payload.pop(name, None)
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"training.{name} must be a non-empty string")
        extras[name] = value
    base = parse_lora_training_config(payload)
    return AdapterTrainingConfig(
        schema_version=base.schema_version,
        base_config=base.base_config,
        language=base.language,
        dataset_manifest=base.dataset_manifest,
        output_dir=base.output_dir,
        seed=base.seed,
        training_mode=base.training_mode,
        compute_dtype=base.compute_dtype,
        sequence_length=base.sequence_length,
        micro_batch_size=base.micro_batch_size,
        gradient_accumulation_steps=base.gradient_accumulation_steps,
        epochs=base.epochs,
        learning_rate=base.learning_rate,
        scheduler=base.scheduler,
        warmup_ratio=base.warmup_ratio,
        gradient_checkpointing=base.gradient_checkpointing,
        loss_mode=base.loss_mode,
        lora=base.lora,
        quantization=base.quantization,
        adapter_family=extras["adapter_family"],
        adapter_id=extras["adapter_id"],
        train_records=extras["train_records"],
        validation_records=extras["validation_records"],
    )


class AdapterTrainingError(ValueError):
    """Raised when generic adapter training cannot resolve a safe plan."""


@dataclass(frozen=True, slots=True)
class TrainingDatasetIdentity:
    """Immutable identity required from a frozen training dataset manifest."""

    manifest_id: str
    language: str
    tokenizer_repository: str
    tokenizer_revision: str
    chat_template_sha256: str
    sha256: str

    def __post_init__(self) -> None:
        for field_name, value in (
            ("manifest_id", self.manifest_id),
            ("language", self.language),
            ("tokenizer_repository", self.tokenizer_repository),
            ("tokenizer_revision", self.tokenizer_revision),
        ):
            if not value.strip():
                raise AdapterTrainingError(f"dataset {field_name} must not be empty")
        for field_name, value in (
            ("chat_template_sha256", self.chat_template_sha256),
            ("sha256", self.sha256),
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise AdapterTrainingError(f"dataset {field_name} must be a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class TrainerArtifactPaths:
    """Stable output paths emitted by one adapter training run."""

    output_dir: Path
    checkpoints: Path
    adapter: Path
    dataset_manifest: Path
    training_config: Path
    training_metrics: Path
    run_manifest: Path
    adapter_manifest: Path


@dataclass(frozen=True, slots=True)
class AdapterTrainingPlan:
    """Fully resolved CPU-only inputs for one generic LoRA/QLoRA run."""

    config_path: Path
    config: AdapterTrainingConfig
    config_sha256: str
    language: str
    target: InspectionTarget
    dataset: TrainingDatasetIdentity
    train_records: Path
    validation_records: Path
    artifacts: TrainerArtifactPaths

    def resolved_config_payload(self) -> dict[str, object]:
        """Return the deterministic config/provenance payload written beside training artifacts."""

        return {
            "schema_version": 1,
            "source_config": str(self.config_path),
            "config_sha256": self.config_sha256,
            "config": asdict(self.config),
            "base": asdict(self.target),
            "dataset": asdict(self.dataset),
        }


def _mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise AdapterTrainingError(f"{context} must be a mapping")
    output: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise AdapterTrainingError(f"{context} keys must be strings")
        output[key] = item
    return output


def _required_string(mapping: dict[str, object], key: str, *, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise AdapterTrainingError(f"{context}.{key} must be a non-empty string")
    return value


def load_training_dataset_identity(path: Path) -> TrainingDatasetIdentity:
    """Read the language/tokenizer identity from any compatible frozen dataset manifest."""

    try:
        payload: object = yaml.safe_load(path.read_text(encoding="utf-8"))
        raw_bytes = path.read_bytes()
    except (OSError, yaml.YAMLError) as exc:
        raise AdapterTrainingError(f"could not read dataset manifest {path}: {exc}") from exc

    root = _mapping(payload, context="dataset manifest")
    tokenizer = _mapping(root.get("tokenizer"), context="dataset manifest.tokenizer")
    identity = TrainingDatasetIdentity(
        manifest_id=_required_string(root, "manifest_id", context="dataset manifest"),
        language=_required_string(root, "language", context="dataset manifest"),
        tokenizer_repository=_required_string(
            tokenizer, "repository", context="dataset manifest.tokenizer"
        ),
        tokenizer_revision=_required_string(
            tokenizer, "revision", context="dataset manifest.tokenizer"
        ),
        chat_template_sha256=_required_string(
            tokenizer, "chat_template_sha256", context="dataset manifest.tokenizer"
        ),
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )
    return identity


def load_training_language_registry(
    languages_dir: Path = Path("configs/languages"),
) -> LanguageRegistry:
    """Load every declarative language plugin from the canonical language-config directory."""

    try:
        config_paths = tuple(sorted(languages_dir.glob("*.yaml")))
    except OSError as exc:
        raise AdapterTrainingError(f"could not enumerate language configs in {languages_dir}") from exc
    if not config_paths:
        raise AdapterTrainingError(f"no language configs found in {languages_dir}")
    return LanguageRegistry(load_language_plugin(path) for path in config_paths)


def _artifact_paths(output_dir: Path) -> TrainerArtifactPaths:
    return TrainerArtifactPaths(
        output_dir=output_dir,
        checkpoints=output_dir / "checkpoints",
        adapter=output_dir / "adapter",
        dataset_manifest=output_dir / "dataset-manifest.json",
        training_config=output_dir / "training-config.json",
        training_metrics=output_dir / "training-metrics.jsonl",
        run_manifest=output_dir / "run-manifest.json",
        adapter_manifest=output_dir / "adapter-manifest.json",
    )


def resolve_adapter_training_plan(
    config_path: Path,
    *,
    registry: LanguageRegistry | None = None,
) -> AdapterTrainingPlan:
    """Resolve and validate all CPU-only inputs before expensive model loading."""

    config = _load_adapter_training_config(config_path)
    language_registry = registry or load_training_language_registry()
    plugin = language_registry.resolve(config.language)
    language = plugin.spec.id
    if language != config.language:
        raise AdapterTrainingError(
            f"training config must use canonical language ID {language!r}, got {config.language!r}"
        )

    target = load_inspection_target(Path(config.base_config))
    dataset_path = Path(config.dataset_manifest)
    dataset = load_training_dataset_identity(dataset_path)
    if dataset.language != language:
        raise AdapterTrainingError(
            f"dataset language {dataset.language!r} does not match configured language {language!r}"
        )
    if dataset.tokenizer_repository != target.tokenizer_repository:
        raise AdapterTrainingError(
            "dataset tokenizer repository does not match canonical base tokenizer repository"
        )
    if dataset.tokenizer_revision != target.tokenizer_revision:
        raise AdapterTrainingError(
            "dataset tokenizer revision does not match canonical base tokenizer revision"
        )

    config_sha256 = hashlib.sha256(canonical_config_json(config).encode("utf-8")).hexdigest()
    return AdapterTrainingPlan(
        config_path=config_path,
        config=config,
        config_sha256=config_sha256,
        language=language,
        target=target,
        dataset=dataset,
        train_records=Path(config.train_records),
        validation_records=Path(config.validation_records),
        artifacts=_artifact_paths(Path(config.output_dir)),
    )


def _message_dicts(record: NormalizedTrainingRecord) -> list[dict[str, str]]:
    return [{"role": message.role, "content": message.content} for message in record.messages]


def training_rows(
    records: tuple[NormalizedTrainingRecord, ...],
    *,
    loss_mode: LossMode,
) -> tuple[TrainingRow, ...]:
    """Convert normalized records to TRL conversational rows without language-specific logic."""

    rows: list[TrainingRow] = []
    for index, record in enumerate(records):
        messages = _message_dicts(record)
        if loss_mode == "assistant_only":
            rows.append({"messages": messages})
            continue

        if not messages or messages[-1]["role"] != "assistant":
            raise AdapterTrainingError(
                f"record {index} completion-only loss requires a final assistant message"
            )
        prompt = messages[:-1]
        if not prompt:
            raise AdapterTrainingError(
                f"record {index} completion-only loss requires a non-empty prompt"
            )
        rows.append({"prompt": prompt, "completion": [messages[-1]]})
    return tuple(rows)


def resolved_config_json(plan: AdapterTrainingPlan) -> str:
    """Serialize the resolved training-plan configuration deterministically."""

    return json.dumps(plan.resolved_config_payload(), indent=2, sort_keys=True) + "\n"
