"""Bounded, non-promotable GPU smoke training for a resolved adapter plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NoReturn

import torch
import yaml

from tiny_qwen_coder.identities import AdapterIdentity, BaseModelIdentity
from tiny_qwen_coder.reporting import create_run_manifest, write_run_manifest
from tiny_qwen_coder.reproducibility import seed_everything
from tiny_qwen_coder.training.plan import (
    AdapterTrainingError,
    AdapterTrainingPlan,
    resolve_adapter_training_plan,
)
from tiny_qwen_coder.training.preflight import run_training_preflight, training_preflight_json
from tiny_qwen_coder.training.runtime import (
    AdapterTrainingRuntimeOptions,
    _load_training_runtime,
    _write_metrics,
    _validation_loss,
)

_SCHEMA_VERSION = 1
_MAX_SMOKE_STEPS = 4
_MAX_TRAIN_SAMPLES = 64
_MAX_VALIDATION_SAMPLES = 16


@dataclass(frozen=True, slots=True)
class TrainingSmokeConfig:
    """Strict bounds layered on top of one canonical training configuration."""

    schema_version: int
    training_config: str
    output_dir: str
    max_steps: int
    train_samples: int
    validation_samples: int

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise AdapterTrainingError(
                f"unsupported smoke schema_version {self.schema_version}; expected {_SCHEMA_VERSION}"
            )
        for field_name, value in (
            ("training_config", self.training_config),
            ("output_dir", self.output_dir),
        ):
            if not value.strip() or value != value.strip():
                raise AdapterTrainingError(f"smoke {field_name} must be a non-empty exact path")
        if not 1 <= self.max_steps <= _MAX_SMOKE_STEPS:
            raise AdapterTrainingError(
                f"smoke max_steps must be between 1 and {_MAX_SMOKE_STEPS}"
            )
        if not 1 <= self.train_samples <= _MAX_TRAIN_SAMPLES:
            raise AdapterTrainingError(
                f"smoke train_samples must be between 1 and {_MAX_TRAIN_SAMPLES}"
            )
        if not 1 <= self.validation_samples <= _MAX_VALIDATION_SAMPLES:
            raise AdapterTrainingError(
                "smoke validation_samples must be between 1 and "
                f"{_MAX_VALIDATION_SAMPLES}"
            )


@dataclass(frozen=True, slots=True)
class SmokeArtifactDigest:
    """One persisted smoke-training file included in the evidence fingerprint."""

    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class TrainingSmokeReport:
    """Machine-readable acceptance evidence for P7-005."""

    schema_version: int
    smoke_only: bool
    promotable: bool
    run_id: str
    source_training_config: str
    source_training_config_sha256: str
    smoke_config_sha256: str
    language: str
    adapter_id: str
    base_model: BaseModelIdentity
    sequence_length: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    max_steps: int
    train_samples: int
    validation_samples: int
    global_steps: int
    training_loss: float
    validation_loss: float
    logged_training_losses: tuple[float, ...]
    total_runtime_seconds: float
    peak_allocated_vram_bytes: int
    peak_reserved_vram_bytes: int
    checkpoint_dir: str
    adapter_dir: str
    persisted_artifacts: tuple[SmokeArtifactDigest, ...]
    artifact_set_sha256: str


def _strict_mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise AdapterTrainingError(f"{context} must be a mapping")
    mapping: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise AdapterTrainingError(f"{context} keys must be strings")
        mapping[key] = item
    return mapping


def _required_int(mapping: dict[str, object], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise AdapterTrainingError(f"smoke.{key} must be an integer")
    return value


def _required_str(mapping: dict[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str):
        raise AdapterTrainingError(f"smoke.{key} must be a string")
    return value


def load_training_smoke_config(path: Path) -> TrainingSmokeConfig:
    """Load a bounded smoke configuration with strict unknown-field handling."""

    try:
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise AdapterTrainingError(f"could not load smoke config {path}: {exc}") from exc
    mapping = _strict_mapping(raw, context="smoke config")
    expected = {
        "schema_version",
        "training_config",
        "output_dir",
        "max_steps",
        "train_samples",
        "validation_samples",
    }
    unknown = sorted(set(mapping) - expected)
    missing = sorted(expected - set(mapping))
    if unknown:
        raise AdapterTrainingError(f"smoke config contains unknown field(s): {', '.join(unknown)}")
    if missing:
        raise AdapterTrainingError(f"smoke config is missing field(s): {', '.join(missing)}")
    return TrainingSmokeConfig(
        schema_version=_required_int(mapping, "schema_version"),
        training_config=_required_str(mapping, "training_config"),
        output_dir=_required_str(mapping, "output_dir"),
        max_steps=_required_int(mapping, "max_steps"),
        train_samples=_required_int(mapping, "train_samples"),
        validation_samples=_required_int(mapping, "validation_samples"),
    )


def training_smoke_config_sha256(config: TrainingSmokeConfig) -> str:
    """Fingerprint the exact parsed smoke bounds deterministically."""

    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_smoke_bounds(config: TrainingSmokeConfig, plan: AdapterTrainingPlan) -> None:
    examples_per_step = plan.config.micro_batch_size * plan.config.gradient_accumulation_steps
    required_train_samples = config.max_steps * examples_per_step
    if config.train_samples < required_train_samples:
        raise AdapterTrainingError(
            "smoke train_samples must cover every bounded optimizer step without recycling; "
            f"need at least {required_train_samples}, got {config.train_samples}"
        )
    canonical_output = Path(plan.config.output_dir).resolve(strict=False)
    smoke_output = Path(config.output_dir).resolve(strict=False)
    if smoke_output == canonical_output:
        raise AdapterTrainingError("smoke output must not reuse the canonical training output")


def _require_finite_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AdapterTrainingError(f"smoke {field_name} must be numeric")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise AdapterTrainingError(f"smoke {field_name} must be finite")
    return resolved


def _training_losses(log_history: object) -> tuple[float, ...]:
    if not isinstance(log_history, list):
        raise AdapterTrainingError("trainer log history must be a list")
    values: list[float] = []
    for item in log_history:
        if not isinstance(item, dict) or "loss" not in item:
            continue
        values.append(_require_finite_float(item["loss"], field_name="logged training loss"))
    if not values:
        raise AdapterTrainingError("smoke training did not emit a logged training loss")
    return tuple(values)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_digests(root: Path, directories: tuple[Path, ...]) -> tuple[SmokeArtifactDigest, ...]:
    files: list[Path] = []
    for directory in directories:
        files.extend(path for path in directory.rglob("*") if path.is_file())
    ordered = sorted(set(files), key=lambda path: path.relative_to(root).as_posix())
    if not ordered:
        raise AdapterTrainingError("smoke training did not persist checkpoint/adapter files")
    return tuple(
        SmokeArtifactDigest(
            path=path.relative_to(root).as_posix(),
            size_bytes=path.stat().st_size,
            sha256=_sha256_file(path),
        )
        for path in ordered
    )


def _artifact_set_sha256(artifacts: tuple[SmokeArtifactDigest, ...]) -> str:
    payload = json.dumps(
        [asdict(item) for item in artifacts],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def training_smoke_report_json(report: TrainingSmokeReport) -> str:
    """Serialize P7-005 evidence in stable field order."""

    return json.dumps(asdict(report), indent=2, sort_keys=True) + "\n"


def _prepare_smoke_output(
    *,
    output_dir: Path,
    smoke_config: TrainingSmokeConfig,
    smoke_config_sha256: str,
    preflight_json: str,
    dataset_manifest: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "checkpoints").mkdir(exist_ok=False)
    (output_dir / "smoke-config.json").write_text(
        json.dumps(
            {
                "config_sha256": smoke_config_sha256,
                "config": asdict(smoke_config),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "training-preflight.json").write_text(preflight_json, encoding="utf-8")
    shutil.copyfile(dataset_manifest, output_dir / "dataset-manifest.json")


def _verify_saved_training_artifacts(
    output_dir: Path,
    *,
    global_steps: int,
) -> tuple[Path, Path]:
    checkpoints = output_dir / "checkpoints"
    final_checkpoint = checkpoints / f"checkpoint-{global_steps}"
    if not final_checkpoint.is_dir() or not any(final_checkpoint.iterdir()):
        raise AdapterTrainingError(
            f"smoke trainer did not save final checkpoint {final_checkpoint}"
        )

    adapter_dir = output_dir / "adapter"
    if not (adapter_dir / "adapter_config.json").is_file():
        raise AdapterTrainingError("smoke trainer did not save adapter_config.json")
    adapter_weights = (
        adapter_dir / "adapter_model.safetensors",
        adapter_dir / "adapter_model.bin",
    )
    if not any(path.is_file() and path.stat().st_size > 0 for path in adapter_weights):
        raise AdapterTrainingError("smoke trainer did not save adapter weights")
    return final_checkpoint, adapter_dir


def run_training_smoke(
    smoke_config_path: Path,
    *,
    repo_root: Path = Path("."),
) -> TrainingSmokeReport:
    """Run one bounded real-GPU training proof without producing a promotable adapter."""

    smoke = load_training_smoke_config(smoke_config_path)
    plan = resolve_adapter_training_plan(Path(smoke.training_config))
    _validate_smoke_bounds(smoke, plan)
    output_dir = Path(smoke.output_dir)
    preflight = run_training_preflight(plan, repo_root=repo_root, output_dir=output_dir)
    smoke_sha256 = training_smoke_config_sha256(smoke)

    seed_everything(plan.config.seed)
    _prepare_smoke_output(
        output_dir=output_dir,
        smoke_config=smoke,
        smoke_config_sha256=smoke_sha256,
        preflight_json=training_preflight_json(preflight),
        dataset_manifest=Path(plan.config.dataset_manifest),
    )
    run_manifest = create_run_manifest(
        run_kind="training",
        base_model=BaseModelIdentity(
            repository=plan.target.model_repository,
            revision=plan.target.model_revision,
            tokenizer_repository=plan.target.tokenizer_repository,
            tokenizer_revision=plan.target.tokenizer_revision,
        ),
        language=plan.language,
        adapter=AdapterIdentity(
            family=plan.config.adapter_family,
            adapter_id=plan.config.adapter_id,
        ),
        seed=plan.config.seed,
        repo_root=repo_root,
    )
    write_run_manifest(run_manifest, output_dir)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    started = time.monotonic()
    trainer, _ = _load_training_runtime(
        plan,
        options=AdapterTrainingRuntimeOptions(
            output_dir=output_dir,
            max_steps=smoke.max_steps,
            train_sample_limit=smoke.train_samples,
            validation_sample_limit=smoke.validation_samples,
        ),
    )
    train_output: Any = trainer.train()
    evaluation_metrics: Any = trainer.evaluate()
    trainer.save_model(str(output_dir / "adapter"))
    torch.cuda.synchronize(0)
    total_runtime_seconds = time.monotonic() - started

    global_steps_value = getattr(trainer.state, "global_step", None)
    if isinstance(global_steps_value, bool) or not isinstance(global_steps_value, int):
        raise AdapterTrainingError("smoke trainer did not report an integer global step count")
    if global_steps_value != smoke.max_steps:
        raise AdapterTrainingError(
            f"smoke trainer completed {global_steps_value} steps; expected {smoke.max_steps}"
        )

    training_loss = _require_finite_float(
        getattr(train_output, "training_loss", None),
        field_name="training loss",
    )
    validation_loss = _validation_loss(evaluation_metrics)
    if validation_loss is None:
        raise AdapterTrainingError("smoke evaluation did not report eval_loss")
    validation_loss = _require_finite_float(validation_loss, field_name="validation loss")
    log_history = getattr(trainer.state, "log_history", None)
    logged_losses = _training_losses(log_history)
    _write_metrics(log_history, output_dir / "training-metrics.jsonl")

    peak_allocated = int(torch.cuda.max_memory_allocated(0))
    peak_reserved = int(torch.cuda.max_memory_reserved(0))
    if peak_allocated <= 0 or peak_reserved <= 0:
        raise AdapterTrainingError("smoke trainer did not record positive CUDA peak memory")
    if peak_allocated > peak_reserved:
        raise AdapterTrainingError("CUDA peak allocated memory exceeds peak reserved memory")

    final_checkpoint, adapter_dir = _verify_saved_training_artifacts(
        output_dir,
        global_steps=global_steps_value,
    )
    persisted = _artifact_digests(output_dir, (final_checkpoint, adapter_dir))
    report = TrainingSmokeReport(
        schema_version=_SCHEMA_VERSION,
        smoke_only=True,
        promotable=False,
        run_id=run_manifest.run_id,
        source_training_config=smoke.training_config,
        source_training_config_sha256=plan.config_sha256,
        smoke_config_sha256=smoke_sha256,
        language=plan.language,
        adapter_id=plan.config.adapter_id,
        base_model=BaseModelIdentity(
            repository=plan.target.model_repository,
            revision=plan.target.model_revision,
            tokenizer_repository=plan.target.tokenizer_repository,
            tokenizer_revision=plan.target.tokenizer_revision,
        ),
        sequence_length=plan.config.sequence_length,
        micro_batch_size=plan.config.micro_batch_size,
        gradient_accumulation_steps=plan.config.gradient_accumulation_steps,
        max_steps=smoke.max_steps,
        train_samples=smoke.train_samples,
        validation_samples=smoke.validation_samples,
        global_steps=global_steps_value,
        training_loss=training_loss,
        validation_loss=validation_loss,
        logged_training_losses=logged_losses,
        total_runtime_seconds=total_runtime_seconds,
        peak_allocated_vram_bytes=peak_allocated,
        peak_reserved_vram_bytes=peak_reserved,
        checkpoint_dir=final_checkpoint.relative_to(output_dir).as_posix(),
        adapter_dir=adapter_dir.relative_to(output_dir).as_posix(),
        persisted_artifacts=persisted,
        artifact_set_sha256=_artifact_set_sha256(persisted),
    )
    (output_dir / "smoke-training-report.json").write_text(
        training_smoke_report_json(report),
        encoding="utf-8",
    )
    return report


def training_smoke_main(argv: list[str] | None = None) -> NoReturn:
    """CLI entry point for P7-005 bounded GPU smoke training."""

    parser = argparse.ArgumentParser(description="Run bounded Tiny Qwen Coder GPU smoke training")
    parser.add_argument("--config", type=Path, required=True, help="Smoke-training YAML config")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Git repository root used for provenance/output safety",
    )
    args = parser.parse_args(argv)
    report = run_training_smoke(args.config, repo_root=args.repo_root)
    print(training_smoke_report_json(report), end="")
    raise SystemExit(0)
