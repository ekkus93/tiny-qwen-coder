"""P7-006 full-training orchestration and completion evidence.

This module intentionally wraps the already-proven generic adapter trainer rather
than duplicating or mutating its training mechanics. It adds fail-closed evidence
collection required by the full Python P0 acceptance contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NoReturn

import torch

from tiny_qwen_coder.identities import BaseModelIdentity
from tiny_qwen_coder.training.plan import (
    AdapterTrainingError,
    AdapterTrainingPlan,
    resolve_adapter_training_plan,
)
from tiny_qwen_coder.training.runtime import run_adapter_training


@dataclass(frozen=True, slots=True)
class TrainingArtifactDigest:
    """One persisted P7-006 artifact included in the evidence fingerprint."""

    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class FullTrainingReport:
    """Machine-readable completion evidence for one full adapter-training run."""

    schema_version: int
    run_id: str
    source_training_config: str
    source_training_config_sha256: str
    language: str
    adapter_id: str
    base_model: BaseModelIdentity
    dataset_manifest_id: str
    dataset_manifest_sha256: str
    train_records: int
    validation_records: int
    sequence_length: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    effective_batch_size: int
    epochs: float
    global_steps: int
    training_loss: float
    validation_loss: float
    logged_training_losses: tuple[float, ...]
    train_runtime_seconds: float
    total_runtime_seconds: float
    train_samples_per_second: float
    train_steps_per_second: float
    peak_allocated_vram_bytes: int
    peak_reserved_vram_bytes: int
    final_checkpoint_dir: str
    adapter_dir: str
    run_manifest_path: str
    adapter_manifest_path: str
    training_metrics_path: str
    persisted_artifacts: tuple[TrainingArtifactDigest, ...]
    artifact_set_sha256: str


def _require_finite_float(
    value: object,
    *,
    field_name: str,
    positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AdapterTrainingError(f"{field_name} must be numeric")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise AdapterTrainingError(f"{field_name} must be finite")
    if positive and resolved <= 0:
        raise AdapterTrainingError(f"{field_name} must be greater than zero")
    return resolved


def _load_json_object(path: Path, *, context: str) -> dict[str, object]:
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise AdapterTrainingError(f"could not read {context} {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise AdapterTrainingError(f"{context} is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise AdapterTrainingError(f"{context} must be a JSON object: {path}")
    return payload


def _load_metric_history(path: Path) -> tuple[dict[str, object], ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise AdapterTrainingError(f"could not read training metrics {path}: {exc}") from exc
    if not lines:
        raise AdapterTrainingError("training metrics are empty")

    history: list[dict[str, object]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            payload: object = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AdapterTrainingError(
                f"training metrics line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(payload, dict):
            raise AdapterTrainingError(f"training metrics line {line_number} must be a JSON object")
        history.append(payload)
    return tuple(history)


def _training_metrics(
    history: tuple[dict[str, object], ...],
) -> tuple[float, float, tuple[float, ...], float, float, float]:
    logged_losses: list[float] = []
    train_summary: dict[str, object] | None = None
    validation_loss: float | None = None

    for item in history:
        if "loss" in item:
            logged_losses.append(
                _require_finite_float(item["loss"], field_name="logged training loss")
            )
        if "train_loss" in item:
            _require_finite_float(item["train_loss"], field_name="logged train_loss")
            train_summary = item
        if "eval_loss" in item:
            validation_loss = _require_finite_float(
                item["eval_loss"], field_name="logged validation loss"
            )

    if not logged_losses:
        raise AdapterTrainingError("training did not emit a logged training loss")
    if train_summary is None:
        raise AdapterTrainingError("training did not emit a final train summary")
    if validation_loss is None:
        raise AdapterTrainingError("training did not emit a validation loss")

    training_loss = _require_finite_float(
        train_summary.get("train_loss"), field_name="training loss"
    )
    train_runtime = _require_finite_float(
        train_summary.get("train_runtime"),
        field_name="train runtime",
        positive=True,
    )
    train_samples_per_second = _require_finite_float(
        train_summary.get("train_samples_per_second"),
        field_name="train samples per second",
        positive=True,
    )
    train_steps_per_second = _require_finite_float(
        train_summary.get("train_steps_per_second"),
        field_name="train steps per second",
        positive=True,
    )
    return (
        training_loss,
        validation_loss,
        tuple(logged_losses),
        train_runtime,
        train_samples_per_second,
        train_steps_per_second,
    )


def _positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AdapterTrainingError(f"{field_name} must be a positive integer")
    return value


def _preflight_counts(path: Path) -> tuple[int, int]:
    payload = _load_json_object(path, context="training preflight")
    dataset = payload.get("dataset")
    if not isinstance(dataset, dict):
        raise AdapterTrainingError("training preflight dataset must be an object")
    return (
        _positive_int(dataset.get("train_records"), field_name="preflight train_records"),
        _positive_int(dataset.get("validation_records"), field_name="preflight validation_records"),
    )


def _verify_full_training_outputs(
    plan: AdapterTrainingPlan,
    *,
    global_steps: int,
) -> Path:
    final_checkpoint = plan.artifacts.checkpoints / f"checkpoint-{global_steps}"
    if not final_checkpoint.is_dir() or not any(final_checkpoint.iterdir()):
        raise AdapterTrainingError(f"trainer did not save final checkpoint {final_checkpoint}")

    adapter_config_path = plan.artifacts.adapter / "adapter_config.json"
    adapter_config = _load_json_object(adapter_config_path, context="adapter config")
    if str(adapter_config.get("peft_type", "")).upper() != "LORA":
        raise AdapterTrainingError("saved adapter is not declared as a PEFT LoRA adapter")

    adapter_weights = (
        plan.artifacts.adapter / "adapter_model.safetensors",
        plan.artifacts.adapter / "adapter_model.bin",
    )
    if not any(path.is_file() and path.stat().st_size > 0 for path in adapter_weights):
        raise AdapterTrainingError("trainer did not save non-empty LoRA adapter weights")

    forbidden: list[Path] = [
        plan.artifacts.adapter / "model.safetensors",
        plan.artifacts.adapter / "model.safetensors.index.json",
        plan.artifacts.adapter / "pytorch_model.bin",
        plan.artifacts.adapter / "pytorch_model.bin.index.json",
    ]
    forbidden.extend(plan.artifacts.adapter.glob("model-*.safetensors"))
    forbidden.extend(plan.artifacts.adapter.glob("pytorch_model-*.bin"))
    forbidden.extend(final_checkpoint.rglob("model.safetensors"))
    forbidden.extend(final_checkpoint.rglob("model.safetensors.index.json"))
    forbidden.extend(final_checkpoint.rglob("pytorch_model.bin"))
    forbidden.extend(final_checkpoint.rglob("pytorch_model.bin.index.json"))
    forbidden.extend(final_checkpoint.rglob("model-*.safetensors"))
    forbidden.extend(final_checkpoint.rglob("pytorch_model-*.bin"))
    existing = sorted({path for path in forbidden if path.is_file()})
    if existing:
        rendered = ", ".join(path.name for path in existing)
        raise AdapterTrainingError(f"adapter output contains merged/full-model weights: {rendered}")
    return final_checkpoint


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_digests(
    root: Path,
    paths: tuple[Path, ...],
) -> tuple[TrainingArtifactDigest, ...]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(candidate for candidate in path.rglob("*") if candidate.is_file())
        elif path.is_file():
            files.append(path)
        else:
            raise AdapterTrainingError(f"expected training artifact is missing: {path}")

    ordered = sorted(set(files), key=lambda path: path.relative_to(root).as_posix())
    if not ordered:
        raise AdapterTrainingError("full training did not persist any evidence artifacts")
    return tuple(
        TrainingArtifactDigest(
            path=path.relative_to(root).as_posix(),
            size_bytes=path.stat().st_size,
            sha256=_sha256_file(path),
        )
        for path in ordered
    )


def _artifact_set_sha256(artifacts: tuple[TrainingArtifactDigest, ...]) -> str:
    payload = json.dumps(
        [asdict(item) for item in artifacts],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def full_training_report_json(report: FullTrainingReport) -> str:
    """Serialize P7-006 evidence deterministically."""

    return json.dumps(asdict(report), indent=2, sort_keys=True) + "\n"


def run_full_training(
    config_path: Path,
    *,
    repo_root: Path = Path("."),
) -> FullTrainingReport:
    """Run the canonical generic trainer and freeze P7-006 completion evidence."""

    plan = resolve_adapter_training_plan(config_path)
    started = time.monotonic()
    result = run_adapter_training(config_path, repo_root=repo_root)
    torch.cuda.synchronize(0)
    total_runtime_seconds = time.monotonic() - started
    if not math.isfinite(total_runtime_seconds) or total_runtime_seconds <= 0:
        raise AdapterTrainingError("total training runtime must be finite and greater than zero")

    metrics_history = _load_metric_history(plan.artifacts.training_metrics)
    (
        training_loss,
        validation_loss,
        logged_training_losses,
        train_runtime_seconds,
        train_samples_per_second,
        train_steps_per_second,
    ) = _training_metrics(metrics_history)

    if result.validation_loss is None:
        raise AdapterTrainingError("generic trainer did not report validation loss")
    generic_validation_loss = _require_finite_float(
        result.validation_loss, field_name="generic trainer validation loss"
    )
    if not math.isclose(generic_validation_loss, validation_loss, rel_tol=0.0, abs_tol=1e-12):
        raise AdapterTrainingError(
            "generic trainer validation loss disagrees with persisted metrics"
        )

    peak_allocated = int(torch.cuda.max_memory_allocated(0))
    peak_reserved = int(torch.cuda.max_memory_reserved(0))
    if peak_allocated <= 0 or peak_reserved <= 0:
        raise AdapterTrainingError("trainer did not record positive CUDA peak memory")
    if peak_allocated > peak_reserved:
        raise AdapterTrainingError("CUDA peak allocated memory exceeds peak reserved memory")
    if result.peak_vram_bytes != peak_reserved:
        raise AdapterTrainingError(
            "generic trainer peak VRAM disagrees with CUDA peak reserved memory"
        )

    final_checkpoint = _verify_full_training_outputs(plan, global_steps=result.global_steps)
    preflight_path = plan.artifacts.output_dir / "training-preflight.json"
    train_records, validation_records = _preflight_counts(preflight_path)

    adapter_manifest = _load_json_object(
        plan.artifacts.adapter_manifest, context="adapter manifest"
    )
    if adapter_manifest.get("adapter_id") != plan.config.adapter_id:
        raise AdapterTrainingError("adapter manifest identity does not match training plan")
    summary = adapter_manifest.get("training_summary")
    if not isinstance(summary, dict):
        raise AdapterTrainingError("adapter manifest training_summary must be an object")
    if summary.get("steps") != result.global_steps:
        raise AdapterTrainingError("adapter manifest step count does not match trainer result")
    if summary.get("peak_vram_bytes") != peak_reserved:
        raise AdapterTrainingError("adapter manifest peak VRAM does not match measured peak")

    persisted = _artifact_digests(
        plan.artifacts.output_dir,
        (
            plan.artifacts.adapter,
            plan.artifacts.dataset_manifest,
            plan.artifacts.training_config,
            plan.artifacts.training_metrics,
            preflight_path,
            plan.artifacts.run_manifest,
            plan.artifacts.adapter_manifest,
        ),
    )
    report = FullTrainingReport(
        schema_version=1,
        run_id=result.run_id,
        source_training_config=str(plan.config_path),
        source_training_config_sha256=plan.config_sha256,
        language=plan.language,
        adapter_id=plan.config.adapter_id,
        base_model=BaseModelIdentity(
            repository=plan.target.model_repository,
            revision=plan.target.model_revision,
            tokenizer_repository=plan.target.tokenizer_repository,
            tokenizer_revision=plan.target.tokenizer_revision,
        ),
        dataset_manifest_id=plan.dataset.manifest_id,
        dataset_manifest_sha256=plan.dataset.sha256,
        train_records=train_records,
        validation_records=validation_records,
        sequence_length=plan.config.sequence_length,
        micro_batch_size=plan.config.micro_batch_size,
        gradient_accumulation_steps=plan.config.gradient_accumulation_steps,
        effective_batch_size=(
            plan.config.micro_batch_size * plan.config.gradient_accumulation_steps
        ),
        epochs=plan.config.epochs,
        global_steps=result.global_steps,
        training_loss=training_loss,
        validation_loss=validation_loss,
        logged_training_losses=logged_training_losses,
        train_runtime_seconds=train_runtime_seconds,
        total_runtime_seconds=total_runtime_seconds,
        train_samples_per_second=train_samples_per_second,
        train_steps_per_second=train_steps_per_second,
        peak_allocated_vram_bytes=peak_allocated,
        peak_reserved_vram_bytes=peak_reserved,
        final_checkpoint_dir=final_checkpoint.relative_to(plan.artifacts.output_dir).as_posix(),
        adapter_dir=plan.artifacts.adapter.relative_to(plan.artifacts.output_dir).as_posix(),
        run_manifest_path=plan.artifacts.run_manifest.relative_to(
            plan.artifacts.output_dir
        ).as_posix(),
        adapter_manifest_path=plan.artifacts.adapter_manifest.relative_to(
            plan.artifacts.output_dir
        ).as_posix(),
        training_metrics_path=plan.artifacts.training_metrics.relative_to(
            plan.artifacts.output_dir
        ).as_posix(),
        persisted_artifacts=persisted,
        artifact_set_sha256=_artifact_set_sha256(persisted),
    )
    report_path = plan.artifacts.output_dir / "training-report.json"
    report_path.write_text(full_training_report_json(report), encoding="utf-8")
    return report


def full_training_main(argv: list[str] | None = None) -> NoReturn:
    """CLI entry point for the P7-006 full-training run."""

    parser = argparse.ArgumentParser(description="Run full Tiny Qwen Coder adapter training")
    parser.add_argument("--config", type=Path, required=True, help="LoRA training YAML config")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Git repository root used for run-manifest provenance",
    )
    args = parser.parse_args(argv)
    report = run_full_training(args.config, repo_root=args.repo_root)
    print(full_training_report_json(report), end="")
    raise SystemExit(0)


if __name__ == "__main__":
    full_training_main()
