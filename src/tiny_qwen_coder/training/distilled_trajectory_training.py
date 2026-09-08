"""GPU training for the frozen P9-007C distilled-data trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NoReturn

import torch
from transformers import TrainerCallback

from tiny_qwen_coder.identities import AdapterIdentity, BaseModelIdentity
from tiny_qwen_coder.reporting import create_run_manifest, write_run_manifest
from tiny_qwen_coder.reproducibility import seed_everything
from tiny_qwen_coder.training.distilled_trajectory import (
    DistilledTrajectoryValidation,
    distilled_trajectory_validation_json,
    validate_distilled_trajectory,
)
from tiny_qwen_coder.training.plan import (
    AdapterTrainingError,
    AdapterTrainingPlan,
    resolve_adapter_training_plan,
)
from tiny_qwen_coder.training.preflight import run_training_preflight, training_preflight_json
from tiny_qwen_coder.training.runtime import (
    AdapterTrainingRuntimeOptions,
    _load_training_runtime,
    _validation_loss,
    _write_metrics,
)

_EXPECTED_MAX_STEPS = 185
_EXPECTED_SNAPSHOT_STEPS = (25, 50, 100, 185)
_EXPECTED_MANIFEST_SHA256 = "7e299de17cbb62bff3cf5f50c61ad6ad30ca571c9297b9935ff1307cd52e0eb7"
_EXPECTED_SOURCE_OUTPUT_SHA256 = "7f07f7253e98bf8ed295b12d72ebdf03c44b2e08a8d9f74aaa02dce5b760d966"
_EXPECTED_ACCEPTED_RECORDS = 1557
_EXPECTED_TRAIN_RECORDS = 1479
_EXPECTED_VALIDATION_RECORDS = 78


@dataclass(frozen=True, slots=True)
class SnapshotFileDigest:
    """One adapter-only file persisted for a P9-007C snapshot."""

    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class DistilledSnapshotEvidence:
    """Integrity evidence for one precommitted P9-007C adapter snapshot."""

    step: int
    directory: str
    files: tuple[SnapshotFileDigest, ...]
    artifact_set_sha256: str


@dataclass(frozen=True, slots=True)
class DistilledTrajectoryTrainingReport:
    """Machine-readable completion evidence for one P9-007C trajectory."""

    schema_version: int
    task_id: str
    study_id: str
    run_id: str
    source_training_config: str
    source_training_config_sha256: str
    protocol_sha256: str
    corpus_evidence_sha256: str
    dataset_manifest_sha256: str
    source_output_sha256: str
    accepted_records: int
    train_records: int
    validation_records: int
    adapter_id: str
    base_model: BaseModelIdentity
    trajectory_max_steps: int
    checkpoint_steps: tuple[int, ...]
    global_steps: int
    training_loss: float
    validation_loss: float
    total_runtime_seconds: float
    peak_allocated_vram_bytes: int
    peak_reserved_vram_bytes: int
    snapshots: tuple[DistilledSnapshotEvidence, ...]
    snapshot_count: int
    promotable: bool


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _finite_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AdapterTrainingError(f"P9-007C {field_name} must be numeric")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise AdapterTrainingError(f"P9-007C {field_name} must be finite")
    return resolved


def _snapshot_directory_name(step: int) -> str:
    if step not in _EXPECTED_SNAPSHOT_STEPS:
        raise AdapterTrainingError(f"P9-007C refuses non-precommitted snapshot step {step}")
    return f"step-{step:04d}"


class _DistilledAdapterSnapshotCallback(TrainerCallback):
    """Save PEFT adapter-only snapshots at exactly the frozen optimizer steps."""

    def __init__(self, *, snapshot_root: Path, steps: tuple[int, ...]) -> None:
        self._snapshot_root = snapshot_root
        self._steps = frozenset(steps)
        self._saved: set[int] = set()

    def on_step_end(
        self,
        args: object,
        state: object,
        control: object,
        **kwargs: object,
    ) -> object:
        del args
        step = getattr(state, "global_step", None)
        if isinstance(step, bool) or not isinstance(step, int):
            raise AdapterTrainingError("P9-007C trainer state lacks an integer global_step")
        if step not in self._steps or step in self._saved:
            return control
        model = kwargs.get("model")
        save_pretrained = getattr(model, "save_pretrained", None)
        if not callable(save_pretrained):
            raise AdapterTrainingError("P9-007C trainer model cannot save adapter snapshots")
        destination = self._snapshot_root / _snapshot_directory_name(step)
        if destination.exists():
            raise AdapterTrainingError(f"P9-007C snapshot already exists: {destination}")
        save_pretrained(str(destination), safe_serialization=True)
        self._saved.add(step)
        return control

    @property
    def saved_steps(self) -> tuple[int, ...]:
        return tuple(sorted(self._saved))


def _require_repo_root_cwd(repo_root: Path) -> None:
    if repo_root.resolve() != Path.cwd().resolve():
        raise AdapterTrainingError(
            "P9-007C training must run with the repository root as the current directory"
        )


def _verify_local_corpus(
    plan: AdapterTrainingPlan,
    validation: DistilledTrajectoryValidation,
    *,
    repo_root: Path,
) -> None:
    manifest = repo_root / plan.config.dataset_manifest
    if _sha256_file(manifest) != validation.dataset_manifest_sha256:
        raise AdapterTrainingError("P9-007C local dataset manifest does not match frozen corpus")
    sidecar = manifest.with_suffix(".sha256")
    expected_sidecar = f"{validation.dataset_manifest_sha256}  dataset-manifest.json\n"
    try:
        sidecar_text = sidecar.read_text(encoding="ascii")
    except OSError as exc:
        raise AdapterTrainingError("P9-007C dataset manifest sidecar is unavailable") from exc
    if sidecar_text != expected_sidecar:
        raise AdapterTrainingError("P9-007C dataset manifest sidecar does not match")

    receipt_path = manifest.parent / "import-receipt.json"
    try:
        receipt_value: object = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdapterTrainingError("P9-007C local import receipt is unavailable") from exc
    if not isinstance(receipt_value, dict):
        raise AdapterTrainingError("P9-007C local import receipt must be an object")
    receipt = {str(key): value for key, value in receipt_value.items()}
    expected = {
        "dataset_manifest_sha256": _EXPECTED_MANIFEST_SHA256,
        "source_output_sha256": _EXPECTED_SOURCE_OUTPUT_SHA256,
        "accepted_records": _EXPECTED_ACCEPTED_RECORDS,
        "train_records": _EXPECTED_TRAIN_RECORDS,
        "validation_records": _EXPECTED_VALIDATION_RECORDS,
        "contamination_status": "clean",
        "reasoning_marker_hits": 0,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise AdapterTrainingError(f"P9-007C local import receipt {key} drifted")

    data_paths = (
        repo_root / plan.config.train_records,
        repo_root / plan.config.validation_records,
    )
    expected_lines = (_EXPECTED_TRAIN_RECORDS, _EXPECTED_VALIDATION_RECORDS)
    for path, line_count in zip(data_paths, expected_lines, strict=True):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise AdapterTrainingError(f"P9-007C training data is unavailable: {path}") from exc
        if "<think>" in text or "</think>" in text:
            raise AdapterTrainingError(f"P9-007C reasoning marker reappeared in {path}")
        if len(text.splitlines()) != line_count:
            raise AdapterTrainingError(
                f"P9-007C training-data cardinality drifted for {path}: expected {line_count}"
            )


def _prepare_output(
    *,
    output_dir: Path,
    validation: DistilledTrajectoryValidation,
    preflight_json: str,
    dataset_manifest: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=False)
    snapshot_root = output_dir / "snapshots"
    snapshot_root.mkdir(exist_ok=False)
    (output_dir / "checkpoints").mkdir(exist_ok=False)
    (output_dir / "trajectory-validation.json").write_text(
        distilled_trajectory_validation_json(validation), encoding="utf-8"
    )
    (output_dir / "training-preflight.json").write_text(preflight_json, encoding="utf-8")
    shutil.copyfile(dataset_manifest, output_dir / "dataset-manifest.json")
    return snapshot_root


def _snapshot_file_inventory(snapshot_dir: Path) -> tuple[SnapshotFileDigest, ...]:
    adapter_config = snapshot_dir / "adapter_config.json"
    if not adapter_config.is_file():
        raise AdapterTrainingError(f"P9-007C snapshot lacks adapter_config.json: {snapshot_dir}")
    weights = (
        snapshot_dir / "adapter_model.safetensors",
        snapshot_dir / "adapter_model.bin",
    )
    if not any(path.is_file() and path.stat().st_size > 0 for path in weights):
        raise AdapterTrainingError(
            f"P9-007C snapshot lacks non-empty adapter weights: {snapshot_dir}"
        )

    forbidden = [
        snapshot_dir / "model.safetensors",
        snapshot_dir / "model.safetensors.index.json",
        snapshot_dir / "pytorch_model.bin",
        snapshot_dir / "pytorch_model.bin.index.json",
    ]
    forbidden.extend(snapshot_dir.glob("model-*.safetensors"))
    forbidden.extend(snapshot_dir.glob("pytorch_model-*.bin"))
    if any(path.is_file() for path in forbidden):
        raise AdapterTrainingError(
            f"P9-007C snapshot contains merged/full-model weights: {snapshot_dir}"
        )

    files = sorted(
        (path for path in snapshot_dir.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(snapshot_dir).as_posix(),
    )
    if not files:
        raise AdapterTrainingError(f"P9-007C snapshot is empty: {snapshot_dir}")
    return tuple(
        SnapshotFileDigest(
            path=path.relative_to(snapshot_dir).as_posix(),
            size_bytes=path.stat().st_size,
            sha256=_sha256_file(path),
        )
        for path in files
    )


def _snapshot_evidence(output_dir: Path) -> tuple[DistilledSnapshotEvidence, ...]:
    snapshot_root = output_dir / "snapshots"
    observed_dirs = tuple(sorted(path.name for path in snapshot_root.iterdir() if path.is_dir()))
    expected_dirs = tuple(_snapshot_directory_name(step) for step in _EXPECTED_SNAPSHOT_STEPS)
    if observed_dirs != expected_dirs:
        raise AdapterTrainingError(
            f"P9-007C snapshot directory set mismatch: expected {expected_dirs!r}, "
            f"got {observed_dirs!r}"
        )

    evidence: list[DistilledSnapshotEvidence] = []
    for step in _EXPECTED_SNAPSHOT_STEPS:
        directory = snapshot_root / _snapshot_directory_name(step)
        files = _snapshot_file_inventory(directory)
        artifact_set_sha = hashlib.sha256(
            _canonical_json([asdict(item) for item in files]).encode()
        ).hexdigest()
        evidence.append(
            DistilledSnapshotEvidence(
                step=step,
                directory=directory.relative_to(output_dir).as_posix(),
                files=files,
                artifact_set_sha256=artifact_set_sha,
            )
        )
    return tuple(evidence)


def distilled_trajectory_training_report_json(report: DistilledTrajectoryTrainingReport) -> str:
    """Serialize one completed P9-007C trajectory deterministically."""

    return json.dumps(asdict(report), indent=2, sort_keys=True) + "\n"


def run_distilled_trajectory_training(
    *,
    repo_root: Path = Path("."),
) -> DistilledTrajectoryTrainingReport:
    """Train one epoch and persist the four frozen development candidates."""

    _require_repo_root_cwd(repo_root)
    validation = validate_distilled_trajectory(repo_root=repo_root)
    if validation.trajectory_max_steps != _EXPECTED_MAX_STEPS:
        raise AdapterTrainingError("P9-007C trajectory horizon drifted")
    if validation.checkpoint_steps != _EXPECTED_SNAPSHOT_STEPS:
        raise AdapterTrainingError("P9-007C checkpoint grid drifted")

    plan = resolve_adapter_training_plan(repo_root / validation.training_config)
    if plan.config.adapter_id != validation.adapter_id:
        raise AdapterTrainingError("P9-007C resolved adapter identity drifted")
    if Path(plan.config.output_dir).as_posix() != validation.output_dir:
        raise AdapterTrainingError("P9-007C resolved output directory drifted")
    _verify_local_corpus(plan, validation, repo_root=repo_root)

    output_dir = repo_root / validation.output_dir
    preflight = run_training_preflight(plan, repo_root=repo_root, output_dir=output_dir)
    seed_everything(plan.config.seed)
    snapshot_root = _prepare_output(
        output_dir=output_dir,
        validation=validation,
        preflight_json=training_preflight_json(preflight),
        dataset_manifest=repo_root / plan.config.dataset_manifest,
    )
    base_model = BaseModelIdentity(
        repository=plan.target.model_repository,
        revision=plan.target.model_revision,
        tokenizer_repository=plan.target.tokenizer_repository,
        tokenizer_revision=plan.target.tokenizer_revision,
    )
    run_manifest = create_run_manifest(
        run_kind="training",
        base_model=base_model,
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
    trainer, _ = _load_training_runtime(
        plan,
        options=AdapterTrainingRuntimeOptions(output_dir=output_dir),
    )
    callback = _DistilledAdapterSnapshotCallback(
        snapshot_root=snapshot_root,
        steps=validation.checkpoint_steps,
    )
    trainer.add_callback(callback)

    started = time.monotonic()
    train_output: Any = trainer.train()
    evaluation_metrics: Any = trainer.evaluate()
    torch.cuda.synchronize(0)
    total_runtime_seconds = time.monotonic() - started

    global_steps = getattr(trainer.state, "global_step", None)
    if isinstance(global_steps, bool) or not isinstance(global_steps, int):
        raise AdapterTrainingError("P9-007C trainer did not report an integer global step count")
    if global_steps != validation.trajectory_max_steps:
        raise AdapterTrainingError(
            f"P9-007C completed {global_steps} steps; expected {validation.trajectory_max_steps}"
        )
    if callback.saved_steps != validation.checkpoint_steps:
        raise AdapterTrainingError(
            f"P9-007C saved steps {callback.saved_steps!r}; expected {validation.checkpoint_steps!r}"
        )

    training_loss = _finite_float(
        getattr(train_output, "training_loss", None), field_name="training loss"
    )
    validation_loss = _validation_loss(evaluation_metrics)
    if validation_loss is None:
        raise AdapterTrainingError("P9-007C evaluation did not report eval_loss")
    validation_loss = _finite_float(validation_loss, field_name="validation loss")
    if not math.isfinite(total_runtime_seconds) or total_runtime_seconds <= 0:
        raise AdapterTrainingError("P9-007C total runtime must be finite and positive")

    _write_metrics(
        getattr(trainer.state, "log_history", None), output_dir / "training-metrics.jsonl"
    )
    peak_allocated = int(torch.cuda.max_memory_allocated(0))
    peak_reserved = int(torch.cuda.max_memory_reserved(0))
    if peak_allocated <= 0 or peak_reserved <= 0 or peak_allocated > peak_reserved:
        raise AdapterTrainingError("P9-007C CUDA peak-memory evidence is invalid")

    snapshots = _snapshot_evidence(output_dir)
    report = DistilledTrajectoryTrainingReport(
        schema_version=1,
        task_id=validation.task_id,
        study_id=validation.study_id,
        run_id=run_manifest.run_id,
        source_training_config=validation.training_config,
        source_training_config_sha256=validation.training_config_sha256,
        protocol_sha256=validation.protocol_sha256,
        corpus_evidence_sha256=validation.corpus_evidence_sha256,
        dataset_manifest_sha256=validation.dataset_manifest_sha256,
        source_output_sha256=validation.source_output_sha256,
        accepted_records=validation.accepted_records,
        train_records=validation.train_records,
        validation_records=validation.validation_records,
        adapter_id=validation.adapter_id,
        base_model=base_model,
        trajectory_max_steps=validation.trajectory_max_steps,
        checkpoint_steps=validation.checkpoint_steps,
        global_steps=global_steps,
        training_loss=training_loss,
        validation_loss=validation_loss,
        total_runtime_seconds=total_runtime_seconds,
        peak_allocated_vram_bytes=peak_allocated,
        peak_reserved_vram_bytes=peak_reserved,
        snapshots=snapshots,
        snapshot_count=len(snapshots),
        promotable=False,
    )
    (output_dir / "training-report.json").write_text(
        distilled_trajectory_training_report_json(report), encoding="utf-8"
    )
    return report


def distilled_trajectory_training_main(argv: Sequence[str] | None = None) -> NoReturn:
    """CLI entry point for the local P9-007C one-epoch trajectory."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    report = run_distilled_trajectory_training(repo_root=args.repo_root)
    print(distilled_trajectory_training_report_json(report), end="")
    raise SystemExit(0)


if __name__ == "__main__":
    distilled_trajectory_training_main()
