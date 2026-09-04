"""GPU trajectory training for the frozen P9-004 minimum-intervention study."""

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
from tiny_qwen_coder.training.minimum_intervention import (
    MinimumInterventionError,
    MinimumInterventionValidation,
    ValidatedCandidate,
    minimum_intervention_validation_json,
    validate_minimum_intervention,
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

_SCHEMA_VERSION = 1
_EXPECTED_MAX_STEPS = 1000
_EXPECTED_SNAPSHOT_STEPS = (50, 100, 250, 500, 1000)


@dataclass(frozen=True, slots=True)
class SnapshotFileDigest:
    """One adapter-only file persisted for a P9-004B snapshot."""

    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class AdapterSnapshotEvidence:
    """Integrity evidence for one precommitted adapter snapshot."""

    step: int
    directory: str
    files: tuple[SnapshotFileDigest, ...]
    artifact_set_sha256: str


@dataclass(frozen=True, slots=True)
class MinimumInterventionTrainingReport:
    """Machine-readable completion evidence for one P9-004B trajectory."""

    schema_version: int
    task_id: str
    study_id: str
    run_id: str
    candidate_label: str
    learning_rate: float
    source_training_config: str
    source_training_config_sha256: str
    protocol_sha256: str
    fixed_payload_sha256: str
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
    snapshots: tuple[AdapterSnapshotEvidence, ...]
    snapshot_count: int


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
        raise AdapterTrainingError(f"P9-004B {field_name} must be numeric")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise AdapterTrainingError(f"P9-004B {field_name} must be finite")
    return resolved


def _select_candidate(
    validation: MinimumInterventionValidation,
    *,
    label: str,
) -> ValidatedCandidate:
    matches = tuple(item for item in validation.candidates if item.label == label)
    if len(matches) != 1:
        allowed = ", ".join(item.label for item in validation.candidates)
        raise MinimumInterventionError(
            f"P9-004B candidate {label!r} is not frozen; expected one of: {allowed}"
        )
    return matches[0]


def resolve_minimum_intervention_training_candidate(
    label: str,
    *,
    repo_root: Path = Path("."),
) -> tuple[MinimumInterventionValidation, ValidatedCandidate, AdapterTrainingPlan]:
    """Resolve one candidate only after the frozen P9-004A protocol validates."""

    validation = validate_minimum_intervention(repo_root=repo_root)
    if validation.trajectory_max_steps != _EXPECTED_MAX_STEPS:
        raise MinimumInterventionError("P9-004B trajectory horizon drifted from 1,000 steps")
    if validation.checkpoint_steps != _EXPECTED_SNAPSHOT_STEPS:
        raise MinimumInterventionError("P9-004B snapshot steps drifted from the frozen protocol")
    candidate = _select_candidate(validation, label=label)
    plan = resolve_adapter_training_plan(repo_root / candidate.config_path)
    if plan.config.adapter_id != candidate.adapter_id:
        raise MinimumInterventionError("P9-004B resolved adapter identity differs from protocol")
    if Path(plan.config.output_dir).as_posix() != candidate.output_dir:
        raise MinimumInterventionError("P9-004B resolved output directory differs from protocol")
    if not math.isclose(
        plan.config.learning_rate,
        candidate.learning_rate,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise MinimumInterventionError("P9-004B resolved learning rate differs from protocol")
    return validation, candidate, plan


def _snapshot_directory_name(step: int) -> str:
    if step not in _EXPECTED_SNAPSHOT_STEPS:
        raise AdapterTrainingError(f"P9-004B refuses non-precommitted snapshot step {step}")
    return f"step-{step:04d}"


class _AdapterSnapshotCallback(TrainerCallback):
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
            raise AdapterTrainingError("P9-004B trainer state lacks an integer global_step")
        if step not in self._steps or step in self._saved:
            return control
        model = kwargs.get("model")
        save_pretrained = getattr(model, "save_pretrained", None)
        if not callable(save_pretrained):
            raise AdapterTrainingError("P9-004B trainer model cannot save adapter snapshots")
        destination = self._snapshot_root / _snapshot_directory_name(step)
        if destination.exists():
            raise AdapterTrainingError(f"P9-004B snapshot already exists: {destination}")
        save_pretrained(str(destination), safe_serialization=True)
        self._saved.add(step)
        return control

    @property
    def saved_steps(self) -> tuple[int, ...]:
        return tuple(sorted(self._saved))


def _prepare_output(
    *,
    output_dir: Path,
    validation: MinimumInterventionValidation,
    candidate: ValidatedCandidate,
    preflight_json: str,
    dataset_manifest: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=False)
    snapshot_root = output_dir / "snapshots"
    snapshot_root.mkdir(exist_ok=False)
    (output_dir / "checkpoints").mkdir(exist_ok=False)
    (output_dir / "protocol-validation.json").write_text(
        minimum_intervention_validation_json(validation), encoding="utf-8"
    )
    (output_dir / "candidate.json").write_text(
        json.dumps(asdict(candidate), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "training-preflight.json").write_text(preflight_json, encoding="utf-8")
    shutil.copyfile(dataset_manifest, output_dir / "dataset-manifest.json")
    return snapshot_root


def _snapshot_file_inventory(snapshot_dir: Path) -> tuple[SnapshotFileDigest, ...]:
    adapter_config = snapshot_dir / "adapter_config.json"
    if not adapter_config.is_file():
        raise AdapterTrainingError(f"P9-004B snapshot lacks adapter_config.json: {snapshot_dir}")
    weights = (
        snapshot_dir / "adapter_model.safetensors",
        snapshot_dir / "adapter_model.bin",
    )
    if not any(path.is_file() and path.stat().st_size > 0 for path in weights):
        raise AdapterTrainingError(
            f"P9-004B snapshot lacks non-empty adapter weights: {snapshot_dir}"
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
            f"P9-004B snapshot contains merged/full-model weights: {snapshot_dir}"
        )

    files = sorted(
        (path for path in snapshot_dir.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(snapshot_dir).as_posix(),
    )
    if not files:
        raise AdapterTrainingError(f"P9-004B snapshot is empty: {snapshot_dir}")
    return tuple(
        SnapshotFileDigest(
            path=path.relative_to(snapshot_dir).as_posix(),
            size_bytes=path.stat().st_size,
            sha256=_sha256_file(path),
        )
        for path in files
    )


def _snapshot_evidence(output_dir: Path) -> tuple[AdapterSnapshotEvidence, ...]:
    snapshot_root = output_dir / "snapshots"
    observed_dirs = tuple(sorted(path.name for path in snapshot_root.iterdir() if path.is_dir()))
    expected_dirs = tuple(_snapshot_directory_name(step) for step in _EXPECTED_SNAPSHOT_STEPS)
    if observed_dirs != expected_dirs:
        raise AdapterTrainingError(
            f"P9-004B snapshot directory set mismatch: expected {expected_dirs!r}, got {observed_dirs!r}"
        )

    evidence: list[AdapterSnapshotEvidence] = []
    for step in _EXPECTED_SNAPSHOT_STEPS:
        directory = snapshot_root / _snapshot_directory_name(step)
        files = _snapshot_file_inventory(directory)
        artifact_set_sha = hashlib.sha256(
            _canonical_json([asdict(item) for item in files]).encode()
        ).hexdigest()
        evidence.append(
            AdapterSnapshotEvidence(
                step=step,
                directory=directory.relative_to(output_dir).as_posix(),
                files=files,
                artifact_set_sha256=artifact_set_sha,
            )
        )
    return tuple(evidence)


def minimum_intervention_training_report_json(
    report: MinimumInterventionTrainingReport,
) -> str:
    """Serialize one completed P9-004B trajectory deterministically."""

    return json.dumps(asdict(report), indent=2, sort_keys=True) + "\n"


def run_minimum_intervention_training(
    candidate_label: str,
    *,
    repo_root: Path = Path("."),
) -> MinimumInterventionTrainingReport:
    """Train one frozen low-LR trajectory and persist five adapter-only snapshots."""

    validation, candidate, plan = resolve_minimum_intervention_training_candidate(
        candidate_label, repo_root=repo_root
    )
    output_dir = repo_root / candidate.output_dir
    preflight = run_training_preflight(plan, repo_root=repo_root, output_dir=output_dir)

    seed_everything(plan.config.seed)
    snapshot_root = _prepare_output(
        output_dir=output_dir,
        validation=validation,
        candidate=candidate,
        preflight_json=training_preflight_json(preflight),
        dataset_manifest=repo_root / plan.config.dataset_manifest,
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
    trainer, _ = _load_training_runtime(
        plan,
        options=AdapterTrainingRuntimeOptions(
            output_dir=output_dir,
            max_steps=validation.trajectory_max_steps,
        ),
    )
    callback = _AdapterSnapshotCallback(
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
        raise AdapterTrainingError("P9-004B trainer did not report an integer global step count")
    if global_steps != validation.trajectory_max_steps:
        raise AdapterTrainingError(
            f"P9-004B completed {global_steps} steps; expected {validation.trajectory_max_steps}"
        )
    if callback.saved_steps != validation.checkpoint_steps:
        raise AdapterTrainingError(
            f"P9-004B saved steps {callback.saved_steps!r}; expected {validation.checkpoint_steps!r}"
        )

    training_loss = _finite_float(
        getattr(train_output, "training_loss", None), field_name="training loss"
    )
    validation_loss = _validation_loss(evaluation_metrics)
    if validation_loss is None:
        raise AdapterTrainingError("P9-004B evaluation did not report eval_loss")
    validation_loss = _finite_float(validation_loss, field_name="validation loss")
    if not math.isfinite(total_runtime_seconds) or total_runtime_seconds <= 0:
        raise AdapterTrainingError("P9-004B total runtime must be finite and positive")

    log_history = getattr(trainer.state, "log_history", None)
    _write_metrics(log_history, output_dir / "training-metrics.jsonl")
    peak_allocated = int(torch.cuda.max_memory_allocated(0))
    peak_reserved = int(torch.cuda.max_memory_reserved(0))
    if peak_allocated <= 0 or peak_reserved <= 0 or peak_allocated > peak_reserved:
        raise AdapterTrainingError("P9-004B CUDA peak-memory evidence is invalid")

    snapshots = _snapshot_evidence(output_dir)
    report = MinimumInterventionTrainingReport(
        schema_version=_SCHEMA_VERSION,
        task_id=validation.task_id,
        study_id=validation.study_id,
        run_id=run_manifest.run_id,
        candidate_label=candidate.label,
        learning_rate=candidate.learning_rate,
        source_training_config=candidate.config_path,
        source_training_config_sha256=candidate.config_sha256,
        protocol_sha256=validation.protocol_sha256,
        fixed_payload_sha256=validation.fixed_payload_sha256,
        adapter_id=candidate.adapter_id,
        base_model=BaseModelIdentity(
            repository=plan.target.model_repository,
            revision=plan.target.model_revision,
            tokenizer_repository=plan.target.tokenizer_repository,
            tokenizer_revision=plan.target.tokenizer_revision,
        ),
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
    )
    (output_dir / "training-report.json").write_text(
        minimum_intervention_training_report_json(report), encoding="utf-8"
    )
    return report


def minimum_intervention_training_main(argv: Sequence[str] | None = None) -> int:
    """CLI for one P9-004B low-LR GPU trajectory."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    report = run_minimum_intervention_training(args.candidate, repo_root=args.repo_root)
    print(minimum_intervention_training_report_json(report), end="")
    return 0


def main() -> NoReturn:
    raise SystemExit(minimum_intervention_training_main())


if __name__ == "__main__":
    main()
