"""Development-only checkpoint evaluation for the frozen P9-004 study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, NoReturn, Protocol, cast

import torch
from torch import nn

from tiny_qwen_coder.config import EvaluationConfig, load_evaluation_config
from tiny_qwen_coder.evaluation._baseline_generation import BaselineGenerator
from tiny_qwen_coder.evaluation._baseline_provenance import load_baseline_base_model_identity
from tiny_qwen_coder.evaluation._baseline_runner import _generate_items
from tiny_qwen_coder.evaluation._baseline_types import BaselineGeneratedResponse
from tiny_qwen_coder.evaluation._python_p0_generation import _parameter_dtypes, _resolved_revision
from tiny_qwen_coder.evaluation.execution import DirectExecutionHarness
from tiny_qwen_coder.evaluation.humaneval import (
    HumanEvalCompletion,
    HumanEvalEvaluator,
    HumanEvalProblem,
)
from tiny_qwen_coder.evaluation.mbpp import MBPPCompletion, MBPPEvaluator, MBPPProblem
from tiny_qwen_coder.evaluation.results import GenerationStats
from tiny_qwen_coder.evaluation.settings import (
    FrozenEvaluationSettings,
    evaluation_settings_sha256,
    load_frozen_evaluation_settings,
)
from tiny_qwen_coder.identities import AdapterIdentity, BaseModelIdentity
from tiny_qwen_coder.languages.python import load_python_plugin
from tiny_qwen_coder.reproducibility import seed_everything
from tiny_qwen_coder.training.minimum_intervention import (
    partition_task,
    validate_minimum_intervention,
)

_DEVELOPMENT_MANIFEST = Path("configs/eval/python/p9_minimum_intervention_development_v1.json")
_CHECKPOINT_REGISTRY = Path("configs/eval/python/p9_minimum_intervention_checkpoints_v1.json")
_BASE_EVALUATION = Path("configs/eval/python/base_baseline_v1.yaml")
_EXPECTED_DEVELOPMENT_SHA256 = "260682d773640b28673357c7a441474656bbaffc67f0b5fcbe45eca3a07283de"
_EXPECTED_CHECKPOINT_REGISTRY_SHA256 = (
    "31362b22f3f84fe05a6c499accf63a23b65295b337c3c9d614568732c0643196"
)
_EXPECTED_MEMBERSHIP_SHA256 = "c8765b7a3a69f134be4066e274a64640a15f2d05858374035432207b3521498b"
_EXPECTED_LABELS = ("lr-1e-5", "lr-2e-5", "lr-5e-5", "lr-1e-4", "lr-2e-4")
_EXPECTED_LRS = (0.00001, 0.00002, 0.00005, 0.0001, 0.0002)
_EXPECTED_STEPS = (50, 100, 250, 500, 1000)
_EXPECTED_HE_TOTAL = 45
_EXPECTED_MBPP_TOTAL = 130
_EXPECTED_COMBINED_TOTAL = 175
_EXPECTED_BASE_HE = 33
_EXPECTED_BASE_MBPP = 70
_EXPECTED_BASE_COMBINED = 103
_EXPECTED_MIN_COMBINED = 104
_OUTPUT_ROOT = Path("artifacts/eval/python/p9-minimum-intervention-development-v1")
_ENABLE_THINKING = False


class MinimumInterventionEvaluationError(RuntimeError):
    """Raised when P9-004C evidence or execution violates the frozen contract."""


@dataclass(frozen=True, slots=True)
class SnapshotIdentity:
    step: int
    adapter_config_sha256: str
    adapter_config_size_bytes: int
    adapter_model_sha256: str
    adapter_model_size_bytes: int
    artifact_set_sha256: str


@dataclass(frozen=True, slots=True)
class TrajectoryIdentity:
    label: str
    learning_rate: float
    adapter_id: str
    training_run_id: str
    training_workflow_run_id: int
    training_source_git_sha: str
    training_artifact_id: int
    training_artifact_name: str
    training_artifact_digest: str
    snapshots: tuple[SnapshotIdentity, ...]

    def snapshot(self, step: int) -> SnapshotIdentity:
        matches = tuple(item for item in self.snapshots if item.step == step)
        if len(matches) != 1:
            raise MinimumInterventionEvaluationError(
                f"P9-004C trajectory {self.label} has no unique snapshot at step {step}"
            )
        return matches[0]


@dataclass(frozen=True, slots=True)
class DevelopmentScore:
    label: str
    learning_rate: float
    step: int
    humaneval_passed: int
    humaneval_total: int
    mbpp_passed: int
    mbpp_total: int
    combined_passed: int
    combined_total: int
    eligible: bool


class _GenerateCapable(Protocol):
    def generate(self, **kwargs: object) -> torch.Tensor: ...


class _Tokenizer(Protocol):
    chat_template: object

    def apply_chat_template(self, *args: object, **kwargs: object) -> object: ...

    def decode(self, token_ids: list[int], **kwargs: object) -> object: ...


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _load_json(path: Path, *, expected_sha256: str, context: str) -> dict[str, object]:
    if _file_sha256(path) != expected_sha256:
        raise MinimumInterventionEvaluationError(f"{context} SHA-256 drifted")
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MinimumInterventionEvaluationError(f"could not read {context}") from exc
    if not isinstance(value, dict):
        raise MinimumInterventionEvaluationError(f"{context} must be a JSON object")
    return cast(dict[str, object], value)


def _mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise MinimumInterventionEvaluationError(f"{context} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise MinimumInterventionEvaluationError(f"{context} keys must be strings")
    return cast(dict[str, object], dict(value))


def _string(mapping: Mapping[str, object], key: str, *, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise MinimumInterventionEvaluationError(f"{context}.{key} must be a non-empty string")
    return value


def _integer(mapping: Mapping[str, object], key: str, *, context: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise MinimumInterventionEvaluationError(f"{context}.{key} must be an integer")
    return value


def _number(mapping: Mapping[str, object], key: str, *, context: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise MinimumInterventionEvaluationError(f"{context}.{key} must be numeric")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise MinimumInterventionEvaluationError(f"{context}.{key} must be finite")
    return resolved


def _string_list(value: object, *, context: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise MinimumInterventionEvaluationError(f"{context} must be a string list")
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise MinimumInterventionEvaluationError(f"{context} must contain non-empty strings")
    return cast(tuple[str, ...], result)


def load_development_manifest(repo_root: Path = Path(".")) -> dict[str, object]:
    """Load and mechanically validate the frozen development/qualification membership."""

    manifest = _load_json(
        repo_root / _DEVELOPMENT_MANIFEST,
        expected_sha256=_EXPECTED_DEVELOPMENT_SHA256,
        context="P9-004C development manifest",
    )
    if manifest.get("schema_version") != 1 or manifest.get("task_id") != "P9-004":
        raise MinimumInterventionEvaluationError("P9-004C development manifest identity drifted")
    if manifest.get("study_id") != "python-p9-minimum-intervention-v1":
        raise MinimumInterventionEvaluationError("P9-004C development study identity drifted")
    membership = _mapping(manifest.get("membership"), context="development membership")
    membership_sha = hashlib.sha256(_canonical_json(membership).encode()).hexdigest()
    if (
        membership_sha != _EXPECTED_MEMBERSHIP_SHA256
        or manifest.get("membership_sha256") != membership_sha
    ):
        raise MinimumInterventionEvaluationError("P9-004C membership SHA-256 drifted")

    validation = validate_minimum_intervention(repo_root=repo_root)
    he = _mapping(membership.get("humaneval"), context="membership.humaneval")
    mbpp = _mapping(membership.get("mbpp"), context="membership.mbpp")
    holdout = _mapping(
        membership.get("repository_holdout"), context="membership.repository_holdout"
    )
    he_dev = _string_list(he.get("development"), context="membership.humaneval.development")
    he_qual = _string_list(he.get("qualification"), context="membership.humaneval.qualification")
    mb_dev = _string_list(mbpp.get("development"), context="membership.mbpp.development")
    mb_qual = _string_list(mbpp.get("qualification"), context="membership.mbpp.qualification")
    holdout_dev = _string_list(
        holdout.get("development"), context="membership.repository_holdout.development"
    )
    holdout_qual = _string_list(
        holdout.get("qualification"), context="membership.repository_holdout.qualification"
    )

    expected_he = tuple(f"HumanEval/{index}" for index in range(164))
    expected_mbpp = tuple(f"MBPP/{index}" for index in range(11, 511))
    if tuple(sorted(he_dev + he_qual, key=lambda item: int(item.split("/")[1]))) != expected_he:
        raise MinimumInterventionEvaluationError("P9-004C HumanEval membership is not exhaustive")
    if tuple(sorted(mb_dev + mb_qual, key=lambda item: int(item.split("/")[1]))) != expected_mbpp:
        raise MinimumInterventionEvaluationError("P9-004C MBPP membership is not exhaustive")
    if set(he_dev) & set(he_qual) or set(mb_dev) & set(mb_qual):
        raise MinimumInterventionEvaluationError(
            "P9-004C development/qualification membership overlaps"
        )
    if holdout_dev or len(holdout_qual) != 11:
        raise MinimumInterventionEvaluationError(
            "repository holdout must remain qualification-only"
        )

    recomputed_he = tuple(
        task_id
        for task_id in expected_he
        if partition_task(validation.development_partition, suite="humaneval", task_id=task_id)
        == "development"
    )
    recomputed_mbpp = tuple(
        task_id
        for task_id in expected_mbpp
        if partition_task(
            validation.development_partition,
            suite="mbpp",
            task_id=task_id.removeprefix("MBPP/"),
        )
        == "development"
    )
    if recomputed_he != he_dev or recomputed_mbpp != mb_dev:
        raise MinimumInterventionEvaluationError(
            "P9-004C membership does not match frozen hash partition"
        )
    if (len(he_dev), len(mb_dev)) != (_EXPECTED_HE_TOTAL, _EXPECTED_MBPP_TOTAL):
        raise MinimumInterventionEvaluationError("P9-004C development cardinality drifted")

    baseline = _mapping(
        manifest.get("accepted_base_development"), context="accepted base development"
    )
    base_he = _mapping(baseline.get("humaneval"), context="accepted base humaneval")
    base_mb = _mapping(baseline.get("mbpp"), context="accepted base mbpp")
    base_combined = _mapping(baseline.get("combined"), context="accepted base combined")
    if (
        (
            _integer(base_he, "passed", context="base humaneval"),
            _integer(base_he, "total", context="base humaneval"),
        )
        != (_EXPECTED_BASE_HE, _EXPECTED_HE_TOTAL)
        or (
            _integer(base_mb, "passed", context="base mbpp"),
            _integer(base_mb, "total", context="base mbpp"),
        )
        != (_EXPECTED_BASE_MBPP, _EXPECTED_MBPP_TOTAL)
        or (
            _integer(base_combined, "passed", context="base combined"),
            _integer(base_combined, "total", context="base combined"),
        )
        != (_EXPECTED_BASE_COMBINED, _EXPECTED_COMBINED_TOTAL)
    ):
        raise MinimumInterventionEvaluationError("accepted base development target drifted")
    return manifest


def load_checkpoint_registry(repo_root: Path = Path(".")) -> tuple[TrajectoryIdentity, ...]:
    """Load the exact five trajectories and 25 adapter snapshots."""

    root = _load_json(
        repo_root / _CHECKPOINT_REGISTRY,
        expected_sha256=_EXPECTED_CHECKPOINT_REGISTRY_SHA256,
        context="P9-004C checkpoint registry",
    )
    if root.get("schema_version") != 1 or root.get("task_id") != "P9-004":
        raise MinimumInterventionEvaluationError("P9-004C checkpoint registry identity drifted")
    if root.get("checkpoint_registry_id") != "python-p9-004c-checkpoints-v1":
        raise MinimumInterventionEvaluationError("P9-004C checkpoint registry ID drifted")
    rows = root.get("trajectories")
    if isinstance(rows, str) or not isinstance(rows, Sequence):
        raise MinimumInterventionEvaluationError("checkpoint registry trajectories must be a list")
    trajectories: list[TrajectoryIdentity] = []
    for index, raw in enumerate(rows):
        row = _mapping(raw, context=f"trajectory[{index}]")
        snapshots_raw = row.get("snapshots")
        if isinstance(snapshots_raw, str) or not isinstance(snapshots_raw, Sequence):
            raise MinimumInterventionEvaluationError(
                f"trajectory[{index}].snapshots must be a list"
            )
        snapshots: list[SnapshotIdentity] = []
        for snap_index, raw_snapshot in enumerate(snapshots_raw):
            snap = _mapping(raw_snapshot, context=f"trajectory[{index}].snapshots[{snap_index}]")
            snapshots.append(
                SnapshotIdentity(
                    step=_integer(snap, "step", context="snapshot"),
                    adapter_config_sha256=_string(
                        snap, "adapter_config_sha256", context="snapshot"
                    ),
                    adapter_config_size_bytes=_integer(
                        snap, "adapter_config_size_bytes", context="snapshot"
                    ),
                    adapter_model_sha256=_string(snap, "adapter_model_sha256", context="snapshot"),
                    adapter_model_size_bytes=_integer(
                        snap, "adapter_model_size_bytes", context="snapshot"
                    ),
                    artifact_set_sha256=_string(snap, "artifact_set_sha256", context="snapshot"),
                )
            )
        trajectories.append(
            TrajectoryIdentity(
                label=_string(row, "label", context="trajectory"),
                learning_rate=_number(row, "learning_rate", context="trajectory"),
                adapter_id=_string(row, "adapter_id", context="trajectory"),
                training_run_id=_string(row, "training_run_id", context="trajectory"),
                training_workflow_run_id=_integer(
                    row, "training_workflow_run_id", context="trajectory"
                ),
                training_source_git_sha=_string(
                    row, "training_source_git_sha", context="trajectory"
                ),
                training_artifact_id=_integer(row, "training_artifact_id", context="trajectory"),
                training_artifact_name=_string(row, "training_artifact_name", context="trajectory"),
                training_artifact_digest=_string(
                    row, "training_artifact_digest", context="trajectory"
                ),
                snapshots=tuple(snapshots),
            )
        )
    if tuple(item.label for item in trajectories) != _EXPECTED_LABELS:
        raise MinimumInterventionEvaluationError("P9-004C trajectory order/labels drifted")
    if tuple(item.learning_rate for item in trajectories) != _EXPECTED_LRS:
        raise MinimumInterventionEvaluationError("P9-004C learning-rate grid drifted")
    if any(
        tuple(snapshot.step for snapshot in item.snapshots) != _EXPECTED_STEPS
        for item in trajectories
    ):
        raise MinimumInterventionEvaluationError("P9-004C checkpoint grid drifted")
    if len({(item.label, snap.step) for item in trajectories for snap in item.snapshots}) != 25:
        raise MinimumInterventionEvaluationError("P9-004C checkpoint identities are not unique")

    protocol = validate_minimum_intervention(repo_root=repo_root)
    by_label = {item.label: item for item in protocol.candidates}
    for item in trajectories:
        candidate = by_label.get(item.label)
        if (
            candidate is None
            or candidate.learning_rate != item.learning_rate
            or candidate.adapter_id != item.adapter_id
        ):
            raise MinimumInterventionEvaluationError(
                "checkpoint registry differs from P9-004 protocol"
            )
    return tuple(trajectories)


def _trajectory(label: str, *, repo_root: Path) -> TrajectoryIdentity:
    matches = tuple(item for item in load_checkpoint_registry(repo_root) if item.label == label)
    if len(matches) != 1:
        raise MinimumInterventionEvaluationError(f"unknown P9-004C trajectory {label!r}")
    return matches[0]


def _snapshot_inventory(snapshot_dir: Path) -> list[dict[str, object]]:
    files = sorted(
        (path for path in snapshot_dir.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(snapshot_dir).as_posix(),
    )
    return [
        {
            "path": path.relative_to(snapshot_dir).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for path in files
    ]


def validate_training_artifact(
    training_output: Path,
    trajectory: TrajectoryIdentity,
    *,
    repo_root: Path = Path("."),
) -> dict[int, Path]:
    """Prove that a downloaded trajectory contains the registry's exact five adapters."""

    report_path = training_output / "training-report.json"
    run_manifest_path = training_output / "run-manifest.json"
    try:
        report: object = json.loads(report_path.read_text(encoding="utf-8"))
        run_manifest: object = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MinimumInterventionEvaluationError(
            "training artifact metadata is unreadable"
        ) from exc
    report_map = _mapping(report, context="training report")
    run_map = _mapping(run_manifest, context="training run manifest")
    if (
        report_map.get("task_id") != "P9-004"
        or report_map.get("study_id") != "python-p9-minimum-intervention-v1"
    ):
        raise MinimumInterventionEvaluationError("training report is not P9-004")
    if (
        report_map.get("candidate_label") != trajectory.label
        or report_map.get("adapter_id") != trajectory.adapter_id
    ):
        raise MinimumInterventionEvaluationError("training report trajectory identity mismatch")
    if report_map.get("run_id") != trajectory.training_run_id:
        raise MinimumInterventionEvaluationError("training run ID mismatch")
    run_git = _mapping(run_map.get("git"), context="training run manifest.git")
    if run_git.get("sha") != trajectory.training_source_git_sha:
        raise MinimumInterventionEvaluationError("training source Git SHA mismatch")
    if report_map.get("global_steps") != 1000 or report_map.get("checkpoint_steps") != list(
        _EXPECTED_STEPS
    ):
        raise MinimumInterventionEvaluationError("training report checkpoint horizon drifted")

    resolved: dict[int, Path] = {}
    for snapshot in trajectory.snapshots:
        directory = training_output / "snapshots" / f"step-{snapshot.step:04d}"
        config = directory / "adapter_config.json"
        weights = directory / "adapter_model.safetensors"
        if not config.is_file() or not weights.is_file():
            raise MinimumInterventionEvaluationError(
                f"missing snapshot files for step {snapshot.step}"
            )
        if (
            config.stat().st_size != snapshot.adapter_config_size_bytes
            or _file_sha256(config) != snapshot.adapter_config_sha256
        ):
            raise MinimumInterventionEvaluationError(
                f"adapter config identity mismatch at step {snapshot.step}"
            )
        if (
            weights.stat().st_size != snapshot.adapter_model_size_bytes
            or _file_sha256(weights) != snapshot.adapter_model_sha256
        ):
            raise MinimumInterventionEvaluationError(
                f"adapter weight identity mismatch at step {snapshot.step}"
            )
        try:
            config_value: object = json.loads(config.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MinimumInterventionEvaluationError("adapter config JSON is unreadable") from exc
        config_map = _mapping(config_value, context="adapter config")
        if str(config_map.get("peft_type", "")).upper() != "LORA":
            raise MinimumInterventionEvaluationError("snapshot PEFT type is not LORA")
        if config_map.get("r") != 8 or config_map.get("lora_alpha") != 32:
            raise MinimumInterventionEvaluationError("snapshot LoRA rank/alpha drifted")
        inventory = _snapshot_inventory(directory)
        artifact_set = hashlib.sha256(_canonical_json(inventory).encode()).hexdigest()
        if artifact_set != snapshot.artifact_set_sha256:
            raise MinimumInterventionEvaluationError(
                f"snapshot artifact-set SHA mismatch at step {snapshot.step}"
            )
        resolved[snapshot.step] = directory
    if tuple(sorted(resolved)) != _EXPECTED_STEPS:
        raise MinimumInterventionEvaluationError(
            "training artifact does not contain exactly five snapshots"
        )
    return resolved


def _source_git_sha(repo_root: Path) -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise MinimumInterventionEvaluationError("could not inspect P9-004C source tree") from exc
    if len(sha) != 40 or any(character not in "0123456789abcdef" for character in sha):
        raise MinimumInterventionEvaluationError("P9-004C source SHA is invalid")
    if status.strip():
        raise MinimumInterventionEvaluationError("P9-004C requires a clean source tree")
    return sha


def _evaluation_context(
    trajectory: TrajectoryIdentity,
) -> tuple[EvaluationConfig, FrozenEvaluationSettings, BaseModelIdentity, str]:
    base_evaluation = load_evaluation_config(_BASE_EVALUATION)
    output = _OUTPUT_ROOT / trajectory.label
    evaluation = replace(
        base_evaluation,
        adapter_id=trajectory.adapter_id,
        suites=("humaneval", "mbpp"),
        output_dir=output.as_posix(),
    )
    settings = load_frozen_evaluation_settings()
    base_model = load_baseline_base_model_identity(Path(evaluation.base_config))
    system_prompt = load_python_plugin().spec.config.system_prompt.text
    return evaluation, settings, base_model, system_prompt


def _development_ids(manifest: Mapping[str, object]) -> tuple[frozenset[str], frozenset[str]]:
    membership = _mapping(manifest.get("membership"), context="development membership")
    he = _mapping(membership.get("humaneval"), context="development humaneval")
    mb = _mapping(membership.get("mbpp"), context="development mbpp")
    return (
        frozenset(_string_list(he.get("development"), context="humaneval development")),
        frozenset(_string_list(mb.get("development"), context="mbpp development")),
    )


def _load_development_problems(
    evaluation: EvaluationConfig,
    base_model: BaseModelIdentity,
    trajectory: TrajectoryIdentity,
    settings: FrozenEvaluationSettings,
    manifest: Mapping[str, object],
    *,
    harness: DirectExecutionHarness | None = None,
) -> tuple[
    HumanEvalEvaluator, tuple[HumanEvalProblem, ...], MBPPEvaluator, tuple[MBPPProblem, ...]
]:
    adapter = AdapterIdentity(family="language", adapter_id=trajectory.adapter_id)
    humaneval = HumanEvalEvaluator(
        evaluation,
        base_model=base_model,
        adapter=adapter,
        settings=settings,
        harness=harness,
    )
    mbpp = MBPPEvaluator(
        evaluation,
        base_model=base_model,
        adapter=adapter,
        settings=settings,
        harness=harness,
    )
    he_ids, mb_ids = _development_ids(manifest)
    he = tuple(problem for problem in humaneval.load_problems() if problem.task_id in he_ids)
    mb = tuple(problem for problem in mbpp.load_problems() if problem.task_id in mb_ids)
    if len(he) != _EXPECTED_HE_TOTAL or len(mb) != _EXPECTED_MBPP_TOTAL:
        raise MinimumInterventionEvaluationError("loaded development benchmark cardinality drifted")
    if {item.task_id for item in he} != he_ids or {item.task_id for item in mb} != mb_ids:
        raise MinimumInterventionEvaluationError("loaded development benchmark membership drifted")
    return humaneval, he, mbpp, mb


def _generation_contract(
    *,
    trajectory: TrajectoryIdentity,
    snapshot: SnapshotIdentity,
    base_model: BaseModelIdentity,
    settings: FrozenEvaluationSettings,
    system_prompt: str,
) -> str:
    payload = {
        "schema_version": 1,
        "study_id": "python-p9-minimum-intervention-v1",
        "trajectory": trajectory.label,
        "learning_rate": trajectory.learning_rate,
        "step": snapshot.step,
        "adapter_id": trajectory.adapter_id,
        "adapter_model_sha256": snapshot.adapter_model_sha256,
        "base_model": asdict(base_model),
        "evaluation_settings_sha256": evaluation_settings_sha256(settings),
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
    }
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


class SnapshotTrajectoryGenerator:
    """Load one base model and five immutable r8 snapshots for deterministic dev generation."""

    def __init__(
        self,
        *,
        snapshot_dirs: Mapping[int, Path],
        trajectory: TrajectoryIdentity,
        base_model: BaseModelIdentity,
        settings: FrozenEvaluationSettings,
        device_index: int = 0,
    ) -> None:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise MinimumInterventionEvaluationError("P9-004C generation requires BF16 CUDA")
        if not 0 <= device_index < torch.cuda.device_count():
            raise MinimumInterventionEvaluationError("invalid P9-004C CUDA device index")
        if settings.generation.decoding_strategy != "greedy":
            raise MinimumInterventionEvaluationError("P9-004C requires frozen greedy decoding")
        from peft import PeftModel
        from transformers import AutoModelForMultimodalLM, AutoTokenizer, PreTrainedTokenizerBase

        self._device = torch.device("cuda", device_index)
        self._settings = settings
        self._trajectory = trajectory
        torch.cuda.set_device(self._device)
        seed_everything(settings.seed)
        tokenizer_obj: object = AutoTokenizer.from_pretrained(
            base_model.tokenizer_repository, revision=base_model.tokenizer_revision
        )
        if not isinstance(tokenizer_obj, PreTrainedTokenizerBase):
            raise MinimumInterventionEvaluationError("Transformers returned unexpected tokenizer")
        tokenizer = cast(_Tokenizer, tokenizer_obj)
        if not isinstance(tokenizer.chat_template, str) or not tokenizer.chat_template:
            raise MinimumInterventionEvaluationError("canonical tokenizer lacks chat template")
        self._tokenizer = tokenizer

        loaded: object = cast(Any, AutoModelForMultimodalLM).from_pretrained(
            base_model.repository,
            revision=base_model.revision,
            dtype=torch.bfloat16,
            device_map={"": device_index},
            low_cpu_mem_usage=True,
        )
        if not isinstance(loaded, nn.Module):
            raise MinimumInterventionEvaluationError("Transformers returned unexpected model")
        base = loaded
        base.eval()
        if _resolved_revision(base) != base_model.revision:
            raise MinimumInterventionEvaluationError("loaded base revision is not canonical")
        if _parameter_dtypes(base) != ("torch.bfloat16",):
            raise MinimumInterventionEvaluationError("P9-004C base model is not pure BF16")

        first = _EXPECTED_STEPS[0]
        first_name = self._adapter_name(first)
        adapted_obj: object = cast(Any, PeftModel).from_pretrained(
            base,
            str(snapshot_dirs[first]),
            adapter_name=first_name,
            is_trainable=False,
        )
        if not isinstance(adapted_obj, nn.Module):
            raise MinimumInterventionEvaluationError("PEFT returned unexpected model")
        self._model = adapted_obj
        load_adapter = getattr(self._model, "load_adapter", None)
        if not callable(load_adapter):
            raise MinimumInterventionEvaluationError("PEFT model cannot load multiple adapters")
        for step in _EXPECTED_STEPS[1:]:
            cast(Any, self._model).load_adapter(
                str(snapshot_dirs[step]), adapter_name=self._adapter_name(step), is_trainable=False
            )
        self._model.eval()
        self._model.requires_grad_(False)
        self._active_step: int | None = None

    @staticmethod
    def _adapter_name(step: int) -> str:
        return f"step-{step:04d}"

    def select_step(self, step: int) -> None:
        if step not in _EXPECTED_STEPS:
            raise MinimumInterventionEvaluationError(f"non-frozen P9-004C step {step}")
        setter = getattr(self._model, "set_adapter", None)
        if not callable(setter):
            raise MinimumInterventionEvaluationError("PEFT model cannot switch adapters")
        setter(self._adapter_name(step))
        status_getter = getattr(self._model, "get_model_status", None)
        if not callable(status_getter):
            raise MinimumInterventionEvaluationError("PEFT model lacks status reporting")
        status = status_getter()
        active = tuple(getattr(status, "active_adapters", ()))
        if active != (self._adapter_name(step),):
            raise MinimumInterventionEvaluationError(
                "PEFT active adapter does not match requested step"
            )
        if getattr(status, "trainable_params", None) != 0:
            raise MinimumInterventionEvaluationError("P9-004C adapter unexpectedly trainable")
        self._active_step = step
        seed_everything(self._settings.seed)

    def _prepare_inputs(self, system_prompt: str, user_prompt: str) -> dict[str, torch.Tensor]:
        encoded = self._tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=_ENABLE_THINKING,
        )
        if not isinstance(encoded, Mapping):
            raise MinimumInterventionEvaluationError("chat template returned non-mapping")
        inputs = {
            str(key): value.to(self._device)
            for key, value in encoded.items()
            if isinstance(value, torch.Tensor)
        }
        ids = inputs.get("input_ids")
        if ids is None or ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] <= 0:
            raise MinimumInterventionEvaluationError("P9-004C prompt tokenization is invalid")
        return inputs

    def generate(self, *, system_prompt: str, user_prompt: str) -> BaselineGeneratedResponse:
        if self._active_step is None:
            raise MinimumInterventionEvaluationError("P9-004C adapter step was not selected")
        inputs = self._prepare_inputs(system_prompt, user_prompt)
        prompt_tokens = int(inputs["input_ids"].shape[1])
        kwargs: dict[str, object] = dict(inputs)
        kwargs.update(
            {
                "max_new_tokens": self._settings.generation.max_new_tokens,
                "do_sample": False,
                "num_beams": 1,
                "use_cache": True,
            }
        )
        torch.cuda.synchronize(self._device)
        started = time.perf_counter()
        with torch.inference_mode():
            output = cast(_GenerateCapable, self._model).generate(**kwargs)
        torch.cuda.synchronize(self._device)
        latency = time.perf_counter() - started
        if not isinstance(output, torch.Tensor) or output.ndim != 2 or output.shape[0] != 1:
            raise MinimumInterventionEvaluationError("model.generate returned invalid tensor")
        if output.shape[1] <= prompt_tokens or latency <= 0:
            raise MinimumInterventionEvaluationError("P9-004C generation produced no completion")
        token_ids = [int(item) for item in output[0, prompt_tokens:].detach().cpu().tolist()]
        decoded = self._tokenizer.decode(
            token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        if not isinstance(decoded, str) or not token_ids:
            raise MinimumInterventionEvaluationError(
                "P9-004C tokenizer returned invalid completion"
            )
        return BaselineGeneratedResponse(
            generated_text=decoded,
            generation=GenerationStats(
                prompt_tokens=prompt_tokens,
                generated_tokens=len(token_ids),
                latency_seconds=latency,
                tokens_per_second=len(token_ids) / latency,
            ),
        )


class _CheckpointOnlyGenerator(BaselineGenerator):
    def generate(self, *, system_prompt: str, user_prompt: str) -> BaselineGeneratedResponse:
        del system_prompt, user_prompt
        raise MinimumInterventionEvaluationError(
            "P9-004C scoring is missing transported GPU responses; refusing regeneration"
        )


def _checkpoint_output(trajectory: TrajectoryIdentity, step: int) -> Path:
    return _OUTPUT_ROOT / trajectory.label / f"step-{step:04d}"


def generate_trajectory(
    *,
    label: str,
    training_output: Path,
    repo_root: Path = Path("."),
    device_index: int = 0,
) -> tuple[Path, ...]:
    """Generate development responses for all five snapshots of one LR trajectory."""

    manifest = load_development_manifest(repo_root)
    trajectory = _trajectory(label, repo_root=repo_root)
    snapshot_dirs = validate_training_artifact(training_output, trajectory, repo_root=repo_root)
    evaluation, settings, base_model, system_prompt = _evaluation_context(trajectory)
    source_sha = _source_git_sha(repo_root)
    humaneval, he, mbpp, mb = _load_development_problems(
        evaluation, base_model, trajectory, settings, manifest
    )
    generator = SnapshotTrajectoryGenerator(
        snapshot_dirs=snapshot_dirs,
        trajectory=trajectory,
        base_model=base_model,
        settings=settings,
        device_index=device_index,
    )
    stage_paths: list[Path] = []
    for snapshot in trajectory.snapshots:
        generator.select_step(snapshot.step)
        output_dir = _checkpoint_output(trajectory, snapshot.step)
        output_dir.mkdir(parents=True, exist_ok=True)
        contract = _generation_contract(
            trajectory=trajectory,
            snapshot=snapshot,
            base_model=base_model,
            settings=settings,
            system_prompt=system_prompt,
        )
        he_responses = _generate_items(
            suite_id="humaneval-development",
            prompts=tuple(
                (problem.task_id, humaneval.prompt_for(problem).user_content) for problem in he
            ),
            generator=generator,
            system_prompt=system_prompt,
            generation_contract=contract,
            output_dir=output_dir,
        )
        mb_responses = _generate_items(
            suite_id="mbpp-development",
            prompts=tuple(
                (problem.task_id, mbpp.prompt_for(problem).user_content) for problem in mb
            ),
            generator=generator,
            system_prompt=system_prompt,
            generation_contract=contract,
            output_dir=output_dir,
        )
        if len(he_responses) + len(mb_responses) != _EXPECTED_COMBINED_TOTAL:
            raise MinimumInterventionEvaluationError("P9-004C generation cardinality is incomplete")
        stage = {
            "schema_version": 1,
            "task_id": "P9-004",
            "stage": "development-generation",
            "source_git_sha": source_sha,
            "development_manifest_sha256": _EXPECTED_DEVELOPMENT_SHA256,
            "membership_sha256": _EXPECTED_MEMBERSHIP_SHA256,
            "checkpoint_registry_sha256": _EXPECTED_CHECKPOINT_REGISTRY_SHA256,
            "trajectory": trajectory.label,
            "learning_rate": trajectory.learning_rate,
            "step": snapshot.step,
            "adapter_id": trajectory.adapter_id,
            "adapter_model_sha256": snapshot.adapter_model_sha256,
            "generation_contract_sha256": contract,
            "humaneval_requests": len(he_responses),
            "mbpp_requests": len(mb_responses),
            "repository_holdout_requests": 0,
        }
        stage_path = output_dir / "generation-stage.json"
        stage_path.write_text(json.dumps(stage, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        stage_paths.append(stage_path)
    return tuple(stage_paths)


def _load_stage(output_dir: Path) -> dict[str, object]:
    try:
        value: object = json.loads(
            (output_dir / "generation-stage.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise MinimumInterventionEvaluationError("P9-004C generation stage is unreadable") from exc
    return _mapping(value, context="P9-004C generation stage")


def score_checkpoint(
    *,
    label: str,
    step: int,
    repo_root: Path = Path("."),
) -> DevelopmentScore:
    """Score one transported checkpoint on development HumanEval+MBPP only."""

    manifest = load_development_manifest(repo_root)
    trajectory = _trajectory(label, repo_root=repo_root)
    snapshot = trajectory.snapshot(step)
    evaluation, settings, base_model, system_prompt = _evaluation_context(trajectory)
    output_dir = _checkpoint_output(trajectory, step)
    stage = _load_stage(output_dir)
    if (
        stage.get("task_id") != "P9-004"
        or stage.get("stage") != "development-generation"
        or stage.get("trajectory") != label
        or stage.get("step") != step
        or stage.get("adapter_model_sha256") != snapshot.adapter_model_sha256
        or stage.get("development_manifest_sha256") != _EXPECTED_DEVELOPMENT_SHA256
        or stage.get("checkpoint_registry_sha256") != _EXPECTED_CHECKPOINT_REGISTRY_SHA256
        or stage.get("repository_holdout_requests") != 0
    ):
        raise MinimumInterventionEvaluationError("P9-004C generation stage identity drifted")
    expected_contract = _generation_contract(
        trajectory=trajectory,
        snapshot=snapshot,
        base_model=base_model,
        settings=settings,
        system_prompt=system_prompt,
    )
    if stage.get("generation_contract_sha256") != expected_contract:
        raise MinimumInterventionEvaluationError("P9-004C generation contract drifted")

    harness = DirectExecutionHarness(allow_reduced_isolation=True)
    humaneval, he, mbpp, mb = _load_development_problems(
        evaluation, base_model, trajectory, settings, manifest, harness=harness
    )
    checkpoint_only = _CheckpointOnlyGenerator()
    he_responses = _generate_items(
        suite_id="humaneval-development",
        prompts=tuple(
            (problem.task_id, humaneval.prompt_for(problem).user_content) for problem in he
        ),
        generator=checkpoint_only,
        system_prompt=system_prompt,
        generation_contract=expected_contract,
        output_dir=output_dir,
    )
    mb_responses = _generate_items(
        suite_id="mbpp-development",
        prompts=tuple((problem.task_id, mbpp.prompt_for(problem).user_content) for problem in mb),
        generator=checkpoint_only,
        system_prompt=system_prompt,
        generation_contract=expected_contract,
        output_dir=output_dir,
    )
    he_result = humaneval.evaluate_suite(
        he,
        tuple(
            HumanEvalCompletion(
                task_id=problem.task_id,
                generated_text=response.generated_text,
                generation=response.generation,
            )
            for problem, response in zip(he, he_responses, strict=True)
        ),
    )
    mb_result = mbpp.evaluate_suite(
        mb,
        tuple(
            MBPPCompletion(
                task_id=problem.task_id,
                generated_text=response.generated_text,
                generation=response.generation,
            )
            for problem, response in zip(mb, mb_responses, strict=True)
        ),
    )
    if he_result.aggregate.harness_errors or mb_result.aggregate.harness_errors:
        raise MinimumInterventionEvaluationError(
            "P9-004C direct scoring encountered harness errors"
        )
    humaneval.write_artifacts(he_result, output_dir / "humaneval-development")
    mbpp.write_artifacts(mb_result, output_dir / "mbpp-development")
    he_passed = he_result.aggregate.passed
    mb_passed = mb_result.aggregate.passed
    combined = he_passed + mb_passed
    eligible = (
        combined >= _EXPECTED_MIN_COMBINED
        and he_passed >= _EXPECTED_BASE_HE
        and mb_passed >= _EXPECTED_BASE_MBPP
    )
    score = DevelopmentScore(
        label=trajectory.label,
        learning_rate=trajectory.learning_rate,
        step=step,
        humaneval_passed=he_passed,
        humaneval_total=_EXPECTED_HE_TOTAL,
        mbpp_passed=mb_passed,
        mbpp_total=_EXPECTED_MBPP_TOTAL,
        combined_passed=combined,
        combined_total=_EXPECTED_COMBINED_TOTAL,
        eligible=eligible,
    )
    payload = {
        "schema_version": 1,
        "task_id": "P9-004",
        "stage": "development-score",
        **asdict(score),
        "base": {
            "humaneval_passed": _EXPECTED_BASE_HE,
            "mbpp_passed": _EXPECTED_BASE_MBPP,
            "combined_passed": _EXPECTED_BASE_COMBINED,
        },
        "delta": {
            "humaneval_passed": he_passed - _EXPECTED_BASE_HE,
            "mbpp_passed": mb_passed - _EXPECTED_BASE_MBPP,
            "combined_passed": combined - _EXPECTED_BASE_COMBINED,
        },
        "development_manifest_sha256": _EXPECTED_DEVELOPMENT_SHA256,
        "membership_sha256": _EXPECTED_MEMBERSHIP_SHA256,
        "checkpoint_registry_sha256": _EXPECTED_CHECKPOINT_REGISTRY_SHA256,
        "adapter_model_sha256": snapshot.adapter_model_sha256,
        "repository_holdout_evaluated": False,
    }
    (output_dir / "development-score.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return score


def _score_from_payload(value: object, *, context: str) -> DevelopmentScore:
    row = _mapping(value, context=context)
    if row.get("task_id") != "P9-004" or row.get("stage") != "development-score":
        raise MinimumInterventionEvaluationError(f"{context} identity drifted")
    if row.get("repository_holdout_evaluated") is not False:
        raise MinimumInterventionEvaluationError(f"{context} touched qualification holdout")
    eligible_value = row.get("eligible")
    if not isinstance(eligible_value, bool):
        raise MinimumInterventionEvaluationError(f"{context}.eligible must be boolean")
    score = DevelopmentScore(
        label=_string(row, "label", context=context),
        learning_rate=_number(row, "learning_rate", context=context),
        step=_integer(row, "step", context=context),
        humaneval_passed=_integer(row, "humaneval_passed", context=context),
        humaneval_total=_integer(row, "humaneval_total", context=context),
        mbpp_passed=_integer(row, "mbpp_passed", context=context),
        mbpp_total=_integer(row, "mbpp_total", context=context),
        combined_passed=_integer(row, "combined_passed", context=context),
        combined_total=_integer(row, "combined_total", context=context),
        eligible=eligible_value,
    )
    expected_eligible = (
        score.combined_passed >= _EXPECTED_MIN_COMBINED
        and score.humaneval_passed >= _EXPECTED_BASE_HE
        and score.mbpp_passed >= _EXPECTED_BASE_MBPP
    )
    if (
        score.humaneval_total != _EXPECTED_HE_TOTAL
        or score.mbpp_total != _EXPECTED_MBPP_TOTAL
        or score.combined_total != _EXPECTED_COMBINED_TOTAL
        or score.combined_passed != score.humaneval_passed + score.mbpp_passed
        or score.eligible is not expected_eligible
    ):
        raise MinimumInterventionEvaluationError(f"{context} score arithmetic/policy is invalid")
    return score


def select_development_candidate(scores: Sequence[DevelopmentScore]) -> DevelopmentScore | None:
    """Apply the precommitted development gate and deterministic tie breakers."""

    items = tuple(scores)
    identities = {(item.label, item.step) for item in items}
    expected = {(label, step) for label in _EXPECTED_LABELS for step in _EXPECTED_STEPS}
    if len(items) != 25 or identities != expected:
        raise MinimumInterventionEvaluationError(
            "P9-004C selection requires exactly the frozen 25 scores"
        )
    lr_by_label = dict(zip(_EXPECTED_LABELS, _EXPECTED_LRS, strict=True))
    if any(item.learning_rate != lr_by_label[item.label] for item in items):
        raise MinimumInterventionEvaluationError("P9-004C score learning-rate identity drifted")
    eligible = tuple(item for item in items if item.eligible)
    if not eligible:
        return None
    return sorted(
        eligible,
        key=lambda item: (-item.combined_passed, item.step, item.learning_rate),
    )[0]


def select_from_root(scores_root: Path, *, repo_root: Path = Path(".")) -> dict[str, object]:
    load_development_manifest(repo_root)
    load_checkpoint_registry(repo_root)
    files = sorted(scores_root.rglob("development-score.json"))
    scores: list[DevelopmentScore] = []
    for index, path in enumerate(files):
        try:
            value: object = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MinimumInterventionEvaluationError(
                f"could not read score artifact {path}"
            ) from exc
        scores.append(_score_from_payload(value, context=f"score[{index}]"))
    selected = select_development_candidate(scores)
    payload: dict[str, object] = {
        "schema_version": 1,
        "task_id": "P9-004",
        "stage": "development-selection",
        "development_manifest_sha256": _EXPECTED_DEVELOPMENT_SHA256,
        "membership_sha256": _EXPECTED_MEMBERSHIP_SHA256,
        "checkpoint_registry_sha256": _EXPECTED_CHECKPOINT_REGISTRY_SHA256,
        "base": {
            "humaneval_passed": _EXPECTED_BASE_HE,
            "mbpp_passed": _EXPECTED_BASE_MBPP,
            "combined_passed": _EXPECTED_BASE_COMBINED,
        },
        "scores": [
            asdict(item)
            for item in sorted(
                scores, key=lambda item: (_EXPECTED_LABELS.index(item.label), item.step)
            )
        ],
        "selected": None if selected is None else asdict(selected),
        "qualification_authorized": selected is not None,
        "repository_holdout_evaluated": False,
        "no_candidate_action": (
            "stop P0 LR/rank/epoch search and prioritize dataset/objective redesign"
            if selected is None
            else None
        ),
    }
    output = scores_root / "p9-004c-development-selection.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("--repo-root", type=Path, default=Path("."))
    generate_parser = sub.add_parser("generate-trajectory")
    generate_parser.add_argument("--label", required=True)
    generate_parser.add_argument("--training-output", type=Path, required=True)
    generate_parser.add_argument("--repo-root", type=Path, default=Path("."))
    generate_parser.add_argument("--device-index", type=int, default=0)
    score_parser = sub.add_parser("score")
    score_parser.add_argument("--label", required=True)
    score_parser.add_argument("--step", type=int, required=True)
    score_parser.add_argument("--repo-root", type=Path, default=Path("."))
    select_parser = sub.add_parser("select")
    select_parser.add_argument("--scores-root", type=Path, required=True)
    select_parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args(argv)

    if args.command == "validate":
        manifest = load_development_manifest(args.repo_root)
        registry = load_checkpoint_registry(args.repo_root)
        print(
            json.dumps(
                {
                    "development_manifest_sha256": _EXPECTED_DEVELOPMENT_SHA256,
                    "membership_sha256": manifest["membership_sha256"],
                    "checkpoint_registry_sha256": _EXPECTED_CHECKPOINT_REGISTRY_SHA256,
                    "trajectory_count": len(registry),
                    "checkpoint_count": sum(len(item.snapshots) for item in registry),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "generate-trajectory":
        paths = generate_trajectory(
            label=args.label,
            training_output=args.training_output,
            repo_root=args.repo_root,
            device_index=args.device_index,
        )
        print(json.dumps([path.as_posix() for path in paths], indent=2))
        return 0
    if args.command == "score":
        score = score_checkpoint(label=args.label, step=args.step, repo_root=args.repo_root)
        print(json.dumps(asdict(score), indent=2, sort_keys=True))
        return 0
    if args.command == "select":
        print(
            json.dumps(
                select_from_root(args.scores_root, repo_root=args.repo_root),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    raise MinimumInterventionEvaluationError("unsupported P9-004C command")


def main() -> NoReturn:
    raise SystemExit(_main())


if __name__ == "__main__":
    main()
