"""Frozen protocol validation for the P9-007C distilled-data trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NoReturn, cast

import yaml

from tiny_qwen_coder.evaluation.python_minimum_intervention import load_development_manifest
from tiny_qwen_coder.training.plan import resolve_adapter_training_plan

_PROTOCOL_PATH = Path("configs/train/python/p9_distilled_v4_2000_trajectory_v1.yaml")
_EXPECTED_PROTOCOL_SHA256 = "a77ac259c144a9eec8160854451c758b0927f6761801f553e3782117284a81d4"
_EXPECTED_TRAINING_CONFIG = "configs/train/python/p9_distilled_v4_2000_r8_lr1e5.yaml"
_EXPECTED_CORPUS_EVIDENCE = "docs/evidence/P9_007_DISTILLED_V4_2000_CORPUS.json"
_EXPECTED_DEVELOPMENT_MANIFEST = "configs/eval/python/p9_minimum_intervention_development_v1.json"
_EXPECTED_DEVELOPMENT_MANIFEST_SHA256 = (
    "260682d773640b28673357c7a441474656bbaffc67f0b5fcbe45eca3a07283de"
)
_EXPECTED_DATASET_MANIFEST_SHA256 = (
    "7e299de17cbb62bff3cf5f50c61ad6ad30ca571c9297b9935ff1307cd52e0eb7"
)
_EXPECTED_SOURCE_OUTPUT_SHA256 = (
    "7f07f7253e98bf8ed295b12d72ebdf03c44b2e08a8d9f74aaa02dce5b760d966"
)
_EXPECTED_TRAIN_RECORDS = 1479
_EXPECTED_VALIDATION_RECORDS = 78
_EXPECTED_ACCEPTED_RECORDS = 1557
_EXPECTED_EFFECTIVE_BATCH_SIZE = 8
_EXPECTED_MAX_STEPS = 185
_EXPECTED_CHECKPOINT_STEPS = (25, 50, 100, 185)
_EXPECTED_ADAPTER_ID = "language/python/p9-distilled-v4-2000-r8-lr1e5"
_EXPECTED_OUTPUT_DIR = "artifacts/train/python/p9-distilled-v4-2000-r8-lr1e5"


class DistilledTrajectoryError(ValueError):
    """Raised when the P9-007C experiment contract drifts."""


@dataclass(frozen=True, slots=True)
class DistilledSelectionPolicy:
    primary_metric: str
    minimum_combined_passed: int
    minimum_humaneval_passed: int
    minimum_mbpp_passed: int
    tie_breakers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DistilledQualificationPolicy:
    one_shot: bool
    evaluate_only_selected_checkpoint: bool
    repository_holdout_qualification_only: bool


@dataclass(frozen=True, slots=True)
class DistilledTrajectoryValidation:
    schema_version: int
    task_id: str
    study_id: str
    protocol_path: str
    protocol_sha256: str
    training_config: str
    training_config_sha256: str
    corpus_evidence: str
    corpus_evidence_sha256: str
    dataset_manifest_sha256: str
    source_output_sha256: str
    accepted_records: int
    train_records: int
    validation_records: int
    effective_batch_size: int
    derived_one_pass_steps: int
    trajectory_max_steps: int
    checkpoint_steps: tuple[int, ...]
    adapter_id: str
    output_dir: str
    development_manifest: str
    development_manifest_sha256: str
    selection: DistilledSelectionPolicy
    qualification: DistilledQualificationPolicy


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise DistilledTrajectoryError(f"could not read {path}") from exc


def _mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise DistilledTrajectoryError(f"{context} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise DistilledTrajectoryError(f"{context} keys must be strings")
    return cast(dict[str, object], dict(value))


def _load_yaml(path: Path, *, context: str) -> dict[str, object]:
    try:
        value: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DistilledTrajectoryError(f"could not read {context} {path}") from exc
    return _mapping(value, context=context)


def _load_json(path: Path, *, context: str) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DistilledTrajectoryError(f"could not read {context} {path}") from exc
    return _mapping(value, context=context)


def _require_keys(mapping: Mapping[str, object], expected: frozenset[str], *, context: str) -> None:
    unknown = sorted(set(mapping) - expected)
    missing = sorted(expected - set(mapping))
    if unknown:
        raise DistilledTrajectoryError(f"{context} contains unknown fields: {', '.join(unknown)}")
    if missing:
        raise DistilledTrajectoryError(f"{context} is missing fields: {', '.join(missing)}")


def _string(mapping: Mapping[str, object], key: str, *, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise DistilledTrajectoryError(f"{context}.{key} must be a non-empty string")
    return value


def _integer(mapping: Mapping[str, object], key: str, *, context: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise DistilledTrajectoryError(f"{context}.{key} must be an integer")
    return value


def _boolean(mapping: Mapping[str, object], key: str, *, context: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise DistilledTrajectoryError(f"{context}.{key} must be a boolean")
    return value


def _string_sequence(value: object, *, context: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise DistilledTrajectoryError(f"{context} must be a sequence")
    items = tuple(value)
    if any(not isinstance(item, str) or not item for item in items):
        raise DistilledTrajectoryError(f"{context} must contain non-empty strings")
    return cast(tuple[str, ...], items)


def _int_sequence(value: object, *, context: str) -> tuple[int, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise DistilledTrajectoryError(f"{context} must be a sequence")
    items = tuple(value)
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in items):
        raise DistilledTrajectoryError(f"{context} must contain positive integers")
    resolved = cast(tuple[int, ...], items)
    if len(resolved) != len(set(resolved)):
        raise DistilledTrajectoryError(f"{context} must not contain duplicates")
    return resolved


def _selection(row: Mapping[str, object]) -> DistilledSelectionPolicy:
    expected = frozenset(
        {
            "primary_metric",
            "minimum_combined_passed",
            "minimum_humaneval_passed",
            "minimum_mbpp_passed",
            "tie_breakers",
        }
    )
    _require_keys(row, expected, context="selection")
    return DistilledSelectionPolicy(
        primary_metric=_string(row, "primary_metric", context="selection"),
        minimum_combined_passed=_integer(row, "minimum_combined_passed", context="selection"),
        minimum_humaneval_passed=_integer(row, "minimum_humaneval_passed", context="selection"),
        minimum_mbpp_passed=_integer(row, "minimum_mbpp_passed", context="selection"),
        tie_breakers=_string_sequence(row.get("tie_breakers"), context="selection.tie_breakers"),
    )


def _qualification(row: Mapping[str, object]) -> DistilledQualificationPolicy:
    expected = frozenset(
        {
            "one_shot",
            "evaluate_only_selected_checkpoint",
            "repository_holdout_qualification_only",
        }
    )
    _require_keys(row, expected, context="qualification")
    return DistilledQualificationPolicy(
        one_shot=_boolean(row, "one_shot", context="qualification"),
        evaluate_only_selected_checkpoint=_boolean(
            row, "evaluate_only_selected_checkpoint", context="qualification"
        ),
        repository_holdout_qualification_only=_boolean(
            row, "repository_holdout_qualification_only", context="qualification"
        ),
    )


def _validate_training_shape(training: Mapping[str, object]) -> tuple[int, int]:
    required = {
        "adapter_id": _EXPECTED_ADAPTER_ID,
        "output_dir": _EXPECTED_OUTPUT_DIR,
        "training_mode": "qlora_4bit",
        "compute_dtype": "bfloat16",
        "loss_mode": "assistant_only",
    }
    for key, expected in required.items():
        if training.get(key) != expected:
            raise DistilledTrajectoryError(f"training config {key} drifted")
    if training.get("learning_rate") != 0.00001:
        raise DistilledTrajectoryError("training config learning_rate must remain 1e-5")
    if training.get("epochs") != 1.0:
        raise DistilledTrajectoryError("training config epochs must remain 1.0")
    lora = _mapping(training.get("lora"), context="training config.lora")
    if lora.get("rank") != 8 or lora.get("alpha") != 32:
        raise DistilledTrajectoryError("training config must remain LoRA r8/alpha32")
    if training.get("micro_batch_size") != 1 or training.get("gradient_accumulation_steps") != 8:
        raise DistilledTrajectoryError("training config effective-batch shape drifted")
    micro_batch = _integer(training, "micro_batch_size", context="training config")
    accumulation = _integer(training, "gradient_accumulation_steps", context="training config")
    return micro_batch, accumulation


def validate_distilled_trajectory(
    *, repo_root: Path = Path(".")
) -> DistilledTrajectoryValidation:
    """Validate the exact P9-007C corpus, optimizer shape, step horizon, and dev gate."""

    protocol_path = repo_root / _PROTOCOL_PATH
    protocol_sha = _sha256(protocol_path)
    if protocol_sha != _EXPECTED_PROTOCOL_SHA256:
        raise DistilledTrajectoryError("P9-007C protocol SHA-256 drifted")
    root = _load_yaml(protocol_path, context="P9-007C protocol")
    _require_keys(
        root,
        frozenset(
            {
                "schema_version",
                "task_id",
                "study_id",
                "training_config",
                "corpus_evidence",
                "expected_train_records",
                "effective_batch_size",
                "trajectory_max_steps",
                "checkpoint_steps",
                "development_manifest",
                "development_manifest_sha256",
                "selection",
                "qualification",
            }
        ),
        context="P9-007C protocol",
    )
    if root.get("schema_version") != 1 or root.get("task_id") != "P9-007C":
        raise DistilledTrajectoryError("P9-007C protocol identity drifted")
    if root.get("study_id") != "python-p9-distilled-v4-2000-trajectory-v1":
        raise DistilledTrajectoryError("P9-007C study identity drifted")

    training_config = _string(root, "training_config", context="P9-007C protocol")
    corpus_evidence = _string(root, "corpus_evidence", context="P9-007C protocol")
    development_manifest = _string(root, "development_manifest", context="P9-007C protocol")
    if training_config != _EXPECTED_TRAINING_CONFIG:
        raise DistilledTrajectoryError("P9-007C training config path drifted")
    if corpus_evidence != _EXPECTED_CORPUS_EVIDENCE:
        raise DistilledTrajectoryError("P9-007C corpus evidence path drifted")
    if development_manifest != _EXPECTED_DEVELOPMENT_MANIFEST:
        raise DistilledTrajectoryError("P9-007C development manifest path drifted")

    training_path = repo_root / training_config
    training = _load_yaml(training_path, context="P9-007C training config")
    micro_batch, accumulation = _validate_training_shape(training)
    effective_batch = micro_batch * accumulation
    if effective_batch != _EXPECTED_EFFECTIVE_BATCH_SIZE:
        raise DistilledTrajectoryError("P9-007C effective batch size drifted")
    if _integer(root, "effective_batch_size", context="P9-007C protocol") != effective_batch:
        raise DistilledTrajectoryError("P9-007C protocol effective batch disagrees with training config")

    evidence_path = repo_root / corpus_evidence
    evidence = _load_json(evidence_path, context="P9-007C corpus evidence")
    if evidence.get("corpus_id") != "qwen38-27b-v4-2000-salvage-v1":
        raise DistilledTrajectoryError("P9-007C corpus identity drifted")
    if evidence.get("dataset_manifest_sha256") != _EXPECTED_DATASET_MANIFEST_SHA256:
        raise DistilledTrajectoryError("P9-007C dataset manifest identity drifted")
    if evidence.get("source_output_sha256") != _EXPECTED_SOURCE_OUTPUT_SHA256:
        raise DistilledTrajectoryError("P9-007C source output identity drifted")
    final_counts = _mapping(evidence.get("final_counts"), context="P9-007C final counts")
    counts = (
        _integer(final_counts, "accepted_records", context="P9-007C final counts"),
        _integer(final_counts, "train_records", context="P9-007C final counts"),
        _integer(final_counts, "validation_records", context="P9-007C final counts"),
    )
    if counts != (_EXPECTED_ACCEPTED_RECORDS, _EXPECTED_TRAIN_RECORDS, _EXPECTED_VALIDATION_RECORDS):
        raise DistilledTrajectoryError("P9-007C frozen corpus counts drifted")
    qualification_evidence = _mapping(evidence.get("qualification"), context="corpus qualification")
    if qualification_evidence.get("qualified") is not True:
        raise DistilledTrajectoryError("P9-007C corpus is not qualified")
    if qualification_evidence.get("contamination_status") != "clean":
        raise DistilledTrajectoryError("P9-007C corpus contamination status is not clean")

    train_records = _integer(root, "expected_train_records", context="P9-007C protocol")
    if train_records != _EXPECTED_TRAIN_RECORDS:
        raise DistilledTrajectoryError("P9-007C expected train count drifted")
    derived_steps = math.ceil(train_records / effective_batch)
    trajectory_max_steps = _integer(root, "trajectory_max_steps", context="P9-007C protocol")
    checkpoints = _int_sequence(root.get("checkpoint_steps"), context="checkpoint_steps")
    if derived_steps != _EXPECTED_MAX_STEPS or trajectory_max_steps != derived_steps:
        raise DistilledTrajectoryError("P9-007C trajectory is not exactly one derived data pass")
    if checkpoints != _EXPECTED_CHECKPOINT_STEPS or checkpoints[-1] != trajectory_max_steps:
        raise DistilledTrajectoryError("P9-007C checkpoint grid drifted")

    development_sha = _sha256(repo_root / development_manifest)
    if development_sha != _EXPECTED_DEVELOPMENT_MANIFEST_SHA256:
        raise DistilledTrajectoryError("P9-007C development manifest SHA-256 drifted")
    if root.get("development_manifest_sha256") != development_sha:
        raise DistilledTrajectoryError("P9-007C protocol development SHA-256 drifted")
    load_development_manifest(repo_root)

    selection = _selection(_mapping(root.get("selection"), context="selection"))
    if selection != DistilledSelectionPolicy(
        primary_metric="combined_passed",
        minimum_combined_passed=104,
        minimum_humaneval_passed=33,
        minimum_mbpp_passed=70,
        tie_breakers=("fewer_steps",),
    ):
        raise DistilledTrajectoryError("P9-007C development selection policy drifted")
    qualification = _qualification(_mapping(root.get("qualification"), context="qualification"))
    if qualification != DistilledQualificationPolicy(True, True, True):
        raise DistilledTrajectoryError("P9-007C qualification policy drifted")

    plan = resolve_adapter_training_plan(training_path)
    if plan.config.adapter_id != _EXPECTED_ADAPTER_ID:
        raise DistilledTrajectoryError("P9-007C resolved adapter identity drifted")
    if Path(plan.config.output_dir).as_posix() != _EXPECTED_OUTPUT_DIR:
        raise DistilledTrajectoryError("P9-007C resolved output path drifted")

    return DistilledTrajectoryValidation(
        schema_version=1,
        task_id="P9-007C",
        study_id="python-p9-distilled-v4-2000-trajectory-v1",
        protocol_path=_PROTOCOL_PATH.as_posix(),
        protocol_sha256=protocol_sha,
        training_config=training_config,
        training_config_sha256=_sha256(training_path),
        corpus_evidence=corpus_evidence,
        corpus_evidence_sha256=_sha256(evidence_path),
        dataset_manifest_sha256=_EXPECTED_DATASET_MANIFEST_SHA256,
        source_output_sha256=_EXPECTED_SOURCE_OUTPUT_SHA256,
        accepted_records=_EXPECTED_ACCEPTED_RECORDS,
        train_records=train_records,
        validation_records=_EXPECTED_VALIDATION_RECORDS,
        effective_batch_size=effective_batch,
        derived_one_pass_steps=derived_steps,
        trajectory_max_steps=trajectory_max_steps,
        checkpoint_steps=checkpoints,
        adapter_id=_EXPECTED_ADAPTER_ID,
        output_dir=_EXPECTED_OUTPUT_DIR,
        development_manifest=development_manifest,
        development_manifest_sha256=development_sha,
        selection=selection,
        qualification=qualification,
    )


def distilled_trajectory_validation_json(validation: DistilledTrajectoryValidation) -> str:
    """Serialize validation evidence deterministically."""

    return json.dumps(asdict(validation), indent=2, sort_keys=True) + "\n"


def distilled_trajectory_main(argv: Sequence[str] | None = None) -> NoReturn:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    print(distilled_trajectory_validation_json(validate_distilled_trajectory(repo_root=args.repo_root)), end="")
    raise SystemExit(0)


if __name__ == "__main__":
    distilled_trajectory_main()
