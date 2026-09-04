"""P9-004 checkpointed low-LR minimum-intervention protocol validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, NoReturn, TypeAlias

import yaml

_PROTOCOL_PATH = Path("configs/train/python/p9_minimum_intervention_v1.yaml")
_CONTROL_CONFIG = "configs/train/python/p9_rank_r8.yaml"
_EXPECTED_PROTOCOL_SHA256 = "51a78114a799fdf8b4fa2fa977f7c3ae6189f697b5d26de3b54d6e83b1b4d47e"
_EXPECTED_FIXED_PAYLOAD_SHA256 = "2f804be5d5b780a65943a9040970a82389a3c5967b1a7133a37c7a7cadfb4cf9"
_EXPECTED_CHECKPOINT_STEPS = (50, 100, 250, 500, 1000)
_EXPECTED_CANDIDATES = (
    ("lr-1e-5", 0.00001, "configs/train/python/p9_min_lr_1e5.yaml"),
    ("lr-2e-5", 0.00002, "configs/train/python/p9_min_lr_2e5.yaml"),
    ("lr-5e-5", 0.00005, "configs/train/python/p9_min_lr_5e5.yaml"),
    ("lr-1e-4", 0.0001, "configs/train/python/p9_min_lr_1e4.yaml"),
    ("lr-2e-4", 0.0002, "configs/train/python/p9_min_lr_2e4.yaml"),
)
_ALLOWED_TRAINING_DIFFERENCES = frozenset({"adapter_id", "output_dir", "learning_rate"})
PartitionAssignment: TypeAlias = Literal["development", "qualification"]


class MinimumInterventionError(ValueError):
    """Raised when the P9-004 minimum-intervention experiment contract drifts."""


@dataclass(frozen=True, slots=True)
class LearningRateCandidate:
    """One exact learning-rate trajectory in the P9-004 grid."""

    label: str
    learning_rate: float
    config_path: str


@dataclass(frozen=True, slots=True)
class DevelopmentPartition:
    """Frozen rule separating tuning tasks from untouched qualification tasks."""

    source_suites: tuple[str, ...]
    qualification_only_suites: tuple[str, ...]
    hash_algorithm: str
    salt: str
    modulus: int
    development_remainders: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SelectionPolicy:
    """Precommitted development-set checkpoint selection rule."""

    primary_metric: str
    require_combined_improvement_over_base: bool
    require_no_suite_regression: bool
    tie_breakers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class QualificationPolicy:
    """Precommitted one-shot qualification rule for the selected checkpoint."""

    one_shot: bool
    evaluate_only_selected_checkpoint: bool
    require_combined_improvement_over_base: bool
    require_no_suite_regression: bool


@dataclass(frozen=True, slots=True)
class MinimumInterventionProtocol:
    """Frozen P9-004 experiment definition."""

    schema_version: int
    task_id: str
    study_id: str
    control_training_config: str
    trajectory_max_steps: int
    checkpoint_steps: tuple[int, ...]
    candidates: tuple[LearningRateCandidate, ...]
    development_partition: DevelopmentPartition
    selection: SelectionPolicy
    qualification: QualificationPolicy


@dataclass(frozen=True, slots=True)
class ValidatedCandidate:
    """One candidate proven to differ from r8 only in allowed study fields."""

    label: str
    learning_rate: float
    config_path: str
    config_sha256: str
    adapter_id: str
    output_dir: str


@dataclass(frozen=True, slots=True)
class MinimumInterventionValidation:
    """Machine-readable proof of the frozen P9-004 study contract."""

    schema_version: int
    task_id: str
    study_id: str
    protocol_path: str
    protocol_sha256: str
    control_training_config: str
    fixed_payload_sha256: str
    trajectory_max_steps: int
    checkpoint_steps: tuple[int, ...]
    candidates: tuple[ValidatedCandidate, ...]
    snapshot_count: int
    development_partition: DevelopmentPartition
    selection: SelectionPolicy
    qualification: QualificationPolicy


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise MinimumInterventionError(f"could not read {path}") from exc


def _mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise MinimumInterventionError(f"{context} must be a mapping")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise MinimumInterventionError(f"{context} keys must be strings")
        result[key] = item
    return result


def _load_yaml(path: Path, *, context: str) -> dict[str, object]:
    try:
        payload: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise MinimumInterventionError(f"could not read {context} {path}") from exc
    return _mapping(payload, context=context)


def _require_keys(
    mapping: Mapping[str, object],
    *,
    required: frozenset[str],
    context: str,
) -> None:
    unknown = sorted(set(mapping) - required)
    missing = sorted(required - set(mapping))
    if unknown:
        raise MinimumInterventionError(f"{context} contains unknown field(s): {', '.join(unknown)}")
    if missing:
        raise MinimumInterventionError(f"{context} is missing field(s): {', '.join(missing)}")


def _string(mapping: Mapping[str, object], key: str, *, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise MinimumInterventionError(f"{context}.{key} must be a non-empty string")
    return value


def _positive_int(mapping: Mapping[str, object], key: str, *, context: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MinimumInterventionError(f"{context}.{key} must be a positive integer")
    return value


def _positive_number(mapping: Mapping[str, object], key: str, *, context: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise MinimumInterventionError(f"{context}.{key} must be a positive number")
    return float(value)


def _boolean(mapping: Mapping[str, object], key: str, *, context: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise MinimumInterventionError(f"{context}.{key} must be a boolean")
    return value


def _sequence(value: object, *, context: str) -> tuple[object, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise MinimumInterventionError(f"{context} must be a sequence")
    return tuple(value)


def _string_sequence(value: object, *, context: str) -> tuple[str, ...]:
    items = _sequence(value, context=context)
    result: list[str] = []
    for index, item in enumerate(items):
        if not isinstance(item, str) or not item.strip():
            raise MinimumInterventionError(f"{context}[{index}] must be a non-empty string")
        result.append(item)
    if len(result) != len(set(result)):
        raise MinimumInterventionError(f"{context} must not contain duplicates")
    return tuple(result)


def _int_sequence(value: object, *, context: str, allow_zero: bool = False) -> tuple[int, ...]:
    items = _sequence(value, context=context)
    result: list[int] = []
    for index, item in enumerate(items):
        minimum = 0 if allow_zero else 1
        if isinstance(item, bool) or not isinstance(item, int) or item < minimum:
            qualifier = "non-negative" if allow_zero else "positive"
            raise MinimumInterventionError(f"{context}[{index}] must be a {qualifier} integer")
        result.append(item)
    if len(result) != len(set(result)):
        raise MinimumInterventionError(f"{context} must not contain duplicates")
    return tuple(result)


def _candidate_rows(value: object) -> tuple[dict[str, object], ...]:
    items = _sequence(value, context="learning_rate_candidates")
    if not items:
        raise MinimumInterventionError("learning_rate_candidates must not be empty")
    return tuple(
        _mapping(item, context=f"learning_rate_candidates[{index}]")
        for index, item in enumerate(items)
    )


def load_minimum_intervention_protocol(path: Path = _PROTOCOL_PATH) -> MinimumInterventionProtocol:
    """Load the strict P9-004 protocol without touching model or dataset artifacts."""

    root = _load_yaml(path, context="minimum intervention protocol")
    _require_keys(
        root,
        required=frozenset(
            {
                "schema_version",
                "task_id",
                "study_id",
                "control_training_config",
                "trajectory_max_steps",
                "checkpoint_steps",
                "learning_rate_candidates",
                "development_partition",
                "selection",
                "qualification",
            }
        ),
        context="minimum intervention protocol",
    )

    candidates: list[LearningRateCandidate] = []
    for index, row in enumerate(_candidate_rows(root["learning_rate_candidates"])):
        context = f"learning_rate_candidates[{index}]"
        _require_keys(
            row,
            required=frozenset({"label", "learning_rate", "config"}),
            context=context,
        )
        candidates.append(
            LearningRateCandidate(
                label=_string(row, "label", context=context),
                learning_rate=_positive_number(row, "learning_rate", context=context),
                config_path=_string(row, "config", context=context),
            )
        )

    partition_row = _mapping(root["development_partition"], context="development_partition")
    _require_keys(
        partition_row,
        required=frozenset(
            {
                "source_suites",
                "qualification_only_suites",
                "hash_algorithm",
                "salt",
                "modulus",
                "development_remainders",
            }
        ),
        context="development_partition",
    )
    partition = DevelopmentPartition(
        source_suites=_string_sequence(
            partition_row["source_suites"], context="development_partition.source_suites"
        ),
        qualification_only_suites=_string_sequence(
            partition_row["qualification_only_suites"],
            context="development_partition.qualification_only_suites",
        ),
        hash_algorithm=_string(partition_row, "hash_algorithm", context="development_partition"),
        salt=_string(partition_row, "salt", context="development_partition"),
        modulus=_positive_int(partition_row, "modulus", context="development_partition"),
        development_remainders=_int_sequence(
            partition_row["development_remainders"],
            context="development_partition.development_remainders",
            allow_zero=True,
        ),
    )

    selection_row = _mapping(root["selection"], context="selection")
    _require_keys(
        selection_row,
        required=frozenset(
            {
                "primary_metric",
                "require_combined_improvement_over_base",
                "require_no_suite_regression",
                "tie_breakers",
            }
        ),
        context="selection",
    )
    selection = SelectionPolicy(
        primary_metric=_string(selection_row, "primary_metric", context="selection"),
        require_combined_improvement_over_base=_boolean(
            selection_row, "require_combined_improvement_over_base", context="selection"
        ),
        require_no_suite_regression=_boolean(
            selection_row, "require_no_suite_regression", context="selection"
        ),
        tie_breakers=_string_sequence(
            selection_row["tie_breakers"], context="selection.tie_breakers"
        ),
    )

    qualification_row = _mapping(root["qualification"], context="qualification")
    _require_keys(
        qualification_row,
        required=frozenset(
            {
                "one_shot",
                "evaluate_only_selected_checkpoint",
                "require_combined_improvement_over_base",
                "require_no_suite_regression",
            }
        ),
        context="qualification",
    )
    qualification = QualificationPolicy(
        one_shot=_boolean(qualification_row, "one_shot", context="qualification"),
        evaluate_only_selected_checkpoint=_boolean(
            qualification_row, "evaluate_only_selected_checkpoint", context="qualification"
        ),
        require_combined_improvement_over_base=_boolean(
            qualification_row, "require_combined_improvement_over_base", context="qualification"
        ),
        require_no_suite_regression=_boolean(
            qualification_row, "require_no_suite_regression", context="qualification"
        ),
    )

    return MinimumInterventionProtocol(
        schema_version=_positive_int(
            root, "schema_version", context="minimum intervention protocol"
        ),
        task_id=_string(root, "task_id", context="minimum intervention protocol"),
        study_id=_string(root, "study_id", context="minimum intervention protocol"),
        control_training_config=_string(
            root, "control_training_config", context="minimum intervention protocol"
        ),
        trajectory_max_steps=_positive_int(
            root, "trajectory_max_steps", context="minimum intervention protocol"
        ),
        checkpoint_steps=_int_sequence(root["checkpoint_steps"], context="checkpoint_steps"),
        candidates=tuple(candidates),
        development_partition=partition,
        selection=selection,
        qualification=qualification,
    )


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _normalized_fixed_training_payload(
    mapping: Mapping[str, object], *, context: str
) -> dict[str, object]:
    payload = dict(mapping)
    for key in _ALLOWED_TRAINING_DIFFERENCES:
        if key not in payload:
            raise MinimumInterventionError(f"{context} is missing required training field {key}")
        payload.pop(key)
    return payload


def _validate_rank8_shape(mapping: Mapping[str, object], *, context: str) -> None:
    lora = _mapping(mapping.get("lora"), context=f"{context}.lora")
    if _positive_int(lora, "rank", context=f"{context}.lora") != 8:
        raise MinimumInterventionError(f"{context} must keep lora.rank=8")
    if _positive_int(lora, "alpha", context=f"{context}.lora") != 32:
        raise MinimumInterventionError(f"{context} must keep lora.alpha=32")


def _validate_partition(partition: DevelopmentPartition) -> None:
    if partition.source_suites != ("humaneval", "mbpp"):
        raise MinimumInterventionError("P9-004 development suites must remain HumanEval and MBPP")
    if partition.qualification_only_suites != ("repository_holdout",):
        raise MinimumInterventionError("repository holdout must remain qualification-only")
    if partition.hash_algorithm != "sha256":
        raise MinimumInterventionError("P9-004 partition hash must remain sha256")
    if partition.salt != "python-p9-004-development-v1":
        raise MinimumInterventionError("P9-004 partition salt drifted")
    if partition.modulus != 4 or partition.development_remainders != (0,):
        raise MinimumInterventionError("P9-004 must reserve exactly one quarter for development")


def _validate_selection(protocol: MinimumInterventionProtocol) -> None:
    selection = protocol.selection
    if selection.primary_metric != "combined_passed":
        raise MinimumInterventionError("P9-004 development selection must use combined_passed")
    if not selection.require_combined_improvement_over_base:
        raise MinimumInterventionError("P9-004 must require development improvement over base")
    if not selection.require_no_suite_regression:
        raise MinimumInterventionError("P9-004 must reject development suite regressions")
    if selection.tie_breakers != ("fewer_steps", "lower_learning_rate"):
        raise MinimumInterventionError("P9-004 development tie-breakers drifted")

    qualification = protocol.qualification
    if not qualification.one_shot or not qualification.evaluate_only_selected_checkpoint:
        raise MinimumInterventionError("P9-004 qualification must remain one-shot and winner-only")
    if not qualification.require_combined_improvement_over_base:
        raise MinimumInterventionError("P9-004 qualification must require improvement over base")
    if not qualification.require_no_suite_regression:
        raise MinimumInterventionError("P9-004 qualification must reject suite regressions")


def validate_minimum_intervention(
    protocol_path: Path = _PROTOCOL_PATH,
    *,
    repo_root: Path = Path("."),
) -> MinimumInterventionValidation:
    """Prove that P9-004 varies only LR and observes only precommitted checkpoints."""

    resolved_protocol = repo_root / protocol_path
    protocol_sha = _sha256(resolved_protocol)
    if protocol_sha != _EXPECTED_PROTOCOL_SHA256:
        raise MinimumInterventionError("P9-004 protocol SHA-256 does not match frozen v1")
    protocol = load_minimum_intervention_protocol(resolved_protocol)
    if protocol.schema_version != 1 or protocol.task_id != "P9-004":
        raise MinimumInterventionError("P9-004 protocol identity drifted")
    if protocol.study_id != "python-p9-minimum-intervention-v1":
        raise MinimumInterventionError("P9-004 study_id drifted")
    if protocol.control_training_config != _CONTROL_CONFIG:
        raise MinimumInterventionError("P9-004 must use the completed r8 configuration as control")
    if protocol.trajectory_max_steps != 1000:
        raise MinimumInterventionError("P9-004 trajectory horizon must remain exactly 1,000 steps")
    if protocol.checkpoint_steps != _EXPECTED_CHECKPOINT_STEPS:
        raise MinimumInterventionError(
            f"P9-004 checkpoints must remain exactly {_EXPECTED_CHECKPOINT_STEPS!r}"
        )
    observed_candidates = tuple(
        (item.label, item.learning_rate, item.config_path) for item in protocol.candidates
    )
    if observed_candidates != _EXPECTED_CANDIDATES:
        raise MinimumInterventionError("P9-004 learning-rate grid or ordering drifted")
    _validate_partition(protocol.development_partition)
    _validate_selection(protocol)

    control_path = repo_root / _CONTROL_CONFIG
    control = _load_yaml(control_path, context="P9-004 r8 control config")
    _validate_rank8_shape(control, context="P9-004 r8 control config")
    control_fixed = _normalized_fixed_training_payload(control, context="P9-004 r8 control config")
    fixed_sha = hashlib.sha256(_canonical_json(control_fixed).encode("utf-8")).hexdigest()
    if fixed_sha != _EXPECTED_FIXED_PAYLOAD_SHA256:
        raise MinimumInterventionError("P9-004 fixed payload drifted from completed r8 control")

    candidates: list[ValidatedCandidate] = []
    seen_adapters: set[str] = set()
    seen_outputs: set[str] = set()
    for candidate in protocol.candidates:
        config_path = repo_root / candidate.config_path
        mapping = _load_yaml(config_path, context=f"P9-004 {candidate.label} config")
        _validate_rank8_shape(mapping, context=f"P9-004 {candidate.label} config")
        fixed = _normalized_fixed_training_payload(
            mapping, context=f"P9-004 {candidate.label} config"
        )
        if fixed != control_fixed:
            raise MinimumInterventionError(
                f"P9-004 {candidate.label} changes a training variable besides learning rate"
            )
        learning_rate = _positive_number(
            mapping, "learning_rate", context=f"P9-004 {candidate.label} config"
        )
        if learning_rate != candidate.learning_rate:
            raise MinimumInterventionError(
                f"P9-004 {candidate.label} config learning rate does not match protocol"
            )
        expected_suffix = f"p9-min-{candidate.label}"
        adapter_id = _string(mapping, "adapter_id", context=f"P9-004 {candidate.label} config")
        output_dir = _string(mapping, "output_dir", context=f"P9-004 {candidate.label} config")
        if adapter_id != f"language/python/{expected_suffix}":
            raise MinimumInterventionError(f"P9-004 {candidate.label} adapter_id is not canonical")
        if output_dir != f"artifacts/train/python/{expected_suffix}":
            raise MinimumInterventionError(f"P9-004 {candidate.label} output_dir is not canonical")
        if adapter_id in seen_adapters or output_dir in seen_outputs:
            raise MinimumInterventionError(
                "P9-004 candidate identities/output paths must be unique"
            )
        seen_adapters.add(adapter_id)
        seen_outputs.add(output_dir)
        candidates.append(
            ValidatedCandidate(
                label=candidate.label,
                learning_rate=learning_rate,
                config_path=candidate.config_path,
                config_sha256=_sha256(config_path),
                adapter_id=adapter_id,
                output_dir=output_dir,
            )
        )

    return MinimumInterventionValidation(
        schema_version=1,
        task_id=protocol.task_id,
        study_id=protocol.study_id,
        protocol_path=str(protocol_path),
        protocol_sha256=protocol_sha,
        control_training_config=protocol.control_training_config,
        fixed_payload_sha256=fixed_sha,
        trajectory_max_steps=protocol.trajectory_max_steps,
        checkpoint_steps=protocol.checkpoint_steps,
        candidates=tuple(candidates),
        snapshot_count=len(candidates) * len(protocol.checkpoint_steps),
        development_partition=protocol.development_partition,
        selection=protocol.selection,
        qualification=protocol.qualification,
    )


def partition_task(
    partition: DevelopmentPartition,
    *,
    suite: str,
    task_id: str,
) -> PartitionAssignment:
    """Assign one benchmark task without consulting model outputs or scores."""

    if not task_id:
        raise MinimumInterventionError("task_id must not be empty")
    if suite in partition.qualification_only_suites:
        return "qualification"
    if suite not in partition.source_suites:
        raise MinimumInterventionError(f"suite {suite!r} is not part of the P9-004 partition")
    payload = f"{partition.salt}\0{suite}\0{task_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    remainder = int.from_bytes(digest[:8], "big") % partition.modulus
    if remainder in partition.development_remainders:
        return "development"
    return "qualification"


def minimum_intervention_validation_json(report: MinimumInterventionValidation) -> str:
    """Serialize P9-004 protocol validation deterministically."""

    return json.dumps(asdict(report), indent=2, sort_keys=True) + "\n"


def minimum_intervention_main(argv: Sequence[str] | None = None) -> int:
    """Validate and print the frozen P9-004 protocol."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=_PROTOCOL_PATH)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    report = validate_minimum_intervention(args.protocol, repo_root=args.repo_root)
    print(minimum_intervention_validation_json(report), end="")
    return 0


def main() -> NoReturn:
    raise SystemExit(minimum_intervention_main())


if __name__ == "__main__":
    main()
