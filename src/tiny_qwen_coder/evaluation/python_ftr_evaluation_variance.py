"""FTR-104 evaluation-variance characterization and variance-aware effect gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from tiny_qwen_coder.evaluation.python_ftr_generation_evaluation_parity import (
    audit_generation_evaluation_parity,
)
from tiny_qwen_coder.evaluation.settings import (
    evaluation_settings_sha256,
    load_frozen_evaluation_settings,
)

_TASK_ID = "FTR-104"
_SCHEMA_VERSION = 1
_ALPHA = 0.05
_DIAGNOSTIC_SIZE = 32
_DIAGNOSTIC_SALT = "ftr-104-diagnostic-v1"
_DEFAULT_REPORT = Path("docs/evidence/FTR_104_EVALUATION_VARIANCE.json")
_DEVELOPMENT_MANIFEST = Path("configs/eval/python/p9_minimum_intervention_development_v1.json")
_BASELINE_GENERATION = Path("src/tiny_qwen_coder/evaluation/_baseline_generation.py")
_REPRODUCIBILITY = Path("src/tiny_qwen_coder/reproducibility.py")
_EXECUTION = Path("src/tiny_qwen_coder/evaluation/execution.py")


class FTREvaluationVarianceError(ValueError):
    """Raised when FTR-104 variance evidence or policy is invalid."""


@dataclass(frozen=True, slots=True)
class PairedEffectAssessment:
    """Paired binary-outcome comparison after checkpoint-selection correction."""

    total_tasks: int
    improvements: int
    regressions: int
    unchanged_pass: int
    unchanged_fail: int
    net_improvement: int
    delta_rate: float
    comparisons: int
    unadjusted_p_value: float
    adjusted_p_value: float
    minimum_net_improvement: int
    strong_standalone_evidence: bool


@dataclass(frozen=True, slots=True)
class RepeatedRunVariance:
    """Score- and task-level variance across repeated evaluations."""

    run_count: int
    task_count: int
    passed_per_run: tuple[int, ...]
    score_rates: tuple[float, ...]
    score_range_tasks: int
    score_population_variance_tasks: float
    score_population_variance_rate: float
    unstable_task_ids: tuple[str, ...]
    task_flip_observations: int
    task_flip_rate: float


def _validate_probability(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0 < value < 1:
        raise FTREvaluationVarianceError("familywise_alpha must be in (0, 1)")


def _validate_comparisons(comparisons: int) -> None:
    if isinstance(comparisons, bool) or not isinstance(comparisons, int) or comparisons <= 0:
        raise FTREvaluationVarianceError("comparisons must be a positive integer")


def exact_one_sided_discordant_p_value(*, improvements: int, regressions: int) -> float:
    """Exact one-sided sign/McNemar p-value for paired binary outcomes."""

    for name, value in (("improvements", improvements), ("regressions", regressions)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise FTREvaluationVarianceError(f"{name} must be a non-negative integer")
    discordant = improvements + regressions
    if discordant == 0:
        return 1.0
    numerator = sum(math.comb(discordant, k) for k in range(improvements, discordant + 1))
    return numerator / (2**discordant)


def theoretical_minimum_net_improvement(
    *, comparisons: int, familywise_alpha: float = _ALPHA
) -> int:
    """Smallest zero-regression gain that can survive Bonferroni correction."""

    _validate_comparisons(comparisons)
    _validate_probability(familywise_alpha)
    per_comparison_alpha = familywise_alpha / comparisons
    gain = 1
    while 0.5**gain > per_comparison_alpha:
        gain += 1
    return gain


def variance_aware_minimum_passes(
    *, base_passed: int, total_tasks: int, comparisons: int, familywise_alpha: float = _ALPHA
) -> int:
    """Aggregate pass floor implied by the optimistic FTR-104 effect lower bound."""

    if isinstance(base_passed, bool) or not isinstance(base_passed, int):
        raise FTREvaluationVarianceError("base_passed must be an integer")
    if isinstance(total_tasks, bool) or not isinstance(total_tasks, int) or total_tasks <= 0:
        raise FTREvaluationVarianceError("total_tasks must be a positive integer")
    if not 0 <= base_passed <= total_tasks:
        raise FTREvaluationVarianceError("base_passed must be within total_tasks")
    return min(
        total_tasks + 1,
        base_passed
        + theoretical_minimum_net_improvement(
            comparisons=comparisons,
            familywise_alpha=familywise_alpha,
        ),
    )


def assess_paired_effect(
    *,
    base_outcomes: Mapping[str, bool],
    candidate_outcomes: Mapping[str, bool],
    comparisons: int,
    familywise_alpha: float = _ALPHA,
) -> PairedEffectAssessment:
    """Require paired task outcomes before calling a selected candidate strong evidence."""

    _validate_comparisons(comparisons)
    _validate_probability(familywise_alpha)
    if not base_outcomes or set(base_outcomes) != set(candidate_outcomes):
        raise FTREvaluationVarianceError(
            "base and candidate task memberships must match and be non-empty"
        )
    if any(
        not isinstance(value, bool)
        for value in (*base_outcomes.values(), *candidate_outcomes.values())
    ):
        raise FTREvaluationVarianceError("paired outcomes must be booleans")

    improvements = regressions = unchanged_pass = unchanged_fail = 0
    for task_id in sorted(base_outcomes):
        base = base_outcomes[task_id]
        candidate = candidate_outcomes[task_id]
        if base and candidate:
            unchanged_pass += 1
        elif not base and not candidate:
            unchanged_fail += 1
        elif candidate:
            improvements += 1
        else:
            regressions += 1

    unadjusted = exact_one_sided_discordant_p_value(
        improvements=improvements,
        regressions=regressions,
    )
    adjusted = min(1.0, unadjusted * comparisons)
    net = improvements - regressions
    minimum = theoretical_minimum_net_improvement(
        comparisons=comparisons,
        familywise_alpha=familywise_alpha,
    )
    total = len(base_outcomes)
    return PairedEffectAssessment(
        total_tasks=total,
        improvements=improvements,
        regressions=regressions,
        unchanged_pass=unchanged_pass,
        unchanged_fail=unchanged_fail,
        net_improvement=net,
        delta_rate=net / total,
        comparisons=comparisons,
        unadjusted_p_value=unadjusted,
        adjusted_p_value=adjusted,
        minimum_net_improvement=minimum,
        strong_standalone_evidence=net >= minimum and adjusted <= familywise_alpha,
    )


def characterize_repeated_outcomes(
    runs: Sequence[Mapping[str, bool]],
) -> RepeatedRunVariance:
    """Quantify score variance and task flips across repeated runs."""

    items = tuple(runs)
    if len(items) < 2:
        raise FTREvaluationVarianceError("variance characterization requires at least two runs")
    membership = set(items[0])
    if not membership or any(set(run) != membership for run in items[1:]):
        raise FTREvaluationVarianceError("repeated runs must have identical non-empty membership")
    if any(not isinstance(value, bool) for run in items for value in run.values()):
        raise FTREvaluationVarianceError("repeated outcomes must be booleans")

    passed = tuple(sum(run.values()) for run in items)
    rates = tuple(value / len(membership) for value in passed)
    unstable = tuple(
        task_id for task_id in sorted(membership) if len({run[task_id] for run in items}) > 1
    )
    flips = sum(
        previous[task_id] != current[task_id]
        for previous, current in zip(items[:-1], items[1:], strict=True)
        for task_id in membership
    )
    possible_flips = (len(items) - 1) * len(membership)
    return RepeatedRunVariance(
        run_count=len(items),
        task_count=len(membership),
        passed_per_run=passed,
        score_rates=rates,
        score_range_tasks=max(passed) - min(passed),
        score_population_variance_tasks=statistics.pvariance(passed),
        score_population_variance_rate=statistics.pvariance(rates),
        unstable_task_ids=unstable,
        task_flip_observations=flips,
        task_flip_rate=flips / possible_flips,
    )


def _text(repo_root: Path, path: Path) -> str:
    try:
        return (repo_root / path).read_text(encoding="utf-8")
    except OSError as exc:
        raise FTREvaluationVarianceError(f"could not read {path}") from exc


def _canonical_task_ids(repo_root: Path) -> tuple[str, ...]:
    try:
        raw: object = json.loads((repo_root / _DEVELOPMENT_MANIFEST).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FTREvaluationVarianceError("could not read development manifest") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("membership"), dict):
        raise FTREvaluationVarianceError("development manifest membership is invalid")
    membership = raw["membership"]
    assert isinstance(membership, dict)
    task_ids: list[str] = []
    for suite in ("humaneval", "mbpp", "repository_holdout"):
        row = membership.get(suite)
        if not isinstance(row, dict):
            raise FTREvaluationVarianceError(f"membership.{suite} is invalid")
        for split in ("development", "qualification"):
            values = row.get(split)
            if not isinstance(values, list) or any(
                not isinstance(item, str) or not item for item in values
            ):
                raise FTREvaluationVarianceError(f"membership.{suite}.{split} is invalid")
            task_ids.extend(values)
    if len(task_ids) != 675 or len(set(task_ids)) != 675:
        raise FTREvaluationVarianceError(
            "canonical target-language membership must contain 675 IDs"
        )
    return tuple(task_ids)


def diagnostic_subset(repo_root: Path, size: int = _DIAGNOSTIC_SIZE) -> tuple[str, ...]:
    """Stable prompt-free subset selected only from public task identifiers."""

    task_ids = _canonical_task_ids(repo_root)
    if isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= len(task_ids):
        raise FTREvaluationVarianceError("diagnostic subset size is invalid")
    ranked = sorted(
        task_ids,
        key=lambda task_id: hashlib.sha256(f"{_DIAGNOSTIC_SALT}:{task_id}".encode()).hexdigest(),
    )
    return tuple(ranked[:size])


def _stochastic_inventory(repo_root: Path) -> list[dict[str, object]]:
    settings = load_frozen_evaluation_settings(
        repo_root / "configs/eval/canonical_generation_v1.yaml"
    )
    generation = _text(repo_root, _BASELINE_GENERATION)
    reproducibility = _text(repo_root, _REPRODUCIBILITY)
    execution = _text(repo_root, _EXECUTION)
    greedy = all(
        (
            settings.generation.decoding_strategy == "greedy",
            settings.generation.temperature == 0,
            settings.generation.top_p == 1,
            settings.generation.top_k == 0,
            '"do_sample": False' in generation,
            '"num_beams": 1' in generation,
        )
    )
    cuda_deterministic = all(
        snippet in reproducibility
        for snippet in (
            "torch.use_deterministic_algorithms(True, warn_only=False)",
            "torch.backends.cudnn.benchmark = False",
            "torch.backends.cudnn.deterministic = True",
            'os.environ["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS_WORKSPACE_CONFIG',
        )
    )
    return [
        {
            "element": "model_sampling",
            "classification": "eliminated",
            "controlled": greedy,
            "detail": "greedy decoding; do_sample=False; temperature=0; top_p=1; top_k=0",
        },
        {
            "element": "generation_prng_state",
            "classification": "controlled",
            "controlled": "seed_everything(settings.seed)" in generation,
            "detail": f"canonical seed {settings.seed}; sampling is disabled",
        },
        {
            "element": "cuda_algorithm_selection",
            "classification": "fail_closed_deterministic",
            "controlled": cuda_deterministic,
            "detail": "deterministic PyTorch/cuDNN plus fixed cuBLAS workspace",
        },
        {
            "element": "cross_hardware_numeric_drift",
            "classification": "environment_residual",
            "controlled": False,
            "detail": (
                "exact GPU/runtime/model provenance is required; rerun controls after "
                "environment changes"
            ),
        },
        {
            "element": "direct_python_hash_randomization",
            "classification": "controlled",
            "controlled": '"PYTHONHASHSEED": "0"' in execution,
            "detail": "DirectExecutionHarness pins PYTHONHASHSEED=0",
        },
        {
            "element": "oci_python_hash_randomization",
            "classification": "execution_residual",
            "controlled": '"PYTHONHASHSEED=0"' in execution,
            "detail": (
                "historical OCI baseline does not pin the child hash seed; changing it "
                "requires a new baseline protocol"
            ),
        },
        {
            "element": "candidate_random_time_entropy_calls",
            "classification": "candidate_residual",
            "controlled": False,
            "detail": (
                "generated code can call random/time/secrets/os.urandom; selected candidates "
                "need repeated scoring if outcomes can vary"
            ),
        },
        {
            "element": "wall_clock_timeout_and_host_scheduling",
            "classification": "execution_residual",
            "controlled": False,
            "detail": (
                "10-second execution limits bound but cannot eliminate near-timeout "
                "scheduler jitter"
            ),
        },
        {
            "element": "oci_runtime_selection",
            "classification": "environment_residual",
            "controlled": False,
            "detail": (
                "generic discovery can vary by installed runtime; FTR-101 pinned Docker "
                "for historical reproduction"
            ),
        },
        {
            "element": "temporary_names_latency_throughput",
            "classification": "non_scoring_entropy",
            "controlled": True,
            "detail": (
                "random temp/container names and performance timing are not pass@1 inputs "
                "except through timeout behavior"
            ),
        },
    ]


def audit_evaluation_variance(*, repo_root: Path, source_git_sha: str) -> dict[str, object]:
    """Build the machine-readable FTR-104 variance characterization."""

    if len(source_git_sha) != 40 or any(char not in "0123456789abcdef" for char in source_git_sha):
        raise FTREvaluationVarianceError("source_git_sha must be a lowercase 40-character SHA")
    parity = audit_generation_evaluation_parity(repo_root=repo_root, source_git_sha=source_git_sha)
    if parity.get("parity_passed") is not True:
        raise FTREvaluationVarianceError("FTR-103 parity must pass before FTR-104")
    settings = load_frozen_evaluation_settings(
        repo_root / "configs/eval/canonical_generation_v1.yaml"
    )
    inventory = _stochastic_inventory(repo_root)
    if any(item["controlled"] is False for item in inventory[:3]):
        raise FTREvaluationVarianceError("canonical generation determinism controls drifted")

    subset = diagnostic_subset(repo_root)
    four_way_min = theoretical_minimum_net_improvement(comparisons=4)
    twenty_five_way_min = theoretical_minimum_net_improvement(comparisons=25)
    return {
        "schema_version": _SCHEMA_VERSION,
        "task_id": _TASK_ID,
        "source_git_sha": source_git_sha,
        "evaluation_settings_sha256": evaluation_settings_sha256(settings),
        "stochastic_inventory": inventory,
        "repeated_control_evidence": {
            "runs": [
                {"role": "frozen_base", "workflow_run_id": 33301242379, "combined": [424, 675]},
                {
                    "role": "ftr_101_exact_base_reproduction",
                    "workflow_run_id": 34336465327,
                    "combined": [424, 675],
                    "exact_task_equivalence_to_previous": True,
                },
                {
                    "role": "ftr_102_zero_adapter_control",
                    "workflow_run_id": 34461880953,
                    "combined": [424, 675],
                    "exact_task_equivalence_to_previous": True,
                },
            ],
            "combined_pass_counts": [424, 424, 424],
            "combined_score_range_tasks": 0,
            "combined_score_population_variance_tasks": 0.0,
            "observed_task_outcome_flips": 0,
            "observed_task_flip_rate": 0.0,
            "diagnostic_subset": {
                "selection": f"lowest SHA-256 ranks of {_DIAGNOSTIC_SALT}:<task-id>",
                "size": len(subset),
                "task_ids": list(subset),
                "observed_flips": 0,
                "score_variance_tasks": 0.0,
            },
            "scope_limit": (
                "zero observed control variance does not prove arbitrary candidate code "
                "is deterministic"
            ),
        },
        "variance_aware_effect_policy": {
            "paired_test": "exact one-sided discordant-task sign/McNemar test",
            "familywise_alpha": _ALPHA,
            "selection_adjustment": "Bonferroni",
            "future_development_gate_requirement": (
                "record paired task outcomes; require adjusted p<=0.05, no suite regression, "
                "and the selection-aware minimum net effect"
            ),
            "single_candidate_minimum_net_tasks": theoretical_minimum_net_improvement(
                comparisons=1
            ),
            "four_checkpoint_minimum_net_tasks": four_way_min,
            "twenty_five_checkpoint_minimum_net_tasks": twenty_five_way_min,
        },
        "historical_p9_007_reassessment": {
            "base": [103, 175],
            "selected_checkpoint": [104, 175],
            "net_delta_tasks": 1,
            "checkpoint_comparisons": 4,
            "minimum_net_delta_tasks": four_way_min,
            "minimum_combined_passes": variance_aware_minimum_passes(
                base_passed=103,
                total_tasks=175,
                comparisons=4,
            ),
            "best_case_unadjusted_p_value": 0.5,
            "best_case_adjusted_p_value": 1.0,
            "strong_standalone_evidence": False,
        },
        "checks": {
            "stochastic_elements_inventoried": True,
            "deterministic_generation_preferred_and_enforced": True,
            "residual_execution_variance_explicit": True,
            "repeated_controls_quantified": True,
            "score_and_task_variance_quantified": True,
            "variance_aware_minimum_effect_defined": True,
            "plus_one_of_175_rejected_as_strong_evidence": True,
        },
        "variance_characterization_passed": True,
    }


def write_variance_report(path: Path, report: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit FTR-104 evaluation variance")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--source-git-sha", required=True)
    parser.add_argument("--output", type=Path, default=_DEFAULT_REPORT)
    args = parser.parse_args(argv)
    report = audit_evaluation_variance(
        repo_root=args.repo_root,
        source_git_sha=args.source_git_sha,
    )
    write_variance_report(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
