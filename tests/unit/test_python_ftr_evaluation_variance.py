"""CPU-only FTR-104 variance characterization and effect-gate tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from tiny_qwen_coder.evaluation.python_ftr_evaluation_variance import (
    FTREvaluationVarianceError,
    assess_paired_effect,
    audit_evaluation_variance,
    characterize_repeated_outcomes,
    diagnostic_subset,
    exact_one_sided_discordant_p_value,
    theoretical_minimum_net_improvement,
    variance_aware_minimum_passes,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SHA = "87ae718704d86ce2fe17bb821fc26ce6e82974de"


def test_canonical_variance_audit_passes_and_rejects_plus_one_of_175() -> None:
    report = audit_evaluation_variance(repo_root=_REPO_ROOT, source_git_sha=_SHA)

    assert report["variance_characterization_passed"] is True
    assert report["checks"]["deterministic_generation_preferred_and_enforced"] is True
    assert report["repeated_control_evidence"]["combined_pass_counts"] == [424, 424, 424]
    assert report["repeated_control_evidence"]["observed_task_outcome_flips"] == 0
    reassessment = report["historical_p9_007_reassessment"]
    assert reassessment["net_delta_tasks"] == 1
    assert reassessment["minimum_net_delta_tasks"] == 7
    assert reassessment["minimum_combined_passes"] == 110
    assert reassessment["strong_standalone_evidence"] is False


def test_bonferroni_effect_floor_scales_with_checkpoint_selection() -> None:
    assert theoretical_minimum_net_improvement(comparisons=1) == 5
    assert theoretical_minimum_net_improvement(comparisons=4) == 7
    assert theoretical_minimum_net_improvement(comparisons=25) == 9
    assert (
        variance_aware_minimum_passes(
            base_passed=103,
            total_tasks=175,
            comparisons=4,
        )
        == 110
    )
    assert (
        variance_aware_minimum_passes(
            base_passed=103,
            total_tasks=175,
            comparisons=25,
        )
        == 112
    )


def test_paired_exact_gate_requires_task_level_evidence_after_selection() -> None:
    base = {f"task-{index}": False for index in range(20)}
    six_gain = dict(base)
    for index in range(6):
        six_gain[f"task-{index}"] = True
    seven_gain = dict(base)
    for index in range(7):
        seven_gain[f"task-{index}"] = True

    weak = assess_paired_effect(
        base_outcomes=base,
        candidate_outcomes=six_gain,
        comparisons=4,
    )
    strong = assess_paired_effect(
        base_outcomes=base,
        candidate_outcomes=seven_gain,
        comparisons=4,
    )

    assert weak.net_improvement == 6
    assert weak.adjusted_p_value == pytest.approx(0.0625)
    assert weak.strong_standalone_evidence is False
    assert strong.net_improvement == 7
    assert strong.adjusted_p_value == pytest.approx(0.03125)
    assert strong.strong_standalone_evidence is True
    assert exact_one_sided_discordant_p_value(improvements=1, regressions=0) == 0.5


def test_repeated_run_variance_quantifies_task_flips_and_score_variance() -> None:
    stable = {"a": True, "b": False, "c": True}
    zero = characterize_repeated_outcomes((stable, dict(stable), dict(stable)))
    assert zero.passed_per_run == (2, 2, 2)
    assert zero.score_range_tasks == 0
    assert zero.score_population_variance_tasks == 0.0
    assert zero.unstable_task_ids == ()
    assert zero.task_flip_rate == 0.0

    changed = dict(stable)
    changed["b"] = True
    variance = characterize_repeated_outcomes((stable, changed, changed))
    assert variance.passed_per_run == (2, 3, 3)
    assert variance.score_range_tasks == 1
    assert variance.unstable_task_ids == ("b",)
    assert variance.task_flip_observations == 1
    assert variance.task_flip_rate == pytest.approx(1 / 6)


def test_diagnostic_subset_is_stable_prompt_free_and_bounded() -> None:
    first = diagnostic_subset(_REPO_ROOT)
    second = diagnostic_subset(_REPO_ROOT)

    assert first == second
    assert len(first) == 32
    assert len(set(first)) == 32
    assert all(item.startswith(("HumanEval/", "MBPP/", "repository-holdout/")) for item in first)

    with pytest.raises(FTREvaluationVarianceError, match="subset size"):
        diagnostic_subset(_REPO_ROOT, size=0)
