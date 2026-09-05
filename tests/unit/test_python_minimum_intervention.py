"""CPU contract tests for P9-004C development-only checkpoint evaluation."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tiny_qwen_coder.evaluation.python_minimum_intervention import (
    DevelopmentScore,
    MinimumInterventionEvaluationError,
    load_checkpoint_registry,
    load_development_manifest,
    select_development_candidate,
)

_LABELS = ("lr-1e-5", "lr-2e-5", "lr-5e-5", "lr-1e-4", "lr-2e-4")
_LRS = (0.00001, 0.00002, 0.00005, 0.0001, 0.0002)
_STEPS = (50, 100, 250, 500, 1000)


def _score(label: str, learning_rate: float, step: int) -> DevelopmentScore:
    return DevelopmentScore(
        label=label,
        learning_rate=learning_rate,
        step=step,
        humaneval_passed=33,
        humaneval_total=45,
        mbpp_passed=70,
        mbpp_total=130,
        combined_passed=103,
        combined_total=175,
        eligible=False,
    )


def _grid() -> list[DevelopmentScore]:
    return [
        _score(label, learning_rate, step)
        for label, learning_rate in zip(_LABELS, _LRS, strict=True)
        for step in _STEPS
    ]


def test_frozen_p9_004c_manifests_are_valid_and_keep_holdout_qualification_only() -> None:
    manifest = load_development_manifest()
    registry = load_checkpoint_registry()

    membership = manifest["membership"]
    assert isinstance(membership, dict)
    humaneval = membership["humaneval"]
    mbpp = membership["mbpp"]
    holdout = membership["repository_holdout"]
    assert isinstance(humaneval, dict)
    assert isinstance(mbpp, dict)
    assert isinstance(holdout, dict)
    assert len(humaneval["development"]) == 45
    assert len(mbpp["development"]) == 130
    assert holdout["development"] == []
    assert len(holdout["qualification"]) == 11
    assert len(registry) == 5
    assert sum(len(item.snapshots) for item in registry) == 25
    assert tuple(item.label for item in registry) == _LABELS
    assert all(tuple(snapshot.step for snapshot in item.snapshots) == _STEPS for item in registry)


def test_selection_requires_strict_combined_improvement_and_no_suite_regression() -> None:
    scores = _grid()
    scores[0] = replace(
        scores[0],
        humaneval_passed=33,
        mbpp_passed=71,
        combined_passed=104,
        eligible=True,
    )
    selected = select_development_candidate(scores)

    assert selected is not None
    assert selected.label == "lr-1e-5"
    assert selected.step == 50


def test_selection_rejects_combined_gain_with_suite_regression() -> None:
    scores = _grid()
    scores[0] = replace(
        scores[0],
        humaneval_passed=32,
        mbpp_passed=73,
        combined_passed=105,
        eligible=False,
    )

    assert select_development_candidate(scores) is None


def test_selection_tie_breaks_by_fewer_steps_then_lower_learning_rate() -> None:
    scores = _grid()
    first = next(
        index for index, item in enumerate(scores) if item.label == "lr-2e-5" and item.step == 100
    )
    second = next(
        index for index, item in enumerate(scores) if item.label == "lr-1e-5" and item.step == 100
    )
    later = next(
        index for index, item in enumerate(scores) if item.label == "lr-1e-5" and item.step == 250
    )
    for index in (first, second, later):
        scores[index] = replace(
            scores[index],
            humaneval_passed=34,
            mbpp_passed=72,
            combined_passed=106,
            eligible=True,
        )

    selected = select_development_candidate(scores)

    assert selected is not None
    assert selected.label == "lr-1e-5"
    assert selected.step == 100


def test_selection_refuses_incomplete_or_duplicate_grid() -> None:
    scores = _grid()
    with pytest.raises(MinimumInterventionEvaluationError, match="exactly the frozen 25"):
        select_development_candidate(scores[:-1])

    duplicated = list(scores)
    duplicated[-1] = duplicated[0]
    with pytest.raises(MinimumInterventionEvaluationError, match="exactly the frozen 25"):
        select_development_candidate(duplicated)
