"""CPU tests for P9-007C development selection and qualification isolation."""

from __future__ import annotations

import pytest

from tiny_qwen_coder.evaluation.python_distilled_trajectory import (
    DistilledDevelopmentScore,
    DistilledTrajectoryEvaluationError,
    select_development_candidate,
)
from tiny_qwen_coder.evaluation.python_minimum_intervention import load_development_manifest


def _score(
    step: int,
    *,
    humaneval: int,
    mbpp: int,
    eligible: bool,
) -> DistilledDevelopmentScore:
    return DistilledDevelopmentScore(
        label="distilled-v4-2000-r8-lr1e5",
        learning_rate=0.00001,
        step=step,
        humaneval_passed=humaneval,
        humaneval_total=45,
        mbpp_passed=mbpp,
        mbpp_total=130,
        combined_passed=humaneval + mbpp,
        combined_total=175,
        eligible=eligible,
    )


def test_distilled_selection_uses_combined_then_fewer_steps() -> None:
    scores = (
        _score(25, humaneval=33, mbpp=71, eligible=True),
        _score(50, humaneval=34, mbpp=70, eligible=True),
        _score(100, humaneval=34, mbpp=71, eligible=True),
        _score(185, humaneval=35, mbpp=70, eligible=True),
    )

    selected = select_development_candidate(scores)

    assert selected is not None
    assert selected.combined_passed == 105
    assert selected.step == 100


def test_distilled_selection_returns_none_when_no_checkpoint_is_eligible() -> None:
    scores = tuple(
        _score(step, humaneval=33, mbpp=70, eligible=False) for step in (25, 50, 100, 185)
    )

    assert select_development_candidate(scores) is None


def test_distilled_selection_requires_exact_frozen_checkpoint_grid() -> None:
    scores = (
        _score(25, humaneval=33, mbpp=71, eligible=True),
        _score(50, humaneval=33, mbpp=71, eligible=True),
        _score(100, humaneval=33, mbpp=71, eligible=True),
    )

    with pytest.raises(DistilledTrajectoryEvaluationError, match="exactly four scores"):
        select_development_candidate(scores)


def test_distilled_evaluation_reuses_frozen_dev_membership_only() -> None:
    manifest = load_development_manifest()
    membership = manifest["membership"]
    assert isinstance(membership, dict)
    holdout = membership["repository_holdout"]
    assert isinstance(holdout, dict)

    assert holdout["development"] == []
    assert len(holdout["qualification"]) == 11
    assert len(membership["humaneval"]["development"]) == 45
    assert len(membership["mbpp"]["development"]) == 130
