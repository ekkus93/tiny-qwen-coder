"""CPU-only tests for winner-only P9-007E qualification."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tiny_qwen_coder.evaluation import python_distilled_qualification as qualification

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_clean_replay_authorizes_exactly_step_185() -> None:
    authorization = qualification.load_qualification_authorization(_REPO_ROOT)

    assert authorization.workflow_run_id == 34255423595
    assert authorization.workflow_run_attempt == 2
    assert authorization.workflow_job_id == 102160760346
    assert authorization.selected_step == 185
    assert (
        authorization.selected_adapter_model_sha256
        == "2ee57b8c6fd10237e6e2faf11a9ff376fe7115f58062a8ed54eb962c3be6478a"
    )


def test_qualification_membership_is_the_untouched_500_task_complement() -> None:
    he_ids, mb_ids, holdout_ids = qualification.qualification_membership(_REPO_ROOT)

    assert len(he_ids) == 119
    assert len(mb_ids) == 370
    assert len(holdout_ids) == 11
    assert len(he_ids) + len(mb_ids) + len(holdout_ids) == 500


def test_target_language_gate_uses_full_phase8_thresholds() -> None:
    passing = qualification.evaluate_target_language_gate(
        humaneval_passed=105,
        mbpp_passed=223,
        repository_holdout_passed=6,
    )
    just_below_combined = qualification.evaluate_target_language_gate(
        humaneval_passed=104,
        mbpp_passed=223,
        repository_holdout_passed=6,
    )

    assert passing.full_combined_passed == 438
    assert passing.full_humaneval_passed == 139
    assert passing.full_mbpp_passed == 293
    assert passing.full_repository_holdout_passed == 6
    assert passing.target_language_gate_passed is True
    assert just_below_combined.full_combined_passed == 437
    assert just_below_combined.target_language_gate_passed is False


def test_qualification_rejects_any_attempt_to_substitute_a_runner_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _REPO_ROOT / "docs/evidence/P9_007_DEVELOPMENT_RUN_34255423595.json"
    evidence = json.loads(source.read_text(encoding="utf-8"))
    evidence["selected"]["step"] = 100
    modified = tmp_path / "authorization.json"
    modified.write_text(json.dumps(evidence), encoding="utf-8")
    monkeypatch.setattr(qualification, "_AUTHORIZATION_PATH", modified)

    with pytest.raises(
        qualification.DistilledQualificationError,
        match="selected development winner step drifted",
    ):
        qualification.load_qualification_authorization(_REPO_ROOT)


def test_invalid_qualification_counts_fail_closed() -> None:
    with pytest.raises(qualification.DistilledQualificationError, match="humaneval pass count"):
        qualification.evaluate_target_language_gate(
            humaneval_passed=120,
            mbpp_passed=220,
            repository_holdout_passed=6,
        )
