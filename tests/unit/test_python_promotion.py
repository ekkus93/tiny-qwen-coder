"""CPU-only contract tests for P8-005 Python adapter promotion."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from tiny_qwen_coder.evaluation.promotion import (
    FROZEN_PYTHON_PROMOTION_POLICY_PATH,
    FROZEN_PYTHON_PROMOTION_POLICY_SHA256,
    PromotionPolicyError,
    PythonPromotionEvidence,
    evaluate_python_adapter_promotion,
    load_frozen_python_promotion_policy,
    parse_python_promotion_policy,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DECISION_PATH = _REPO_ROOT / "docs/evidence/P8_005_P0_PROMOTION_DECISION.json"


def _p0_evidence() -> PythonPromotionEvidence:
    return PythonPromotionEvidence(
        adapter_id="language/python/p0",
        humaneval_passed=88,
        humaneval_total=164,
        mbpp_passed=97,
        mbpp_total=500,
        repository_holdout_passed=2,
        repository_holdout_total=11,
        combined_passed=187,
        combined_total=675,
        general_tool_passed=2,
        general_tool_total=12,
        general_tool_regressions=0,
        cross_language_conclusion="no_catastrophic_regression",
        typescript_semantic_passed=3,
        typescript_semantic_total=3,
        rust_semantic_passed=3,
        rust_semantic_total=3,
        contamination_status="not_run",
        adapter_load_validated=True,
    )


def test_frozen_policy_uses_baseline_anchored_thresholds() -> None:
    policy = load_frozen_python_promotion_policy(repo_root=_REPO_ROOT)

    assert policy.policy_id == "python-promotion-v1"
    assert policy.combined.passed == 424
    assert policy.combined.total == 675
    assert policy.minimum_combined_absolute_gain == 0.02
    assert policy.minimum_combined_passed == 438
    assert policy.humaneval.passed == 128
    assert policy.mbpp.passed == 290
    assert policy.repository_holdout.passed == 6
    assert policy.general_tool_minimum_passed == 2
    assert policy.general_tool_maximum_regressions == 0
    assert policy.required_contamination_status == "clean"


def test_p0_rejection_matches_committed_decision_record() -> None:
    policy = load_frozen_python_promotion_policy(repo_root=_REPO_ROOT)
    decision = evaluate_python_adapter_promotion(policy, _p0_evidence())
    record = json.loads(_DECISION_PATH.read_text(encoding="utf-8"))

    normalized_decision = json.loads(json.dumps(asdict(decision), sort_keys=True))
    assert record["policy"]["sha256"] == FROZEN_PYTHON_PROMOTION_POLICY_SHA256
    assert record["decision"] == normalized_decision
    assert decision.disposition == "reject"
    assert decision.recommended_adapter_id is None
    assert decision.rejection_reasons == (
        "python_combined_gain",
        "humaneval_preservation",
        "mbpp_preservation",
        "repository_holdout_preservation",
        "contamination_evidence",
    )


def test_promotion_requires_every_gate() -> None:
    policy = load_frozen_python_promotion_policy(repo_root=_REPO_ROOT)
    evidence = PythonPromotionEvidence(
        adapter_id="language/python/p1",
        humaneval_passed=130,
        humaneval_total=164,
        mbpp_passed=300,
        mbpp_total=500,
        repository_holdout_passed=8,
        repository_holdout_total=11,
        combined_passed=450,
        combined_total=675,
        general_tool_passed=2,
        general_tool_total=12,
        general_tool_regressions=0,
        cross_language_conclusion="no_catastrophic_regression",
        typescript_semantic_passed=2,
        typescript_semantic_total=3,
        rust_semantic_passed=2,
        rust_semantic_total=3,
        contamination_status="clean",
        adapter_load_validated=True,
    )

    decision = evaluate_python_adapter_promotion(policy, evidence)

    assert decision.disposition == "promote"
    assert decision.recommended_adapter_id == "language/python/p1"
    assert decision.rejection_reasons == ()
    assert all(check.passed for check in decision.checks)


def test_clean_scores_do_not_bypass_missing_contamination_evidence() -> None:
    policy = load_frozen_python_promotion_policy(repo_root=_REPO_ROOT)
    evidence = PythonPromotionEvidence(
        adapter_id="language/python/p1",
        humaneval_passed=130,
        humaneval_total=164,
        mbpp_passed=300,
        mbpp_total=500,
        repository_holdout_passed=8,
        repository_holdout_total=11,
        combined_passed=450,
        combined_total=675,
        general_tool_passed=2,
        general_tool_total=12,
        general_tool_regressions=0,
        cross_language_conclusion="no_catastrophic_regression",
        typescript_semantic_passed=3,
        typescript_semantic_total=3,
        rust_semantic_passed=3,
        rust_semantic_total=3,
        contamination_status="not_run",
        adapter_load_validated=True,
    )

    decision = evaluate_python_adapter_promotion(policy, evidence)

    assert decision.disposition == "reject"
    assert decision.recommended_adapter_id is None
    assert decision.rejection_reasons == ("contamination_evidence",)


def test_policy_parser_rejects_unknown_fields() -> None:
    path = _REPO_ROOT / FROZEN_PYTHON_PROMOTION_POLICY_PATH
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    value["silent_fallback"] = True

    with pytest.raises(PromotionPolicyError, match="unknown field"):
        parse_python_promotion_policy(value)


def test_frozen_loader_rejects_byte_drift(tmp_path: Path) -> None:
    source = _REPO_ROOT / FROZEN_PYTHON_PROMOTION_POLICY_PATH
    destination = tmp_path / FROZEN_PYTHON_PROMOTION_POLICY_PATH
    destination.parent.mkdir(parents=True)
    destination.write_bytes(source.read_bytes() + b"# drift\n")

    with pytest.raises(PromotionPolicyError, match="SHA-256 mismatch"):
        load_frozen_python_promotion_policy(repo_root=tmp_path)
