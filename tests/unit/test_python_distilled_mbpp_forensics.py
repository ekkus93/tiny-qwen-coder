"""CPU tests for deterministic P9-008 MBPP regression forensics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tiny_qwen_coder.evaluation import python_distilled_mbpp_forensics as forensics


def _row(
    problem_id: str,
    *,
    adapter_id: str | None,
    passed: bool,
    category: str | None = None,
    generated_tokens: int = 100,
    code: str | None = None,
) -> dict[str, object]:
    error_category = "none" if passed else (category or "test")
    return {
        "problem_id": problem_id,
        "base_model": {
            "repository": "Qwen/Qwen3.5-4B",
            "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        },
        "adapter": {"family": None if adapter_id is None else "language", "adapter_id": adapter_id},
        "error_category": error_category,
        "error_message": None if passed else "synthetic failure",
        "tests": {"passed": 3 if passed else 0, "total": 3},
        "generation": {"generated_tokens": generated_tokens},
        "generated_code": code if code is not None else f"def f_{problem_id[5:]}(): pass",
    }


def _synthetic_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    base: list[dict[str, object]] = []
    adapter: list[dict[str, object]] = []
    adapter_id = "language/python/p9-distilled-v4-2000-r8-lr1e5"
    for task in range(1, 501):
        base_passed = task <= 220 or 371 <= task <= 440
        base.append(_row(f"MBPP/{task}", adapter_id=None, passed=base_passed))
    for task in range(1, 371):
        adapter_passed = task <= 179 or 221 <= task <= 237
        category = "parse" if task == 180 else "test"
        code = "def repeated(): return 0" if task in (180, 181) else None
        adapter.append(
            _row(
                f"MBPP/{task}",
                adapter_id=adapter_id,
                passed=adapter_passed,
                category=category,
                generated_tokens=512 if task == 180 else 100,
                code=code,
            )
        )
    return base, adapter


def test_analyze_reproduces_expected_flip_arithmetic(monkeypatch: pytest.MonkeyPatch) -> None:
    base, adapter = _synthetic_rows()
    membership = forensics._membership_sha256([str(row["problem_id"]) for row in adapter])
    monkeypatch.setattr(forensics, "_EXPECTED_MEMBERSHIP_SHA256", membership)

    summary = forensics.analyze_mbpp_results(base, adapter)

    assert summary["flip_matrix"] == {
        "retained_pass": 179,
        "retained_fail": 133,
        "regressions": 41,
        "improvements": 17,
        "net_pass_delta": -24,
    }
    assert summary["regression_error_categories"] == {"parse": 1, "test": 40}
    assert summary["improvement_base_error_categories"] == {"test": 17}
    assert summary["conclusion"] == {
        "dominant_failure_mode": "semantic_or_test_failure",
        "test_failure_fraction_of_regressions": 40 / 41,
        "truncation_is_dominant": False,
        "harness_failure_detected": False,
        "recommended_next_experiment": (
            "redesign distillation around independently executable semantic verification; "
            "do not run another rank/LR sweep or scale teacher data yet"
        ),
    }


def test_analyze_rejects_membership_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    base, adapter = _synthetic_rows()
    monkeypatch.setattr(forensics, "_EXPECTED_MEMBERSHIP_SHA256", "0" * 64)

    with pytest.raises(forensics.MbppForensicsError, match="membership drifted"):
        forensics.analyze_mbpp_results(base, adapter)


def test_analyze_rejects_success_without_all_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    base, adapter = _synthetic_rows()
    membership = forensics._membership_sha256([str(row["problem_id"]) for row in adapter])
    monkeypatch.setattr(forensics, "_EXPECTED_MEMBERSHIP_SHA256", membership)
    adapter[0]["tests"] = {"passed": 2, "total": 3}

    with pytest.raises(forensics.MbppForensicsError, match="without all tests passing"):
        forensics.analyze_mbpp_results(base, adapter)


def test_load_result_rows_and_write_forensics(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_text(
        json.dumps(_row("MBPP/1", adapter_id=None, passed=True)) + "\n",
        encoding="utf-8",
    )
    loaded = forensics.load_result_rows(source)
    assert len(loaded) == 1
    assert loaded[0]["problem_id"] == "MBPP/1"

    output = tmp_path / "out"
    summary: dict[str, object] = {
        "regression_rows": [{"problem_id": "MBPP/2"}],
        "improvement_rows": [{"problem_id": "MBPP/3"}],
    }
    summary_path, flips_path = forensics.write_forensics(summary, output)
    assert json.loads(summary_path.read_text(encoding="utf-8"))["regression_rows"] == [
        {"problem_id": "MBPP/2"}
    ]
    lines = [json.loads(line) for line in flips_path.read_text(encoding="utf-8").splitlines()]
    assert lines == [
        {"disposition": "regression", "problem_id": "MBPP/2"},
        {"disposition": "improvement", "problem_id": "MBPP/3"},
    ]
