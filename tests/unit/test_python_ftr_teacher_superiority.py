"""CPU-only tests for the FTR-202 benchmark protocol and FTR-203 gate."""

from __future__ import annotations

import json
from pathlib import Path

from tiny_qwen_coder.evaluation.python_ftr_teacher_superiority import (
    audit_teacher_benchmark_protocol,
    compare_teacher_to_base,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SHA = "a" * 40


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _coding_row(task_id: str, passed: bool) -> dict[str, object]:
    return {
        "problem_id": task_id,
        "tests": {"passed": 1 if passed else 0, "total": 1},
    }


def _write_coding_suite(root: Path, suite: str, names: list[str], outcomes: list[bool]) -> None:
    if suite == "repository_holdout":
        directory = root / "repository-holdout"
        stem = "repository-holdout"
    else:
        directory = root / suite
        stem = suite
    _write_jsonl(
        directory / f"{stem}-results.jsonl",
        [_coding_row(name, passed) for name, passed in zip(names, outcomes, strict=True)],
    )
    _write_json(
        directory / f"{stem}-aggregate.json",
        {
            "passed": sum(outcomes),
            "total_problems": len(outcomes),
            "harness_errors": 0,
        },
    )


def _write_regression(root: Path, outcomes: dict[str, bool]) -> None:
    directory = root / "general-tool-regression"
    _write_jsonl(
        directory / "general-tool-regression-results.jsonl",
        [{"case_id": case_id, "passed": passed} for case_id, passed in outcomes.items()],
    )
    _write_json(
        directory / "general-tool-regression-aggregate.json",
        {"total_cases": len(outcomes), "passed": sum(outcomes.values())},
    )


def _small_gate(path: Path) -> None:
    _write_json(
        path,
        {
            "suite_constraints": {
                "humaneval": {"base_passed": 1, "total_tasks": 2, "minimum_delta_tasks": 0},
                "mbpp": {"base_passed": 1, "total_tasks": 3, "minimum_delta_tasks": 0},
                "repository_holdout": {
                    "base_passed": 1,
                    "total_tasks": 2,
                    "maximum_pass_drop_tasks": 1,
                },
                "general_tool_regression": {"maximum_pass_drop_cases": 1},
            },
            "primary_capability": {
                "total_tasks": 5,
                "minimum_teacher_passed": 5,
                "minimum_net_improvement_tasks": 3,
                "minimum_absolute_pass_rate_gain": 0.6,
                "maximum_one_sided_p_value": 0.2,
            },
            "decision": {"pass": "authorize", "fail": "stop"},
        },
    )


def _fixture_pair(tmp_path: Path, *, teacher_mbpp: list[bool]) -> tuple[Path, Path, Path]:
    base = tmp_path / "base"
    teacher = tmp_path / "teacher"
    gate = tmp_path / "gate.json"
    _small_gate(gate)
    _write_coding_suite(base, "humaneval", ["HumanEval/0", "HumanEval/1"], [True, False])
    _write_coding_suite(teacher, "humaneval", ["HumanEval/0", "HumanEval/1"], [True, True])
    _write_coding_suite(base, "mbpp", ["MBPP/0", "MBPP/1", "MBPP/2"], [True, False, False])
    _write_coding_suite(teacher, "mbpp", ["MBPP/0", "MBPP/1", "MBPP/2"], teacher_mbpp)
    _write_coding_suite(base, "repository_holdout", ["holdout/0", "holdout/1"], [True, False])
    _write_coding_suite(teacher, "repository_holdout", ["holdout/0", "holdout/1"], [False, True])
    _write_regression(base, {"a": True, "b": True})
    _write_regression(teacher, {"a": True, "b": False})
    return base, teacher, gate


def test_repository_protocol_and_gate_are_precommitted() -> None:
    report = audit_teacher_benchmark_protocol(repo_root=_REPO_ROOT, source_git_sha=_SHA)

    assert report["protocol_ready"] is True
    checks = report["checks"]
    assert isinstance(checks, dict)
    assert checks["repository_holdout_one_shot_status_frozen"] is True
    assert checks["ftr_203_gate_precommitted"] is True
    assert checks["eight_point_public_coding_margin_frozen"] is True


def test_teacher_gate_authorizes_only_when_all_precommitted_checks_pass(tmp_path: Path) -> None:
    base, teacher, gate = _fixture_pair(tmp_path, teacher_mbpp=[True, True, True])

    report = compare_teacher_to_base(base_dir=base, teacher_dir=teacher, gate_path=gate)

    assert report["teacher_superiority_demonstrated"] is True
    assert report["decision"] == "authorize"
    primary = report["public_coding"]
    assert isinstance(primary, dict)
    assert primary["base_passed"] == 2
    assert primary["teacher_passed"] == 5
    assert primary["net_improvement_tasks"] == 3
    assert primary["one_sided_p_value"] == 0.125


def test_teacher_gate_stops_when_practical_margin_is_not_met(tmp_path: Path) -> None:
    base, teacher, gate = _fixture_pair(tmp_path, teacher_mbpp=[True, True, False])

    report = compare_teacher_to_base(base_dir=base, teacher_dir=teacher, gate_path=gate)

    assert report["teacher_superiority_demonstrated"] is False
    assert report["decision"] == "stop"
    checks = report["gate_checks"]
    assert isinstance(checks, dict)
    assert checks["primary_minimum_net_effect"] is False
    assert checks["primary_minimum_teacher_score"] is False


def test_pairwise_report_preserves_changed_task_evidence(tmp_path: Path) -> None:
    base, teacher, gate = _fixture_pair(tmp_path, teacher_mbpp=[True, True, True])

    report = compare_teacher_to_base(base_dir=base, teacher_dir=teacher, gate_path=gate)
    primary = report["public_coding"]
    assert isinstance(primary, dict)
    changed = primary["changed_tasks"]
    assert isinstance(changed, list)
    changed_ids = {row["task_id"] for row in changed if isinstance(row, dict)}
    assert changed_ids == {"HumanEval/1", "MBPP/1", "MBPP/2"}
