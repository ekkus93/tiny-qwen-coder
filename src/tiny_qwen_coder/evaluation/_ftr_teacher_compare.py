"""Task-level FTR-202 teacher/base comparison and FTR-203 gate application."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from tiny_qwen_coder.evaluation._ftr_teacher_common import (
    FTRTeacherSuperiorityError,
    expect_int,
    expect_number,
    expect_str,
    read_json,
    sha256_file,
    strict_mapping,
)

_SUITE_PATHS = {
    "humaneval": (
        Path("humaneval/humaneval-results.jsonl"),
        Path("humaneval/humaneval-aggregate.json"),
    ),
    "mbpp": (Path("mbpp/mbpp-results.jsonl"), Path("mbpp/mbpp-aggregate.json")),
    "repository_holdout": (
        Path("repository-holdout/repository-holdout-results.jsonl"),
        Path("repository-holdout/repository-holdout-aggregate.json"),
    ),
}
_REGRESSION_RESULTS = Path("general-tool-regression/general-tool-regression-results.jsonl")
_REGRESSION_AGGREGATE = Path("general-tool-regression/general-tool-regression-aggregate.json")


def _read_jsonl(path: Path, *, context: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value: object = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise FTRTeacherSuperiorityError(
                        f"invalid JSON in {context} line {line_number}: {path}"
                    ) from exc
                rows.append(strict_mapping(value, context=f"{context}[{line_number}]"))
    except OSError as exc:
        raise FTRTeacherSuperiorityError(f"could not read {context}: {path}") from exc
    if not rows:
        raise FTRTeacherSuperiorityError(f"{context} is empty")
    return rows


def _coding_outcomes(path: Path, *, suite_id: str) -> dict[str, bool]:
    outcomes: dict[str, bool] = {}
    for index, row in enumerate(_read_jsonl(path, context=f"{suite_id} results")):
        context = f"{suite_id}[{index}]"
        problem_id = expect_str(row, "problem_id", context=context)
        tests = strict_mapping(row.get("tests"), context=f"{context}.tests")
        passed = expect_int(tests, "passed", context=f"{context}.tests")
        total = expect_int(tests, "total", context=f"{context}.tests")
        if total <= 0 or passed < 0 or passed > total:
            raise FTRTeacherSuperiorityError(f"invalid test counts for {problem_id}")
        if problem_id in outcomes:
            raise FTRTeacherSuperiorityError(f"duplicate task ID in {suite_id}: {problem_id}")
        outcomes[problem_id] = passed == total
    return outcomes


def _aggregate_counts(path: Path, *, expected_total: int, context: str) -> tuple[int, int]:
    aggregate = read_json(path, context=context)
    passed = expect_int(aggregate, "passed", context=context)
    total = expect_int(aggregate, "total_problems", context=context)
    if total != expected_total or not 0 <= passed <= total:
        raise FTRTeacherSuperiorityError(f"{context} aggregate counts are invalid")
    if expect_int(aggregate, "harness_errors", context=context) != 0:
        raise FTRTeacherSuperiorityError(f"{context} contains harness errors")
    return passed, total


def _paired_p_value(*, improvements: int, regressions: int) -> float:
    discordant = improvements + regressions
    if discordant == 0:
        return 1.0
    numerator: int = sum(math.comb(discordant, k) for k in range(improvements, discordant + 1))
    return float(numerator) / float(2**discordant)


def _paired_comparison(
    *, base: Mapping[str, bool], teacher: Mapping[str, bool], context: str
) -> dict[str, object]:
    if set(base) != set(teacher) or not base:
        raise FTRTeacherSuperiorityError(f"{context} task memberships must match exactly")
    improvements = regressions = unchanged_pass = unchanged_fail = 0
    changed: list[dict[str, object]] = []
    for task_id in sorted(base):
        before = base[task_id]
        after = teacher[task_id]
        if before and after:
            unchanged_pass += 1
        elif not before and not after:
            unchanged_fail += 1
        elif after:
            improvements += 1
            changed.append({"task_id": task_id, "base_passed": False, "teacher_passed": True})
        else:
            regressions += 1
            changed.append({"task_id": task_id, "base_passed": True, "teacher_passed": False})
    net = improvements - regressions
    return {
        "tasks": len(base),
        "base_passed": sum(base.values()),
        "teacher_passed": sum(teacher.values()),
        "improvements": improvements,
        "regressions": regressions,
        "unchanged_pass": unchanged_pass,
        "unchanged_fail": unchanged_fail,
        "net_improvement_tasks": net,
        "absolute_pass_rate_gain": net / len(base),
        "one_sided_p_value": _paired_p_value(improvements=improvements, regressions=regressions),
        "changed_tasks": changed,
    }


def _regression_outcomes(path: Path) -> dict[str, bool]:
    outcomes: dict[str, bool] = {}
    for index, row in enumerate(_read_jsonl(path, context="general-tool regression results")):
        context = f"general-tool regression[{index}]"
        case_id = expect_str(row, "case_id", context=context)
        passed = row.get("passed")
        if not isinstance(passed, bool):
            raise FTRTeacherSuperiorityError(f"{context}.passed must be boolean")
        if case_id in outcomes:
            raise FTRTeacherSuperiorityError(f"duplicate regression case {case_id}")
        outcomes[case_id] = passed
    return outcomes


def _validate_regression_aggregate(path: Path, *, expected: Mapping[str, bool]) -> None:
    aggregate = read_json(path, context="general-tool regression aggregate")
    total = expect_int(aggregate, "total_cases", context="general-tool regression aggregate")
    passed = expect_int(aggregate, "passed", context="general-tool regression aggregate")
    if total != len(expected) or passed != sum(expected.values()):
        raise FTRTeacherSuperiorityError("general-tool regression aggregate/result mismatch")


def compare_teacher_to_base(
    *, base_dir: Path, teacher_dir: Path, gate_path: Path
) -> dict[str, object]:
    """Compare exact task outcomes and apply the precommitted FTR-203 gate."""

    gate = read_json(gate_path, context="FTR-203 gate")
    suite_constraints = strict_mapping(gate.get("suite_constraints"), context="FTR-203 suites")
    comparisons: dict[str, dict[str, object]] = {}
    base_outcomes: dict[str, dict[str, bool]] = {}
    teacher_outcomes: dict[str, dict[str, bool]] = {}

    for suite_id, (results_relative, aggregate_relative) in _SUITE_PATHS.items():
        constraint = strict_mapping(suite_constraints.get(suite_id), context=f"FTR-203 {suite_id}")
        expected_total = expect_int(constraint, "total_tasks", context=f"FTR-203 {suite_id}")
        base_rows = _coding_outcomes(base_dir / results_relative, suite_id=f"base {suite_id}")
        teacher_rows = _coding_outcomes(
            teacher_dir / results_relative, suite_id=f"teacher {suite_id}"
        )
        if len(base_rows) != expected_total or len(teacher_rows) != expected_total:
            raise FTRTeacherSuperiorityError(f"{suite_id} task count drifted")
        base_agg = _aggregate_counts(
            base_dir / aggregate_relative,
            expected_total=expected_total,
            context=f"base {suite_id} aggregate",
        )
        teacher_agg = _aggregate_counts(
            teacher_dir / aggregate_relative,
            expected_total=expected_total,
            context=f"teacher {suite_id} aggregate",
        )
        if base_agg[0] != sum(base_rows.values()) or teacher_agg[0] != sum(teacher_rows.values()):
            raise FTRTeacherSuperiorityError(f"{suite_id} aggregate/result mismatch")
        expected_base = expect_int(constraint, "base_passed", context=f"FTR-203 {suite_id}")
        if base_agg[0] != expected_base:
            raise FTRTeacherSuperiorityError(f"frozen base {suite_id} score drifted")
        base_outcomes[suite_id] = base_rows
        teacher_outcomes[suite_id] = teacher_rows
        comparisons[suite_id] = _paired_comparison(
            base=base_rows, teacher=teacher_rows, context=suite_id
        )

    public_base = {**base_outcomes["humaneval"], **base_outcomes["mbpp"]}
    public_teacher = {**teacher_outcomes["humaneval"], **teacher_outcomes["mbpp"]}
    public_comparison = _paired_comparison(
        base=public_base, teacher=public_teacher, context="public coding"
    )
    combined_base = {**public_base, **base_outcomes["repository_holdout"]}
    combined_teacher = {**public_teacher, **teacher_outcomes["repository_holdout"]}
    combined_comparison = _paired_comparison(
        base=combined_base, teacher=combined_teacher, context="combined coding"
    )

    base_regression = _regression_outcomes(base_dir / _REGRESSION_RESULTS)
    teacher_regression = _regression_outcomes(teacher_dir / _REGRESSION_RESULTS)
    if set(base_regression) != set(teacher_regression):
        raise FTRTeacherSuperiorityError("general-tool regression membership drifted")
    _validate_regression_aggregate(base_dir / _REGRESSION_AGGREGATE, expected=base_regression)
    _validate_regression_aggregate(teacher_dir / _REGRESSION_AGGREGATE, expected=teacher_regression)
    regression = _paired_comparison(
        base=base_regression,
        teacher=teacher_regression,
        context="general-tool regression",
    )

    primary = strict_mapping(gate.get("primary_capability"), context="FTR-203 primary")
    he = strict_mapping(suite_constraints.get("humaneval"), context="FTR-203 HumanEval")
    mbpp = strict_mapping(suite_constraints.get("mbpp"), context="FTR-203 MBPP")
    holdout = strict_mapping(
        suite_constraints.get("repository_holdout"), context="FTR-203 repository holdout"
    )
    regression_gate = strict_mapping(
        suite_constraints.get("general_tool_regression"), context="FTR-203 regression"
    )
    checks = {
        "primary_membership_exact": cast(int, public_comparison["tasks"])
        == expect_int(primary, "total_tasks", context="FTR-203 primary"),
        "primary_minimum_teacher_score": cast(int, public_comparison["teacher_passed"])
        >= expect_int(primary, "minimum_teacher_passed", context="FTR-203 primary"),
        "primary_minimum_net_effect": cast(int, public_comparison["net_improvement_tasks"])
        >= expect_int(primary, "minimum_net_improvement_tasks", context="FTR-203 primary"),
        "primary_minimum_rate_effect": cast(float, public_comparison["absolute_pass_rate_gain"])
        >= expect_number(primary, "minimum_absolute_pass_rate_gain", context="FTR-203 primary"),
        "primary_paired_significance": cast(float, public_comparison["one_sided_p_value"])
        <= expect_number(primary, "maximum_one_sided_p_value", context="FTR-203 primary"),
        "humaneval_non_regression": cast(int, comparisons["humaneval"]["net_improvement_tasks"])
        >= expect_int(he, "minimum_delta_tasks", context="FTR-203 HumanEval"),
        "mbpp_non_regression": cast(int, comparisons["mbpp"]["net_improvement_tasks"])
        >= expect_int(mbpp, "minimum_delta_tasks", context="FTR-203 MBPP"),
        "repository_holdout_non_catastrophic": cast(
            int, comparisons["repository_holdout"]["net_improvement_tasks"]
        )
        >= -expect_int(holdout, "maximum_pass_drop_tasks", context="FTR-203 repository holdout"),
        "general_tool_regression_non_catastrophic": cast(int, regression["net_improvement_tasks"])
        >= -expect_int(
            regression_gate,
            "maximum_pass_drop_cases",
            context="FTR-203 regression",
        ),
    }
    passed = all(checks.values())
    decision = strict_mapping(gate.get("decision"), context="FTR-203 decision")
    return {
        "schema_version": 1,
        "task_id": "FTR-202/FTR-203",
        "gate_sha256": sha256_file(gate_path),
        "coding_suites": comparisons,
        "public_coding": public_comparison,
        "combined_coding": combined_comparison,
        "general_tool_regression": regression,
        "gate_checks": checks,
        "teacher_superiority_demonstrated": passed,
        "decision": expect_str(decision, "pass" if passed else "fail", context="FTR-203 decision"),
    }
