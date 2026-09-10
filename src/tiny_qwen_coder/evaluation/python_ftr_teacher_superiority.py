"""FTR-202/FTR-203 teacher benchmark comparison and precommitted superiority gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import yaml

_PROTOCOL_PATH = Path("configs/eval/python/ftr_202_teacher_benchmark_v1.json")
_GATE_PATH = Path("configs/eval/python/ftr_203_teacher_superiority_gate_v1.json")
_FTR201_EVIDENCE = Path("docs/evidence/FTR_201_DIRECT_TEACHER_EVALUATION_SUPPORT.json")
_FTR201_EVAL = Path("configs/eval/python/ftr_201_teacher_direct_v1.yaml")
_EXPECTED_TEACHER_REPOSITORY = "Qwen/Qwen3.8-27B"
_EXPECTED_TEACHER_REVISION = "72a217afab8029b39e4af1c7273a829995a3dbaf"
_EXPECTED_BASE_SCORES = {
    "humaneval": (128, 164),
    "mbpp": (290, 500),
    "repository_holdout": (6, 11),
    "public_coding": (418, 664),
    "combined_coding": (424, 675),
}
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


class FTRTeacherSuperiorityError(RuntimeError):
    """Raised when teacher benchmark evidence or gate configuration is invalid."""


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise FTRTeacherSuperiorityError(f"could not hash {path}") from exc


def _mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise FTRTeacherSuperiorityError(f"{context} must be a mapping")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise FTRTeacherSuperiorityError(f"{context} keys must be strings")
        result[key] = item
    return result


def _read_json(path: Path, *, context: str) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FTRTeacherSuperiorityError(f"could not read {context}: {path}") from exc
    return _mapping(value, context=context)


def _read_yaml(path: Path, *, context: str) -> dict[str, object]:
    try:
        value: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise FTRTeacherSuperiorityError(f"could not read {context}: {path}") from exc
    return _mapping(value, context=context)


def _expect_str(mapping: Mapping[str, object], key: str, *, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise FTRTeacherSuperiorityError(f"{context}.{key} must be a non-empty string")
    return value


def _expect_int(mapping: Mapping[str, object], key: str, *, context: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise FTRTeacherSuperiorityError(f"{context}.{key} must be an integer")
    return value


def _expect_number(mapping: Mapping[str, object], key: str, *, context: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise FTRTeacherSuperiorityError(f"{context}.{key} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise FTRTeacherSuperiorityError(f"{context}.{key} must be finite")
    return numeric


def _score_pair(value: object, *, context: str) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise FTRTeacherSuperiorityError(f"{context} must be [passed, total]")
    pair = cast(list[int], value)
    return pair[0], pair[1]


def _validate_source_sha(source_git_sha: str) -> None:
    if len(source_git_sha) != 40 or any(char not in "0123456789abcdef" for char in source_git_sha):
        raise FTRTeacherSuperiorityError("source_git_sha must be a lowercase 40-character SHA")


def audit_teacher_benchmark_protocol(
    *, repo_root: Path, source_git_sha: str
) -> dict[str, object]:
    """Validate FTR-202 protocol and prove the FTR-203 gate was frozen with it."""

    _validate_source_sha(source_git_sha)
    protocol_path = repo_root / _PROTOCOL_PATH
    gate_path = repo_root / _GATE_PATH
    protocol = _read_json(protocol_path, context="FTR-202 protocol")
    gate = _read_json(gate_path, context="FTR-203 gate")
    ftr201 = _read_json(repo_root / _FTR201_EVIDENCE, context="FTR-201 evidence")
    teacher_eval = _read_yaml(repo_root / _FTR201_EVAL, context="FTR-201 teacher evaluation")

    if _expect_str(protocol, "task_id", context="FTR-202 protocol") != "FTR-202":
        raise FTRTeacherSuperiorityError("unexpected FTR-202 task_id")
    if _expect_str(gate, "task_id", context="FTR-203 gate") != "FTR-203":
        raise FTRTeacherSuperiorityError("unexpected FTR-203 task_id")
    if ftr201.get("support_ready") is not True:
        raise FTRTeacherSuperiorityError("FTR-201 direct teacher support is not frozen as ready")

    teacher = _mapping(protocol.get("teacher"), context="FTR-202 teacher")
    if (
        _expect_str(teacher, "repository", context="FTR-202 teacher")
        != _EXPECTED_TEACHER_REPOSITORY
    ):
        raise FTRTeacherSuperiorityError("FTR-202 teacher repository drifted")
    if _expect_str(teacher, "revision", context="FTR-202 teacher") != _EXPECTED_TEACHER_REVISION:
        raise FTRTeacherSuperiorityError("FTR-202 teacher revision drifted")

    output_dir = _expect_str(teacher_eval, "output_dir", context="FTR-201 evaluation")
    if output_dir != "artifacts/eval/python/ftr-201-teacher-direct-v1":
        raise FTRTeacherSuperiorityError("FTR-201 teacher artifact root drifted")

    base = _mapping(protocol.get("frozen_base_reference"), context="FTR-202 frozen base")
    scores = _mapping(base.get("scores"), context="FTR-202 frozen base scores")
    for suite_id, expected in _EXPECTED_BASE_SCORES.items():
        observed = _score_pair(scores.get(suite_id), context=f"FTR-202 base score {suite_id}")
        if observed != expected:
            raise FTRTeacherSuperiorityError(f"frozen base score drifted for {suite_id}")

    holdout = _mapping(
        protocol.get("repository_holdout_policy"), context="FTR-202 holdout policy"
    )
    if holdout.get("eligible") is not True or holdout.get("may_tune_after_observation") is not False:
        raise FTRTeacherSuperiorityError("repository holdout must remain one-shot and non-tuning")

    generation = _mapping(protocol.get("generation"), context="FTR-202 generation")
    scoring = _mapping(protocol.get("scoring"), context="FTR-202 scoring")
    minimum_bytes = _expect_int(
        generation, "minimum_cuda_total_bytes", context="FTR-202 generation"
    )
    if minimum_bytes < 75 * 1024**3:
        raise FTRTeacherSuperiorityError("FTR-202 generation must require an A100-class 80 GB GPU")
    if generation.get("execute_generated_code") is not False:
        raise FTRTeacherSuperiorityError("FTR-202 GPU generation must not execute generated code")
    if scoring.get("require_oci_isolation") is not True or scoring.get("network_enabled") is not False:
        raise FTRTeacherSuperiorityError("FTR-202 scoring isolation contract drifted")
    if scoring.get("execute_with_drive_or_cloud_credentials_mounted") is not False:
        raise FTRTeacherSuperiorityError("FTR-202 scoring must exclude mounted cloud credentials")

    primary = _mapping(gate.get("primary_capability"), context="FTR-203 primary capability")
    if gate.get("precommitted_before_teacher_results") is not True:
        raise FTRTeacherSuperiorityError("FTR-203 gate must be precommitted")
    if _expect_int(gate, "fixed_teacher_comparisons", context="FTR-203 gate") != 1:
        raise FTRTeacherSuperiorityError("FTR-203 must cover exactly one fixed teacher comparison")
    expected_primary = {
        "base_passed": 418,
        "total_tasks": 664,
        "minimum_net_improvement_tasks": 54,
        "minimum_teacher_passed": 472,
    }
    for key, expected in expected_primary.items():
        if _expect_int(primary, key, context="FTR-203 primary") != expected:
            raise FTRTeacherSuperiorityError(f"FTR-203 primary {key} drifted")
    if _expect_number(primary, "minimum_absolute_pass_rate_gain", context="FTR-203 primary") != 0.08:
        raise FTRTeacherSuperiorityError("FTR-203 absolute gain floor drifted")

    checks = {
        "ftr_201_support_frozen": True,
        "exact_teacher_identity_pinned": True,
        "frozen_base_reference_pinned": True,
        "a100_80gb_generation_required": True,
        "generation_does_not_execute_candidates": True,
        "isolated_network_disabled_scoring_required": True,
        "cloud_credentials_excluded_from_candidate_execution": True,
        "repository_holdout_one_shot_status_frozen": True,
        "ftr_203_gate_precommitted": True,
        "eight_point_public_coding_margin_frozen": True,
    }
    return {
        "schema_version": 1,
        "task_id": "FTR-202/FTR-203",
        "source_git_sha": source_git_sha,
        "protocol_path": _PROTOCOL_PATH.as_posix(),
        "protocol_sha256": _sha256(protocol_path),
        "gate_path": _GATE_PATH.as_posix(),
        "gate_sha256": _sha256(gate_path),
        "checks": checks,
        "protocol_ready": all(checks.values()),
    }


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
                rows.append(_mapping(value, context=f"{context}[{line_number}]"))
    except OSError as exc:
        raise FTRTeacherSuperiorityError(f"could not read {context}: {path}") from exc
    if not rows:
        raise FTRTeacherSuperiorityError(f"{context} is empty")
    return rows


def _coding_outcomes(path: Path, *, suite_id: str) -> dict[str, bool]:
    outcomes: dict[str, bool] = {}
    for index, row in enumerate(_read_jsonl(path, context=f"{suite_id} results")):
        context = f"{suite_id}[{index}]"
        problem_id = _expect_str(row, "problem_id", context=context)
        tests = _mapping(row.get("tests"), context=f"{context}.tests")
        passed = _expect_int(tests, "passed", context=f"{context}.tests")
        total = _expect_int(tests, "total", context=f"{context}.tests")
        if total <= 0 or passed < 0 or passed > total:
            raise FTRTeacherSuperiorityError(f"invalid test counts for {problem_id}")
        if problem_id in outcomes:
            raise FTRTeacherSuperiorityError(f"duplicate task ID in {suite_id}: {problem_id}")
        outcomes[problem_id] = passed == total
    return outcomes


def _aggregate_counts(path: Path, *, expected_total: int, context: str) -> tuple[int, int]:
    aggregate = _read_json(path, context=context)
    passed = _expect_int(aggregate, "passed", context=context)
    total = _expect_int(aggregate, "total_problems", context=context)
    if total != expected_total or not 0 <= passed <= total:
        raise FTRTeacherSuperiorityError(f"{context} aggregate counts are invalid")
    if _expect_int(aggregate, "harness_errors", context=context) != 0:
        raise FTRTeacherSuperiorityError(f"{context} contains harness errors")
    return passed, total


def _paired_p_value(*, improvements: int, regressions: int) -> float:
    discordant = improvements + regressions
    if discordant == 0:
        return 1.0
    numerator = sum(math.comb(discordant, k) for k in range(improvements, discordant + 1))
    return numerator / (2**discordant)


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
        "one_sided_p_value": _paired_p_value(
            improvements=improvements, regressions=regressions
        ),
        "changed_tasks": changed,
    }


def _regression_outcomes(path: Path) -> dict[str, bool]:
    outcomes: dict[str, bool] = {}
    for index, row in enumerate(_read_jsonl(path, context="general-tool regression results")):
        context = f"general-tool regression[{index}]"
        case_id = _expect_str(row, "case_id", context=context)
        passed = row.get("passed")
        if not isinstance(passed, bool):
            raise FTRTeacherSuperiorityError(f"{context}.passed must be boolean")
        if case_id in outcomes:
            raise FTRTeacherSuperiorityError(f"duplicate regression case {case_id}")
        outcomes[case_id] = passed
    return outcomes


def _validate_regression_aggregate(path: Path, *, expected: Mapping[str, bool]) -> None:
    aggregate = _read_json(path, context="general-tool regression aggregate")
    total = _expect_int(aggregate, "total_cases", context="general-tool regression aggregate")
    passed = _expect_int(aggregate, "passed", context="general-tool regression aggregate")
    if total != len(expected) or passed != sum(expected.values()):
        raise FTRTeacherSuperiorityError("general-tool regression aggregate/result mismatch")


def compare_teacher_to_base(
    *, base_dir: Path, teacher_dir: Path, gate_path: Path
) -> dict[str, object]:
    """Compare exact task outcomes and apply the precommitted FTR-203 gate."""

    gate = _read_json(gate_path, context="FTR-203 gate")
    suite_constraints = _mapping(gate.get("suite_constraints"), context="FTR-203 suites")
    comparisons: dict[str, dict[str, object]] = {}
    base_outcomes: dict[str, dict[str, bool]] = {}
    teacher_outcomes: dict[str, dict[str, bool]] = {}

    for suite_id, (results_relative, aggregate_relative) in _SUITE_PATHS.items():
        constraint = _mapping(suite_constraints.get(suite_id), context=f"FTR-203 {suite_id}")
        expected_total = _expect_int(constraint, "total_tasks", context=f"FTR-203 {suite_id}")
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
        expected_base = _expect_int(constraint, "base_passed", context=f"FTR-203 {suite_id}")
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
    _validate_regression_aggregate(
        teacher_dir / _REGRESSION_AGGREGATE, expected=teacher_regression
    )
    regression_comparison = _paired_comparison(
        base=base_regression,
        teacher=teacher_regression,
        context="general-tool regression",
    )

    primary = _mapping(gate.get("primary_capability"), context="FTR-203 primary")
    he_constraint = _mapping(suite_constraints.get("humaneval"), context="FTR-203 HumanEval")
    mbpp_constraint = _mapping(suite_constraints.get("mbpp"), context="FTR-203 MBPP")
    holdout_constraint = _mapping(
        suite_constraints.get("repository_holdout"), context="FTR-203 repository holdout"
    )
    regression_constraint = _mapping(
        suite_constraints.get("general_tool_regression"), context="FTR-203 regression"
    )

    checks = {
        "primary_membership_exact": cast(int, public_comparison["tasks"])
        == _expect_int(primary, "total_tasks", context="FTR-203 primary"),
        "primary_minimum_teacher_score": cast(int, public_comparison["teacher_passed"])
        >= _expect_int(primary, "minimum_teacher_passed", context="FTR-203 primary"),
        "primary_minimum_net_effect": cast(int, public_comparison["net_improvement_tasks"])
        >= _expect_int(primary, "minimum_net_improvement_tasks", context="FTR-203 primary"),
        "primary_minimum_rate_effect": cast(float, public_comparison["absolute_pass_rate_gain"])
        >= _expect_number(primary, "minimum_absolute_pass_rate_gain", context="FTR-203 primary"),
        "primary_paired_significance": cast(float, public_comparison["one_sided_p_value"])
        <= _expect_number(primary, "maximum_one_sided_p_value", context="FTR-203 primary"),
        "humaneval_non_regression": cast(int, comparisons["humaneval"]["net_improvement_tasks"])
        >= _expect_int(he_constraint, "minimum_delta_tasks", context="FTR-203 HumanEval"),
        "mbpp_non_regression": cast(int, comparisons["mbpp"]["net_improvement_tasks"])
        >= _expect_int(mbpp_constraint, "minimum_delta_tasks", context="FTR-203 MBPP"),
        "repository_holdout_non_catastrophic": cast(
            int, comparisons["repository_holdout"]["net_improvement_tasks"]
        )
        >= -_expect_int(
            holdout_constraint,
            "maximum_pass_drop_tasks",
            context="FTR-203 repository holdout",
        ),
        "general_tool_regression_non_catastrophic": cast(
            int, regression_comparison["net_improvement_tasks"]
        )
        >= -_expect_int(
            regression_constraint,
            "maximum_pass_drop_cases",
            context="FTR-203 regression",
        ),
    }
    passed = all(checks.values())
    decision = _mapping(gate.get("decision"), context="FTR-203 decision")
    return {
        "schema_version": 1,
        "task_id": "FTR-202/FTR-203",
        "gate_sha256": _sha256(gate_path),
        "coding_suites": comparisons,
        "public_coding": public_comparison,
        "combined_coding": combined_comparison,
        "general_tool_regression": regression_comparison,
        "gate_checks": checks,
        "teacher_superiority_demonstrated": passed,
        "decision": _expect_str(
            decision,
            "pass" if passed else "fail",
            context="FTR-203 decision",
        ),
    }


def write_report(path: Path, report: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FTR-202/FTR-203 teacher superiority protocol")
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--repo-root", type=Path, default=Path("."))
    audit.add_argument("--source-git-sha", required=True)
    audit.add_argument("--output", type=Path, default=None)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--repo-root", type=Path, default=Path("."))
    compare.add_argument("--base-dir", type=Path, required=True)
    compare.add_argument("--teacher-dir", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "audit":
        report = audit_teacher_benchmark_protocol(
            repo_root=cast(Path, args.repo_root), source_git_sha=cast(str, args.source_git_sha)
        )
        output = cast(Path | None, args.output)
        if output is not None:
            write_report(output, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    repo_root = cast(Path, args.repo_root)
    report = compare_teacher_to_base(
        base_dir=cast(Path, args.base_dir),
        teacher_dir=cast(Path, args.teacher_dir),
        gate_path=repo_root / _GATE_PATH,
    )
    write_report(cast(Path, args.output), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
