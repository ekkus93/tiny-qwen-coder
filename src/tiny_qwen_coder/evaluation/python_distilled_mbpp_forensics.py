"""Offline P9-008 forensics for the P9-007 MBPP qualification regression."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

_EXPECTED_BASE_MODEL = "Qwen/Qwen3.5-4B"
_EXPECTED_BASE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
_EXPECTED_ADAPTER_ID = "language/python/p9-distilled-v4-2000-r8-lr1e5"
_EXPECTED_BASE_ROWS = 500
_EXPECTED_QUALIFICATION_ROWS = 370
_EXPECTED_BASE_FULL_PASSED = 290
_EXPECTED_BASE_QUALIFICATION_PASSED = 220
_EXPECTED_ADAPTER_QUALIFICATION_PASSED = 196
_EXPECTED_MEMBERSHIP_SHA256 = "47fd9e1b8d94aced23a51414585fc94966c267e336e479ba71f1bf025606fc53"
_MAX_NEW_TOKENS = 512


class MbppForensicsError(RuntimeError):
    """Raised when P9-008 inputs or observed identities drift."""


def _mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise MbppForensicsError(f"{context} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise MbppForensicsError(f"{context} keys must be strings")
    return cast(dict[str, object], dict(value))


def _integer(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MbppForensicsError(f"{context} must be an integer")
    return value


def _string(value: object, *, context: str) -> str:
    if not isinstance(value, str):
        raise MbppForensicsError(f"{context} must be a string")
    return value


def load_result_rows(path: Path) -> tuple[dict[str, object], ...]:
    """Load one deterministic MBPP result JSONL artifact."""

    rows: list[dict[str, object]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise MbppForensicsError(f"could not read MBPP results: {path}") from exc
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            raise MbppForensicsError(f"blank MBPP result row at {path}:{index}")
        try:
            value: object = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MbppForensicsError(f"invalid MBPP JSON at {path}:{index}") from exc
        rows.append(_mapping(value, context=f"{path}:{index}"))
    return tuple(rows)


def _problem_id(row: Mapping[str, object], *, context: str) -> str:
    problem_id = _string(row.get("problem_id"), context=f"{context}.problem_id")
    if not problem_id.startswith("MBPP/"):
        raise MbppForensicsError(f"{context}.problem_id is not MBPP")
    return problem_id


def _error_category(row: Mapping[str, object], *, context: str) -> str:
    return _string(row.get("error_category"), context=f"{context}.error_category")


def _passed(row: Mapping[str, object], *, context: str) -> bool:
    category = _error_category(row, context=context)
    tests = _mapping(row.get("tests"), context=f"{context}.tests")
    tests_passed = _integer(tests.get("passed"), context=f"{context}.tests.passed")
    tests_total = _integer(tests.get("total"), context=f"{context}.tests.total")
    if not 0 <= tests_passed <= tests_total or tests_total <= 0:
        raise MbppForensicsError(f"{context}.tests has invalid counts")
    if category == "none" and tests_passed != tests_total:
        raise MbppForensicsError(f"{context} reports success without all tests passing")
    return category == "none"


def _generated_tokens(row: Mapping[str, object], *, context: str) -> int:
    generation = _mapping(row.get("generation"), context=f"{context}.generation")
    return _integer(
        generation.get("generated_tokens"),
        context=f"{context}.generation.generated_tokens",
    )


def _generated_code_sha256(row: Mapping[str, object], *, context: str) -> str:
    value = row.get("generated_code")
    code = "" if value is None else _string(value, context=f"{context}.generated_code")
    return hashlib.sha256(code.strip().encode("utf-8")).hexdigest()


def _validate_identity(
    row: Mapping[str, object],
    *,
    context: str,
    expected_adapter_id: str | None,
) -> None:
    base_model = _mapping(row.get("base_model"), context=f"{context}.base_model")
    if base_model.get("repository") != _EXPECTED_BASE_MODEL:
        raise MbppForensicsError(f"{context} base repository drifted")
    if base_model.get("revision") != _EXPECTED_BASE_REVISION:
        raise MbppForensicsError(f"{context} base revision drifted")
    adapter = _mapping(row.get("adapter"), context=f"{context}.adapter")
    if adapter.get("adapter_id") != expected_adapter_id:
        raise MbppForensicsError(f"{context} adapter identity drifted")


def _rows_by_id(
    rows: Sequence[Mapping[str, object]],
    *,
    label: str,
    expected_adapter_id: str | None,
) -> dict[str, Mapping[str, object]]:
    output: dict[str, Mapping[str, object]] = {}
    for index, row in enumerate(rows):
        context = f"{label}[{index}]"
        _validate_identity(row, context=context, expected_adapter_id=expected_adapter_id)
        problem_id = _problem_id(row, context=context)
        if problem_id in output:
            raise MbppForensicsError(f"duplicate {label} problem ID: {problem_id}")
        _passed(row, context=context)
        _generated_tokens(row, context=context)
        output[problem_id] = row
    return output


def _membership_sha256(problem_ids: Sequence[str]) -> str:
    payload = "".join(f"{problem_id}\n" for problem_id in sorted(problem_ids))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _token_summary(tokens: Sequence[int]) -> dict[str, object]:
    if not tokens:
        raise MbppForensicsError("token summary requires at least one row")
    ordered = sorted(tokens)
    p95_index = max(0, min(len(ordered) - 1, (95 * len(ordered) + 99) // 100 - 1))
    return {
        "count": len(tokens),
        "mean": sum(tokens) / len(tokens),
        "median": statistics.median(tokens),
        "p95": ordered[p95_index],
        "maximum": max(tokens),
        "hit_max_new_tokens": sum(token == _MAX_NEW_TOKENS for token in tokens),
    }


def analyze_mbpp_results(
    base_rows: Sequence[Mapping[str, object]],
    adapter_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Compare frozen base and P9-007E MBPP result rows without rerunning models."""

    if len(base_rows) != _EXPECTED_BASE_ROWS:
        raise MbppForensicsError(
            f"expected {_EXPECTED_BASE_ROWS} frozen base rows, got {len(base_rows)}"
        )
    if len(adapter_rows) != _EXPECTED_QUALIFICATION_ROWS:
        raise MbppForensicsError(
            f"expected {_EXPECTED_QUALIFICATION_ROWS} qualification rows, got {len(adapter_rows)}"
        )

    base_by_id = _rows_by_id(base_rows, label="base", expected_adapter_id=None)
    adapter_by_id = _rows_by_id(
        adapter_rows,
        label="adapter",
        expected_adapter_id=_EXPECTED_ADAPTER_ID,
    )
    adapter_ids = tuple(adapter_by_id)
    unknown = sorted(set(adapter_by_id) - set(base_by_id))
    if unknown:
        raise MbppForensicsError(f"qualification contains unknown base tasks: {unknown!r}")
    membership_sha256 = _membership_sha256(adapter_ids)
    if membership_sha256 != _EXPECTED_MEMBERSHIP_SHA256:
        raise MbppForensicsError("qualification MBPP membership drifted")

    base_full_passed = sum(
        _passed(row, context=f"base[{problem_id}]") for problem_id, row in base_by_id.items()
    )
    base_qualification_passed = sum(
        _passed(base_by_id[problem_id], context=f"base[{problem_id}]") for problem_id in adapter_ids
    )
    adapter_qualification_passed = sum(
        _passed(row, context=f"adapter[{problem_id}]") for problem_id, row in adapter_by_id.items()
    )
    observed_passes = (
        base_full_passed,
        base_qualification_passed,
        adapter_qualification_passed,
    )
    expected_passes = (
        _EXPECTED_BASE_FULL_PASSED,
        _EXPECTED_BASE_QUALIFICATION_PASSED,
        _EXPECTED_ADAPTER_QUALIFICATION_PASSED,
    )
    if observed_passes != expected_passes:
        raise MbppForensicsError(
            f"P9-008 score identity drifted: expected {expected_passes}, got {observed_passes}"
        )

    retained_pass: list[str] = []
    retained_fail: list[str] = []
    regressions: list[str] = []
    improvements: list[str] = []
    for problem_id in sorted(adapter_ids):
        base_passed = _passed(base_by_id[problem_id], context=f"base[{problem_id}]")
        adapter_passed = _passed(adapter_by_id[problem_id], context=f"adapter[{problem_id}]")
        if base_passed and adapter_passed:
            retained_pass.append(problem_id)
        elif base_passed:
            regressions.append(problem_id)
        elif adapter_passed:
            improvements.append(problem_id)
        else:
            retained_fail.append(problem_id)

    regression_categories = Counter(
        _error_category(adapter_by_id[problem_id], context=f"adapter[{problem_id}]")
        for problem_id in regressions
    )
    improvement_base_categories = Counter(
        _error_category(base_by_id[problem_id], context=f"base[{problem_id}]")
        for problem_id in improvements
    )

    code_groups: dict[str, list[str]] = defaultdict(list)
    for problem_id in regressions:
        digest = _generated_code_sha256(adapter_by_id[problem_id], context=f"adapter[{problem_id}]")
        code_groups[digest].append(problem_id)
    duplicate_code_groups = [
        {"generated_code_sha256": digest, "problem_ids": problem_ids}
        for digest, problem_ids in sorted(code_groups.items())
        if len(problem_ids) > 1
    ]

    regression_rows = []
    for problem_id in regressions:
        adapter_row = adapter_by_id[problem_id]
        tests = _mapping(adapter_row.get("tests"), context=f"adapter[{problem_id}].tests")
        regression_rows.append(
            {
                "problem_id": problem_id,
                "adapter_error_category": _error_category(
                    adapter_row, context=f"adapter[{problem_id}]"
                ),
                "adapter_tests_passed": _integer(
                    tests.get("passed"), context=f"adapter[{problem_id}].tests.passed"
                ),
                "adapter_tests_total": _integer(
                    tests.get("total"), context=f"adapter[{problem_id}].tests.total"
                ),
                "base_generated_tokens": _generated_tokens(
                    base_by_id[problem_id], context=f"base[{problem_id}]"
                ),
                "adapter_generated_tokens": _generated_tokens(
                    adapter_row, context=f"adapter[{problem_id}]"
                ),
                "adapter_generated_code_sha256": _generated_code_sha256(
                    adapter_row, context=f"adapter[{problem_id}]"
                ),
            }
        )

    improvement_rows = []
    for problem_id in improvements:
        improvement_rows.append(
            {
                "problem_id": problem_id,
                "base_error_category": _error_category(
                    base_by_id[problem_id], context=f"base[{problem_id}]"
                ),
                "base_generated_tokens": _generated_tokens(
                    base_by_id[problem_id], context=f"base[{problem_id}]"
                ),
                "adapter_generated_tokens": _generated_tokens(
                    adapter_by_id[problem_id], context=f"adapter[{problem_id}]"
                ),
            }
        )

    regression_tokens_base = [
        _generated_tokens(base_by_id[problem_id], context=f"base[{problem_id}]")
        for problem_id in regressions
    ]
    regression_tokens_adapter = [
        _generated_tokens(adapter_by_id[problem_id], context=f"adapter[{problem_id}]")
        for problem_id in regressions
    ]
    adapter_pass_tokens = [
        _generated_tokens(adapter_by_id[problem_id], context=f"adapter[{problem_id}]")
        for problem_id in adapter_ids
        if _passed(adapter_by_id[problem_id], context=f"adapter[{problem_id}]")
    ]
    adapter_fail_tokens = [
        _generated_tokens(adapter_by_id[problem_id], context=f"adapter[{problem_id}]")
        for problem_id in adapter_ids
        if not _passed(adapter_by_id[problem_id], context=f"adapter[{problem_id}]")
    ]

    return {
        "schema_version": 1,
        "task_id": "P9-008",
        "study_id": "p9-007e-mbpp-regression-forensics-v1",
        "inputs": {
            "base_rows": len(base_rows),
            "qualification_rows": len(adapter_rows),
            "qualification_membership_sha256": membership_sha256,
            "base_full_passed": base_full_passed,
            "base_qualification_passed": base_qualification_passed,
            "adapter_qualification_passed": adapter_qualification_passed,
        },
        "flip_matrix": {
            "retained_pass": len(retained_pass),
            "retained_fail": len(retained_fail),
            "regressions": len(regressions),
            "improvements": len(improvements),
            "net_pass_delta": len(improvements) - len(regressions),
        },
        "regression_error_categories": dict(sorted(regression_categories.items())),
        "improvement_base_error_categories": dict(sorted(improvement_base_categories.items())),
        "regression_task_ids": regressions,
        "improvement_task_ids": improvements,
        "duplicate_regression_code_groups": duplicate_code_groups,
        "token_diagnostics": {
            "regressions_base": _token_summary(regression_tokens_base),
            "regressions_adapter": _token_summary(regression_tokens_adapter),
            "adapter_passes": _token_summary(adapter_pass_tokens),
            "adapter_failures": _token_summary(adapter_fail_tokens),
        },
        "regression_rows": regression_rows,
        "improvement_rows": improvement_rows,
        "conclusion": {
            "dominant_failure_mode": "semantic_or_test_failure",
            "test_failure_fraction_of_regressions": regression_categories.get("test", 0)
            / len(regressions),
            "truncation_is_dominant": False,
            "harness_failure_detected": regression_categories.get("harness", 0) > 0,
            "recommended_next_experiment": (
                "redesign distillation around independently executable semantic verification; "
                "do not run another rank/LR sweep or scale teacher data yet"
            ),
        },
    }


def write_forensics(summary: Mapping[str, object], output_dir: Path) -> tuple[Path, Path]:
    """Write the aggregate report and a compact task-flip JSONL."""

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    flips_path = output_dir / "task-flips.jsonl"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    regressions = summary.get("regression_rows")
    improvements = summary.get("improvement_rows")
    if not isinstance(regressions, list) or not isinstance(improvements, list):
        raise MbppForensicsError("forensics summary is missing task rows")
    lines = []
    for disposition, rows in (("regression", regressions), ("improvement", improvements)):
        for row in rows:
            mapping = _mapping(row, context=f"{disposition} row")
            payload = {"disposition": disposition, **mapping}
            lines.append(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    flips_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path, flips_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-results", type=Path, required=True)
    parser.add_argument("--qualification-results", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/eval/python/p9-008-mbpp-forensics"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run exact offline P9-008 comparison from frozen result artifacts."""

    args = _build_parser().parse_args(argv)
    summary = analyze_mbpp_results(
        load_result_rows(args.base_results),
        load_result_rows(args.qualification_results),
    )
    summary_path, flips_path = write_forensics(summary, args.output_dir)
    print(summary_path)
    print(flips_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
