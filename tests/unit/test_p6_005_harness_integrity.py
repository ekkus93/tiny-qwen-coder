"""Regression tests for P6-005 baseline execution integrity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tiny_qwen_coder.evaluation._baseline_artifacts import _require_valid_coding_aggregates
from tiny_qwen_coder.evaluation._baseline_types import PythonBaselineError
from tiny_qwen_coder.evaluation.execution import _COPY_AND_EXEC_SCRIPT

_CODING_AGGREGATES = (
    "humaneval/humaneval-aggregate.json",
    "mbpp/mbpp-aggregate.json",
    "repository-holdout/repository-holdout-aggregate.json",
)


def _write_coding_aggregates(
    root: Path,
    *,
    harness_errors: int = 0,
    pass_at_1: object = 0.0,
) -> None:
    payload = json.dumps({"harness_errors": harness_errors, "pass_at_1": pass_at_1}) + "\n"
    for relative_path in _CODING_AGGREGATES:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")


def test_workspace_bootstrap_never_chmods_the_tmpfs_mount_root() -> None:
    assert "chmod -R u+rwX /workspace;" not in _COPY_AND_EXEC_SCRIPT
    assert 'chmod -R u+rwX "$path"' in _COPY_AND_EXEC_SCRIPT


def test_valid_coding_aggregates_allow_zero_pass_rate(tmp_path: Path) -> None:
    _write_coding_aggregates(tmp_path, harness_errors=0, pass_at_1=0.0)
    _require_valid_coding_aggregates(tmp_path)


@pytest.mark.parametrize("harness_errors", [1, 164])
def test_coding_aggregate_harness_errors_fail_closed(
    tmp_path: Path,
    harness_errors: int,
) -> None:
    _write_coding_aggregates(tmp_path, harness_errors=harness_errors, pass_at_1=None)
    with pytest.raises(PythonBaselineError, match="harness errors"):
        _require_valid_coding_aggregates(tmp_path)


def test_null_pass_at_1_without_harness_errors_fails_closed(tmp_path: Path) -> None:
    _write_coding_aggregates(tmp_path, harness_errors=0, pass_at_1=None)
    with pytest.raises(PythonBaselineError, match="pass_at_1 must be a numeric value"):
        _require_valid_coding_aggregates(tmp_path)
