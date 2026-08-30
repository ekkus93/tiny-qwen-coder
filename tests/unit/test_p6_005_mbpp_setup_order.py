"""Regression coverage for MBPP setup code that depends on candidate symbols."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tiny_qwen_coder.evaluation._mbpp_common import _MBPP_RUNNER_SOURCE


def _run_runner(
    tmp_path: Path,
    *,
    candidate: str,
    setup: str,
    tests: tuple[str, ...],
) -> subprocess.CompletedProcess[str]:
    (tmp_path / "runner.py").write_text(_MBPP_RUNNER_SOURCE, encoding="utf-8")
    (tmp_path / "candidate.py").write_text(candidate, encoding="utf-8")
    (tmp_path / "setup.py").write_text(setup, encoding="utf-8")
    (tmp_path / "tests.json").write_text(json.dumps(list(tests)), encoding="utf-8")
    return subprocess.run(
        [sys.executable, "-I", "-B", "runner.py"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )


def test_runner_executes_candidate_before_setup_that_uses_candidate_class(tmp_path: Path) -> None:
    completed = _run_runner(
        tmp_path,
        candidate=(
            "class Node:\n"
            "    def __init__(self, data):\n"
            "        self.data = data\n"
            "        self.left = None\n"
        ),
        setup="root = Node(1)\nroot.left = Node(2)\n",
        tests=(
            "assert root.data == 1",
            "assert root.left.data == 2",
            "assert isinstance(root, Node)",
        ),
    )

    assert completed.returncode == 0
    assert "__TQC_MBPP_TEST_OK_V1__:3" in completed.stderr
    assert "__TQC_MBPP_PASS_V1__" in completed.stderr


def test_missing_candidate_symbol_in_setup_is_test_failure_not_harness_error(
    tmp_path: Path,
) -> None:
    completed = _run_runner(
        tmp_path,
        candidate="from typing import Optional\n",
        setup="root = Node(1)\n",
        tests=("assert root is not None", "assert True", "assert True"),
    )

    assert completed.returncode == 12
    assert "__TQC_MBPP_PARSE_OK_V1__" in completed.stderr
    assert "__TQC_MBPP_COMPILE_OK_V1__" in completed.stderr
    assert "__TQC_MBPP_ERROR_V1__:setup:NameError:name 'Node' is not defined" in completed.stderr
    assert "__TQC_MBPP_ERROR_V1__:benchmark:" not in completed.stderr
