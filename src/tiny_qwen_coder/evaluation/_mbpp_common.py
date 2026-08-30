"""Shared constants, runner source, and validation helpers for MBPP."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Protocol

from tiny_qwen_coder.config import ExecutionConfig
from tiny_qwen_coder.evaluation.execution import (
    ExecutionLimits,
    ExecutionRequest,
    ExecutionResult,
)

_MBPP_SCHEMA_VERSION = 1
_MBPP_RUNNER_ID = "mbpp"
_MBPP_RUNNER_VERSION = 1
_MBPP_BENCHMARK_ID = "mbpp"
_MBPP_DATASET_ID = "google-research-datasets/mbpp"
_MBPP_BENCHMARK_CONFIG_PATH = Path("configs/eval/python/mbpp.yaml")
_MBPP_RUNNER_CONFIG_PATH = Path("configs/eval/python/mbpp_runner_v1.yaml")
_FROZEN_MBPP_RUNNER_SHA256 = "1aef848a963ebe07fdf4c9631df94b8668b24af7fa3d9edcba650bd1fe622061"
_TASK_ID_PATTERN = re.compile(r"^MBPP/([1-9][0-9]*)$")
_GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_IMAGE_DIGEST_PATTERN = re.compile(r"^.+@sha256:[0-9a-f]{64}$")
_PYTHON_FENCE_PATTERN = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_PARSE_OK_MARKER = "__TQC_MBPP_PARSE_OK_V1__"
_COMPILE_OK_MARKER = "__TQC_MBPP_COMPILE_OK_V1__"
_TEST_OK_MARKER = "__TQC_MBPP_TEST_OK_V1__:"
_SUCCESS_MARKER = "__TQC_MBPP_PASS_V1__"
_ERROR_MARKER = "__TQC_MBPP_ERROR_V1__:"
_RUNNER_PARSE_EXIT = 10
_RUNNER_COMPILE_EXIT = 11
_RUNNER_TEST_EXIT = 12
_RUNNER_RUNTIME_EXIT = 13
_RUNNER_HARNESS_EXIT = 20

_MBPP_RUNNER_SOURCE = f"""\
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

PARSE_OK = {_PARSE_OK_MARKER!r}
COMPILE_OK = {_COMPILE_OK_MARKER!r}
TEST_OK = {_TEST_OK_MARKER!r}
PASS = {_SUCCESS_MARKER!r}
ERROR = {_ERROR_MARKER!r}


def fail(code: int, stage: str, exc: BaseException) -> None:
    detail = str(exc).replace("\\n", "\\\\n")[:2000]
    print(f"{{ERROR}}{{stage}}:{{type(exc).__name__}}:{{detail}}", file=sys.stderr, flush=True)
    raise SystemExit(code)


candidate_source = Path("candidate.py").read_text(encoding="utf-8")
setup_source = Path("setup.py").read_text(encoding="utf-8")
tests = json.loads(Path("tests.json").read_text(encoding="utf-8"))
if not isinstance(tests, list) or not all(isinstance(test, str) for test in tests):
    fail(
        {_RUNNER_HARNESS_EXIT},
        "benchmark",
        TypeError("tests.json must contain a list of strings"),
    )

try:
    ast.parse(candidate_source, filename="candidate.py")
except SyntaxError as exc:
    fail({_RUNNER_PARSE_EXIT}, "parse", exc)
print(PARSE_OK, file=sys.stderr, flush=True)

try:
    candidate_code = compile(candidate_source, "candidate.py", "exec")
except BaseException as exc:
    fail({_RUNNER_COMPILE_EXIT}, "compile", exc)
print(COMPILE_OK, file=sys.stderr, flush=True)

try:
    setup_code = compile(setup_source, "setup.py", "exec") if setup_source else None
    test_code = [compile(test, f"test_{{index}}.py", "exec") for index, test in enumerate(tests)]
except BaseException as exc:
    fail({_RUNNER_HARNESS_EXIT}, "benchmark", exc)

namespace: dict[str, object] = {{}}
try:
    if setup_code is not None:
        exec(setup_code, namespace)
except BaseException as exc:
    fail({_RUNNER_HARNESS_EXIT}, "benchmark", exc)

try:
    exec(candidate_code, namespace)
except BaseException as exc:
    fail({_RUNNER_RUNTIME_EXIT}, "runtime", exc)

for index, compiled_test in enumerate(test_code, start=1):
    try:
        exec(compiled_test, namespace)
    except BaseException as exc:
        fail({_RUNNER_TEST_EXIT}, f"test[{{index}}]", exc)
    print(f"{{TEST_OK}}{{index}}", file=sys.stderr, flush=True)

print(PASS, file=sys.stderr, flush=True)
"""

DatasetRow = Mapping[str, object]
MBPPDatasetRowsLoader = Callable[..., Iterable[DatasetRow]]


class MBPPError(ValueError):
    """Raised when MBPP configuration, data, or scoring is invalid."""


class MBPPHarness(Protocol):
    """Minimal execution-harness protocol used for testable MBPP scoring."""

    def run(
        self,
        request: ExecutionRequest,
        execution: ExecutionConfig,
        *,
        limits: ExecutionLimits | None = None,
    ) -> ExecutionResult: ...


def _require_non_empty(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MBPPError(f"{field_name} must be a non-empty string")
    if value != value.strip():
        raise MBPPError(f"{field_name} must not contain outer whitespace")
    return value


def _require_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise MBPPError(f"{field_name} must be a non-empty string")
    if "\x00" in value:
        raise MBPPError(f"{field_name} must not contain NUL bytes")
    return value


def _require_sha256(value: str, *, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise MBPPError(f"{field_name} must be a lowercase SHA-256 digest")


def _strict_mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise MBPPError(f"{context} must be a mapping")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise MBPPError(f"{context} keys must be strings")
        result[key] = item
    return result


def _validate_keys(
    mapping: Mapping[str, object],
    *,
    required: frozenset[str],
    context: str,
) -> None:
    unknown = sorted(set(mapping) - required)
    missing = sorted(required - set(mapping))
    if unknown:
        raise MBPPError(f"{context} contains unknown field(s): {', '.join(unknown)}")
    if missing:
        raise MBPPError(f"{context} is missing required field(s): {', '.join(missing)}")


def _expect_int(mapping: Mapping[str, object], key: str, *, context: str) -> int:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise MBPPError(f"{context}.{key} must be an integer")
    return value


def _expect_float(mapping: Mapping[str, object], key: str, *, context: str) -> float:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise MBPPError(f"{context}.{key} must be a number")
    return float(value)


def _expect_bool(mapping: Mapping[str, object], key: str, *, context: str) -> bool:
    value = mapping[key]
    if not isinstance(value, bool):
        raise MBPPError(f"{context}.{key} must be a boolean")
    return value


def _expect_str(mapping: Mapping[str, object], key: str, *, context: str) -> str:
    return _require_non_empty(mapping[key], field_name=f"{context}.{key}")


def _expect_str_tuple(mapping: Mapping[str, object], key: str, *, context: str) -> tuple[str, ...]:
    value = mapping[key]
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise MBPPError(f"{context}.{key} must be a list of strings")
    items: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise MBPPError(f"{context}.{key}[{index}] must be a string")
        items.append(item)
    return tuple(items)
