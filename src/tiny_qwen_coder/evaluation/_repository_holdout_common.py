"""Shared contract, runner source, and validation helpers for the Python holdout."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from tiny_qwen_coder.config import ExecutionConfig
from tiny_qwen_coder.evaluation.execution import (
    ExecutionLimits,
    ExecutionRequest,
    ExecutionResult,
)

_REPOSITORY_HOLDOUT_SCHEMA_VERSION = 1
_REPOSITORY_HOLDOUT_SUITE_ID = "repository-holdout"
_REPOSITORY_HOLDOUT_SUITE_VERSION = 1
_REPOSITORY_HOLDOUT_DATASET_ID = "repository://tiny-qwen-coder/python-holdout"
_REPOSITORY_HOLDOUT_DATASET_REVISION = "repository-holdout-v1"
_REPOSITORY_HOLDOUT_BENCHMARK_CONFIG_PATH = Path("configs/eval/python/repository_holdout.yaml")
_REPOSITORY_HOLDOUT_SUITE_CONFIG_PATH = Path(
    "configs/eval/python/repository_holdout_suite_v1.yaml"
)
_REPOSITORY_HOLDOUT_ASSET_ROOT = Path("benchmarks/python/repository_holdout_v1")
_FROZEN_REPOSITORY_HOLDOUT_SUITE_SHA256 = (
    "91d87aa5d1fb5041d9d26e6c8bfbeb958fa406a3aa24f0f1966ded9816f8252e"
)
_TASK_ID_PATTERN = re.compile(r"^[a-z][a-z0-9-]*$")
_IMAGE_DIGEST_PATTERN = re.compile(r"^.+@sha256:[0-9a-f]{64}$")
_PYTHON_FENCE_PATTERN = re.compile(
    r"```(?:python)?\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)
_EXPECTED_CATEGORIES = frozenset(
    {
        "standard-library-data-transforms",
        "pathlib",
        "json",
        "regex",
        "dataclasses-typing",
        "generators-decorators-context-managers",
        "exceptions",
        "async-await",
        "subprocess",
        "sqlite",
        "pytest-oriented",
    }
)

_PARSE_OK_MARKER = "__TQC_REPOSITORY_HOLDOUT_PARSE_OK_V1__"
_COMPILE_OK_MARKER = "__TQC_REPOSITORY_HOLDOUT_COMPILE_OK_V1__"
_TEST_OK_MARKER = "__TQC_REPOSITORY_HOLDOUT_TEST_OK_V1__:"
_SUCCESS_MARKER = "__TQC_REPOSITORY_HOLDOUT_PASS_V1__"
_ERROR_MARKER = "__TQC_REPOSITORY_HOLDOUT_ERROR_V1__:"
_RUNNER_PARSE_EXIT = 10
_RUNNER_COMPILE_EXIT = 11
_RUNNER_TEST_EXIT = 12
_RUNNER_RUNTIME_EXIT = 13
_RUNNER_HARNESS_EXIT = 20

_REPOSITORY_HOLDOUT_RUNNER_SOURCE = f"""\
from __future__ import annotations

import ast
import json
import sys
import types
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
test_source = Path("tests.py").read_text(encoding="utf-8")
metadata = json.loads(Path("metadata.json").read_text(encoding="utf-8"))
expected_tests = metadata.get("expected_tests")
if not isinstance(expected_tests, int) or isinstance(expected_tests, bool) or expected_tests <= 0:
    fail({_RUNNER_HARNESS_EXIT}, "benchmark", TypeError("expected_tests must be positive int"))

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
    test_code = compile(test_source, "tests.py", "exec")
except BaseException as exc:
    fail({_RUNNER_HARNESS_EXIT}, "benchmark", exc)

candidate_module = types.ModuleType("candidate")
namespace = candidate_module.__dict__
namespace["__candidate_source__"] = candidate_source
sys.modules["candidate"] = candidate_module
try:
    if setup_code is not None:
        exec(setup_code, namespace)
except BaseException as exc:
    fail({_RUNNER_HARNESS_EXIT}, "setup", exc)

try:
    exec(candidate_code, namespace)
except BaseException as exc:
    fail({_RUNNER_RUNTIME_EXIT}, "runtime", exc)

try:
    exec(test_code, namespace)
except BaseException as exc:
    fail({_RUNNER_HARNESS_EXIT}, "benchmark", exc)

tests = namespace.get("TESTS")
if not isinstance(tests, (tuple, list)) or len(tests) != expected_tests:
    fail(
        {_RUNNER_HARNESS_EXIT},
        "benchmark",
        TypeError("TESTS must contain exactly expected_tests callables"),
    )
if not all(callable(test) for test in tests):
    fail({_RUNNER_HARNESS_EXIT}, "benchmark", TypeError("TESTS entries must be callable"))

for index, test in enumerate(tests, start=1):
    test_name = getattr(test, "__name__", f"test_{{index}}")
    try:
        test()
    except BaseException as exc:
        fail({_RUNNER_TEST_EXIT}, f"test[{{index}}:{{test_name}}]", exc)
    print(f"{{TEST_OK}}{{index}}", file=sys.stderr, flush=True)

print(PASS, file=sys.stderr, flush=True)
"""


class RepositoryHoldoutError(ValueError):
    """Raised when the repository-owned Python holdout contract is invalid."""


class RepositoryHoldoutHarness(Protocol):
    """Minimal constrained-execution protocol used by the evaluator."""

    def run(
        self,
        request: ExecutionRequest,
        execution: ExecutionConfig,
        *,
        limits: ExecutionLimits | None = None,
    ) -> ExecutionResult: ...


def _require_non_empty(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RepositoryHoldoutError(f"{field_name} must be a non-empty string")
    if value != value.strip():
        raise RepositoryHoldoutError(f"{field_name} must not contain outer whitespace")
    return value


def _require_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise RepositoryHoldoutError(f"{field_name} must be a non-empty string")
    if "\x00" in value:
        raise RepositoryHoldoutError(f"{field_name} must not contain NUL bytes")
    return value


def _require_sha256(value: str, *, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise RepositoryHoldoutError(f"{field_name} must be a lowercase SHA-256 digest")


def _strict_mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise RepositoryHoldoutError(f"{context} must be a mapping")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise RepositoryHoldoutError(f"{context} keys must be strings")
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
        raise RepositoryHoldoutError(f"{context} contains unknown field(s): {', '.join(unknown)}")
    if missing:
        raise RepositoryHoldoutError(
            f"{context} is missing required field(s): {', '.join(missing)}"
        )


def _expect_int(mapping: Mapping[str, object], key: str, *, context: str) -> int:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise RepositoryHoldoutError(f"{context}.{key} must be an integer")
    return value


def _expect_float(mapping: Mapping[str, object], key: str, *, context: str) -> float:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RepositoryHoldoutError(f"{context}.{key} must be a number")
    return float(value)


def _expect_bool(mapping: Mapping[str, object], key: str, *, context: str) -> bool:
    value = mapping[key]
    if not isinstance(value, bool):
        raise RepositoryHoldoutError(f"{context}.{key} must be a boolean")
    return value


def _expect_str(mapping: Mapping[str, object], key: str, *, context: str) -> str:
    return _require_non_empty(mapping[key], field_name=f"{context}.{key}")


def _expect_text(mapping: Mapping[str, object], key: str, *, context: str) -> str:
    return _require_text(mapping[key], field_name=f"{context}.{key}")


def _expect_optional_str(
    mapping: Mapping[str, object],
    key: str,
    *,
    context: str,
) -> str | None:
    value = mapping[key]
    if value is None:
        return None
    return _require_non_empty(value, field_name=f"{context}.{key}")


def _parse_limits(value: object) -> ExecutionLimits:
    context = "repository holdout.limits"
    mapping = _strict_mapping(value, context=context)
    _validate_keys(
        mapping,
        required=frozenset(
            {
                "cpus",
                "memory_mebibytes",
                "pids",
                "workspace_mebibytes",
                "temp_mebibytes",
                "max_output_bytes",
                "max_input_bytes",
                "open_files",
                "cleanup_timeout_seconds",
            }
        ),
        context=context,
    )
    return ExecutionLimits(
        cpus=_expect_float(mapping, "cpus", context=context),
        memory_mebibytes=_expect_int(mapping, "memory_mebibytes", context=context),
        pids=_expect_int(mapping, "pids", context=context),
        workspace_mebibytes=_expect_int(mapping, "workspace_mebibytes", context=context),
        temp_mebibytes=_expect_int(mapping, "temp_mebibytes", context=context),
        max_output_bytes=_expect_int(mapping, "max_output_bytes", context=context),
        max_input_bytes=_expect_int(mapping, "max_input_bytes", context=context),
        open_files=_expect_int(mapping, "open_files", context=context),
        cleanup_timeout_seconds=_expect_float(
            mapping,
            "cleanup_timeout_seconds",
            context=context,
        ),
    )


def _safe_asset_path(path_text: str, *, field_name: str) -> Path:
    path = Path(path_text)
    if path.is_absolute() or ".." in path.parts:
        raise RepositoryHoldoutError(f"{field_name} must be a repository-relative safe path")
    try:
        path.relative_to(_REPOSITORY_HOLDOUT_ASSET_ROOT)
    except ValueError as exc:
        raise RepositoryHoldoutError(
            f"{field_name} must stay under {_REPOSITORY_HOLDOUT_ASSET_ROOT.as_posix()}"
        ) from exc
    return path


def _read_asset(path_text: str, *, field_name: str) -> tuple[str, str]:
    path = _safe_asset_path(path_text, field_name=field_name)
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RepositoryHoldoutError(f"could not read {field_name} {path}: {exc}") from exc
    if not source:
        raise RepositoryHoldoutError(f"{field_name} must not be empty")
    try:
        compile(source, path.as_posix(), "exec")
    except SyntaxError as exc:
        raise RepositoryHoldoutError(f"{field_name} is not valid Python: {exc}") from exc
    return source, hashlib.sha256(source.encode("utf-8")).hexdigest()
