"""Deterministic HumanEval loading, prompting, sandboxed scoring, and artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, cast

import yaml

from tiny_qwen_coder.config import EvaluationConfig, ExecutionConfig
from tiny_qwen_coder.evaluation.execution import (
    ConstrainedExecutionHarness,
    ExecutionFile,
    ExecutionHarnessError,
    ExecutionLimits,
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
)
from tiny_qwen_coder.evaluation.protected_benchmarks import (
    ProtectedBenchmark,
    load_protected_benchmark_config,
)
from tiny_qwen_coder.evaluation.results import (
    EvaluationErrorCategory,
    EvaluationResult,
    EvaluationStageStatus,
    EvaluationTestSummary,
    GenerationStats,
    create_evaluation_result,
)
from tiny_qwen_coder.evaluation.settings import (
    FrozenEvaluationSettings,
    load_frozen_evaluation_settings,
    validate_evaluation_config_settings,
)
from tiny_qwen_coder.identities import AdapterIdentity, BaseModelIdentity

_HUMANEVAL_SCHEMA_VERSION = 1
_HUMANEVAL_RUNNER_ID = "humaneval"
_HUMANEVAL_RUNNER_VERSION = 1
_HUMANEVAL_BENCHMARK_ID = "humaneval"
_HUMANEVAL_DATASET_ID = "openai/openai_humaneval"
_HUMANEVAL_BENCHMARK_CONFIG_PATH = Path("configs/eval/python/humaneval.yaml")
_HUMANEVAL_RUNNER_CONFIG_PATH = Path("configs/eval/python/humaneval_runner_v1.yaml")
_FROZEN_HUMANEVAL_RUNNER_SHA256 = "b2f405cdd05551ac858d75eb13aedab591b79522b493fbe0ad247a0f3b677e19"
_TASK_ID_PATTERN = re.compile(r"^HumanEval/(0|[1-9][0-9]*)$")
_GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_IMAGE_DIGEST_PATTERN = re.compile(r"^.+@sha256:[0-9a-f]{64}$")
_PYTHON_FENCE_PATTERN = re.compile(r"```python\n(.*?)```", re.DOTALL)
_PARSE_OK_MARKER = "__TQC_HUMANEVAL_PARSE_OK_V1__"
_COMPILE_OK_MARKER = "__TQC_HUMANEVAL_COMPILE_OK_V1__"
_SUCCESS_MARKER = "__TQC_HUMANEVAL_PASS_V1__"
_ERROR_MARKER = "__TQC_HUMANEVAL_ERROR_V1__:"
_RUNNER_PARSE_EXIT = 10
_RUNNER_COMPILE_EXIT = 11
_RUNNER_TEST_EXIT = 12
_RUNNER_HARNESS_EXIT = 20

_HUMANEVAL_RUNNER_SOURCE = f"""\
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

PARSE_OK = {_PARSE_OK_MARKER!r}
COMPILE_OK = {_COMPILE_OK_MARKER!r}
PASS = {_SUCCESS_MARKER!r}
ERROR = {_ERROR_MARKER!r}


def fail(code: int, stage: str, exc: BaseException) -> None:
    detail = str(exc).replace("\\n", "\\\\n")[:2000]
    print(f"{{ERROR}}{{stage}}:{{type(exc).__name__}}:{{detail}}", file=sys.stderr, flush=True)
    raise SystemExit(code)


candidate_source = Path("candidate.py").read_text(encoding="utf-8")
test_source = Path("tests.py").read_text(encoding="utf-8")
metadata = json.loads(Path("metadata.json").read_text(encoding="utf-8"))
entry_point = metadata["entry_point"]

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
    test_code = compile(test_source, "tests.py", "exec")
except BaseException as exc:
    fail({_RUNNER_HARNESS_EXIT}, "benchmark", exc)

namespace: dict[str, object] = {{}}
try:
    exec(candidate_code, namespace)
    exec(test_code, namespace)
    checker = namespace["check"]
    candidate = namespace[entry_point]
    checker(candidate)
except BaseException as exc:
    fail({_RUNNER_TEST_EXIT}, "test", exc)

print(PASS, file=sys.stderr, flush=True)
"""

DatasetRow = Mapping[str, object]
DatasetRowsLoader = Callable[..., Iterable[DatasetRow]]


class HumanEvalError(ValueError):
    """Raised when HumanEval configuration, data, or scoring is invalid."""


class HumanEvalHarness(Protocol):
    """Minimal execution-harness protocol used for testable HumanEval scoring."""

    def run(
        self,
        request: ExecutionRequest,
        execution: ExecutionConfig,
        *,
        limits: ExecutionLimits | None = None,
    ) -> ExecutionResult: ...


@dataclass(frozen=True, slots=True)
class HumanEvalRunnerConfig:
    """Frozen HumanEval protocol and constrained-runtime policy."""

    schema_version: int
    runner_id: str
    runner_version: int
    frozen: bool
    benchmark_id: str
    reference_repository: str
    reference_revision: str
    dataset_split: str
    expected_problem_count: int
    prompt_version: str
    completion_normalizer_version: str
    instruction: str
    execution_image: str
    execution_timeout_seconds: float
    limits: ExecutionLimits

    def __post_init__(self) -> None:
        if self.schema_version != _HUMANEVAL_SCHEMA_VERSION:
            raise HumanEvalError("unsupported HumanEval runner schema version")
        if self.runner_id != _HUMANEVAL_RUNNER_ID:
            raise HumanEvalError(f"HumanEval runner_id must be {_HUMANEVAL_RUNNER_ID!r}")
        if self.runner_version != _HUMANEVAL_RUNNER_VERSION:
            raise HumanEvalError(f"HumanEval runner_version must be {_HUMANEVAL_RUNNER_VERSION}")
        if not self.frozen:
            raise HumanEvalError("HumanEval runner configuration must be frozen")
        if self.benchmark_id != _HUMANEVAL_BENCHMARK_ID:
            raise HumanEvalError(f"HumanEval benchmark_id must be {_HUMANEVAL_BENCHMARK_ID!r}")
        _require_non_empty(self.reference_repository, field_name="reference_repository")
        if not _GIT_SHA_PATTERN.fullmatch(self.reference_revision):
            raise HumanEvalError("reference_revision must be a lowercase 40-character Git SHA")
        if self.dataset_split != "test":
            raise HumanEvalError("HumanEval dataset_split must be 'test'")
        if self.expected_problem_count <= 0:
            raise HumanEvalError("expected_problem_count must be greater than zero")
        _require_non_empty(self.prompt_version, field_name="prompt_version")
        _require_non_empty(
            self.completion_normalizer_version,
            field_name="completion_normalizer_version",
        )
        _require_non_empty(self.instruction, field_name="instruction")
        if not _IMAGE_DIGEST_PATTERN.fullmatch(self.execution_image):
            raise HumanEvalError("execution_image must be pinned by an exact sha256 digest")
        if self.execution_timeout_seconds <= 0:
            raise HumanEvalError("execution_timeout_seconds must be greater than zero")
        if not isinstance(self.limits, ExecutionLimits):
            raise HumanEvalError("limits must be ExecutionLimits")


@dataclass(frozen=True, slots=True)
class HumanEvalProblem:
    """One protected HumanEval problem without its canonical solution."""

    task_id: str
    prompt: str
    test: str
    entry_point: str

    def __post_init__(self) -> None:
        if not _TASK_ID_PATTERN.fullmatch(self.task_id):
            raise HumanEvalError("task_id must match HumanEval/<non-negative integer>")
        _require_text(self.prompt, field_name="prompt")
        _require_text(self.test, field_name="test")
        if not self.entry_point.isidentifier():
            raise HumanEvalError("entry_point must be a valid Python identifier")

    @property
    def task_index(self) -> int:
        """Return the numeric HumanEval task index."""

        return int(self.task_id.partition("/")[2])


@dataclass(frozen=True, slots=True)
class HumanEvalPrompt:
    """Normalized single-user-message prompt for one HumanEval task."""

    task_id: str
    prompt_version: str
    user_content: str

    def __post_init__(self) -> None:
        if not _TASK_ID_PATTERN.fullmatch(self.task_id):
            raise HumanEvalError("prompt task_id must match HumanEval/<non-negative integer>")
        _require_non_empty(self.prompt_version, field_name="prompt_version")
        _require_text(self.user_content, field_name="user_content")


@dataclass(frozen=True, slots=True)
class HumanEvalCompletion:
    """One generated response plus common generation statistics."""

    task_id: str
    generated_text: str
    generation: GenerationStats
    generation_error: str | None = None

    def __post_init__(self) -> None:
        if not _TASK_ID_PATTERN.fullmatch(self.task_id):
            raise HumanEvalError("completion task_id must match HumanEval/<non-negative integer>")
        if not isinstance(self.generated_text, str):
            raise HumanEvalError("generated_text must be a string")
        if not isinstance(self.generation, GenerationStats):
            raise HumanEvalError("generation must be GenerationStats")
        if self.generation_error is not None:
            _require_non_empty(self.generation_error, field_name="generation_error")


@dataclass(frozen=True, slots=True)
class HumanEvalAggregate:
    """Deterministic aggregate HumanEval pass@1 artifact."""

    schema_version: int
    benchmark_id: str
    dataset_id: str
    dataset_revision: str
    dataset_split: str
    reference_repository: str
    reference_revision: str
    runner_config_sha256: str
    runner_source_sha256: str
    evaluation_settings_sha256: str
    prompt_version: str
    completion_normalizer_version: str
    execution_image: str
    base_model: BaseModelIdentity
    adapter: AdapterIdentity
    total_problems: int
    passed: int
    failed: int
    timed_out: int
    harness_errors: int
    pass_at_1: float | None
    problem_set_sha256: str
    results_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != _HUMANEVAL_SCHEMA_VERSION:
            raise HumanEvalError("unsupported HumanEval aggregate schema version")
        if self.benchmark_id != _HUMANEVAL_BENCHMARK_ID:
            raise HumanEvalError("HumanEval aggregate benchmark_id is invalid")
        if self.dataset_id != _HUMANEVAL_DATASET_ID:
            raise HumanEvalError("HumanEval aggregate dataset_id is invalid")
        if not _GIT_SHA_PATTERN.fullmatch(self.dataset_revision):
            raise HumanEvalError(
                "HumanEval aggregate dataset_revision must be an immutable Git SHA"
            )
        if self.dataset_split != "test":
            raise HumanEvalError("HumanEval aggregate dataset_split must be 'test'")
        _require_non_empty(self.reference_repository, field_name="reference_repository")
        if not _GIT_SHA_PATTERN.fullmatch(self.reference_revision):
            raise HumanEvalError("HumanEval aggregate reference_revision must be a Git SHA")
        _require_non_empty(self.prompt_version, field_name="prompt_version")
        _require_non_empty(
            self.completion_normalizer_version,
            field_name="completion_normalizer_version",
        )
        if not _IMAGE_DIGEST_PATTERN.fullmatch(self.execution_image):
            raise HumanEvalError("HumanEval aggregate execution_image must be digest-pinned")
        if not isinstance(self.base_model, BaseModelIdentity):
            raise HumanEvalError("HumanEval aggregate base_model must be BaseModelIdentity")
        if not isinstance(self.adapter, AdapterIdentity):
            raise HumanEvalError("HumanEval aggregate adapter must be AdapterIdentity")
        if self.total_problems <= 0:
            raise HumanEvalError("HumanEval aggregate total_problems must be positive")
        if self.passed < 0 or self.failed < 0 or self.passed + self.failed != self.total_problems:
            raise HumanEvalError("HumanEval aggregate pass/fail counts are inconsistent")
        if not 0 <= self.timed_out <= self.failed:
            raise HumanEvalError("HumanEval aggregate timed_out count is inconsistent")
        if not 0 <= self.harness_errors <= self.failed:
            raise HumanEvalError("HumanEval aggregate harness_errors count is inconsistent")
        if self.harness_errors:
            if self.pass_at_1 is not None:
                raise HumanEvalError("pass_at_1 must be null when harness errors occurred")
        else:
            expected = self.passed / self.total_problems
            if self.pass_at_1 != expected:
                raise HumanEvalError("HumanEval aggregate pass_at_1 is inconsistent")
        for field_name, value in (
            ("runner_config_sha256", self.runner_config_sha256),
            ("runner_source_sha256", self.runner_source_sha256),
            ("evaluation_settings_sha256", self.evaluation_settings_sha256),
            ("problem_set_sha256", self.problem_set_sha256),
            ("results_sha256", self.results_sha256),
        ):
            _require_sha256(value, field_name=field_name)


@dataclass(frozen=True, slots=True)
class HumanEvalSuiteResult:
    """Per-problem common results plus one aggregate HumanEval score."""

    results: tuple[EvaluationResult, ...]
    aggregate: HumanEvalAggregate

    def __post_init__(self) -> None:
        if len(self.results) != self.aggregate.total_problems:
            raise HumanEvalError("HumanEval result count must match aggregate total_problems")
        if humaneval_results_sha256(self.results) != self.aggregate.results_sha256:
            raise HumanEvalError("HumanEval aggregate results_sha256 does not match results")
        passed = sum(
            result.error_category is EvaluationErrorCategory.NONE for result in self.results
        )
        timed_out = sum(
            result.error_category is EvaluationErrorCategory.TIMEOUT for result in self.results
        )
        harness_errors = sum(
            result.error_category is EvaluationErrorCategory.HARNESS for result in self.results
        )
        if passed != self.aggregate.passed:
            raise HumanEvalError("HumanEval aggregate passed count does not match results")
        if timed_out != self.aggregate.timed_out:
            raise HumanEvalError("HumanEval aggregate timed_out count does not match results")
        if harness_errors != self.aggregate.harness_errors:
            raise HumanEvalError("HumanEval aggregate harness_errors count does not match results")


@dataclass(frozen=True, slots=True)
class HumanEvalArtifactPaths:
    """Materialized per-problem and aggregate HumanEval artifacts."""

    per_problem: Path
    aggregate: Path


def _require_non_empty(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HumanEvalError(f"{field_name} must be a non-empty string")
    if value != value.strip():
        raise HumanEvalError(f"{field_name} must not contain outer whitespace")
    return value


def _require_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise HumanEvalError(f"{field_name} must be a non-empty string")
    if "\x00" in value:
        raise HumanEvalError(f"{field_name} must not contain NUL bytes")
    return value


def _require_sha256(value: str, *, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise HumanEvalError(f"{field_name} must be a lowercase SHA-256 digest")


def _strict_mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise HumanEvalError(f"{context} must be a mapping")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise HumanEvalError(f"{context} keys must be strings")
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
        raise HumanEvalError(f"{context} contains unknown field(s): {', '.join(unknown)}")
    if missing:
        raise HumanEvalError(f"{context} is missing required field(s): {', '.join(missing)}")


def _expect_int(mapping: Mapping[str, object], key: str, *, context: str) -> int:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise HumanEvalError(f"{context}.{key} must be an integer")
    return value


def _expect_float(mapping: Mapping[str, object], key: str, *, context: str) -> float:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HumanEvalError(f"{context}.{key} must be a number")
    return float(value)


def _expect_bool(mapping: Mapping[str, object], key: str, *, context: str) -> bool:
    value = mapping[key]
    if not isinstance(value, bool):
        raise HumanEvalError(f"{context}.{key} must be a boolean")
    return value


def _expect_str(mapping: Mapping[str, object], key: str, *, context: str) -> str:
    return _require_non_empty(mapping[key], field_name=f"{context}.{key}")


def _parse_limits(value: object) -> ExecutionLimits:
    context = "HumanEval runner.limits"
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


def load_humaneval_runner_config(path: Path) -> HumanEvalRunnerConfig:
    """Load one strict HumanEval runner definition."""

    try:
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise HumanEvalError(f"could not read HumanEval runner config {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise HumanEvalError(f"invalid YAML in HumanEval runner config {path}: {exc}") from exc
    context = "HumanEval runner"
    mapping = _strict_mapping(raw, context=context)
    _validate_keys(
        mapping,
        required=frozenset(
            {
                "schema_version",
                "runner_id",
                "runner_version",
                "frozen",
                "benchmark_id",
                "reference_repository",
                "reference_revision",
                "dataset_split",
                "expected_problem_count",
                "prompt_version",
                "completion_normalizer_version",
                "instruction",
                "execution_image",
                "execution_timeout_seconds",
                "limits",
            }
        ),
        context=context,
    )
    return HumanEvalRunnerConfig(
        schema_version=_expect_int(mapping, "schema_version", context=context),
        runner_id=_expect_str(mapping, "runner_id", context=context),
        runner_version=_expect_int(mapping, "runner_version", context=context),
        frozen=_expect_bool(mapping, "frozen", context=context),
        benchmark_id=_expect_str(mapping, "benchmark_id", context=context),
        reference_repository=_expect_str(mapping, "reference_repository", context=context),
        reference_revision=_expect_str(mapping, "reference_revision", context=context),
        dataset_split=_expect_str(mapping, "dataset_split", context=context),
        expected_problem_count=_expect_int(mapping, "expected_problem_count", context=context),
        prompt_version=_expect_str(mapping, "prompt_version", context=context),
        completion_normalizer_version=_expect_str(
            mapping,
            "completion_normalizer_version",
            context=context,
        ),
        instruction=_expect_str(mapping, "instruction", context=context),
        execution_image=_expect_str(mapping, "execution_image", context=context),
        execution_timeout_seconds=_expect_float(
            mapping,
            "execution_timeout_seconds",
            context=context,
        ),
        limits=_parse_limits(mapping["limits"]),
    )


def humaneval_runner_config_json(config: HumanEvalRunnerConfig) -> str:
    """Serialize one HumanEval runner definition deterministically."""

    return json.dumps(asdict(config), indent=2, sort_keys=True) + "\n"


def humaneval_runner_config_sha256(config: HumanEvalRunnerConfig) -> str:
    """Return the semantic SHA-256 of a HumanEval runner definition."""

    return hashlib.sha256(humaneval_runner_config_json(config).encode("utf-8")).hexdigest()


def load_frozen_humaneval_runner_config(
    path: Path = _HUMANEVAL_RUNNER_CONFIG_PATH,
) -> HumanEvalRunnerConfig:
    """Load the frozen P6-002 HumanEval runner and fail closed on drift."""

    config = load_humaneval_runner_config(path)
    fingerprint = humaneval_runner_config_sha256(config)
    if fingerprint != _FROZEN_HUMANEVAL_RUNNER_SHA256:
        raise HumanEvalError(
            "frozen HumanEval runner fingerprint mismatch; increment runner_version and "
            "explicitly update the frozen fingerprint before evaluation"
        )
    return config


def _load_huggingface_rows(
    repository: str,
    *,
    revision: str,
    split: str,
    streaming: bool,
) -> Iterable[DatasetRow]:
    from datasets import load_dataset  # type: ignore[import-untyped]

    loaded = load_dataset(
        repository,
        revision=revision,
        split=split,
        streaming=streaming,
    )
    return cast(Iterable[DatasetRow], loaded)


def _row_str(row: Mapping[str, object], key: str, *, task_context: str) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise HumanEvalError(f"{task_context}.{key} must be a string")
    return value


def _parse_problem(row: Mapping[str, object], *, row_index: int) -> HumanEvalProblem:
    expected_fields = {"task_id", "prompt", "canonical_solution", "test", "entry_point"}
    actual_fields = set(row)
    if actual_fields != expected_fields:
        missing = sorted(expected_fields - actual_fields)
        unknown = sorted(actual_fields - expected_fields)
        detail: list[str] = []
        if missing:
            detail.append(f"missing={missing!r}")
        if unknown:
            detail.append(f"unknown={unknown!r}")
        raise HumanEvalError(
            f"HumanEval row {row_index} schema does not match pinned dataset: {', '.join(detail)}"
        )
    context = f"HumanEval row {row_index}"
    canonical_solution = _row_str(row, "canonical_solution", task_context=context)
    _require_text(canonical_solution, field_name=f"{context}.canonical_solution")
    return HumanEvalProblem(
        task_id=_row_str(row, "task_id", task_context=context),
        prompt=_row_str(row, "prompt", task_context=context),
        test=_row_str(row, "test", task_context=context),
        entry_point=_row_str(row, "entry_point", task_context=context),
    )


def load_humaneval_problems(
    benchmark: ProtectedBenchmark,
    runner: HumanEvalRunnerConfig,
    *,
    dataset_loader: DatasetRowsLoader = _load_huggingface_rows,
) -> tuple[HumanEvalProblem, ...]:
    """Load and validate the exact protected HumanEval test split."""

    _validate_benchmark_contract(benchmark, runner)
    rows = dataset_loader(
        benchmark.dataset_id,
        revision=benchmark.dataset_revision,
        split=runner.dataset_split,
        streaming=False,
    )
    problems = tuple(
        sorted(
            (_parse_problem(row, row_index=index) for index, row in enumerate(rows)),
            key=lambda problem: problem.task_index,
        )
    )
    if len(problems) != runner.expected_problem_count:
        raise HumanEvalError(
            f"HumanEval expected {runner.expected_problem_count} problems; got {len(problems)}"
        )
    task_ids = tuple(problem.task_id for problem in problems)
    expected_ids = tuple(f"HumanEval/{index}" for index in range(runner.expected_problem_count))
    if task_ids != expected_ids:
        raise HumanEvalError("HumanEval task IDs do not match the canonical contiguous task set")
    return problems


def create_humaneval_prompt(
    problem: HumanEvalProblem,
    runner: HumanEvalRunnerConfig,
) -> HumanEvalPrompt:
    """Create the frozen single-user-message HumanEval prompt."""

    return HumanEvalPrompt(
        task_id=problem.task_id,
        prompt_version=runner.prompt_version,
        user_content=f"{runner.instruction}\n{problem.prompt}",
    )


def normalize_humaneval_completion(
    problem: HumanEvalProblem,
    generated_text: str,
) -> str:
    """Normalize chat-style HumanEval output into a classic completion suffix."""

    if not isinstance(generated_text, str):
        raise HumanEvalError("generated_text must be a string")
    fenced = _PYTHON_FENCE_PATTERN.search(generated_text)
    extracted = fenced.group(1) if fenced is not None else generated_text
    if extracted.startswith(problem.prompt):
        return extracted[len(problem.prompt) :]

    function_position = extracted.find(f"def {problem.entry_point}")
    body_marker_position = extracted.find(":\n    ", function_position)
    if function_position >= 0 and body_marker_position >= 0:
        return extracted[body_marker_position + 2 :]
    return extracted


def _candidate_source(problem: HumanEvalProblem, completion: str) -> str:
    source = problem.prompt + completion
    return source if source.endswith("\n") else source + "\n"


def humaneval_problem_set_sha256(problems: tuple[HumanEvalProblem, ...]) -> str:
    """Hash protected prompts/tests without embedding them in public result artifacts."""

    payload = json.dumps(
        [asdict(problem) for problem in problems],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _result_json_line(result: EvaluationResult) -> str:
    return json.dumps(
        asdict(result),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def humaneval_results_sha256(results: tuple[EvaluationResult, ...]) -> str:
    """Hash ordered common per-problem result records."""

    payload = "".join(_result_json_line(result) + "\n" for result in results).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def humaneval_aggregate_json(aggregate: HumanEvalAggregate) -> str:
    """Serialize a HumanEval aggregate deterministically."""

    return json.dumps(asdict(aggregate), indent=2, sort_keys=True) + "\n"


def _runner_source_sha256() -> str:
    return hashlib.sha256(_HUMANEVAL_RUNNER_SOURCE.encode("utf-8")).hexdigest()


def _execution_error_message(result: ExecutionResult) -> str:
    for line in reversed(result.stderr.splitlines()):
        if line.startswith(_ERROR_MARKER):
            return line.removeprefix(_ERROR_MARKER)
    if result.exit_code is None:
        return "HumanEval candidate exceeded the execution timeout"
    return f"HumanEval candidate process exited with code {result.exit_code}"


def _stage_statuses(result: ExecutionResult) -> tuple[EvaluationStageStatus, EvaluationStageStatus]:
    stderr_lines = set(result.stderr.splitlines())
    parse_status = (
        EvaluationStageStatus.PASSED
        if _PARSE_OK_MARKER in stderr_lines
        else EvaluationStageStatus.NOT_RUN
    )
    compile_status = (
        EvaluationStageStatus.PASSED
        if _COMPILE_OK_MARKER in stderr_lines
        else EvaluationStageStatus.NOT_RUN
    )
    if result.exit_code == _RUNNER_PARSE_EXIT:
        parse_status = EvaluationStageStatus.FAILED
        compile_status = EvaluationStageStatus.NOT_RUN
    elif result.exit_code == _RUNNER_COMPILE_EXIT:
        parse_status = EvaluationStageStatus.PASSED
        compile_status = EvaluationStageStatus.FAILED
    return parse_status, compile_status


def _validate_benchmark_contract(
    benchmark: ProtectedBenchmark,
    runner: HumanEvalRunnerConfig,
) -> None:
    if benchmark.language != "python":
        raise HumanEvalError("HumanEval protected benchmark must belong to Python")
    if benchmark.id != runner.benchmark_id:
        raise HumanEvalError("HumanEval protected benchmark ID does not match runner config")
    if benchmark.dataset_id != _HUMANEVAL_DATASET_ID:
        raise HumanEvalError(f"HumanEval protected dataset must be {_HUMANEVAL_DATASET_ID!r}")
    if not _GIT_SHA_PATTERN.fullmatch(benchmark.dataset_revision):
        raise HumanEvalError("HumanEval dataset revision must be an immutable Git SHA")


def _load_base_identity(path: Path) -> BaseModelIdentity:
    try:
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise HumanEvalError(f"could not read base-model config {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise HumanEvalError(f"invalid YAML in base-model config {path}: {exc}") from exc
    mapping = _strict_mapping(raw, context="base-model config")
    model = _strict_mapping(mapping.get("model"), context="base-model config.model")
    tokenizer = _strict_mapping(
        mapping.get("tokenizer"),
        context="base-model config.tokenizer",
    )
    return BaseModelIdentity(
        repository=_expect_str(model, "repository", context="base-model config.model"),
        revision=_expect_str(model, "revision", context="base-model config.model"),
        tokenizer_repository=_expect_str(
            tokenizer,
            "repository",
            context="base-model config.tokenizer",
        ),
        tokenizer_revision=_expect_str(
            tokenizer,
            "revision",
            context="base-model config.tokenizer",
        ),
    )


def _validate_evaluator_contract(
    evaluation: EvaluationConfig,
    runner: HumanEvalRunnerConfig,
    benchmark: ProtectedBenchmark,
    settings: FrozenEvaluationSettings,
    base_model: BaseModelIdentity,
    adapter: AdapterIdentity,
) -> str:
    _validate_benchmark_contract(benchmark, runner)
    if evaluation.language != "python":
        raise HumanEvalError("HumanEval evaluation language must be 'python'")
    if runner.benchmark_id not in evaluation.suites:
        raise HumanEvalError("HumanEval evaluation suites must include 'humaneval'")
    if evaluation.execution.network_enabled:
        raise HumanEvalError("HumanEval execution must keep networking disabled")
    if evaluation.execution.timeout_seconds != runner.execution_timeout_seconds:
        raise HumanEvalError(
            "HumanEval execution timeout does not match the frozen runner configuration"
        )
    expected_base = _load_base_identity(Path(evaluation.base_config))
    if base_model != expected_base:
        raise HumanEvalError("HumanEval base_model identity does not match evaluation base_config")
    if evaluation.adapter_id != adapter.adapter_id:
        raise HumanEvalError("HumanEval adapter identity does not match evaluation adapter_id")
    if evaluation.adapter_id is None:
        if adapter.family is not None:
            raise HumanEvalError("base-only HumanEval evaluation must not define adapter family")
    elif adapter.family != "language":
        raise HumanEvalError("HumanEval language adapter evaluation requires family='language'")
    return validate_evaluation_config_settings(evaluation, settings)


class HumanEvalEvaluator:
    """Evaluate exactly one completion per HumanEval task under frozen settings."""

    def __init__(
        self,
        evaluation: EvaluationConfig,
        *,
        base_model: BaseModelIdentity,
        adapter: AdapterIdentity,
        benchmark: ProtectedBenchmark | None = None,
        runner: HumanEvalRunnerConfig | None = None,
        settings: FrozenEvaluationSettings | None = None,
        harness: HumanEvalHarness | None = None,
    ) -> None:
        resolved_runner = runner if runner is not None else load_frozen_humaneval_runner_config()
        resolved_settings = settings if settings is not None else load_frozen_evaluation_settings()
        if benchmark is None:
            benchmark = load_protected_benchmark_config(_HUMANEVAL_BENCHMARK_CONFIG_PATH)
        settings_sha256 = _validate_evaluator_contract(
            evaluation,
            resolved_runner,
            benchmark,
            resolved_settings,
            base_model,
            adapter,
        )
        self._evaluation = evaluation
        self._base_model = base_model
        self._adapter = adapter
        self._benchmark = benchmark
        self._runner = resolved_runner
        self._settings = resolved_settings
        self._settings_sha256 = settings_sha256
        self._harness = harness if harness is not None else ConstrainedExecutionHarness()

    @property
    def runner(self) -> HumanEvalRunnerConfig:
        return self._runner

    @property
    def benchmark(self) -> ProtectedBenchmark:
        return self._benchmark

    def load_problems(
        self,
        *,
        dataset_loader: DatasetRowsLoader = _load_huggingface_rows,
    ) -> tuple[HumanEvalProblem, ...]:
        """Load the exact protected HumanEval problem set."""

        return load_humaneval_problems(
            self._benchmark,
            self._runner,
            dataset_loader=dataset_loader,
        )

    def prompt_for(self, problem: HumanEvalProblem) -> HumanEvalPrompt:
        """Return the frozen prompt presented to the model for one task."""

        return create_humaneval_prompt(problem, self._runner)

    def evaluate_completion(
        self,
        problem: HumanEvalProblem,
        completion: HumanEvalCompletion,
    ) -> EvaluationResult:
        """Score one model completion with the constrained execution harness."""

        if completion.task_id != problem.task_id:
            raise HumanEvalError("completion task_id does not match HumanEval problem")
        if completion.generation_error is not None:
            return create_evaluation_result(
                problem_id=problem.task_id,
                language="python",
                generated_text=completion.generated_text,
                generated_code=None,
                parse_status=EvaluationStageStatus.NOT_RUN,
                compile_status=EvaluationStageStatus.NOT_RUN,
                tests=EvaluationTestSummary(passed=0, total=1),
                error_category=EvaluationErrorCategory.GENERATION,
                error_message=completion.generation_error,
                generation=completion.generation,
                base_model=self._base_model,
                adapter=self._adapter,
            )

        normalized = normalize_humaneval_completion(problem, completion.generated_text)
        candidate_source = _candidate_source(problem, normalized)
        request = ExecutionRequest(
            image=self._runner.execution_image,
            command=("python", "-I", "-B", "runner.py"),
            files=(
                ExecutionFile.from_text("runner.py", _HUMANEVAL_RUNNER_SOURCE),
                ExecutionFile.from_text("candidate.py", candidate_source),
                ExecutionFile.from_text("tests.py", problem.test),
                ExecutionFile.from_text(
                    "metadata.json",
                    json.dumps({"entry_point": problem.entry_point}, sort_keys=True) + "\n",
                ),
            ),
        )
        execution = ExecutionConfig(
            timeout_seconds=self._runner.execution_timeout_seconds,
            network_enabled=False,
        )
        try:
            executed = self._harness.run(request, execution, limits=self._runner.limits)
        except ExecutionHarnessError as exc:
            return create_evaluation_result(
                problem_id=problem.task_id,
                language="python",
                generated_text=completion.generated_text,
                generated_code=candidate_source,
                parse_status=EvaluationStageStatus.NOT_RUN,
                compile_status=EvaluationStageStatus.NOT_RUN,
                tests=EvaluationTestSummary(passed=0, total=1),
                error_category=EvaluationErrorCategory.HARNESS,
                error_message=f"HumanEval execution harness failed: {exc}",
                generation=completion.generation,
                base_model=self._base_model,
                adapter=self._adapter,
            )

        parse_status, compile_status = _stage_statuses(executed)
        stderr_lines = set(executed.stderr.splitlines())
        if executed.status is ExecutionStatus.SUCCEEDED and _SUCCESS_MARKER in stderr_lines:
            return create_evaluation_result(
                problem_id=problem.task_id,
                language="python",
                generated_text=completion.generated_text,
                generated_code=candidate_source,
                parse_status=EvaluationStageStatus.PASSED,
                compile_status=EvaluationStageStatus.PASSED,
                tests=EvaluationTestSummary(passed=1, total=1),
                error_category=EvaluationErrorCategory.NONE,
                error_message=None,
                generation=completion.generation,
                base_model=self._base_model,
                adapter=self._adapter,
            )

        if executed.status is ExecutionStatus.TIMED_OUT:
            category = EvaluationErrorCategory.TIMEOUT
        elif executed.exit_code == _RUNNER_PARSE_EXIT:
            category = EvaluationErrorCategory.PARSE
        elif executed.exit_code == _RUNNER_COMPILE_EXIT:
            category = EvaluationErrorCategory.COMPILE
        elif executed.exit_code == _RUNNER_TEST_EXIT:
            category = EvaluationErrorCategory.TEST
        elif executed.exit_code == _RUNNER_HARNESS_EXIT:
            category = EvaluationErrorCategory.HARNESS
        elif (
            parse_status is EvaluationStageStatus.NOT_RUN
            and compile_status is EvaluationStageStatus.NOT_RUN
        ):
            category = EvaluationErrorCategory.HARNESS
        else:
            category = EvaluationErrorCategory.RUNTIME

        return create_evaluation_result(
            problem_id=problem.task_id,
            language="python",
            generated_text=completion.generated_text,
            generated_code=candidate_source,
            parse_status=parse_status,
            compile_status=compile_status,
            tests=EvaluationTestSummary(passed=0, total=1),
            error_category=category,
            error_message=_execution_error_message(executed),
            generation=completion.generation,
            base_model=self._base_model,
            adapter=self._adapter,
        )

    def evaluate_suite(
        self,
        problems: tuple[HumanEvalProblem, ...],
        completions: Iterable[HumanEvalCompletion],
    ) -> HumanEvalSuiteResult:
        """Evaluate one and only one completion for every supplied HumanEval task."""

        completion_by_id: dict[str, HumanEvalCompletion] = {}
        for completion in completions:
            if completion.task_id in completion_by_id:
                raise HumanEvalError(f"duplicate HumanEval completion for {completion.task_id}")
            completion_by_id[completion.task_id] = completion
        problem_ids = tuple(problem.task_id for problem in problems)
        if len(problem_ids) != len(set(problem_ids)):
            raise HumanEvalError("HumanEval problems must not contain duplicate task IDs")
        missing = sorted(set(problem_ids) - set(completion_by_id))
        extra = sorted(set(completion_by_id) - set(problem_ids))
        if missing or extra:
            raise HumanEvalError(
                f"HumanEval completions must exactly match problems; missing={missing!r}, "
                f"extra={extra!r}"
            )

        results = tuple(
            self.evaluate_completion(problem, completion_by_id[problem.task_id])
            for problem in problems
        )
        passed = sum(result.error_category is EvaluationErrorCategory.NONE for result in results)
        timed_out = sum(
            result.error_category is EvaluationErrorCategory.TIMEOUT for result in results
        )
        harness_errors = sum(
            result.error_category is EvaluationErrorCategory.HARNESS for result in results
        )
        total = len(results)
        if total == 0:
            raise HumanEvalError("HumanEval suite must contain at least one problem")
        aggregate = HumanEvalAggregate(
            schema_version=_HUMANEVAL_SCHEMA_VERSION,
            benchmark_id=self._benchmark.id,
            dataset_id=self._benchmark.dataset_id,
            dataset_revision=self._benchmark.dataset_revision,
            dataset_split=self._runner.dataset_split,
            reference_repository=self._runner.reference_repository,
            reference_revision=self._runner.reference_revision,
            runner_config_sha256=humaneval_runner_config_sha256(self._runner),
            runner_source_sha256=_runner_source_sha256(),
            evaluation_settings_sha256=self._settings_sha256,
            prompt_version=self._runner.prompt_version,
            completion_normalizer_version=self._runner.completion_normalizer_version,
            execution_image=self._runner.execution_image,
            base_model=self._base_model,
            adapter=self._adapter,
            total_problems=total,
            passed=passed,
            failed=total - passed,
            timed_out=timed_out,
            harness_errors=harness_errors,
            pass_at_1=None if harness_errors else passed / total,
            problem_set_sha256=humaneval_problem_set_sha256(problems),
            results_sha256=humaneval_results_sha256(results),
        )
        return HumanEvalSuiteResult(results=results, aggregate=aggregate)

    def write_artifacts(
        self,
        suite: HumanEvalSuiteResult,
        output_dir: Path | None = None,
    ) -> HumanEvalArtifactPaths:
        """Write deterministic per-problem JSONL and aggregate JSON artifacts."""

        destination = output_dir if output_dir is not None else Path(self._evaluation.output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        per_problem_path = destination / "humaneval-results.jsonl"
        aggregate_path = destination / "humaneval-aggregate.json"
        _write_atomic(
            per_problem_path,
            "".join(_result_json_line(result) + "\n" for result in suite.results),
        )
        _write_atomic(aggregate_path, humaneval_aggregate_json(suite.aggregate))
        return HumanEvalArtifactPaths(
            per_problem=per_problem_path,
            aggregate=aggregate_path,
        )


def _write_atomic(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


__all__ = [
    "DatasetRowsLoader",
    "HumanEvalAggregate",
    "HumanEvalArtifactPaths",
    "HumanEvalCompletion",
    "HumanEvalError",
    "HumanEvalEvaluator",
    "HumanEvalHarness",
    "HumanEvalProblem",
    "HumanEvalPrompt",
    "HumanEvalRunnerConfig",
    "HumanEvalSuiteResult",
    "create_humaneval_prompt",
    "humaneval_aggregate_json",
    "humaneval_problem_set_sha256",
    "humaneval_results_sha256",
    "humaneval_runner_config_json",
    "humaneval_runner_config_sha256",
    "load_frozen_humaneval_runner_config",
    "load_humaneval_problems",
    "load_humaneval_runner_config",
    "normalize_humaneval_completion",
]
