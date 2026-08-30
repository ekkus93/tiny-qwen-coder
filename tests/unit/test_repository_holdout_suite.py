"""P6-004 tests for the deterministic repo-owned Python holdout suite."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from tiny_qwen_coder.config import EvaluationConfig, ExecutionConfig, GenerationConfig
from tiny_qwen_coder.evaluation import (
    EvaluationErrorCategory,
    EvaluationStageStatus,
    ExecutionLimits,
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    GenerationStats,
    OciRuntime,
    RepositoryHoldoutCompletion,
    RepositoryHoldoutError,
    RepositoryHoldoutEvaluator,
    RepositoryHoldoutTask,
    create_repository_holdout_prompt,
    load_frozen_repository_holdout_suite,
    load_repository_holdout_suite,
    normalize_repository_holdout_completion,
    repository_holdout_suite_sha256,
)
from tiny_qwen_coder.evaluation._repository_holdout_common import (
    _REPOSITORY_HOLDOUT_RUNNER_SOURCE,
)
from tiny_qwen_coder.evaluation.settings import load_frozen_evaluation_settings
from tiny_qwen_coder.identities import AdapterIdentity, BaseModelIdentity

_BASE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
_SUITE_SHA256 = "91d87aa5d1fb5041d9d26e6c8bfbeb958fa406a3aa24f0f1966ded9816f8252e"
_EXPECTED_CATEGORIES = {
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

_CANDIDATES: dict[str, str] = {
    "stdlib-data-transforms": '''\
def summarize_events(events):
    grouped = {}
    for event in events:
        seconds = event["seconds"]
        if seconds < 0:
            raise ValueError("seconds must be non-negative")
        user = event["user"]
        item = grouped.setdefault(user, {"total": 0, "kinds": set()})
        item["total"] += seconds
        item["kinds"].add(event["kind"])
    return [
        {
            "user": user,
            "total_seconds": grouped[user]["total"],
            "kinds": sorted(grouped[user]["kinds"]),
        }
        for user in sorted(grouped)
    ]
''',
    "pathlib-files": '''\
def find_python_modules(root):
    result = []
    for path in root.rglob("*.py"):
        relative = path.relative_to(root)
        if path.is_symlink():
            continue
        if any(part == "__pycache__" or part.startswith(".") for part in relative.parts):
            continue
        if path.is_file():
            result.append(relative.as_posix())
    return sorted(result)
''',
    "json-canonicalize": '''\
import json


def canonicalize_json_object(text, updates):
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("top-level JSON must be an object")
    merged = dict(value)
    merged.update(updates)
    return json.dumps(merged, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
''',
    "regex-issue-refs": '''\
import re

_PATTERN = re.compile(r"(?<!\\w)(?:#|GH-)([1-9][0-9]*)(?!\\w)", re.IGNORECASE)


def extract_issue_refs(text):
    result = []
    seen = set()
    for match in _PATTERN.finditer(text):
        item = f"#{int(match.group(1))}"
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
''',
    "dataclasses-typing": '''\
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Metric:
    name: str
    values: tuple[float, ...]
    labels: tuple[tuple[str, str], ...]

    @property
    def mean(self):
        if not self.values:
            raise ValueError("no values")
        return sum(self.values) / len(self.values)


def make_metric(name, values, labels=None):
    normalized_labels = tuple(sorted((labels or {}).items()))
    return Metric(name, tuple(float(value) for value in values), normalized_labels)
''',
    "generators-decorators-context": '''\
from contextlib import contextmanager
from functools import wraps


def chunked(iterable, size):
    if size <= 0:
        raise ValueError("size")
    iterator = iter(iterable)
    while True:
        chunk = []
        try:
            for _ in range(size):
                chunk.append(next(iterator))
        except StopIteration:
            pass
        if not chunk:
            return
        yield tuple(chunk)
        if len(chunk) < size:
            return


def count_calls(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        return func(*args, **kwargs)

    wrapper.calls = 0
    return wrapper


@contextmanager
def temporary_mapping(mapping, updates):
    original = dict(mapping)
    mapping.update(updates)
    try:
        yield mapping
    finally:
        mapping.clear()
        mapping.update(original)
''',
    "exceptions-config-path": '''\
from collections.abc import Mapping


class ConfigPathError(LookupError):
    def __init__(self, path, segment):
        self.path = path
        self.segment = segment
        super().__init__(f"could not resolve {segment!r} in {path!r}")


def require_path(mapping, path):
    if not path:
        raise ValueError("path")
    current = mapping
    for segment in path.split("."):
        if not isinstance(current, Mapping) or segment not in current:
            raise ConfigPathError(path, segment)
        current = current[segment]
    return current
''',
    "async-map-limited": '''\
import asyncio


async def map_limited(items, worker, limit):
    if limit <= 0:
        raise ValueError("limit")
    semaphore = asyncio.Semaphore(limit)

    async def run(item):
        async with semaphore:
            return await worker(item)

    return list(await asyncio.gather(*(run(item) for item in items)))
''',
    "subprocess-run-command": '''\
import subprocess


def run_command(command, timeout):
    if not command:
        raise ValueError("command")
    completed = subprocess.run(
        list(command),
        timeout=timeout,
        capture_output=True,
        text=True,
        check=False,
        shell=False,
    )
    return completed.returncode, completed.stdout, completed.stderr
''',
    "sqlite-domain-counts": '''\
def domain_counts(connection):
    query = """
        SELECT lower(substr(email, instr(email, '@') + 1)) AS domain, count(*) AS n
        FROM users
        WHERE instr(email, '@') > 0
          AND length(substr(email, instr(email, '@') + 1)) > 0
        GROUP BY domain
        ORDER BY n DESC, domain ASC
    """
    return [(domain, count) for domain, count in connection.execute(query)]
''',
    "pytest-contract": '''\
import pytest


def test_slugify_contract(slugify):
    assert slugify("Hello, World!") == "hello-world"
    assert slugify("  already--spaced  ") == "already-spaced"
    assert slugify("Café déjà vu") == "cafe-deja-vu"
    assert slugify("") == ""
    with pytest.raises(TypeError):
        slugify(123)
''',
}


def _base() -> BaseModelIdentity:
    return BaseModelIdentity(
        repository="Qwen/Qwen3.5-4B",
        revision=_BASE_REVISION,
        tokenizer_repository="Qwen/Qwen3.5-4B",
        tokenizer_revision=_BASE_REVISION,
    )


def _evaluation(*, adapter_id: str | None = None) -> EvaluationConfig:
    settings = load_frozen_evaluation_settings()
    frozen = settings.generation
    return EvaluationConfig(
        schema_version=1,
        base_config="configs/base/qwen35-4b.yaml",
        language="python",
        adapter_id=adapter_id,
        suites=("repository-holdout",),
        output_dir="artifacts/eval/python",
        seed=settings.seed,
        generation=GenerationConfig(
            temperature=frozen.temperature,
            top_p=frozen.top_p,
            top_k=frozen.top_k,
            max_new_tokens=frozen.max_new_tokens,
            prompt_version=frozen.prompt_version,
        ),
        execution=ExecutionConfig(timeout_seconds=10.0, network_enabled=False),
    )


def _completion(
    task: RepositoryHoldoutTask,
    text: str | None = None,
) -> RepositoryHoldoutCompletion:
    return RepositoryHoldoutCompletion(
        problem_id=task.problem_id,
        generated_text=text if text is not None else _CANDIDATES[task.task_id],
        generation=GenerationStats(
            prompt_tokens=48,
            generated_tokens=32,
            latency_seconds=0.2,
            tokens_per_second=160.0,
        ),
    )


def _result(
    *,
    status: ExecutionStatus,
    exit_code: int | None,
    stderr: str,
) -> ExecutionResult:
    return ExecutionResult(
        status=status,
        runtime=OciRuntime.PODMAN,
        exit_code=exit_code,
        duration_seconds=0.01,
        stdout="",
        stderr=stderr,
        stdout_truncated=False,
        stderr_truncated=False,
    )


def _success(task: RepositoryHoldoutTask) -> ExecutionResult:
    markers = "".join(
        f"__TQC_REPOSITORY_HOLDOUT_TEST_OK_V1__:{index}\n"
        for index in range(1, task.expected_tests + 1)
    )
    return _result(
        status=ExecutionStatus.SUCCEEDED,
        exit_code=0,
        stderr=(
            "__TQC_REPOSITORY_HOLDOUT_PARSE_OK_V1__\n"
            "__TQC_REPOSITORY_HOLDOUT_COMPILE_OK_V1__\n"
            f"{markers}"
            "__TQC_REPOSITORY_HOLDOUT_PASS_V1__\n"
        ),
    )


class _RecordingHarness:
    def __init__(self, *outcomes: ExecutionResult | Exception) -> None:
        self._outcomes = list(outcomes)
        self.requests: list[ExecutionRequest] = []
        self.executions: list[ExecutionConfig] = []
        self.limits: list[ExecutionLimits | None] = []

    def run(
        self,
        request: ExecutionRequest,
        execution: ExecutionConfig,
        *,
        limits: ExecutionLimits | None = None,
    ) -> ExecutionResult:
        self.requests.append(request)
        self.executions.append(execution)
        self.limits.append(limits)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _evaluator(harness: _RecordingHarness) -> RepositoryHoldoutEvaluator:
    return RepositoryHoldoutEvaluator(
        _evaluation(),
        base_model=_base(),
        adapter=AdapterIdentity(family=None, adapter_id=None),
        harness=harness,
    )


def test_frozen_suite_covers_every_required_category_and_pins_assets() -> None:
    suite = load_frozen_repository_holdout_suite()

    assert len(suite.tasks) == 11
    assert {task.category for task in suite.tasks} == _EXPECTED_CATEGORIES
    assert repository_holdout_suite_sha256(suite) == _SUITE_SHA256
    assert all(task.test_sha256 for task in suite.tasks)
    assert all(
        task.test_path.startswith("benchmarks/python/repository_holdout_v1/")
        for task in suite.tasks
    )
    assert "@sha256:" in suite.execution_image
    assert suite.execution_timeout_seconds == 10.0


def test_suite_config_drift_fails_closed(tmp_path: Path) -> None:
    source = Path("configs/eval/python/repository_holdout_suite_v1.yaml").read_text(
        encoding="utf-8"
    )
    path = tmp_path / "drift.yaml"
    path.write_text(source.replace("expected_tests: 4", "expected_tests: 3", 1), encoding="utf-8")

    with pytest.raises(RepositoryHoldoutError, match="fingerprint mismatch"):
        from tiny_qwen_coder.evaluation.repository_holdout import (
            load_frozen_repository_holdout_suite as load_frozen,
        )

        load_frozen(path)


def test_unknown_suite_fields_fail_closed(tmp_path: Path) -> None:
    source = Path("configs/eval/python/repository_holdout_suite_v1.yaml").read_text(
        encoding="utf-8"
    )
    path = tmp_path / "unknown.yaml"
    path.write_text(source + "surprise: forbidden\n", encoding="utf-8")

    with pytest.raises(RepositoryHoldoutError, match="unknown field"):
        load_repository_holdout_suite(path)


def test_prompt_and_completion_normalization_are_stable() -> None:
    suite = load_frozen_repository_holdout_suite()
    task = suite.tasks[0]
    prompt = create_repository_holdout_prompt(task, suite)

    assert prompt.problem_id == task.problem_id
    assert task.prompt in prompt.user_content
    assert "Return only Python code" in prompt.user_content
    fenced = "```python\ndef answer():\n    return 1\n```"
    assert normalize_repository_holdout_completion(task, fenced, suite) == (
        "def answer():\n    return 1\n"
    )


def test_every_repo_owned_test_asset_executes_against_known_good_candidate(
    tmp_path: Path,
) -> None:
    suite = load_frozen_repository_holdout_suite()

    for task in suite.tasks:
        case_dir = tmp_path / task.task_id
        case_dir.mkdir()
        (case_dir / "runner.py").write_text(_REPOSITORY_HOLDOUT_RUNNER_SOURCE, encoding="utf-8")
        (case_dir / "candidate.py").write_text(_CANDIDATES[task.task_id], encoding="utf-8")
        (case_dir / "setup.py").write_text(task.setup_source, encoding="utf-8")
        (case_dir / "tests.py").write_text(task.test_source, encoding="utf-8")
        (case_dir / "metadata.json").write_text(
            json.dumps({"expected_tests": task.expected_tests}) + "\n",
            encoding="utf-8",
        )
        completed = subprocess.run(
            [sys.executable, "-I", "-B", "runner.py"],
            cwd=case_dir,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        assert completed.returncode == 0, f"{task.task_id}: {completed.stderr}"
        assert "__TQC_REPOSITORY_HOLDOUT_PASS_V1__" in completed.stderr


def test_evaluator_uses_digest_pinned_network_disabled_constrained_request() -> None:
    suite = load_frozen_repository_holdout_suite()
    task = suite.tasks[0]
    harness = _RecordingHarness(_success(task))
    result = _evaluator(harness).evaluate_completion(task, _completion(task))

    assert result.error_category is EvaluationErrorCategory.NONE
    request = harness.requests[0]
    execution = harness.executions[0]
    assert request.image == suite.execution_image
    assert request.command == ("python", "-I", "-B", "runner.py")
    assert {item.path for item in request.files} == {
        "runner.py",
        "candidate.py",
        "setup.py",
        "tests.py",
        "metadata.json",
    }
    assert execution.network_enabled is False
    assert execution.timeout_seconds == 10.0
    assert harness.limits[0] == suite.limits


def test_test_failure_preserves_partial_pass_count() -> None:
    task = load_frozen_repository_holdout_suite().tasks[0]
    harness = _RecordingHarness(
        _result(
            status=ExecutionStatus.FAILED,
            exit_code=12,
            stderr=(
                "__TQC_REPOSITORY_HOLDOUT_PARSE_OK_V1__\n"
                "__TQC_REPOSITORY_HOLDOUT_COMPILE_OK_V1__\n"
                "__TQC_REPOSITORY_HOLDOUT_TEST_OK_V1__:1\n"
                "__TQC_REPOSITORY_HOLDOUT_TEST_OK_V1__:2\n"
                "__TQC_REPOSITORY_HOLDOUT_ERROR_V1__:test[3:x]:AssertionError:bad\n"
            ),
        )
    )

    result = _evaluator(harness).evaluate_completion(task, _completion(task))

    assert result.error_category is EvaluationErrorCategory.TEST
    assert result.tests.passed == 2
    assert result.tests.total == task.expected_tests
    assert result.parse_status is EvaluationStageStatus.PASSED
    assert result.compile_status is EvaluationStageStatus.PASSED


def test_harness_failure_is_not_counted_as_model_miss() -> None:
    suite = load_frozen_repository_holdout_suite()
    outcomes = [_success(task) for task in suite.tasks]
    outcomes[0] = _result(
        status=ExecutionStatus.FAILED,
        exit_code=20,
        stderr="__TQC_REPOSITORY_HOLDOUT_ERROR_V1__:benchmark:TypeError:bad fixture\n",
    )
    evaluator = _evaluator(_RecordingHarness(*outcomes))

    result = evaluator.evaluate_suite(_completion(task) for task in suite.tasks)

    assert result.aggregate.harness_errors == 1
    assert result.aggregate.pass_at_1 is None


def test_suite_results_and_artifacts_are_deterministic(tmp_path: Path) -> None:
    suite = load_frozen_repository_holdout_suite()
    completions = tuple(_completion(task) for task in suite.tasks)
    first = _evaluator(_RecordingHarness(*(_success(task) for task in suite.tasks)))
    second = _evaluator(_RecordingHarness(*(_success(task) for task in suite.tasks)))

    first_result = first.evaluate_suite(completions)
    second_result = second.evaluate_suite(completions)

    assert first_result == second_result
    assert first_result.aggregate.pass_at_1 == 1.0
    first_paths = first.write_artifacts(first_result, tmp_path / "first")
    second_paths = second.write_artifacts(second_result, tmp_path / "second")
    assert first_paths.per_problem.read_bytes() == second_paths.per_problem.read_bytes()
    assert first_paths.aggregate.read_bytes() == second_paths.aggregate.read_bytes()
    aggregate_text = first_paths.aggregate.read_text(encoding="utf-8")
    assert "test_source" not in aggregate_text
    assert "pytest.raises" not in aggregate_text


def test_completions_must_exactly_match_frozen_tasks() -> None:
    suite = load_frozen_repository_holdout_suite()
    evaluator = _evaluator(_RecordingHarness())

    with pytest.raises(RepositoryHoldoutError, match="exactly match tasks"):
        evaluator.evaluate_suite(_completion(task) for task in suite.tasks[:-1])

    duplicate = [_completion(task) for task in suite.tasks]
    duplicate.append(_completion(suite.tasks[0]))
    with pytest.raises(RepositoryHoldoutError, match="duplicate"):
        evaluator.evaluate_suite(duplicate)


def test_evaluator_rejects_network_timeout_suite_and_adapter_drift() -> None:
    adapter = AdapterIdentity(family=None, adapter_id=None)

    with pytest.raises(RepositoryHoldoutError, match="network_enabled=false"):
        RepositoryHoldoutEvaluator(
            replace(
                _evaluation(),
                execution=ExecutionConfig(timeout_seconds=10.0, network_enabled=True),
            ),
            base_model=_base(),
            adapter=adapter,
        )

    with pytest.raises(RepositoryHoldoutError, match="timeout"):
        RepositoryHoldoutEvaluator(
            replace(
                _evaluation(),
                execution=ExecutionConfig(timeout_seconds=9.0, network_enabled=False),
            ),
            base_model=_base(),
            adapter=adapter,
        )

    with pytest.raises(RepositoryHoldoutError, match="suites must include"):
        RepositoryHoldoutEvaluator(
            replace(_evaluation(), suites=("humaneval",)),
            base_model=_base(),
            adapter=adapter,
        )

    with pytest.raises(RepositoryHoldoutError, match="adapter_id"):
        RepositoryHoldoutEvaluator(
            _evaluation(adapter_id="python-p0"),
            base_model=_base(),
            adapter=adapter,
        )
