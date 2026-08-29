"""P6-002 HumanEval evaluator tests."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import deque
from dataclasses import replace
from pathlib import Path

import pytest

from tiny_qwen_coder.config import EvaluationConfig, ExecutionConfig, GenerationConfig
from tiny_qwen_coder.evaluation import (
    EvaluationErrorCategory,
    EvaluationStageStatus,
    GenerationStats,
)
from tiny_qwen_coder.evaluation.execution import (
    ExecutionHarnessUnavailableError,
    ExecutionLimits,
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    OciRuntime,
)
from tiny_qwen_coder.evaluation.humaneval import (
    HumanEvalCompletion,
    HumanEvalError,
    HumanEvalEvaluator,
    HumanEvalProblem,
    humaneval_problem_set_sha256,
    humaneval_results_sha256,
    humaneval_runner_config_sha256,
    load_frozen_humaneval_runner_config,
    load_humaneval_problems,
    load_humaneval_runner_config,
    normalize_humaneval_completion,
)
from tiny_qwen_coder.evaluation.protected_benchmarks import ProtectedBenchmark
from tiny_qwen_coder.evaluation.settings import load_frozen_evaluation_settings
from tiny_qwen_coder.identities import AdapterIdentity, BaseModelIdentity

_BASE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
_HUMANEVAL_REVISION = "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544"
_RUNNER_PATH = Path("configs/eval/python/humaneval_runner_v1.yaml")
_RUNNER_SHA256 = "b2f405cdd05551ac858d75eb13aedab591b79522b493fbe0ad247a0f3b677e19"
_PARSE_OK = "__TQC_HUMANEVAL_PARSE_OK_V1__"
_COMPILE_OK = "__TQC_HUMANEVAL_COMPILE_OK_V1__"
_PASS = "__TQC_HUMANEVAL_PASS_V1__"
_ERROR = "__TQC_HUMANEVAL_ERROR_V1__:"


def _base() -> BaseModelIdentity:
    return BaseModelIdentity(
        repository="Qwen/Qwen3.5-4B",
        revision=_BASE_REVISION,
        tokenizer_repository="Qwen/Qwen3.5-4B",
        tokenizer_revision=_BASE_REVISION,
    )


def _benchmark() -> ProtectedBenchmark:
    return ProtectedBenchmark(
        language="python",
        id="humaneval",
        dataset_id="openai/openai_humaneval",
        dataset_revision=_HUMANEVAL_REVISION,
        source_configs=("configs/eval/python/humaneval.yaml",),
    )


def _evaluation(
    *, timeout_seconds: float = 10.0, network_enabled: bool = False
) -> EvaluationConfig:
    settings = load_frozen_evaluation_settings()
    frozen = settings.generation
    return EvaluationConfig(
        schema_version=1,
        base_config="configs/base/qwen35-4b.yaml",
        language="python",
        adapter_id=None,
        suites=("humaneval",),
        output_dir="artifacts/eval/python/humaneval",
        seed=settings.seed,
        generation=GenerationConfig(
            temperature=frozen.temperature,
            top_p=frozen.top_p,
            top_k=frozen.top_k,
            max_new_tokens=frozen.max_new_tokens,
            prompt_version=frozen.prompt_version,
        ),
        execution=ExecutionConfig(
            timeout_seconds=timeout_seconds,
            network_enabled=network_enabled,
        ),
    )


def _problem(index: int = 0) -> HumanEvalProblem:
    return HumanEvalProblem(
        task_id=f"HumanEval/{index}",
        prompt="def add(a, b):\n    \"\"\"Return the sum.\"\"\"\n",
        test="def check(candidate):\n    assert candidate(2, 3) == 5\n",
        entry_point="add",
    )


def _generation() -> GenerationStats:
    return GenerationStats(
        prompt_tokens=32,
        generated_tokens=8,
        latency_seconds=0.1,
        tokens_per_second=80.0,
    )


def _completion(index: int = 0, text: str = "    return a + b\n") -> HumanEvalCompletion:
    return HumanEvalCompletion(
        task_id=f"HumanEval/{index}",
        generated_text=text,
        generation=_generation(),
    )


def _execution_result(
    *,
    status: ExecutionStatus,
    exit_code: int | None,
    stderr: str,
) -> ExecutionResult:
    return ExecutionResult(
        status=status,
        runtime=OciRuntime.DOCKER,
        exit_code=exit_code,
        duration_seconds=0.01,
        stdout="",
        stderr=stderr,
        stdout_truncated=False,
        stderr_truncated=False,
    )


class _RecordingHarness:
    def __init__(self, *outcomes: ExecutionResult | Exception) -> None:
        self.outcomes = deque(outcomes)
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
        outcome = self.outcomes.popleft()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _success() -> ExecutionResult:
    return _execution_result(
        status=ExecutionStatus.SUCCEEDED,
        exit_code=0,
        stderr=f"{_PARSE_OK}\n{_COMPILE_OK}\n{_PASS}\n",
    )


def _test_failure() -> ExecutionResult:
    return _execution_result(
        status=ExecutionStatus.FAILED,
        exit_code=12,
        stderr=f"{_PARSE_OK}\n{_COMPILE_OK}\n{_ERROR}test:AssertionError:\n",
    )


def _evaluator(harness: _RecordingHarness) -> HumanEvalEvaluator:
    return HumanEvalEvaluator(
        _evaluation(),
        base_model=_base(),
        adapter=AdapterIdentity(family=None, adapter_id=None),
        benchmark=_benchmark(),
        harness=harness,
    )


def test_default_evaluator_loads_p6_protected_humaneval_registration() -> None:
    evaluator = HumanEvalEvaluator(
        _evaluation(),
        base_model=_base(),
        adapter=AdapterIdentity(family=None, adapter_id=None),
        harness=_RecordingHarness(_success()),
    )

    assert evaluator.benchmark.dataset_id == "openai/openai_humaneval"
    assert evaluator.benchmark.dataset_revision == _HUMANEVAL_REVISION


def test_frozen_runner_pins_reference_prompt_runtime_and_limits() -> None:
    runner = load_frozen_humaneval_runner_config()

    assert runner.reference_repository == "openai/simple-evals"
    assert runner.reference_revision == "652c89d0ca9df547706735883097e9537d40dc47"
    assert runner.dataset_split == "test"
    assert runner.expected_problem_count == 164
    assert runner.prompt_version == "humaneval-openai-simple-evals-v1"
    assert runner.completion_normalizer_version == "humaneval-completion-normalizer-v1"
    assert runner.execution_image.startswith("python:3.11.14-slim@sha256:")
    assert runner.execution_timeout_seconds == 10.0
    assert runner.limits.memory_mebibytes == 512
    assert humaneval_runner_config_sha256(runner) == _RUNNER_SHA256


def test_runner_config_unknown_fields_and_drift_fail_closed(tmp_path: Path) -> None:
    original = _RUNNER_PATH.read_text(encoding="utf-8")
    unknown = tmp_path / "unknown.yaml"
    unknown.write_text(original + "surprise: forbidden\n", encoding="utf-8")
    with pytest.raises(HumanEvalError, match=r"unknown field\(s\): surprise"):
        load_humaneval_runner_config(unknown)

    drifted = tmp_path / "drifted.yaml"
    drifted.write_text(
        original.replace("execution_timeout_seconds: 10", "execution_timeout_seconds: 11"),
        encoding="utf-8",
    )
    with pytest.raises(HumanEvalError, match="fingerprint mismatch"):
        load_frozen_humaneval_runner_config(drifted)


def test_loader_uses_exact_protected_revision_and_drops_canonical_solution() -> None:
    calls: list[tuple[str, str, str, bool]] = []

    def loader(
        repository: str,
        *,
        revision: str,
        split: str,
        streaming: bool,
    ) -> list[dict[str, object]]:
        calls.append((repository, revision, split, streaming))
        return [
            {
                "task_id": "HumanEval/1",
                "prompt": "def second():\n    \"\"\"Return two.\"\"\"\n",
                "canonical_solution": "    return 2\n",
                "test": "def check(candidate):\n    assert candidate() == 2\n",
                "entry_point": "second",
            },
            {
                "task_id": "HumanEval/0",
                "prompt": "def first():\n    \"\"\"Return one.\"\"\"\n",
                "canonical_solution": "    return 1\n",
                "test": "def check(candidate):\n    assert candidate() == 1\n",
                "entry_point": "first",
            },
        ]

    runner = replace(load_frozen_humaneval_runner_config(), expected_problem_count=2)
    problems = load_humaneval_problems(_benchmark(), runner, dataset_loader=loader)

    assert calls == [("openai/openai_humaneval", _HUMANEVAL_REVISION, "test", False)]
    assert tuple(problem.task_id for problem in problems) == ("HumanEval/0", "HumanEval/1")
    assert not hasattr(problems[0], "canonical_solution")


def test_loader_rejects_schema_drift_wrong_count_and_noncanonical_ids() -> None:
    runner = replace(load_frozen_humaneval_runner_config(), expected_problem_count=1)

    def schema_drift(*args: object, **kwargs: object) -> list[dict[str, object]]:
        row = {
            "task_id": "HumanEval/0",
            "prompt": "def f():\n    pass\n",
            "canonical_solution": "    return 1\n",
            "test": "def check(candidate):\n    pass\n",
            "entry_point": "f",
            "extra": "unexpected",
        }
        return [row]

    with pytest.raises(HumanEvalError, match="schema does not match pinned dataset"):
        load_humaneval_problems(_benchmark(), runner, dataset_loader=schema_drift)

    def empty(*args: object, **kwargs: object) -> list[dict[str, object]]:
        return []

    with pytest.raises(HumanEvalError, match="expected 1 problems; got 0"):
        load_humaneval_problems(_benchmark(), runner, dataset_loader=empty)

    def wrong_id(*args: object, **kwargs: object) -> list[dict[str, object]]:
        return [
            {
                "task_id": "HumanEval/1",
                "prompt": "def f():\n    pass\n",
                "canonical_solution": "    return 1\n",
                "test": "def check(candidate):\n    pass\n",
                "entry_point": "f",
            }
        ]

    with pytest.raises(HumanEvalError, match="canonical contiguous task set"):
        load_humaneval_problems(_benchmark(), runner, dataset_loader=wrong_id)


def test_prompt_is_stable_single_user_content_and_does_not_expose_solution() -> None:
    evaluator = _evaluator(_RecordingHarness(_success()))
    prompt = evaluator.prompt_for(_problem())

    assert prompt.task_id == "HumanEval/0"
    assert prompt.prompt_version == evaluator.runner.prompt_version
    assert prompt.user_content.startswith(evaluator.runner.instruction + "\n")
    assert prompt.user_content.endswith(_problem().prompt)
    assert "return a + b" not in prompt.user_content


def test_completion_normalizer_accepts_raw_suffix_fence_and_full_function() -> None:
    problem = _problem()
    raw = "    return a + b\n"
    assert normalize_humaneval_completion(problem, raw) == raw

    fenced = "Here is the code:\n```python\ndef add(a, b):\n    return a + b\n```\n"
    assert normalize_humaneval_completion(problem, fenced) == "    return a + b\n"

    repeated_prefix = problem.prompt + "    return a + b\n"
    assert normalize_humaneval_completion(problem, repeated_prefix) == "    return a + b\n"


def test_success_uses_pinned_sandbox_and_emits_common_result() -> None:
    harness = _RecordingHarness(_success())
    result = _evaluator(harness).evaluate_completion(_problem(), _completion())

    assert result.problem_id == "HumanEval/0"
    assert result.parse_status is EvaluationStageStatus.PASSED
    assert result.compile_status is EvaluationStageStatus.PASSED
    assert result.tests.passed == result.tests.total == 1
    assert result.error_category is EvaluationErrorCategory.NONE
    assert result.generated_text == "    return a + b\n"
    assert result.generated_code == _problem().prompt + "    return a + b\n"

    request = harness.requests[0]
    assert request.image == load_frozen_humaneval_runner_config().execution_image
    assert request.command == ("python", "-I", "-B", "runner.py")
    file_map = {item.path: item.content.decode("utf-8") for item in request.files}
    assert file_map["candidate.py"] == result.generated_code
    assert file_map["tests.py"] == _problem().test
    assert json.loads(file_map["metadata.json"]) == {"entry_point": "add"}
    assert harness.executions[0] == ExecutionConfig(timeout_seconds=10.0, network_enabled=False)
    assert harness.limits[0] == load_frozen_humaneval_runner_config().limits


@pytest.mark.parametrize(
    ("executed", "category", "parse_status", "compile_status"),
    (
        (
            _execution_result(
                status=ExecutionStatus.FAILED,
                exit_code=10,
                stderr=f"{_ERROR}parse:SyntaxError:invalid syntax\n",
            ),
            EvaluationErrorCategory.PARSE,
            EvaluationStageStatus.FAILED,
            EvaluationStageStatus.NOT_RUN,
        ),
        (
            _execution_result(
                status=ExecutionStatus.FAILED,
                exit_code=11,
                stderr=f"{_PARSE_OK}\n{_ERROR}compile:SyntaxError:return outside function\n",
            ),
            EvaluationErrorCategory.COMPILE,
            EvaluationStageStatus.PASSED,
            EvaluationStageStatus.FAILED,
        ),
        (
            _test_failure(),
            EvaluationErrorCategory.TEST,
            EvaluationStageStatus.PASSED,
            EvaluationStageStatus.PASSED,
        ),
        (
            _execution_result(
                status=ExecutionStatus.TIMED_OUT,
                exit_code=None,
                stderr=f"{_PARSE_OK}\n{_COMPILE_OK}\n",
            ),
            EvaluationErrorCategory.TIMEOUT,
            EvaluationStageStatus.PASSED,
            EvaluationStageStatus.PASSED,
        ),
        (
            _execution_result(
                status=ExecutionStatus.FAILED,
                exit_code=9,
                stderr=f"{_PARSE_OK}\n{_COMPILE_OK}\n",
            ),
            EvaluationErrorCategory.RUNTIME,
            EvaluationStageStatus.PASSED,
            EvaluationStageStatus.PASSED,
        ),
        (
            _execution_result(
                status=ExecutionStatus.FAILED,
                exit_code=125,
                stderr="OCI runtime failed before runner startup\n",
            ),
            EvaluationErrorCategory.HARNESS,
            EvaluationStageStatus.NOT_RUN,
            EvaluationStageStatus.NOT_RUN,
        ),
    ),
)
def test_execution_outcomes_map_to_common_error_schema(
    executed: ExecutionResult,
    category: EvaluationErrorCategory,
    parse_status: EvaluationStageStatus,
    compile_status: EvaluationStageStatus,
) -> None:
    result = _evaluator(_RecordingHarness(executed)).evaluate_completion(_problem(), _completion())

    assert result.error_category is category
    assert result.parse_status is parse_status
    assert result.compile_status is compile_status
    assert result.tests.passed == 0
    assert result.tests.total == 1
    assert result.error_message is not None


def test_generation_and_harness_failures_are_explicit() -> None:
    generation_error = replace(_completion(), generated_text="", generation_error="model failed")
    unused_harness = _RecordingHarness(_success())
    evaluator = _evaluator(unused_harness)
    result = evaluator.evaluate_completion(_problem(), generation_error)
    assert result.error_category is EvaluationErrorCategory.GENERATION
    assert result.generated_code is None
    assert len(unused_harness.outcomes) == 1

    harness = _RecordingHarness(ExecutionHarnessUnavailableError("no OCI runtime"))
    result = _evaluator(harness).evaluate_completion(_problem(), _completion())
    assert result.error_category is EvaluationErrorCategory.HARNESS
    assert result.error_message == "HumanEval execution harness failed: no OCI runtime"


def test_trusted_runner_executes_controlled_candidate_and_tests(tmp_path: Path) -> None:
    harness = _RecordingHarness(_success())
    _evaluator(harness).evaluate_completion(_problem(), _completion())
    request = harness.requests[0]
    for item in request.files:
        destination = tmp_path / item.path
        destination.write_bytes(item.content)

    completed = subprocess.run(
        [sys.executable, "-I", "-B", "runner.py"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    assert completed.returncode == 0
    assert _PARSE_OK in completed.stderr
    assert _COMPILE_OK in completed.stderr
    assert _PASS in completed.stderr


def test_suite_aggregate_and_artifacts_are_deterministic_and_content_safe(tmp_path: Path) -> None:
    problems = (_problem(0), replace(_problem(1), task_id="HumanEval/1"))
    completions = (_completion(0), _completion(1))
    harness = _RecordingHarness(_success(), _test_failure())
    evaluator = _evaluator(harness)

    suite = evaluator.evaluate_suite(problems, completions)
    aggregate = suite.aggregate
    assert aggregate.total_problems == 2
    assert aggregate.passed == 1
    assert aggregate.failed == 1
    assert aggregate.pass_at_1 == 0.5
    assert aggregate.reference_repository == "openai/simple-evals"
    assert aggregate.reference_revision == "652c89d0ca9df547706735883097e9537d40dc47"
    assert aggregate.problem_set_sha256 == humaneval_problem_set_sha256(problems)
    assert aggregate.results_sha256 == humaneval_results_sha256(suite.results)
    assert len(aggregate.runner_source_sha256) == 64

    first = evaluator.write_artifacts(suite, tmp_path)
    first_results = first.per_problem.read_text(encoding="utf-8")
    first_aggregate = first.aggregate.read_text(encoding="utf-8")
    second = evaluator.write_artifacts(suite, tmp_path)
    assert second.per_problem.read_text(encoding="utf-8") == first_results
    assert second.aggregate.read_text(encoding="utf-8") == first_aggregate
    assert "assert candidate" not in first_results
    assert "assert candidate" not in first_aggregate
    assert "canonical_solution" not in first_results
    assert "canonical_solution" not in first_aggregate


def test_repeated_deterministic_runs_produce_identical_suite_results() -> None:
    problems = (_problem(0), replace(_problem(1), task_id="HumanEval/1"))
    completions = (_completion(0), _completion(1))

    first = _evaluator(_RecordingHarness(_success(), _test_failure())).evaluate_suite(
        problems, completions
    )
    second = _evaluator(_RecordingHarness(_success(), _test_failure())).evaluate_suite(
        problems, completions
    )

    assert first == second


def test_harness_error_invalidates_aggregate_pass_at_1() -> None:
    harness = _RecordingHarness(ExecutionHarnessUnavailableError("missing runtime"))
    suite = _evaluator(harness).evaluate_suite((_problem(),), (_completion(),))

    assert suite.aggregate.passed == 0
    assert suite.aggregate.harness_errors == 1
    assert suite.aggregate.pass_at_1 is None


def test_suite_requires_exactly_one_completion_per_problem() -> None:
    evaluator = _evaluator(_RecordingHarness(_success()))
    problems = (_problem(0), _problem(1))

    with pytest.raises(HumanEvalError, match="exactly match problems"):
        evaluator.evaluate_suite(problems, (_completion(0),))
    with pytest.raises(HumanEvalError, match="duplicate HumanEval completion"):
        evaluator.evaluate_suite(problems, (_completion(0), _completion(0), _completion(1)))


def test_evaluator_rejects_network_timeout_suite_base_and_adapter_drift() -> None:
    with pytest.raises(HumanEvalError, match="networking disabled"):
        HumanEvalEvaluator(
            _evaluation(network_enabled=True),
            base_model=_base(),
            adapter=AdapterIdentity(family=None, adapter_id=None),
            benchmark=_benchmark(),
            harness=_RecordingHarness(_success()),
        )

    with pytest.raises(HumanEvalError, match="timeout"):
        HumanEvalEvaluator(
            _evaluation(timeout_seconds=9.0),
            base_model=_base(),
            adapter=AdapterIdentity(family=None, adapter_id=None),
            benchmark=_benchmark(),
            harness=_RecordingHarness(_success()),
        )

    no_suite = replace(_evaluation(), suites=("mbpp",))
    with pytest.raises(HumanEvalError, match="must include 'humaneval'"):
        HumanEvalEvaluator(
            no_suite,
            base_model=_base(),
            adapter=AdapterIdentity(family=None, adapter_id=None),
            benchmark=_benchmark(),
            harness=_RecordingHarness(_success()),
        )

    wrong_base = replace(_base(), repository="Qwen/other")
    with pytest.raises(HumanEvalError, match="base_model identity"):
        HumanEvalEvaluator(
            _evaluation(),
            base_model=wrong_base,
            adapter=AdapterIdentity(family=None, adapter_id=None),
            benchmark=_benchmark(),
            harness=_RecordingHarness(_success()),
        )

    adapter_config = replace(_evaluation(), adapter_id="language/python/p0")
    with pytest.raises(HumanEvalError, match="adapter identity"):
        HumanEvalEvaluator(
            adapter_config,
            base_model=_base(),
            adapter=AdapterIdentity(family=None, adapter_id=None),
            benchmark=_benchmark(),
            harness=_RecordingHarness(_success()),
        )
