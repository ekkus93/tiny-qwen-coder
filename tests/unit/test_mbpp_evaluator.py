"""P6-003 tests for deterministic constrained MBPP evaluation."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

import pytest

from tiny_qwen_coder.config import EvaluationConfig, ExecutionConfig, GenerationConfig
from tiny_qwen_coder.evaluation import (
    EvaluationErrorCategory,
    EvaluationStageStatus,
    ExecutionHarnessUnavailableError,
    ExecutionLimits,
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    GenerationStats,
    MBPPCompletion,
    MBPPError,
    MBPPEvaluator,
    MBPPProblem,
    OciRuntime,
    create_mbpp_prompt,
    load_frozen_mbpp_runner_config,
    load_mbpp_problems,
    load_mbpp_runner_config,
    mbpp_problem_set_sha256,
    mbpp_runner_config_sha256,
    normalize_mbpp_completion,
)
from tiny_qwen_coder.evaluation._mbpp_common import _MBPP_RUNNER_SOURCE
from tiny_qwen_coder.evaluation.settings import load_frozen_evaluation_settings
from tiny_qwen_coder.identities import AdapterIdentity, BaseModelIdentity

_MBPP_REVISION = "4bb6404fdc6cacfda99d4ac4205087b89d32030c"
_BASE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
_RUNNER_SHA256 = "1aef848a963ebe07fdf4c9631df94b8668b24af7fa3d9edcba650bd1fe622061"


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
        suites=("mbpp",),
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


def _problem(task: int = 11) -> MBPPProblem:
    return MBPPProblem(
        task_id=f"MBPP/{task}",
        description="Write a function to add two integers.",
        tests=(
            "assert add(2, 3) == 5",
            "assert add(-1, 1) == 0",
            "assert add(0, 0) == 0",
        ),
        test_setup_code="OFFSET = 1\n",
    )


def _completion(task: int = 11, text: str = "def add(a, b):\n    return a + b\n") -> MBPPCompletion:
    return MBPPCompletion(
        task_id=f"MBPP/{task}",
        generated_text=text,
        generation=GenerationStats(
            prompt_tokens=32,
            generated_tokens=12,
            latency_seconds=0.1,
            tokens_per_second=120.0,
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


def _success() -> ExecutionResult:
    return _result(
        status=ExecutionStatus.SUCCEEDED,
        exit_code=0,
        stderr=(
            "__TQC_MBPP_PARSE_OK_V1__\n"
            "__TQC_MBPP_COMPILE_OK_V1__\n"
            "__TQC_MBPP_TEST_OK_V1__:1\n"
            "__TQC_MBPP_TEST_OK_V1__:2\n"
            "__TQC_MBPP_TEST_OK_V1__:3\n"
            "__TQC_MBPP_PASS_V1__\n"
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


def _evaluator(harness: _RecordingHarness) -> MBPPEvaluator:
    return MBPPEvaluator(
        _evaluation(),
        base_model=_base(),
        adapter=AdapterIdentity(family=None, adapter_id=None),
        harness=harness,
    )


def _row(task_id: int) -> dict[str, object]:
    return {
        "task_id": task_id,
        "text": f"Problem {task_id}",
        "code": "def answer():\n    return 1\n",
        "test_list": [
            "assert answer() == 1",
            "assert answer() != 2",
            "assert callable(answer)",
        ],
        "test_setup_code": "",
        "challenge_test_list": ["assert answer() == 1"],
    }


def test_default_evaluator_loads_p6_protected_mbpp_registration() -> None:
    evaluator = _evaluator(_RecordingHarness(_success()))

    assert evaluator.benchmark.dataset_id == "google-research-datasets/mbpp"
    assert evaluator.benchmark.dataset_revision == _MBPP_REVISION


def test_frozen_runner_pins_bigcode_split_prompt_runtime_and_limits() -> None:
    runner = load_frozen_mbpp_runner_config()

    assert runner.reference_repository == "bigcode-project/bigcode-evaluation-harness"
    assert runner.reference_revision == "8fc5bae6479c4fbbb28c3f8b644f6a15b3f3b5bd"
    assert runner.dataset_config == "full"
    assert runner.dataset_split == "test"
    assert (runner.task_id_start, runner.task_id_end, runner.expected_problem_count) == (
        11,
        510,
        500,
    )
    assert runner.tests_per_problem == 3
    assert runner.prompt_version == "mbpp-bigcode-incoder-v1"
    assert "@sha256:" in runner.execution_image
    assert runner.execution_timeout_seconds == 10.0
    assert runner.limits.max_output_bytes == 65536
    assert mbpp_runner_config_sha256(runner) == _RUNNER_SHA256


def test_runner_config_unknown_fields_and_drift_fail_closed(tmp_path: Path) -> None:
    source = Path("configs/eval/python/mbpp_runner_v1.yaml").read_text(encoding="utf-8")
    unknown = tmp_path / "unknown.yaml"
    unknown.write_text(source + "surprise: forbidden\n", encoding="utf-8")
    with pytest.raises(MBPPError, match="unknown field"):
        load_mbpp_runner_config(unknown)

    drift = tmp_path / "drift.yaml"
    drift.write_text(
        source.replace("tests_per_problem: 3", "tests_per_problem: 4"),
        encoding="utf-8",
    )
    with pytest.raises(MBPPError, match="fingerprint mismatch"):
        load_frozen_mbpp_runner_config(drift)


def test_loader_uses_exact_protected_revision_config_split_and_drops_answers() -> None:
    runner = replace(
        load_frozen_mbpp_runner_config(),
        task_id_start=11,
        task_id_end=12,
        expected_problem_count=2,
    )
    calls: list[tuple[object, ...]] = []

    def loader(
        repository: str,
        dataset_config: str,
        *,
        revision: str,
        split: str,
        streaming: bool,
    ) -> Iterable[dict[str, object]]:
        calls.append((repository, dataset_config, revision, split, streaming))
        return (_row(12), _row(11))

    problems = load_mbpp_problems(
        _evaluator(_RecordingHarness(_success())).benchmark,
        runner,
        dataset_loader=loader,
    )

    assert calls == [("google-research-datasets/mbpp", "full", _MBPP_REVISION, "test", False)]
    assert tuple(problem.task_id for problem in problems) == ("MBPP/11", "MBPP/12")
    assert not hasattr(problems[0], "code")
    assert not hasattr(problems[0], "challenge_test_list")


def test_loader_rejects_schema_test_count_wrong_count_and_noncanonical_ids() -> None:
    benchmark = _evaluator(_RecordingHarness(_success())).benchmark
    runner = replace(
        load_frozen_mbpp_runner_config(),
        task_id_start=11,
        task_id_end=12,
        expected_problem_count=2,
    )

    bad_schema = _row(11)
    bad_schema["extra"] = "nope"
    with pytest.raises(MBPPError, match="schema"):
        load_mbpp_problems(
            benchmark,
            runner,
            dataset_loader=lambda *args, **kwargs: (bad_schema, _row(12)),
        )

    bad_tests = _row(11)
    bad_tests["test_list"] = ["assert True"]
    with pytest.raises(MBPPError, match="exactly 3 tests"):
        load_mbpp_problems(
            benchmark,
            runner,
            dataset_loader=lambda *args, **kwargs: (bad_tests, _row(12)),
        )

    with pytest.raises(MBPPError, match="expected 2 problems"):
        load_mbpp_problems(benchmark, runner, dataset_loader=lambda *args, **kwargs: (_row(11),))

    with pytest.raises(MBPPError, match="canonical contiguous"):
        load_mbpp_problems(
            benchmark,
            runner,
            dataset_loader=lambda *args, **kwargs: (_row(11), _row(13)),
        )


def test_prompt_exactly_matches_bigcode_incoder_style() -> None:
    runner = load_frozen_mbpp_runner_config()
    prompt = create_mbpp_prompt(_problem(), runner)

    assert prompt.user_content == (
        '"""\nWrite a function to add two integers.\nassert add(2, 3) == 5\n"""\n'
    )
    assert "return a + b" not in prompt.user_content


def test_completion_normalizer_accepts_raw_fence_echo_and_reference_stop_words() -> None:
    problem = _problem()
    runner = load_frozen_mbpp_runner_config()
    code = "def add(a, b):\n    return a + b\n"
    prompt = create_mbpp_prompt(problem, runner).user_content

    assert normalize_mbpp_completion(problem, code, runner) == code
    assert normalize_mbpp_completion(problem, f"```python\n{code}```", runner) == code
    assert normalize_mbpp_completion(problem, prompt + code, runner) == code
    assert normalize_mbpp_completion(problem, code + "\nassert False\n", runner) == code


def test_success_uses_pinned_sandbox_and_emits_common_result() -> None:
    harness = _RecordingHarness(_success())
    result = _evaluator(harness).evaluate_completion(_problem(), _completion())

    assert result.error_category is EvaluationErrorCategory.NONE
    assert result.parse_status is EvaluationStageStatus.PASSED
    assert result.compile_status is EvaluationStageStatus.PASSED
    assert (result.tests.passed, result.tests.total) == (3, 3)
    request = harness.requests[0]
    assert request.image == load_frozen_mbpp_runner_config().execution_image
    assert request.command == ("python", "-I", "-B", "runner.py")
    assert {item.path for item in request.files} == {
        "runner.py",
        "candidate.py",
        "setup.py",
        "tests.json",
    }
    assert harness.executions[0].network_enabled is False
    assert harness.limits[0] == load_frozen_mbpp_runner_config().limits


@pytest.mark.parametrize(
    ("executed", "category", "parse_status", "compile_status", "passed"),
    (
        (
            _result(
                status=ExecutionStatus.FAILED,
                exit_code=10,
                stderr="__TQC_MBPP_ERROR_V1__:parse:SyntaxError:oops\n",
            ),
            EvaluationErrorCategory.PARSE,
            EvaluationStageStatus.FAILED,
            EvaluationStageStatus.NOT_RUN,
            0,
        ),
        (
            _result(
                status=ExecutionStatus.FAILED,
                exit_code=11,
                stderr=(
                    "__TQC_MBPP_PARSE_OK_V1__\n"
                    "__TQC_MBPP_ERROR_V1__:compile:ValueError:oops\n"
                ),
            ),
            EvaluationErrorCategory.COMPILE,
            EvaluationStageStatus.PASSED,
            EvaluationStageStatus.FAILED,
            0,
        ),
        (
            _result(
                status=ExecutionStatus.FAILED,
                exit_code=12,
                stderr=(
                    "__TQC_MBPP_PARSE_OK_V1__\n"
                    "__TQC_MBPP_COMPILE_OK_V1__\n"
                    "__TQC_MBPP_TEST_OK_V1__:1\n"
                    "__TQC_MBPP_TEST_OK_V1__:2\n"
                    "__TQC_MBPP_ERROR_V1__:test[3]:AssertionError:oops\n"
                ),
            ),
            EvaluationErrorCategory.TEST,
            EvaluationStageStatus.PASSED,
            EvaluationStageStatus.PASSED,
            2,
        ),
        (
            _result(
                status=ExecutionStatus.FAILED,
                exit_code=13,
                stderr=(
                    "__TQC_MBPP_PARSE_OK_V1__\n"
                    "__TQC_MBPP_COMPILE_OK_V1__\n"
                    "__TQC_MBPP_ERROR_V1__:runtime:RuntimeError:oops\n"
                ),
            ),
            EvaluationErrorCategory.RUNTIME,
            EvaluationStageStatus.PASSED,
            EvaluationStageStatus.PASSED,
            0,
        ),
        (
            _result(
                status=ExecutionStatus.TIMED_OUT,
                exit_code=None,
                stderr="__TQC_MBPP_PARSE_OK_V1__\n__TQC_MBPP_COMPILE_OK_V1__\n",
            ),
            EvaluationErrorCategory.TIMEOUT,
            EvaluationStageStatus.PASSED,
            EvaluationStageStatus.PASSED,
            0,
        ),
    ),
)
def test_execution_outcomes_map_to_common_error_schema(
    executed: ExecutionResult,
    category: EvaluationErrorCategory,
    parse_status: EvaluationStageStatus,
    compile_status: EvaluationStageStatus,
    passed: int,
) -> None:
    result = _evaluator(_RecordingHarness(executed)).evaluate_completion(_problem(), _completion())

    assert result.error_category is category
    assert result.parse_status is parse_status
    assert result.compile_status is compile_status
    assert result.tests.passed == passed
    assert result.tests.total == 3


def test_generation_harness_and_pre_runner_failures_are_explicit() -> None:
    generation_failure = replace(_completion(), generation_error="generation failed")
    result = _evaluator(_RecordingHarness(_success())).evaluate_completion(
        _problem(),
        generation_failure,
    )
    assert result.error_category is EvaluationErrorCategory.GENERATION

    result = _evaluator(
        _RecordingHarness(ExecutionHarnessUnavailableError("missing runtime"))
    ).evaluate_completion(_problem(), _completion())
    assert result.error_category is EvaluationErrorCategory.HARNESS

    result = _evaluator(
        _RecordingHarness(
            _result(status=ExecutionStatus.FAILED, exit_code=125, stderr="OCI launch failed\n")
        )
    ).evaluate_completion(_problem(), _completion())
    assert result.error_category is EvaluationErrorCategory.HARNESS


def test_trusted_runner_executes_setup_candidate_and_each_test(tmp_path: Path) -> None:
    (tmp_path / "runner.py").write_text(_MBPP_RUNNER_SOURCE, encoding="utf-8")
    (tmp_path / "candidate.py").write_text(
        "def shifted(value):\n    return value + OFFSET\n",
        encoding="utf-8",
    )
    (tmp_path / "setup.py").write_text("OFFSET = 1\n", encoding="utf-8")
    (tmp_path / "tests.json").write_text(
        json.dumps(
            [
                "assert shifted(1) == 2",
                "assert shifted(2) == 3",
                "assert shifted(3) == 4",
            ]
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [sys.executable, "-I", "-B", "runner.py"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "__TQC_MBPP_TEST_OK_V1__:3" in completed.stderr
    assert "__TQC_MBPP_PASS_V1__" in completed.stderr


def test_suite_aggregate_artifacts_and_repeated_runs_are_deterministic(tmp_path: Path) -> None:
    problems = (_problem(11), _problem(12))
    completions = (_completion(11), _completion(12))
    failure = _result(
        status=ExecutionStatus.FAILED,
        exit_code=12,
        stderr=(
            "__TQC_MBPP_PARSE_OK_V1__\n"
            "__TQC_MBPP_COMPILE_OK_V1__\n"
            "__TQC_MBPP_TEST_OK_V1__:1\n"
            "__TQC_MBPP_ERROR_V1__:test[2]:AssertionError:oops\n"
        ),
    )
    first = _evaluator(_RecordingHarness(_success(), failure)).evaluate_suite(
        problems,
        completions,
    )
    second = _evaluator(_RecordingHarness(_success(), failure)).evaluate_suite(
        problems,
        completions,
    )

    assert first == second
    assert first.aggregate.total_problems == 2
    assert first.aggregate.passed == 1
    assert first.aggregate.pass_at_1 == 0.5
    assert len(first.aggregate.problem_set_sha256) == 64
    assert first.aggregate.problem_set_sha256 == mbpp_problem_set_sha256(problems)

    paths = _evaluator(_RecordingHarness(_success())).write_artifacts(first, tmp_path)
    per_problem = paths.per_problem.read_text(encoding="utf-8")
    aggregate = paths.aggregate.read_text(encoding="utf-8")
    assert "assert add(2, 3) == 5" not in per_problem + aggregate
    assert "challenge_test_list" not in per_problem + aggregate
    assert "canonical_solution" not in per_problem + aggregate
    assert json.loads(aggregate)["pass_at_1"] == 0.5


def test_harness_error_invalidates_aggregate_pass_at_1() -> None:
    suite = _evaluator(
        _RecordingHarness(ExecutionHarnessUnavailableError("missing runtime"))
    ).evaluate_suite((_problem(),), (_completion(),))

    assert suite.aggregate.harness_errors == 1
    assert suite.aggregate.pass_at_1 is None


def test_suite_requires_exactly_one_completion_per_problem() -> None:
    evaluator = _evaluator(_RecordingHarness(_success()))
    with pytest.raises(MBPPError, match="exactly match"):
        evaluator.evaluate_suite((_problem(11), _problem(12)), (_completion(11),))
    with pytest.raises(MBPPError, match="duplicate MBPP completion"):
        evaluator.evaluate_suite((_problem(11),), (_completion(11), _completion(11)))


def test_evaluator_rejects_network_timeout_suite_base_and_adapter_drift() -> None:
    base = _evaluation()
    adapter = AdapterIdentity(family=None, adapter_id=None)

    with pytest.raises(MBPPError, match="network_enabled=false"):
        MBPPEvaluator(
            replace(base, execution=ExecutionConfig(timeout_seconds=10.0, network_enabled=True)),
            base_model=_base(),
            adapter=adapter,
        )
    with pytest.raises(MBPPError, match="timeout"):
        MBPPEvaluator(
            replace(base, execution=ExecutionConfig(timeout_seconds=5.0, network_enabled=False)),
            base_model=_base(),
            adapter=adapter,
        )
    with pytest.raises(MBPPError, match="suites"):
        MBPPEvaluator(replace(base, suites=("other",)), base_model=_base(), adapter=adapter)
    wrong_base = replace(_base(), repository="Qwen/not-the-base")
    with pytest.raises(MBPPError, match="base model identity"):
        MBPPEvaluator(base, base_model=wrong_base, adapter=adapter)
    with pytest.raises(MBPPError, match="adapter_id"):
        MBPPEvaluator(
            base,
            base_model=_base(),
            adapter=AdapterIdentity(family="language", adapter_id="language/python/p0"),
        )
