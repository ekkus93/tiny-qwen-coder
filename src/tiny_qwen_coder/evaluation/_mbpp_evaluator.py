"""Constrained scoring and deterministic artifact materialization for MBPP."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from tiny_qwen_coder.config import EvaluationConfig, ExecutionConfig
from tiny_qwen_coder.evaluation._mbpp_common import (
    _MBPP_BENCHMARK_CONFIG_PATH,
    _MBPP_RUNNER_SOURCE,
    _MBPP_SCHEMA_VERSION,
    _RUNNER_COMPILE_EXIT,
    _RUNNER_HARNESS_EXIT,
    _RUNNER_PARSE_EXIT,
    _RUNNER_RUNTIME_EXIT,
    _RUNNER_TEST_EXIT,
    _SUCCESS_MARKER,
    MBPPDatasetRowsLoader,
    MBPPError,
    MBPPHarness,
)
from tiny_qwen_coder.evaluation._mbpp_config import (
    load_frozen_mbpp_runner_config,
    mbpp_runner_config_sha256,
)
from tiny_qwen_coder.evaluation._mbpp_data import (
    _load_huggingface_rows,
    _result_json_line,
    create_mbpp_prompt,
    load_mbpp_problems,
    mbpp_aggregate_json,
    mbpp_problem_set_sha256,
    mbpp_results_sha256,
    normalize_mbpp_completion,
)
from tiny_qwen_coder.evaluation._mbpp_runtime import (
    _execution_error_message,
    _passed_test_count,
    _runner_source_sha256,
    _stage_statuses,
    _validate_evaluator_contract,
)
from tiny_qwen_coder.evaluation._mbpp_types import (
    MBPPAggregate,
    MBPPArtifactPaths,
    MBPPCompletion,
    MBPPProblem,
    MBPPPrompt,
    MBPPRunnerConfig,
    MBPPSuiteResult,
)
from tiny_qwen_coder.evaluation.execution import (
    ConstrainedExecutionHarness,
    ExecutionFile,
    ExecutionHarnessError,
    ExecutionRequest,
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
    create_evaluation_result,
)
from tiny_qwen_coder.evaluation.settings import (
    FrozenEvaluationSettings,
    load_frozen_evaluation_settings,
)
from tiny_qwen_coder.identities import AdapterIdentity, BaseModelIdentity


class MBPPEvaluator:
    """Evaluate exactly one completion per MBPP task under frozen settings."""

    def __init__(
        self,
        evaluation: EvaluationConfig,
        *,
        base_model: BaseModelIdentity,
        adapter: AdapterIdentity,
        benchmark: ProtectedBenchmark | None = None,
        runner: MBPPRunnerConfig | None = None,
        settings: FrozenEvaluationSettings | None = None,
        harness: MBPPHarness | None = None,
    ) -> None:
        resolved_runner = runner if runner is not None else load_frozen_mbpp_runner_config()
        resolved_settings = settings if settings is not None else load_frozen_evaluation_settings()
        if benchmark is None:
            benchmark = load_protected_benchmark_config(_MBPP_BENCHMARK_CONFIG_PATH)
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
    def runner(self) -> MBPPRunnerConfig:
        return self._runner

    @property
    def benchmark(self) -> ProtectedBenchmark:
        return self._benchmark

    def load_problems(
        self,
        *,
        dataset_loader: MBPPDatasetRowsLoader = _load_huggingface_rows,
    ) -> tuple[MBPPProblem, ...]:
        """Load the exact protected MBPP problem set."""

        return load_mbpp_problems(
            self._benchmark,
            self._runner,
            dataset_loader=dataset_loader,
        )

    def prompt_for(self, problem: MBPPProblem) -> MBPPPrompt:
        return create_mbpp_prompt(problem, self._runner)

    def evaluate_completion(
        self,
        problem: MBPPProblem,
        completion: MBPPCompletion,
    ) -> EvaluationResult:
        """Score one model completion with the constrained execution harness."""

        if completion.task_id != problem.task_id:
            raise MBPPError("completion task_id does not match MBPP problem")
        total_tests = len(problem.tests)
        if total_tests != self._runner.tests_per_problem:
            raise MBPPError("problem test count does not match the frozen MBPP runner")
        if completion.generation_error is not None:
            return create_evaluation_result(
                problem_id=problem.task_id,
                language="python",
                generated_text=completion.generated_text,
                generated_code=None,
                parse_status=EvaluationStageStatus.NOT_RUN,
                compile_status=EvaluationStageStatus.NOT_RUN,
                tests=EvaluationTestSummary(passed=0, total=total_tests),
                error_category=EvaluationErrorCategory.GENERATION,
                error_message=completion.generation_error,
                generation=completion.generation,
                base_model=self._base_model,
                adapter=self._adapter,
            )

        candidate_source = normalize_mbpp_completion(
            problem,
            completion.generated_text,
            self._runner,
        )
        request = ExecutionRequest(
            image=self._runner.execution_image,
            command=("python", "-I", "-B", "runner.py"),
            files=(
                ExecutionFile.from_text("runner.py", _MBPP_RUNNER_SOURCE),
                ExecutionFile.from_text("candidate.py", candidate_source),
                ExecutionFile.from_text("setup.py", problem.test_setup_code),
                ExecutionFile.from_text(
                    "tests.json",
                    json.dumps(list(problem.tests), ensure_ascii=False) + "\n",
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
                tests=EvaluationTestSummary(passed=0, total=total_tests),
                error_category=EvaluationErrorCategory.HARNESS,
                error_message=f"MBPP execution harness failed: {exc}",
                generation=completion.generation,
                base_model=self._base_model,
                adapter=self._adapter,
            )

        parse_status, compile_status = _stage_statuses(executed)
        passed_tests = _passed_test_count(executed, total=total_tests)
        stderr_lines = set(executed.stderr.splitlines())
        if executed.status is ExecutionStatus.SUCCEEDED and _SUCCESS_MARKER in stderr_lines:
            if (
                parse_status is not EvaluationStageStatus.PASSED
                or compile_status is not EvaluationStageStatus.PASSED
                or passed_tests != total_tests
            ):
                return create_evaluation_result(
                    problem_id=problem.task_id,
                    language="python",
                    generated_text=completion.generated_text,
                    generated_code=candidate_source,
                    parse_status=EvaluationStageStatus.PASSED,
                    compile_status=EvaluationStageStatus.PASSED,
                    tests=EvaluationTestSummary(passed=passed_tests, total=total_tests),
                    error_category=EvaluationErrorCategory.HARNESS,
                    error_message="MBPP runner reported success without all test-pass markers",
                    generation=completion.generation,
                    base_model=self._base_model,
                    adapter=self._adapter,
                )
            return create_evaluation_result(
                problem_id=problem.task_id,
                language="python",
                generated_text=completion.generated_text,
                generated_code=candidate_source,
                parse_status=EvaluationStageStatus.PASSED,
                compile_status=EvaluationStageStatus.PASSED,
                tests=EvaluationTestSummary(passed=total_tests, total=total_tests),
                error_category=EvaluationErrorCategory.NONE,
                error_message=None,
                generation=completion.generation,
                base_model=self._base_model,
                adapter=self._adapter,
            )

        if executed.status is ExecutionStatus.SUCCEEDED:
            category = EvaluationErrorCategory.HARNESS
        elif executed.status is ExecutionStatus.TIMED_OUT:
            category = EvaluationErrorCategory.TIMEOUT
        elif executed.exit_code == _RUNNER_PARSE_EXIT:
            category = EvaluationErrorCategory.PARSE
        elif executed.exit_code == _RUNNER_COMPILE_EXIT:
            category = EvaluationErrorCategory.COMPILE
        elif executed.exit_code == _RUNNER_TEST_EXIT:
            category = EvaluationErrorCategory.TEST
        elif executed.exit_code == _RUNNER_RUNTIME_EXIT:
            category = EvaluationErrorCategory.RUNTIME
        elif executed.exit_code == _RUNNER_HARNESS_EXIT or (
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
            tests=EvaluationTestSummary(passed=passed_tests, total=total_tests),
            error_category=category,
            error_message=_execution_error_message(executed),
            generation=completion.generation,
            base_model=self._base_model,
            adapter=self._adapter,
        )

    def evaluate_suite(
        self,
        problems: tuple[MBPPProblem, ...],
        completions: Iterable[MBPPCompletion],
    ) -> MBPPSuiteResult:
        """Evaluate one and only one completion for every supplied MBPP task."""

        completion_by_id: dict[str, MBPPCompletion] = {}
        for completion in completions:
            if completion.task_id in completion_by_id:
                raise MBPPError(f"duplicate MBPP completion for {completion.task_id}")
            completion_by_id[completion.task_id] = completion
        problem_ids = tuple(problem.task_id for problem in problems)
        if len(problem_ids) != len(set(problem_ids)):
            raise MBPPError("MBPP problems must not contain duplicate task IDs")
        missing = sorted(set(problem_ids) - set(completion_by_id))
        extra = sorted(set(completion_by_id) - set(problem_ids))
        if missing or extra:
            raise MBPPError(
                "MBPP completions must exactly match problems; "
                f"missing={missing!r}, extra={extra!r}"
            )

        results = tuple(
            self.evaluate_completion(problem, completion_by_id[problem.task_id])
            for problem in problems
        )
        total = len(results)
        if total == 0:
            raise MBPPError("MBPP suite must contain at least one problem")
        passed = sum(result.error_category is EvaluationErrorCategory.NONE for result in results)
        timed_out = sum(
            result.error_category is EvaluationErrorCategory.TIMEOUT for result in results
        )
        harness_errors = sum(
            result.error_category is EvaluationErrorCategory.HARNESS for result in results
        )
        aggregate = MBPPAggregate(
            schema_version=_MBPP_SCHEMA_VERSION,
            benchmark_id=self._benchmark.id,
            dataset_id=self._benchmark.dataset_id,
            dataset_revision=self._benchmark.dataset_revision,
            dataset_config=self._runner.dataset_config,
            dataset_split=self._runner.dataset_split,
            reference_repository=self._runner.reference_repository,
            reference_revision=self._runner.reference_revision,
            runner_config_sha256=mbpp_runner_config_sha256(self._runner),
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
            problem_set_sha256=mbpp_problem_set_sha256(problems),
            results_sha256=mbpp_results_sha256(results),
        )
        return MBPPSuiteResult(results=results, aggregate=aggregate)

    def write_artifacts(
        self,
        suite: MBPPSuiteResult,
        output_dir: Path | None = None,
    ) -> MBPPArtifactPaths:
        """Write deterministic per-problem JSONL and aggregate JSON artifacts."""

        destination = output_dir if output_dir is not None else Path(self._evaluation.output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        per_problem_path = destination / "mbpp-results.jsonl"
        aggregate_path = destination / "mbpp-aggregate.json"
        _write_atomic(
            per_problem_path,
            "".join(_result_json_line(result) + "\n" for result in suite.results),
        )
        _write_atomic(aggregate_path, mbpp_aggregate_json(suite.aggregate))
        return MBPPArtifactPaths(per_problem=per_problem_path, aggregate=aggregate_path)


def _write_atomic(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
