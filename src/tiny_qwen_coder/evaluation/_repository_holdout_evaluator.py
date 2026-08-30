"""Constrained scoring and deterministic artifact materialization for the holdout."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from tiny_qwen_coder.config import EvaluationConfig, ExecutionConfig
from tiny_qwen_coder.evaluation._repository_holdout_common import (
    _REPOSITORY_HOLDOUT_BENCHMARK_CONFIG_PATH,
    _REPOSITORY_HOLDOUT_RUNNER_SOURCE,
    _REPOSITORY_HOLDOUT_SCHEMA_VERSION,
    _RUNNER_COMPILE_EXIT,
    _RUNNER_HARNESS_EXIT,
    _RUNNER_PARSE_EXIT,
    _RUNNER_RUNTIME_EXIT,
    _RUNNER_TEST_EXIT,
    _SUCCESS_MARKER,
    RepositoryHoldoutError,
    RepositoryHoldoutHarness,
)
from tiny_qwen_coder.evaluation._repository_holdout_config import (
    load_frozen_repository_holdout_suite,
    repository_holdout_suite_sha256,
)
from tiny_qwen_coder.evaluation._repository_holdout_runtime import (
    _execution_error_message,
    _passed_test_count,
    _result_json_line,
    _runner_source_sha256,
    _stage_statuses,
    _validate_evaluator_contract,
    create_repository_holdout_prompt,
    normalize_repository_holdout_completion,
    repository_holdout_aggregate_json,
    repository_holdout_results_sha256,
)
from tiny_qwen_coder.evaluation._repository_holdout_types import (
    RepositoryHoldoutAggregate,
    RepositoryHoldoutArtifactPaths,
    RepositoryHoldoutCompletion,
    RepositoryHoldoutPrompt,
    RepositoryHoldoutSuiteConfig,
    RepositoryHoldoutSuiteResult,
    RepositoryHoldoutTask,
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


class RepositoryHoldoutEvaluator:
    """Evaluate exactly one completion for each frozen repository-owned Python task."""

    def __init__(
        self,
        evaluation: EvaluationConfig,
        *,
        base_model: BaseModelIdentity,
        adapter: AdapterIdentity,
        benchmark: ProtectedBenchmark | None = None,
        suite: RepositoryHoldoutSuiteConfig | None = None,
        settings: FrozenEvaluationSettings | None = None,
        harness: RepositoryHoldoutHarness | None = None,
    ) -> None:
        resolved_suite = suite if suite is not None else load_frozen_repository_holdout_suite()
        resolved_settings = settings if settings is not None else load_frozen_evaluation_settings()
        if benchmark is None:
            benchmark = load_protected_benchmark_config(_REPOSITORY_HOLDOUT_BENCHMARK_CONFIG_PATH)
        settings_sha256 = _validate_evaluator_contract(
            evaluation,
            resolved_suite,
            benchmark,
            resolved_settings,
            base_model,
            adapter,
        )
        self._evaluation = evaluation
        self._base_model = base_model
        self._adapter = adapter
        self._benchmark = benchmark
        self._suite = resolved_suite
        self._settings_sha256 = settings_sha256
        self._harness = harness if harness is not None else ConstrainedExecutionHarness()

    @property
    def suite(self) -> RepositoryHoldoutSuiteConfig:
        return self._suite

    @property
    def benchmark(self) -> ProtectedBenchmark:
        return self._benchmark

    def prompt_for(self, task: RepositoryHoldoutTask) -> RepositoryHoldoutPrompt:
        return create_repository_holdout_prompt(task, self._suite)

    def evaluate_completion(
        self,
        task: RepositoryHoldoutTask,
        completion: RepositoryHoldoutCompletion,
    ) -> EvaluationResult:
        """Score one generated Python module in the constrained execution harness."""

        if completion.problem_id != task.problem_id:
            raise RepositoryHoldoutError("completion problem_id does not match holdout task")
        total_tests = task.expected_tests
        if completion.generation_error is not None:
            return create_evaluation_result(
                problem_id=task.problem_id,
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

        candidate_source = normalize_repository_holdout_completion(
            task,
            completion.generated_text,
            self._suite,
        )
        request = ExecutionRequest(
            image=self._suite.execution_image,
            command=("python", "-I", "-B", "runner.py"),
            files=(
                ExecutionFile.from_text("runner.py", _REPOSITORY_HOLDOUT_RUNNER_SOURCE),
                ExecutionFile.from_text("candidate.py", candidate_source),
                ExecutionFile.from_text("setup.py", task.setup_source),
                ExecutionFile.from_text("tests.py", task.test_source),
                ExecutionFile.from_text(
                    "metadata.json",
                    json.dumps({"expected_tests": total_tests}, sort_keys=True) + "\n",
                ),
            ),
        )
        execution = ExecutionConfig(
            timeout_seconds=self._suite.execution_timeout_seconds,
            network_enabled=False,
        )
        try:
            executed = self._harness.run(request, execution, limits=self._suite.limits)
        except ExecutionHarnessError as exc:
            return create_evaluation_result(
                problem_id=task.problem_id,
                language="python",
                generated_text=completion.generated_text,
                generated_code=candidate_source,
                parse_status=EvaluationStageStatus.NOT_RUN,
                compile_status=EvaluationStageStatus.NOT_RUN,
                tests=EvaluationTestSummary(passed=0, total=total_tests),
                error_category=EvaluationErrorCategory.HARNESS,
                error_message=f"repository holdout execution harness failed: {exc}",
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
                    problem_id=task.problem_id,
                    language="python",
                    generated_text=completion.generated_text,
                    generated_code=candidate_source,
                    parse_status=EvaluationStageStatus.PASSED,
                    compile_status=EvaluationStageStatus.PASSED,
                    tests=EvaluationTestSummary(passed=passed_tests, total=total_tests),
                    error_category=EvaluationErrorCategory.HARNESS,
                    error_message="holdout runner reported success without all test-pass markers",
                    generation=completion.generation,
                    base_model=self._base_model,
                    adapter=self._adapter,
                )
            return create_evaluation_result(
                problem_id=task.problem_id,
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
            problem_id=task.problem_id,
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
        completions: Iterable[RepositoryHoldoutCompletion],
    ) -> RepositoryHoldoutSuiteResult:
        """Evaluate one and only one completion for every frozen holdout task."""

        completion_by_id: dict[str, RepositoryHoldoutCompletion] = {}
        for completion in completions:
            if completion.problem_id in completion_by_id:
                raise RepositoryHoldoutError(
                    f"duplicate repository holdout completion for {completion.problem_id}"
                )
            completion_by_id[completion.problem_id] = completion
        problem_ids = tuple(task.problem_id for task in self._suite.tasks)
        missing = sorted(set(problem_ids) - set(completion_by_id))
        extra = sorted(set(completion_by_id) - set(problem_ids))
        if missing or extra:
            raise RepositoryHoldoutError(
                "repository holdout completions must exactly match tasks; "
                f"missing={missing!r}, extra={extra!r}"
            )

        results = tuple(
            self.evaluate_completion(task, completion_by_id[task.problem_id])
            for task in self._suite.tasks
        )
        passed = sum(result.error_category is EvaluationErrorCategory.NONE for result in results)
        timed_out = sum(
            result.error_category is EvaluationErrorCategory.TIMEOUT for result in results
        )
        harness_errors = sum(
            result.error_category is EvaluationErrorCategory.HARNESS for result in results
        )
        total = len(results)
        aggregate = RepositoryHoldoutAggregate(
            schema_version=_REPOSITORY_HOLDOUT_SCHEMA_VERSION,
            benchmark_id=self._benchmark.id,
            dataset_id=self._benchmark.dataset_id,
            dataset_revision=self._benchmark.dataset_revision,
            suite_version=self._suite.suite_version,
            suite_sha256=repository_holdout_suite_sha256(self._suite),
            runner_source_sha256=_runner_source_sha256(),
            evaluation_settings_sha256=self._settings_sha256,
            prompt_version=self._suite.prompt_version,
            completion_normalizer_version=self._suite.completion_normalizer_version,
            execution_image=self._suite.execution_image,
            base_model=self._base_model,
            adapter=self._adapter,
            total_problems=total,
            passed=passed,
            failed=total - passed,
            timed_out=timed_out,
            harness_errors=harness_errors,
            pass_at_1=None if harness_errors else passed / total,
            results_sha256=repository_holdout_results_sha256(results),
        )
        return RepositoryHoldoutSuiteResult(results=results, aggregate=aggregate)

    def write_artifacts(
        self,
        suite: RepositoryHoldoutSuiteResult,
        output_dir: Path | None = None,
    ) -> RepositoryHoldoutArtifactPaths:
        """Write deterministic per-task JSONL and aggregate JSON artifacts."""

        destination = output_dir if output_dir is not None else Path(self._evaluation.output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        per_problem_path = destination / "repository-holdout-results.jsonl"
        aggregate_path = destination / "repository-holdout-aggregate.json"
        _write_atomic(
            per_problem_path,
            "".join(_result_json_line(result) + "\n" for result in suite.results),
        )
        _write_atomic(aggregate_path, repository_holdout_aggregate_json(suite.aggregate))
        return RepositoryHoldoutArtifactPaths(
            per_problem=per_problem_path,
            aggregate=aggregate_path,
        )


def _write_atomic(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
