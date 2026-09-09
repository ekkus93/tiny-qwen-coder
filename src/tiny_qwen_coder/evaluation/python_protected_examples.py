"""Load Python benchmark examples solely for post-generation contamination checks.

Protected benchmark text is loaded only in memory after teacher generation. The
returned examples are consumed by the contamination checker; benchmark prompts and
solutions are never written into the distilled training corpus or its manifest.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import cast

from tiny_qwen_coder.data.records import TrainingMessage
from tiny_qwen_coder.evaluation.contamination import ProtectedBenchmarkExample
from tiny_qwen_coder.evaluation.humaneval import (
    create_humaneval_prompt,
    load_frozen_humaneval_runner_config,
    load_humaneval_problems,
)
from tiny_qwen_coder.evaluation.mbpp import (
    create_mbpp_prompt,
    load_frozen_mbpp_runner_config,
    load_mbpp_problems,
)
from tiny_qwen_coder.evaluation.protected_benchmarks import (
    ProtectedBenchmark,
    ProtectedBenchmarkRegistry,
)
from tiny_qwen_coder.evaluation.repository_holdout import (
    create_repository_holdout_prompt,
    load_frozen_repository_holdout_suite,
)

DatasetRow = Mapping[str, object]
HumanEvalRowsLoader = Callable[..., Iterable[DatasetRow]]
MBPPRowsLoader = Callable[..., Iterable[DatasetRow]]


def _load_humaneval_rows(
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


def _load_mbpp_rows(
    repository: str,
    dataset_config: str,
    *,
    revision: str,
    split: str,
    streaming: bool,
) -> Iterable[DatasetRow]:
    from datasets import load_dataset

    loaded = load_dataset(
        repository,
        dataset_config,
        revision=revision,
        split=split,
        streaming=streaming,
    )
    return cast(Iterable[DatasetRow], loaded)


def _required_row_string(row: DatasetRow, key: str, *, context: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context}.{key} must be a non-empty string")
    return value


def _row_string_tuple(
    row: DatasetRow,
    key: str,
    *,
    context: str,
    required: bool = False,
) -> tuple[str, ...]:
    value = row.get(key)
    if value is None and not required:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{context}.{key} must be a sequence of strings")
    output: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{context}.{key}[{index}] must be a non-empty string")
        output.append(item)
    if required and not output:
        raise ValueError(f"{context}.{key} must contain at least one test")
    return tuple(output)


def _human_eval_examples(
    benchmark: ProtectedBenchmark,
    *,
    dataset_loader: HumanEvalRowsLoader,
) -> tuple[ProtectedBenchmarkExample, ...]:
    runner = load_frozen_humaneval_runner_config()
    rows = tuple(
        dataset_loader(
            benchmark.dataset_id,
            revision=benchmark.dataset_revision,
            split=runner.dataset_split,
            streaming=False,
        )
    )

    def cached_loader(
        repository: str,
        *,
        revision: str,
        split: str,
        streaming: bool,
    ) -> Iterable[DatasetRow]:
        del repository, revision, split, streaming
        return rows

    problems = load_humaneval_problems(benchmark, runner, dataset_loader=cached_loader)
    solutions: dict[str, str] = {}
    tests: dict[str, tuple[str, ...]] = {}
    for index, row in enumerate(rows):
        context = f"HumanEval protected row {index}"
        task_id = _required_row_string(row, "task_id", context=context)
        solution = _required_row_string(row, "canonical_solution", context=context)
        solutions[task_id] = solution
        tests[task_id] = (_required_row_string(row, "test", context=context),)

    return tuple(
        ProtectedBenchmarkExample(
            language="python",
            benchmark_id=benchmark.id,
            dataset_id=benchmark.dataset_id,
            dataset_revision=benchmark.dataset_revision,
            record_id=problem.task_id,
            prompt_messages=(
                TrainingMessage(
                    role="user",
                    content=create_humaneval_prompt(problem, runner).user_content,
                ),
            ),
            solution=solutions[problem.task_id],
            test_texts=tests[problem.task_id],
        )
        for problem in problems
    )


def _mbpp_examples(
    benchmark: ProtectedBenchmark,
    *,
    dataset_loader: MBPPRowsLoader,
) -> tuple[ProtectedBenchmarkExample, ...]:
    runner = load_frozen_mbpp_runner_config()
    rows = tuple(
        dataset_loader(
            benchmark.dataset_id,
            runner.dataset_config,
            revision=benchmark.dataset_revision,
            split=runner.dataset_split,
            streaming=False,
        )
    )

    def cached_loader(
        repository: str,
        dataset_config: str,
        *,
        revision: str,
        split: str,
        streaming: bool,
    ) -> Iterable[DatasetRow]:
        del repository, dataset_config, revision, split, streaming
        return rows

    problems = load_mbpp_problems(benchmark, runner, dataset_loader=cached_loader)
    solutions: dict[str, str] = {}
    tests: dict[str, tuple[str, ...]] = {}
    for index, row in enumerate(rows):
        context = f"MBPP protected row {index}"
        task_id = row.get("task_id")
        if isinstance(task_id, bool) or not isinstance(task_id, int):
            raise ValueError(f"{context}.task_id must be an integer")
        problem_id = f"MBPP/{task_id}"
        solutions[problem_id] = _required_row_string(row, "code", context=context)
        tests[problem_id] = (
            *_row_string_tuple(row, "test_list", context=context, required=True),
            *_row_string_tuple(row, "challenge_test_list", context=context),
        )

    return tuple(
        ProtectedBenchmarkExample(
            language="python",
            benchmark_id=benchmark.id,
            dataset_id=benchmark.dataset_id,
            dataset_revision=benchmark.dataset_revision,
            record_id=problem.task_id,
            prompt_messages=(
                TrainingMessage(
                    role="user",
                    content=create_mbpp_prompt(problem, runner).user_content,
                ),
            ),
            solution=solutions[problem.task_id],
            test_texts=tests[problem.task_id],
        )
        for problem in problems
    )


def _repository_holdout_examples(
    benchmark: ProtectedBenchmark,
) -> tuple[ProtectedBenchmarkExample, ...]:
    suite = load_frozen_repository_holdout_suite()
    if benchmark.dataset_revision != suite.dataset_revision:
        raise ValueError("repository holdout protected revision does not match frozen suite")
    if benchmark.id != suite.benchmark_id:
        raise ValueError("repository holdout protected benchmark ID does not match frozen suite")

    return tuple(
        ProtectedBenchmarkExample(
            language="python",
            benchmark_id=benchmark.id,
            dataset_id=benchmark.dataset_id,
            dataset_revision=benchmark.dataset_revision,
            record_id=task.problem_id,
            prompt_messages=(
                TrainingMessage(
                    role="user",
                    content=create_repository_holdout_prompt(task, suite).user_content,
                ),
            ),
            solution=None,
            test_texts=tuple(text for text in (task.test_source, task.setup_source) if text),
        )
        for task in suite.tasks
    )


def load_python_protected_examples(
    registry: ProtectedBenchmarkRegistry,
    *,
    humaneval_dataset_loader: HumanEvalRowsLoader = _load_humaneval_rows,
    mbpp_dataset_loader: MBPPRowsLoader = _load_mbpp_rows,
) -> tuple[ProtectedBenchmarkExample, ...]:
    """Load every registered Python benchmark example for contamination checking.

    The function fails closed if a newly registered benchmark has no explicit
    protected-example loader. This prevents a clean contamination status from being
    emitted when registry coverage expands without corresponding example coverage.
    """

    examples: list[ProtectedBenchmarkExample] = []
    for benchmark in registry.list_benchmarks(language="python"):
        if benchmark.id == "humaneval":
            examples.extend(
                _human_eval_examples(benchmark, dataset_loader=humaneval_dataset_loader)
            )
        elif benchmark.id == "mbpp":
            examples.extend(_mbpp_examples(benchmark, dataset_loader=mbpp_dataset_loader))
        elif benchmark.id == "repository-holdout":
            examples.extend(_repository_holdout_examples(benchmark))
        else:
            raise ValueError(
                f"no protected-example loader exists for registered benchmark {benchmark.id!r}"
            )
    return tuple(examples)


__all__ = ["load_python_protected_examples"]
