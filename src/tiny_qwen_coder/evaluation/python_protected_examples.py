"""Load Python benchmark examples solely for post-generation contamination checks.

Protected benchmark text is loaded only in memory after teacher generation. The
returned examples are consumed by the contamination checker; benchmark prompts and
solutions are never written into the distilled training corpus or its manifest.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

from tiny_qwen_coder.data.records import TrainingMessage
from tiny_qwen_coder.evaluation.contamination import ProtectedBenchmarkExample
from tiny_qwen_coder.evaluation.humaneval import (
    HumanEvalProblem,
    create_humaneval_prompt,
    load_frozen_humaneval_runner_config,
    load_humaneval_problems,
)
from tiny_qwen_coder.evaluation.mbpp import (
    MBPPProblem,
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

    return load_dataset(
        repository,
        revision=revision,
        split=split,
        streaming=streaming,
    )


def _load_mbpp_rows(
    repository: str,
    dataset_config: str,
    *,
    revision: str,
    split: str,
    streaming: bool,
) -> Iterable[DatasetRow]:
    from datasets import load_dataset  # type: ignore[import-untyped]

    return load_dataset(
        repository,
        dataset_config,
        revision=revision,
        split=split,
        streaming=streaming,
    )


def _required_row_string(row: DatasetRow, key: str, *, context: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context}.{key} must be a non-empty string")
    return value


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
    for index, row in enumerate(rows):
        context = f"HumanEval protected row {index}"
        task_id = _required_row_string(row, "task_id", context=context)
        solution = _required_row_string(row, "canonical_solution", context=context)
        solutions[task_id] = solution

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
    for index, row in enumerate(rows):
        context = f"MBPP protected row {index}"
        task_id = row.get("task_id")
        if isinstance(task_id, bool) or not isinstance(task_id, int):
            raise ValueError(f"{context}.task_id must be an integer")
        solutions[f"MBPP/{task_id}"] = _required_row_string(row, "code", context=context)

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

    loaders = {
        "humaneval": lambda benchmark: _human_eval_examples(
            benchmark, dataset_loader=humaneval_dataset_loader
        ),
        "mbpp": lambda benchmark: _mbpp_examples(benchmark, dataset_loader=mbpp_dataset_loader),
        "repository-holdout": _repository_holdout_examples,
    }
    examples: list[ProtectedBenchmarkExample] = []
    for benchmark in registry.list_benchmarks(language="python"):
        loader = loaders.get(benchmark.id)
        if loader is None:
            raise ValueError(
                f"no protected-example loader exists for registered benchmark {benchmark.id!r}"
            )
        examples.extend(loader(benchmark))
    return tuple(examples)


__all__ = ["load_python_protected_examples"]
