"""Protected MBPP loading, prompting, normalization, and artifact hashes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from typing import cast

from tiny_qwen_coder.evaluation._mbpp_common import (
    _PYTHON_FENCE_PATTERN,
    DatasetRow,
    MBPPDatasetRowsLoader,
    MBPPError,
    _require_text,
)
from tiny_qwen_coder.evaluation._mbpp_types import (
    MBPPAggregate,
    MBPPProblem,
    MBPPPrompt,
    MBPPRunnerConfig,
)
from tiny_qwen_coder.evaluation.protected_benchmarks import ProtectedBenchmark
from tiny_qwen_coder.evaluation.results import EvaluationResult


def _load_huggingface_rows(
    repository: str,
    dataset_config: str,
    *,
    revision: str,
    split: str,
    streaming: bool,
) -> Iterable[DatasetRow]:
    from datasets import load_dataset  # type: ignore[import-untyped]

    loaded = load_dataset(
        repository,
        dataset_config,
        revision=revision,
        split=split,
        streaming=streaming,
    )
    return cast(Iterable[DatasetRow], loaded)


def _row_str(row: Mapping[str, object], key: str, *, task_context: str) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise MBPPError(f"{task_context}.{key} must be a string")
    return value


def _row_int(row: Mapping[str, object], key: str, *, task_context: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise MBPPError(f"{task_context}.{key} must be an integer")
    return value


def _row_str_tuple(row: Mapping[str, object], key: str, *, task_context: str) -> tuple[str, ...]:
    value = row.get(key)
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise MBPPError(f"{task_context}.{key} must be a list of strings")
    items: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise MBPPError(f"{task_context}.{key}[{index}] must be a string")
        items.append(item)
    return tuple(items)


def _parse_problem(
    row: Mapping[str, object],
    *,
    row_index: int,
    tests_per_problem: int,
) -> MBPPProblem:
    expected_fields = {
        "task_id",
        "text",
        "code",
        "test_list",
        "test_setup_code",
        "challenge_test_list",
    }
    actual_fields = set(row)
    if actual_fields != expected_fields:
        missing = sorted(expected_fields - actual_fields)
        unknown = sorted(actual_fields - expected_fields)
        detail: list[str] = []
        if missing:
            detail.append(f"missing={missing!r}")
        if unknown:
            detail.append(f"unknown={unknown!r}")
        raise MBPPError(
            f"MBPP row {row_index} schema does not match pinned dataset: {', '.join(detail)}"
        )
    context = f"MBPP row {row_index}"
    canonical_solution = _row_str(row, "code", task_context=context)
    _require_text(canonical_solution, field_name=f"{context}.code")
    challenge_tests = _row_str_tuple(row, "challenge_test_list", task_context=context)
    for index, test in enumerate(challenge_tests):
        _require_text(test, field_name=f"{context}.challenge_test_list[{index}]")
    tests = _row_str_tuple(row, "test_list", task_context=context)
    if len(tests) != tests_per_problem:
        raise MBPPError(
            f"{context}.test_list must contain exactly {tests_per_problem} tests; got {len(tests)}"
        )
    return MBPPProblem(
        task_id=f"MBPP/{_row_int(row, 'task_id', task_context=context)}",
        description=_row_str(row, "text", task_context=context),
        tests=tests,
        test_setup_code=_row_str(row, "test_setup_code", task_context=context),
    )


def _validate_benchmark_contract(benchmark: ProtectedBenchmark, runner: MBPPRunnerConfig) -> None:
    if benchmark.language != "python":
        raise MBPPError("MBPP protected benchmark must belong to Python")
    if benchmark.id != runner.benchmark_id:
        raise MBPPError("MBPP protected benchmark ID does not match runner config")
    if benchmark.dataset_id != "google-research-datasets/mbpp":
        raise MBPPError("MBPP protected dataset must be 'google-research-datasets/mbpp'")
    if not benchmark.dataset_revision or len(benchmark.dataset_revision) != 40:
        raise MBPPError("MBPP protected dataset revision must be an immutable Git SHA")


def load_mbpp_problems(
    benchmark: ProtectedBenchmark,
    runner: MBPPRunnerConfig,
    *,
    dataset_loader: MBPPDatasetRowsLoader = _load_huggingface_rows,
) -> tuple[MBPPProblem, ...]:
    """Load and validate the exact protected MBPP full/test split."""

    _validate_benchmark_contract(benchmark, runner)
    rows = dataset_loader(
        benchmark.dataset_id,
        runner.dataset_config,
        revision=benchmark.dataset_revision,
        split=runner.dataset_split,
        streaming=False,
    )
    problems = tuple(
        sorted(
            (
                _parse_problem(
                    row,
                    row_index=index,
                    tests_per_problem=runner.tests_per_problem,
                )
                for index, row in enumerate(rows)
            ),
            key=lambda problem: problem.task_number,
        )
    )
    if len(problems) != runner.expected_problem_count:
        raise MBPPError(
            f"MBPP expected {runner.expected_problem_count} problems; got {len(problems)}"
        )
    task_numbers = tuple(problem.task_number for problem in problems)
    expected_numbers = tuple(range(runner.task_id_start, runner.task_id_end + 1))
    if task_numbers != expected_numbers:
        raise MBPPError("MBPP task IDs do not match the canonical contiguous test task set")
    return problems


def create_mbpp_prompt(problem: MBPPProblem, runner: MBPPRunnerConfig) -> MBPPPrompt:
    """Create the frozen BigCode/InCoder-style prompt for one MBPP task."""

    user_content = f'"""\n{problem.description}\n{problem.tests[0]}\n"""\n'
    return MBPPPrompt(
        task_id=problem.task_id,
        prompt_version=runner.prompt_version,
        user_content=user_content,
    )


def _stop_at_reference_token(text: str, stop_words: tuple[str, ...]) -> str:
    positions = [position for token in stop_words if (position := text.find(token)) >= 0]
    return text[: min(positions)] if positions else text


def normalize_mbpp_completion(
    problem: MBPPProblem,
    generated_text: str,
    runner: MBPPRunnerConfig,
) -> str:
    """Normalize common model response shapes into executable MBPP source."""

    if not isinstance(generated_text, str):
        raise MBPPError("generated_text must be a string")
    prompt = create_mbpp_prompt(problem, runner).user_content
    text = generated_text
    fence = _PYTHON_FENCE_PATTERN.search(text)
    if fence is not None:
        text = fence.group(1)
    if text.startswith(prompt):
        text = text[len(prompt) :]
    text = _stop_at_reference_token(text, runner.stop_words)
    return text.rstrip() + "\n" if text.strip() else ""


def _problem_payload(problem: MBPPProblem) -> dict[str, object]:
    return {
        "task_id": problem.task_id,
        "description": problem.description,
        "test_setup_code": problem.test_setup_code,
        "tests": list(problem.tests),
    }


def mbpp_problem_set_sha256(problems: tuple[MBPPProblem, ...]) -> str:
    payload = json.dumps(
        [_problem_payload(problem) for problem in problems],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _result_json_line(result: EvaluationResult) -> str:
    return json.dumps(asdict(result), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def mbpp_results_sha256(results: tuple[EvaluationResult, ...]) -> str:
    payload = "\n".join(_result_json_line(result) for result in results) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def mbpp_aggregate_json(aggregate: MBPPAggregate) -> str:
    return json.dumps(asdict(aggregate), indent=2, sort_keys=True) + "\n"
