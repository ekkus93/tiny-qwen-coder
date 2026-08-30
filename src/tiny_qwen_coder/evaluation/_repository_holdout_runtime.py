"""Prompt normalization, result hashing, and runtime interpretation for the holdout."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import yaml

from tiny_qwen_coder.config import EvaluationConfig
from tiny_qwen_coder.evaluation._repository_holdout_common import (
    _COMPILE_OK_MARKER,
    _ERROR_MARKER,
    _PARSE_OK_MARKER,
    _PYTHON_FENCE_PATTERN,
    _REPOSITORY_HOLDOUT_DATASET_ID,
    _REPOSITORY_HOLDOUT_DATASET_REVISION,
    _REPOSITORY_HOLDOUT_RUNNER_SOURCE,
    _REPOSITORY_HOLDOUT_SUITE_CONFIG_PATH,
    _REPOSITORY_HOLDOUT_SUITE_ID,
    _RUNNER_COMPILE_EXIT,
    _RUNNER_PARSE_EXIT,
    _TEST_OK_MARKER,
    RepositoryHoldoutError,
    _expect_str,
    _require_non_empty,
    _strict_mapping,
)
from tiny_qwen_coder.evaluation._repository_holdout_types import (
    RepositoryHoldoutAggregate,
    RepositoryHoldoutPrompt,
    RepositoryHoldoutSuiteConfig,
    RepositoryHoldoutTask,
)
from tiny_qwen_coder.evaluation.execution import ExecutionResult
from tiny_qwen_coder.evaluation.protected_benchmarks import ProtectedBenchmark
from tiny_qwen_coder.evaluation.results import EvaluationResult, EvaluationStageStatus
from tiny_qwen_coder.evaluation.settings import (
    FrozenEvaluationSettings,
    validate_evaluation_config_settings,
)
from tiny_qwen_coder.identities import AdapterIdentity, BaseModelIdentity


def create_repository_holdout_prompt(
    task: RepositoryHoldoutTask,
    suite: RepositoryHoldoutSuiteConfig,
) -> RepositoryHoldoutPrompt:
    """Build the frozen single-user prompt for one repository-owned task."""

    return RepositoryHoldoutPrompt(
        problem_id=task.problem_id,
        prompt_version=suite.prompt_version,
        user_content=f"{suite.instruction}\n\n{task.prompt}",
    )


def normalize_repository_holdout_completion(
    task: RepositoryHoldoutTask,
    generated_text: str,
    suite: RepositoryHoldoutSuiteConfig,
) -> str:
    """Normalize a model reply into one Python module deterministically."""

    del task
    _require_non_empty(suite.completion_normalizer_version, field_name="normalizer version")
    text = generated_text.strip()
    fence = _PYTHON_FENCE_PATTERN.search(text)
    if fence is not None:
        text = fence.group(1).strip()
    return text + ("\n" if text else "")


def _result_json_line(result: EvaluationResult) -> str:
    return json.dumps(asdict(result), sort_keys=True, ensure_ascii=False)


def repository_holdout_results_sha256(results: tuple[EvaluationResult, ...]) -> str:
    payload = "".join(_result_json_line(result) + "\n" for result in results)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def repository_holdout_aggregate_json(aggregate: RepositoryHoldoutAggregate) -> str:
    return json.dumps(asdict(aggregate), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _runner_source_sha256() -> str:
    return hashlib.sha256(_REPOSITORY_HOLDOUT_RUNNER_SOURCE.encode("utf-8")).hexdigest()


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


def _passed_test_count(result: ExecutionResult, *, total: int) -> int:
    seen: set[int] = set()
    for line in result.stderr.splitlines():
        if not line.startswith(_TEST_OK_MARKER):
            continue
        suffix = line.removeprefix(_TEST_OK_MARKER)
        try:
            index = int(suffix)
        except ValueError:
            continue
        if 1 <= index <= total:
            seen.add(index)
    return len(seen)


def _execution_error_message(result: ExecutionResult) -> str:
    for line in reversed(result.stderr.splitlines()):
        if line.startswith(_ERROR_MARKER):
            return line.removeprefix(_ERROR_MARKER)
    if result.exit_code is None:
        return "repository holdout candidate exceeded the execution timeout"
    return f"repository holdout candidate process exited with code {result.exit_code}"


def _load_base_identity(path: Path) -> BaseModelIdentity:
    try:
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RepositoryHoldoutError(f"could not load base model config {path}: {exc}") from exc
    mapping = _strict_mapping(raw, context="base config")
    model = _strict_mapping(mapping.get("model"), context="base config.model")
    tokenizer = _strict_mapping(mapping.get("tokenizer"), context="base config.tokenizer")
    return BaseModelIdentity(
        repository=_expect_str(model, "repository", context="base config.model"),
        revision=_expect_str(model, "revision", context="base config.model"),
        tokenizer_repository=_expect_str(
            tokenizer,
            "repository",
            context="base config.tokenizer",
        ),
        tokenizer_revision=_expect_str(
            tokenizer,
            "revision",
            context="base config.tokenizer",
        ),
    )


def _validate_evaluator_contract(
    evaluation: EvaluationConfig,
    suite: RepositoryHoldoutSuiteConfig,
    benchmark: ProtectedBenchmark,
    settings: FrozenEvaluationSettings,
    base_model: BaseModelIdentity,
    adapter: AdapterIdentity,
) -> str:
    if evaluation.language != "python":
        raise RepositoryHoldoutError("repository holdout evaluator requires language='python'")
    if _REPOSITORY_HOLDOUT_SUITE_ID not in evaluation.suites:
        raise RepositoryHoldoutError("evaluation suites must include 'repository-holdout'")
    if evaluation.execution.network_enabled:
        raise RepositoryHoldoutError("repository holdout requires network_enabled=false")
    if evaluation.execution.timeout_seconds != suite.execution_timeout_seconds:
        raise RepositoryHoldoutError("evaluation timeout must match frozen holdout timeout")
    if evaluation.adapter_id != adapter.adapter_id:
        raise RepositoryHoldoutError("evaluation adapter_id does not match adapter identity")
    if adapter.family is not None and adapter.family != "language":
        raise RepositoryHoldoutError("repository holdout adapter family must be 'language'")
    if benchmark.id != _REPOSITORY_HOLDOUT_SUITE_ID:
        raise RepositoryHoldoutError("protected benchmark ID does not match repository holdout")
    if benchmark.dataset_id != _REPOSITORY_HOLDOUT_DATASET_ID:
        raise RepositoryHoldoutError("protected dataset identity does not match repository holdout")
    if benchmark.dataset_revision != _REPOSITORY_HOLDOUT_DATASET_REVISION:
        raise RepositoryHoldoutError("protected dataset revision does not match repository holdout")
    suite_selector = _REPOSITORY_HOLDOUT_SUITE_CONFIG_PATH.as_posix()
    if suite_selector not in benchmark.source_configs:
        raise RepositoryHoldoutError("repository holdout suite config is not protected from SFT")
    if suite.benchmark_id != benchmark.id or suite.dataset_revision != benchmark.dataset_revision:
        raise RepositoryHoldoutError("suite identity does not match protected benchmark")
    if _load_base_identity(Path(evaluation.base_config)) != base_model:
        raise RepositoryHoldoutError("evaluation base model identity does not match base config")
    return validate_evaluation_config_settings(evaluation, settings)
