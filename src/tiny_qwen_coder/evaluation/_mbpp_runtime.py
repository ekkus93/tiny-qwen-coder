"""MBPP execution-result interpretation and evaluator contract validation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from tiny_qwen_coder.config import EvaluationConfig
from tiny_qwen_coder.evaluation._mbpp_common import (
    _COMPILE_OK_MARKER,
    _ERROR_MARKER,
    _GIT_SHA_PATTERN,
    _MBPP_DATASET_ID,
    _MBPP_RUNNER_SOURCE,
    _PARSE_OK_MARKER,
    _RUNNER_COMPILE_EXIT,
    _RUNNER_PARSE_EXIT,
    _TEST_OK_MARKER,
    MBPPError,
    _expect_str,
    _strict_mapping,
)
from tiny_qwen_coder.evaluation._mbpp_types import MBPPRunnerConfig
from tiny_qwen_coder.evaluation.execution import ExecutionResult
from tiny_qwen_coder.evaluation.protected_benchmarks import ProtectedBenchmark
from tiny_qwen_coder.evaluation.results import EvaluationStageStatus
from tiny_qwen_coder.evaluation.settings import (
    FrozenEvaluationSettings,
    validate_evaluation_config_settings,
)
from tiny_qwen_coder.identities import AdapterIdentity, BaseModelIdentity


def _runner_source_sha256() -> str:
    return hashlib.sha256(_MBPP_RUNNER_SOURCE.encode("utf-8")).hexdigest()


def _execution_error_message(result: ExecutionResult) -> str:
    for line in reversed(result.stderr.splitlines()):
        if line.startswith(_ERROR_MARKER):
            return line.removeprefix(_ERROR_MARKER)
    if result.exit_code is None:
        return "MBPP candidate exceeded the execution timeout"
    return f"MBPP candidate process exited with code {result.exit_code}"


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
            test_index = int(suffix)
        except ValueError:
            continue
        if 1 <= test_index <= total:
            seen.add(test_index)
    return len(seen)


def _load_base_identity(path: Path) -> BaseModelIdentity:
    try:
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise MBPPError(f"could not load base model config {path}: {exc}") from exc
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
    runner: MBPPRunnerConfig,
    benchmark: ProtectedBenchmark,
    settings: FrozenEvaluationSettings,
    base_model: BaseModelIdentity,
    adapter: AdapterIdentity,
) -> str:
    if evaluation.language != "python":
        raise MBPPError("MBPP evaluator requires language='python'")
    if "mbpp" not in evaluation.suites:
        raise MBPPError("evaluation suites must include 'mbpp'")
    if evaluation.execution.network_enabled:
        raise MBPPError("MBPP constrained execution requires network_enabled=false")
    if evaluation.execution.timeout_seconds != runner.execution_timeout_seconds:
        raise MBPPError("evaluation timeout must match frozen MBPP runner timeout")
    if evaluation.adapter_id != adapter.adapter_id:
        raise MBPPError("evaluation adapter_id does not match adapter identity")
    if adapter.family is not None and adapter.family != "language":
        raise MBPPError("MBPP adapter family must be 'language'")
    if not _GIT_SHA_PATTERN.fullmatch(benchmark.dataset_revision):
        raise MBPPError("MBPP dataset revision must be an immutable Git SHA")
    if benchmark.dataset_id != _MBPP_DATASET_ID:
        raise MBPPError("MBPP dataset identity does not match the protected registration")
    if _load_base_identity(Path(evaluation.base_config)) != base_model:
        raise MBPPError("evaluation base model identity does not match base config")
    return validate_evaluation_config_settings(evaluation, settings)
