"""Two-stage P6-005 baseline execution for GPU-only and OCI-capable runners."""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import cast

from tiny_qwen_coder.config import EvaluationConfig, load_evaluation_config
from tiny_qwen_coder.evaluation._baseline_artifacts import (
    evaluation_config_sha256,
    file_sha256,
    freeze_python_baseline,
    python_baseline_manifest_json,
    system_prompt_sha256,
    write_regression_baseline_artifacts,
    write_runtime_metadata,
)
from tiny_qwen_coder.evaluation._baseline_generation import (
    BaselineGenerator,
    HuggingFaceBaselineGenerator,
    generation_contract_sha256,
)
from tiny_qwen_coder.evaluation._baseline_provenance import (
    BaselineGpuProvenance,
    BaselineProvenance,
    collect_baseline_provenance,
    load_baseline_base_model_identity,
    write_baseline_provenance,
)
from tiny_qwen_coder.evaluation._baseline_runner import (
    _CANONICAL_BASELINE_CONFIG_PATH,
    _WORK_DIR_NAME,
    _generate_items,
    _preflight_execution_images,
    _preflight_source_tree,
    _regression_aggregate,
    _regression_results,
    _suite_performance,
    _validate_baseline_contract,
    _validate_runtime_metadata,
)
from tiny_qwen_coder.evaluation._baseline_types import (
    BaselineGeneratedResponse,
    BaselineRuntimeMetadata,
    BaselineSuitePerformance,
    PythonBaselineError,
    PythonBaselineManifest,
)
from tiny_qwen_coder.evaluation.execution import ConstrainedExecutionHarness, discover_oci_runtime
from tiny_qwen_coder.evaluation.humaneval import (
    HumanEvalCompletion,
    HumanEvalEvaluator,
    HumanEvalProblem,
)
from tiny_qwen_coder.evaluation.mbpp import MBPPCompletion, MBPPEvaluator, MBPPProblem
from tiny_qwen_coder.evaluation.regression import load_frozen_general_tool_regression_suite
from tiny_qwen_coder.evaluation.repository_holdout import (
    RepositoryHoldoutCompletion,
    RepositoryHoldoutEvaluator,
)
from tiny_qwen_coder.evaluation.settings import (
    FrozenEvaluationSettings,
    evaluation_settings_sha256,
    load_frozen_evaluation_settings,
)
from tiny_qwen_coder.identities import AdapterIdentity, BaseModelIdentity

_GENERATION_STAGE_NAME = "generation-stage.json"
_GENERATION_STAGE_SCHEMA_VERSION = 1
_STAGE_ARTIFACTS = (
    ".baseline-work/humaneval-generation.jsonl",
    ".baseline-work/mbpp-generation.jsonl",
    ".baseline-work/repository-holdout-generation.jsonl",
    ".baseline-work/general-tool-regression-generation.jsonl",
    "runtime-metadata.json",
    "provenance.json",
)


class _CheckpointOnlyGenerator(BaselineGenerator):
    """Fail closed if scoring attempts generation instead of consuming checkpoints."""

    def generate(self, *, system_prompt: str, user_prompt: str) -> BaselineGeneratedResponse:
        del system_prompt, user_prompt
        raise PythonBaselineError(
            "scoring stage is missing a required GPU generation checkpoint; refusing regeneration"
        )


def _strict_mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise PythonBaselineError(f"{context} must be a JSON object")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise PythonBaselineError(f"{context} keys must be strings")
        result[key] = item
    return result


def _expect_str(mapping: Mapping[str, object], key: str, *, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PythonBaselineError(f"{context}.{key} must be a non-empty string")
    return value


def _expect_int(mapping: Mapping[str, object], key: str, *, context: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise PythonBaselineError(f"{context}.{key} must be an integer")
    return value


def _expect_float(mapping: Mapping[str, object], key: str, *, context: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PythonBaselineError(f"{context}.{key} must be numeric")
    return float(value)


def _read_json_mapping(path: Path, *, context: str) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PythonBaselineError(f"could not read {context} {path}") from exc
    return _strict_mapping(value, context=context)


def _parse_base_model(value: object, *, context: str) -> BaseModelIdentity:
    mapping = _strict_mapping(value, context=context)
    return BaseModelIdentity(
        repository=_expect_str(mapping, "repository", context=context),
        revision=_expect_str(mapping, "revision", context=context),
        tokenizer_repository=_expect_str(mapping, "tokenizer_repository", context=context),
        tokenizer_revision=_expect_str(mapping, "tokenizer_revision", context=context),
    )


def _parse_adapter(value: object, *, context: str) -> AdapterIdentity:
    mapping = _strict_mapping(value, context=context)
    family = mapping.get("family")
    adapter_id = mapping.get("adapter_id")
    if family is not None and not isinstance(family, str):
        raise PythonBaselineError(f"{context}.family must be string or null")
    if adapter_id is not None and not isinstance(adapter_id, str):
        raise PythonBaselineError(f"{context}.adapter_id must be string or null")
    return AdapterIdentity(family=family, adapter_id=adapter_id)


def _parse_suite_performance(value: object, *, index: int) -> BaselineSuitePerformance:
    context = f"runtime.suites[{index}]"
    mapping = _strict_mapping(value, context=context)
    return BaselineSuitePerformance(
        suite_id=_expect_str(mapping, "suite_id", context=context),
        requests=_expect_int(mapping, "requests", context=context),
        prompt_tokens=_expect_int(mapping, "prompt_tokens", context=context),
        generated_tokens=_expect_int(mapping, "generated_tokens", context=context),
        generation_latency_seconds=_expect_float(
            mapping, "generation_latency_seconds", context=context
        ),
        tokens_per_second=_expect_float(mapping, "tokens_per_second", context=context),
    )


def _load_runtime_metadata(path: Path) -> BaselineRuntimeMetadata:
    mapping = _read_json_mapping(path, context="baseline runtime metadata")
    raw_dtypes = mapping.get("parameter_dtypes")
    raw_suites = mapping.get("suites")
    if not isinstance(raw_dtypes, list) or any(not isinstance(item, str) for item in raw_dtypes):
        raise PythonBaselineError("runtime.parameter_dtypes must be a list of strings")
    if not isinstance(raw_suites, list):
        raise PythonBaselineError("runtime.suites must be a list")
    return BaselineRuntimeMetadata(
        schema_version=_expect_int(mapping, "schema_version", context="runtime"),
        device=_expect_str(mapping, "device", context="runtime"),
        gpu_name=_expect_str(mapping, "gpu_name", context="runtime"),
        gpu_compute_capability=_expect_str(
            mapping, "gpu_compute_capability", context="runtime"
        ),
        torch_version=_expect_str(mapping, "torch_version", context="runtime"),
        transformers_version=_expect_str(
            mapping, "transformers_version", context="runtime"
        ),
        model_class=_expect_str(mapping, "model_class", context="runtime"),
        parameter_dtypes=tuple(cast(list[str], raw_dtypes)),
        resolved_model_revision=_expect_str(
            mapping, "resolved_model_revision", context="runtime"
        ),
        cuda_total_bytes=_expect_int(mapping, "cuda_total_bytes", context="runtime"),
        cuda_free_before_load_bytes=_expect_int(
            mapping, "cuda_free_before_load_bytes", context="runtime"
        ),
        cuda_free_after_load_bytes=_expect_int(
            mapping, "cuda_free_after_load_bytes", context="runtime"
        ),
        torch_allocated_after_load_bytes=_expect_int(
            mapping, "torch_allocated_after_load_bytes", context="runtime"
        ),
        torch_reserved_after_load_bytes=_expect_int(
            mapping, "torch_reserved_after_load_bytes", context="runtime"
        ),
        load_peak_allocated_bytes=_expect_int(
            mapping, "load_peak_allocated_bytes", context="runtime"
        ),
        load_peak_reserved_bytes=_expect_int(
            mapping, "load_peak_reserved_bytes", context="runtime"
        ),
        generation_peak_allocated_bytes=_expect_int(
            mapping, "generation_peak_allocated_bytes", context="runtime"
        ),
        generation_peak_reserved_bytes=_expect_int(
            mapping, "generation_peak_reserved_bytes", context="runtime"
        ),
        model_load_seconds=_expect_float(mapping, "model_load_seconds", context="runtime"),
        total_wall_seconds=_expect_float(mapping, "total_wall_seconds", context="runtime"),
        total_requests=_expect_int(mapping, "total_requests", context="runtime"),
        total_prompt_tokens=_expect_int(mapping, "total_prompt_tokens", context="runtime"),
        total_generated_tokens=_expect_int(
            mapping, "total_generated_tokens", context="runtime"
        ),
        total_generation_latency_seconds=_expect_float(
            mapping, "total_generation_latency_seconds", context="runtime"
        ),
        overall_tokens_per_second=_expect_float(
            mapping, "overall_tokens_per_second", context="runtime"
        ),
        suites=tuple(
            _parse_suite_performance(item, index=index) for index, item in enumerate(raw_suites)
        ),
    )


def _parse_gpu(value: object, *, index: int) -> BaselineGpuProvenance:
    context = f"provenance.gpus[{index}]"
    mapping = _strict_mapping(value, context=context)
    return BaselineGpuProvenance(
        index=_expect_int(mapping, "index", context=context),
        name=_expect_str(mapping, "name", context=context),
        total_memory_bytes=_expect_int(mapping, "total_memory_bytes", context=context),
        compute_capability=_expect_str(mapping, "compute_capability", context=context),
    )


def _load_provenance(path: Path) -> BaselineProvenance:
    mapping = _read_json_mapping(path, context="baseline provenance")
    raw_gpus = mapping.get("gpus")
    raw_dependencies = mapping.get("dependencies")
    if not isinstance(raw_gpus, list):
        raise PythonBaselineError("provenance.gpus must be a list")
    if not isinstance(raw_dependencies, list):
        raise PythonBaselineError("provenance.dependencies must be a list")
    dependencies: list[tuple[str, str]] = []
    for index, value in enumerate(raw_dependencies):
        if (
            not isinstance(value, list)
            or len(value) != 2
            or not isinstance(value[0], str)
            or not isinstance(value[1], str)
        ):
            raise PythonBaselineError(f"provenance.dependencies[{index}] must be [name, version]")
        dependencies.append((value[0], value[1]))
    source_git_dirty = mapping.get("source_git_dirty")
    cuda_available = mapping.get("cuda_available")
    cuda_runtime = mapping.get("cuda_runtime")
    if not isinstance(source_git_dirty, bool):
        raise PythonBaselineError("provenance.source_git_dirty must be a boolean")
    if not isinstance(cuda_available, bool):
        raise PythonBaselineError("provenance.cuda_available must be a boolean")
    if cuda_runtime is not None and not isinstance(cuda_runtime, str):
        raise PythonBaselineError("provenance.cuda_runtime must be string or null")
    return BaselineProvenance(
        schema_version=_expect_int(mapping, "schema_version", context="provenance"),
        created_at_utc=_expect_str(mapping, "created_at_utc", context="provenance"),
        source_git_sha=_expect_str(mapping, "source_git_sha", context="provenance"),
        source_git_dirty=source_git_dirty,
        base_model=_parse_base_model(mapping.get("base_model"), context="provenance.base_model"),
        adapter=_parse_adapter(mapping.get("adapter"), context="provenance.adapter"),
        language=_expect_str(mapping, "language", context="provenance"),
        seed=_expect_int(mapping, "seed", context="provenance"),
        hostname=_expect_str(mapping, "hostname", context="provenance"),
        system=_expect_str(mapping, "system", context="provenance"),
        release=_expect_str(mapping, "release", context="provenance"),
        machine=_expect_str(mapping, "machine", context="provenance"),
        python_version=_expect_str(mapping, "python_version", context="provenance"),
        cuda_available=cuda_available,
        cuda_runtime=cast(str | None, cuda_runtime),
        gpus=tuple(_parse_gpu(item, index=index) for index, item in enumerate(raw_gpus)),
        dependencies=tuple(dependencies),
    )


def _generation_contract(
    *,
    evaluation: EvaluationConfig,
    settings: FrozenEvaluationSettings,
    base_model: BaseModelIdentity,
) -> tuple[str, str, str]:
    system_prompt_version, system_prompt = _validate_baseline_contract(
        evaluation, settings, base_model
    )
    contract = generation_contract_sha256(
        base_model=base_model,
        settings=settings,
        system_prompt_version=system_prompt_version,
        system_prompt=system_prompt,
    )
    return system_prompt_version, system_prompt, contract


def _stage_manifest_payload(
    *,
    output_dir: Path,
    source_git_sha: str,
    evaluation: EvaluationConfig,
    settings: FrozenEvaluationSettings,
    system_prompt: str,
    generation_contract: str,
) -> dict[str, object]:
    artifacts = []
    for relative_path in _STAGE_ARTIFACTS:
        path = output_dir / relative_path
        if not path.is_file():
            raise PythonBaselineError(
                f"GPU generation stage is incomplete; missing {relative_path!r}"
            )
        artifacts.append({"path": relative_path, "sha256": file_sha256(path)})
    return {
        "schema_version": _GENERATION_STAGE_SCHEMA_VERSION,
        "source_git_sha": source_git_sha,
        "evaluation_config_sha256": evaluation_config_sha256(evaluation),
        "evaluation_settings_sha256": evaluation_settings_sha256(settings),
        "system_prompt_sha256": system_prompt_sha256(system_prompt),
        "generation_contract_sha256": generation_contract,
        "artifacts": artifacts,
    }


def _write_generation_stage_manifest(
    *,
    output_dir: Path,
    source_git_sha: str,
    evaluation: EvaluationConfig,
    settings: FrozenEvaluationSettings,
    system_prompt: str,
    generation_contract: str,
) -> Path:
    payload = _stage_manifest_payload(
        output_dir=output_dir,
        source_git_sha=source_git_sha,
        evaluation=evaluation,
        settings=settings,
        system_prompt=system_prompt,
        generation_contract=generation_contract,
    )
    path = output_dir / _GENERATION_STAGE_NAME
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _validate_generation_stage_manifest(
    *,
    output_dir: Path,
    source_git_sha: str,
    evaluation: EvaluationConfig,
    settings: FrozenEvaluationSettings,
    system_prompt: str,
    generation_contract: str,
) -> None:
    path = output_dir / _GENERATION_STAGE_NAME
    mapping = _read_json_mapping(path, context="GPU generation-stage manifest")
    expected_keys = {
        "schema_version",
        "source_git_sha",
        "evaluation_config_sha256",
        "evaluation_settings_sha256",
        "system_prompt_sha256",
        "generation_contract_sha256",
        "artifacts",
    }
    if set(mapping) != expected_keys:
        raise PythonBaselineError("GPU generation-stage manifest schema drift detected")
    if _expect_int(mapping, "schema_version", context="generation_stage") != 1:
        raise PythonBaselineError("unsupported GPU generation-stage manifest version")
    expected_values = {
        "source_git_sha": source_git_sha,
        "evaluation_config_sha256": evaluation_config_sha256(evaluation),
        "evaluation_settings_sha256": evaluation_settings_sha256(settings),
        "system_prompt_sha256": system_prompt_sha256(system_prompt),
        "generation_contract_sha256": generation_contract,
    }
    for key, expected in expected_values.items():
        if _expect_str(mapping, key, context="generation_stage") != expected:
            raise PythonBaselineError(f"GPU generation-stage {key} drift detected")
    raw_artifacts = mapping.get("artifacts")
    if not isinstance(raw_artifacts, list) or len(raw_artifacts) != len(_STAGE_ARTIFACTS):
        raise PythonBaselineError("GPU generation-stage artifact inventory is invalid")
    observed_paths: list[str] = []
    for index, value in enumerate(raw_artifacts):
        context = f"generation_stage.artifacts[{index}]"
        item = _strict_mapping(value, context=context)
        if set(item) != {"path", "sha256"}:
            raise PythonBaselineError(f"{context} schema is invalid")
        relative_path = _expect_str(item, "path", context=context)
        expected_digest = _expect_str(item, "sha256", context=context)
        observed_paths.append(relative_path)
        if file_sha256(output_dir / relative_path) != expected_digest:
            raise PythonBaselineError(
                f"GPU generation-stage artifact digest mismatch: {relative_path!r}"
            )
    if tuple(observed_paths) != _STAGE_ARTIFACTS:
        raise PythonBaselineError("GPU generation-stage artifact paths are invalid or reordered")


def _load_generation_inputs(
    *,
    evaluation: EvaluationConfig,
    settings: FrozenEvaluationSettings,
    base_model: BaseModelIdentity,
    harness: ConstrainedExecutionHarness | None = None,
) -> tuple[
    HumanEvalEvaluator,
    tuple[HumanEvalProblem, ...],
    MBPPEvaluator,
    tuple[MBPPProblem, ...],
    RepositoryHoldoutEvaluator,
]:
    adapter = AdapterIdentity(family=None, adapter_id=None)
    humaneval = HumanEvalEvaluator(
        evaluation,
        base_model=base_model,
        adapter=adapter,
        settings=settings,
        harness=harness,
    )
    mbpp = MBPPEvaluator(
        evaluation,
        base_model=base_model,
        adapter=adapter,
        settings=settings,
        harness=harness,
    )
    holdout = RepositoryHoldoutEvaluator(
        evaluation,
        base_model=base_model,
        adapter=adapter,
        settings=settings,
        harness=harness,
    )
    return humaneval, humaneval.load_problems(), mbpp, mbpp.load_problems(), holdout


def _generate_all_suites(
    *,
    evaluation: EvaluationConfig,
    generator: BaselineGenerator,
    base_model: BaseModelIdentity,
    settings: FrozenEvaluationSettings,
    system_prompt: str,
    generation_contract: str,
    output_dir: Path,
    harness: ConstrainedExecutionHarness | None = None,
) -> tuple[
    tuple[HumanEvalProblem, ...],
    tuple[BaselineGeneratedResponse, ...],
    tuple[MBPPProblem, ...],
    tuple[BaselineGeneratedResponse, ...],
    RepositoryHoldoutEvaluator,
    tuple[BaselineGeneratedResponse, ...],
    tuple[BaselineGeneratedResponse, ...],
]:
    humaneval, humaneval_problems, mbpp, mbpp_problems, holdout = _load_generation_inputs(
        evaluation=evaluation,
        settings=settings,
        base_model=base_model,
        harness=harness,
    )
    humaneval_responses = _generate_items(
        suite_id="humaneval",
        prompts=tuple(
            (problem.task_id, humaneval.prompt_for(problem).user_content)
            for problem in humaneval_problems
        ),
        generator=generator,
        system_prompt=system_prompt,
        generation_contract=generation_contract,
        output_dir=output_dir,
    )
    mbpp_responses = _generate_items(
        suite_id="mbpp",
        prompts=tuple(
            (problem.task_id, mbpp.prompt_for(problem).user_content) for problem in mbpp_problems
        ),
        generator=generator,
        system_prompt=system_prompt,
        generation_contract=generation_contract,
        output_dir=output_dir,
    )
    holdout_responses = _generate_items(
        suite_id="repository-holdout",
        prompts=tuple(
            (task.problem_id, holdout.prompt_for(task).user_content) for task in holdout.suite.tasks
        ),
        generator=generator,
        system_prompt=system_prompt,
        generation_contract=generation_contract,
        output_dir=output_dir,
    )
    regression_suite = load_frozen_general_tool_regression_suite()
    regression_responses = _generate_items(
        suite_id="general-tool-regression",
        prompts=tuple((case.id, case.prompt) for case in regression_suite.cases),
        generator=generator,
        system_prompt=system_prompt,
        generation_contract=generation_contract,
        output_dir=output_dir,
    )
    return (
        humaneval_problems,
        humaneval_responses,
        mbpp_problems,
        mbpp_responses,
        holdout,
        holdout_responses,
        regression_responses,
    )


def _suite_performance_from_responses(
    *,
    humaneval: tuple[BaselineGeneratedResponse, ...],
    mbpp: tuple[BaselineGeneratedResponse, ...],
    holdout: tuple[BaselineGeneratedResponse, ...],
    regression: tuple[BaselineGeneratedResponse, ...],
) -> tuple[BaselineSuitePerformance, ...]:
    return (
        _suite_performance("humaneval", humaneval),
        _suite_performance("mbpp", mbpp),
        _suite_performance("repository-holdout", holdout),
        _suite_performance("general-tool-regression", regression),
    )


def generate_canonical_python_base_baseline_stage(
    *,
    config_path: Path = _CANONICAL_BASELINE_CONFIG_PATH,
    device_index: int = 0,
    repo_root: Path = Path("."),
) -> Path:
    """Generate all frozen responses on CUDA without requiring an OCI runtime."""

    evaluation = load_evaluation_config(config_path)
    settings = load_frozen_evaluation_settings()
    base_model = load_baseline_base_model_identity(Path(evaluation.base_config))
    _, system_prompt, generation_contract = _generation_contract(
        evaluation=evaluation, settings=settings, base_model=base_model
    )
    source_git_sha, _ = _preflight_source_tree(repo_root)
    output_dir = Path(evaluation.output_dir)
    if (output_dir / "baseline-manifest.json").exists():
        raise PythonBaselineError("GPU generation stage refuses an already-frozen baseline directory")
    generator = HuggingFaceBaselineGenerator(
        base_model=base_model,
        settings=settings,
        device_index=device_index,
    )
    (
        _humaneval_problems,
        humaneval_responses,
        _mbpp_problems,
        mbpp_responses,
        _holdout,
        holdout_responses,
        regression_responses,
    ) = _generate_all_suites(
        evaluation=evaluation,
        generator=generator,
        base_model=base_model,
        settings=settings,
        system_prompt=system_prompt,
        generation_contract=generation_contract,
        output_dir=output_dir,
    )
    suites = _suite_performance_from_responses(
        humaneval=humaneval_responses,
        mbpp=mbpp_responses,
        holdout=holdout_responses,
        regression=regression_responses,
    )
    runtime_metadata = generator.runtime_metadata(suites)
    _validate_runtime_metadata(runtime_metadata, suites)
    write_runtime_metadata(runtime_metadata, output_dir)
    provenance = collect_baseline_provenance(
        base_model=base_model,
        seed=evaluation.seed,
        repo_root=repo_root,
    )
    if provenance.source_git_sha != source_git_sha:
        raise PythonBaselineError("GPU provenance source SHA changed during generation")
    if not provenance.cuda_available or not provenance.gpus:
        raise PythonBaselineError("GPU generation stage did not record CUDA provenance")
    write_baseline_provenance(provenance, output_dir)
    return _write_generation_stage_manifest(
        output_dir=output_dir,
        source_git_sha=source_git_sha,
        evaluation=evaluation,
        settings=settings,
        system_prompt=system_prompt,
        generation_contract=generation_contract,
    )


def score_canonical_python_base_baseline_stage(
    *,
    config_path: Path = _CANONICAL_BASELINE_CONFIG_PATH,
    repo_root: Path = Path("."),
) -> PythonBaselineManifest:
    """Score transported GPU checkpoints under OCI isolation and freeze the baseline."""

    evaluation = load_evaluation_config(config_path)
    settings = load_frozen_evaluation_settings()
    base_model = load_baseline_base_model_identity(Path(evaluation.base_config))
    system_prompt_version, system_prompt, generation_contract = _generation_contract(
        evaluation=evaluation, settings=settings, base_model=base_model
    )
    source_git_sha, _ = _preflight_source_tree(repo_root)
    output_dir = Path(evaluation.output_dir)
    _validate_generation_stage_manifest(
        output_dir=output_dir,
        source_git_sha=source_git_sha,
        evaluation=evaluation,
        settings=settings,
        system_prompt=system_prompt,
        generation_contract=generation_contract,
    )
    runtime_metadata = _load_runtime_metadata(output_dir / "runtime-metadata.json")
    provenance = _load_provenance(output_dir / "provenance.json")
    if provenance.source_git_sha != source_git_sha:
        raise PythonBaselineError("GPU provenance source SHA does not match scoring source SHA")
    if provenance.base_model != base_model:
        raise PythonBaselineError("GPU provenance base-model identity drift detected")
    if provenance.source_git_dirty or not provenance.cuda_available or not provenance.gpus:
        raise PythonBaselineError("GPU provenance is not eligible for baseline freezing")
    if runtime_metadata.resolved_model_revision != base_model.revision:
        raise PythonBaselineError("GPU runtime metadata model revision drift detected")
    if runtime_metadata.gpu_name not in {gpu.name for gpu in provenance.gpus}:
        raise PythonBaselineError("GPU runtime metadata does not match GPU provenance")

    runtime = discover_oci_runtime()
    harness = ConstrainedExecutionHarness(runtime=runtime)
    humaneval, humaneval_problems, mbpp, mbpp_problems, holdout = _load_generation_inputs(
        evaluation=evaluation,
        settings=settings,
        base_model=base_model,
        harness=harness,
    )
    _preflight_execution_images(
        runtime,
        (
            humaneval.runner.execution_image,
            mbpp.runner.execution_image,
            holdout.suite.execution_image,
        ),
    )
    generator = _CheckpointOnlyGenerator()
    (
        resolved_humaneval,
        humaneval_responses,
        resolved_mbpp,
        mbpp_responses,
        resolved_holdout,
        holdout_responses,
        regression_responses,
    ) = _generate_all_suites(
        evaluation=evaluation,
        generator=generator,
        base_model=base_model,
        settings=settings,
        system_prompt=system_prompt,
        generation_contract=generation_contract,
        output_dir=output_dir,
        harness=harness,
    )
    if resolved_humaneval != humaneval_problems or resolved_mbpp != mbpp_problems:
        raise PythonBaselineError("protected benchmark inputs changed during scoring setup")
    if resolved_holdout.suite != holdout.suite:
        raise PythonBaselineError("repository holdout suite changed during scoring setup")

    humaneval_completions = tuple(
        HumanEvalCompletion(
            task_id=problem.task_id,
            generated_text=response.generated_text,
            generation=response.generation,
        )
        for problem, response in zip(resolved_humaneval, humaneval_responses, strict=True)
    )
    humaneval_result = humaneval.evaluate_suite(resolved_humaneval, humaneval_completions)
    humaneval.write_artifacts(humaneval_result, output_dir / "humaneval")

    mbpp_completions = tuple(
        MBPPCompletion(
            task_id=problem.task_id,
            generated_text=response.generated_text,
            generation=response.generation,
        )
        for problem, response in zip(resolved_mbpp, mbpp_responses, strict=True)
    )
    mbpp_result = mbpp.evaluate_suite(resolved_mbpp, mbpp_completions)
    mbpp.write_artifacts(mbpp_result, output_dir / "mbpp")

    holdout_completions = tuple(
        RepositoryHoldoutCompletion(
            problem_id=task.problem_id,
            generated_text=response.generated_text,
            generation=response.generation,
        )
        for task, response in zip(resolved_holdout.suite.tasks, holdout_responses, strict=True)
    )
    holdout_result = holdout.evaluate_suite(holdout_completions)
    holdout.write_artifacts(holdout_result, output_dir / "repository-holdout")

    regression_suite = load_frozen_general_tool_regression_suite()
    regression_results = _regression_results(regression_suite, regression_responses)
    regression_aggregate = _regression_aggregate(
        suite=regression_suite,
        results=regression_results,
        settings=settings,
        system_prompt_version=system_prompt_version,
        system_prompt=system_prompt,
        base_model=base_model,
    )
    write_regression_baseline_artifacts(
        results=regression_results,
        aggregate=regression_aggregate,
        output_dir=output_dir / "general-tool-regression",
    )

    suites = _suite_performance_from_responses(
        humaneval=humaneval_responses,
        mbpp=mbpp_responses,
        holdout=holdout_responses,
        regression=regression_responses,
    )
    _validate_runtime_metadata(runtime_metadata, suites)
    write_runtime_metadata(runtime_metadata, output_dir)
    write_baseline_provenance(provenance, output_dir)
    stage_manifest = output_dir / _GENERATION_STAGE_NAME
    if stage_manifest.exists():
        stage_manifest.unlink()
    work_dir = output_dir / _WORK_DIR_NAME
    if work_dir.exists():
        shutil.rmtree(work_dir)
    return freeze_python_baseline(
        output_dir=output_dir,
        evaluation=evaluation,
        settings=settings,
        system_prompt_version=system_prompt_version,
        system_prompt=system_prompt,
        generation_contract_sha256=generation_contract,
        provenance=provenance,
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one stage of the canonical P6-005 baseline")
    parser.add_argument("stage", choices=("generate", "score"))
    parser.add_argument(
        "--config",
        type=Path,
        default=_CANONICAL_BASELINE_CONFIG_PATH,
    )
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _argument_parser().parse_args(argv)
    config_path = cast(Path, args.config)
    repo_root = cast(Path, args.repo_root)
    if cast(str, args.stage) == "generate":
        path = generate_canonical_python_base_baseline_stage(
            config_path=config_path,
            device_index=cast(int, args.device_index),
            repo_root=repo_root,
        )
        print(path)
        return
    manifest = score_canonical_python_base_baseline_stage(
        config_path=config_path,
        repo_root=repo_root,
    )
    print(python_baseline_manifest_json(manifest), end="")


if __name__ == "__main__":
    main()
