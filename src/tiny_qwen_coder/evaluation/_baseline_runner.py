"""Orchestration for the canonical unchanged-base Python evaluation baseline."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import cast

from tiny_qwen_coder.config import EvaluationConfig, load_evaluation_config
from tiny_qwen_coder.evaluation._baseline_artifacts import (
    freeze_python_baseline,
    regression_baseline_results_sha256,
    system_prompt_sha256,
    validate_python_baseline_artifacts,
    write_regression_baseline_artifacts,
    write_runtime_metadata,
)
from tiny_qwen_coder.evaluation._baseline_generation import (
    BaselineGenerator,
    HuggingFaceBaselineGenerator,
    generation_contract_sha256,
    prompt_sha256,
)
from tiny_qwen_coder.evaluation._baseline_provenance import (
    BaselineProvenance,
    collect_baseline_provenance,
    load_baseline_base_model_identity,
    write_baseline_provenance,
)
from tiny_qwen_coder.evaluation._baseline_types import (
    BaselineGeneratedResponse,
    BaselineGenerationCheckpoint,
    BaselineRuntimeMetadata,
    BaselineSuitePerformance,
    PythonBaselineError,
    PythonBaselineManifest,
    RegressionBaselineAggregate,
    RegressionBaselineCaseResult,
)
from tiny_qwen_coder.evaluation.execution import (
    ConstrainedExecutionHarness,
    OciRuntimeSpec,
    discover_oci_runtime,
)
from tiny_qwen_coder.evaluation.humaneval import (
    HumanEvalCompletion,
    HumanEvalEvaluator,
    HumanEvalProblem,
)
from tiny_qwen_coder.evaluation.mbpp import MBPPCompletion, MBPPEvaluator, MBPPProblem
from tiny_qwen_coder.evaluation.regression import (
    RegressionCategory,
    RegressionCategoryScore,
    RegressionSuite,
    evaluate_regression_response,
    load_frozen_general_tool_regression_suite,
    regression_suite_sha256,
)
from tiny_qwen_coder.evaluation.repository_holdout import (
    RepositoryHoldoutCompletion,
    RepositoryHoldoutEvaluator,
)
from tiny_qwen_coder.evaluation.settings import (
    FrozenEvaluationSettings,
    evaluation_settings_sha256,
    load_frozen_evaluation_settings,
    validate_evaluation_config_settings,
)
from tiny_qwen_coder.identities import AdapterIdentity, BaseModelIdentity
from tiny_qwen_coder.languages.python import load_python_plugin

_CANONICAL_BASELINE_CONFIG_PATH = Path("configs/eval/python/base_baseline_v1.yaml")
_EXPECTED_SUITES = (
    "humaneval",
    "mbpp",
    "repository-holdout",
    "general-tool-regression",
)
_EXPECTED_SYSTEM_PROMPT_VERSION = "python-v1"
_EXPECTED_SYSTEM_PROMPT_SHA256 = (
    "ed10dcc67116e4b3633eed7413228bb083a797fc6917132df03890aa7e05497e"
)
_WORK_DIR_NAME = ".baseline-work"


def _checkpoint_path(output_dir: Path, suite_id: str) -> Path:
    return output_dir / _WORK_DIR_NAME / f"{suite_id}-generation.jsonl"


def _checkpoint_json(checkpoint: BaselineGenerationCheckpoint) -> str:
    return json.dumps(asdict(checkpoint), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _parse_generation(value: object, *, context: str) -> BaselineGeneratedResponse:
    if not isinstance(value, dict):
        raise PythonBaselineError(f"{context}.response must be an object")
    generated_text = value.get("generated_text")
    raw_generation = value.get("generation")
    if not isinstance(generated_text, str):
        raise PythonBaselineError(f"{context}.response.generated_text must be a string")
    if not isinstance(raw_generation, dict):
        raise PythonBaselineError(f"{context}.response.generation must be an object")
    from tiny_qwen_coder.evaluation.results import GenerationStats

    prompt_tokens = raw_generation.get("prompt_tokens")
    if prompt_tokens is not None and (
        isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int)
    ):
        raise PythonBaselineError(f"{context}.response.generation.prompt_tokens is invalid")
    generated_tokens = raw_generation.get("generated_tokens")
    latency_seconds = raw_generation.get("latency_seconds")
    tokens_per_second = raw_generation.get("tokens_per_second")
    if isinstance(generated_tokens, bool) or not isinstance(generated_tokens, int):
        raise PythonBaselineError(f"{context}.response.generation.generated_tokens is invalid")
    if isinstance(latency_seconds, bool) or not isinstance(latency_seconds, int | float):
        raise PythonBaselineError(f"{context}.response.generation.latency_seconds is invalid")
    if isinstance(tokens_per_second, bool) or not isinstance(tokens_per_second, int | float):
        raise PythonBaselineError(f"{context}.response.generation.tokens_per_second is invalid")
    return BaselineGeneratedResponse(
        generated_text=generated_text,
        generation=GenerationStats(
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            latency_seconds=float(latency_seconds),
            tokens_per_second=float(tokens_per_second),
        ),
    )


def _load_checkpoints(path: Path) -> dict[str, BaselineGenerationCheckpoint]:
    if not path.exists():
        return {}
    checkpoints: dict[str, BaselineGenerationCheckpoint] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PythonBaselineError(f"could not read baseline checkpoint {path}: {exc}") from exc
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            raise PythonBaselineError(f"baseline checkpoint {path}:{index} is blank")
        try:
            value: object = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PythonBaselineError(
                f"invalid baseline checkpoint JSON at {path}:{index}"
            ) from exc
        if not isinstance(value, dict):
            raise PythonBaselineError(f"baseline checkpoint {path}:{index} must be an object")
        item_id = value.get("item_id")
        prompt_digest = value.get("prompt_sha256")
        contract_digest = value.get("generation_contract_sha256")
        if not isinstance(item_id, str):
            raise PythonBaselineError(f"baseline checkpoint {path}:{index} item_id is invalid")
        if not isinstance(prompt_digest, str) or not isinstance(contract_digest, str):
            raise PythonBaselineError(f"baseline checkpoint {path}:{index} digest is invalid")
        checkpoint = BaselineGenerationCheckpoint(
            item_id=item_id,
            prompt_sha256=prompt_digest,
            generation_contract_sha256=contract_digest,
            response=_parse_generation(value.get("response"), context=f"checkpoint[{index}]"),
        )
        if checkpoint.item_id in checkpoints:
            raise PythonBaselineError(f"duplicate checkpoint item {checkpoint.item_id!r} in {path}")
        checkpoints[checkpoint.item_id] = checkpoint
    return checkpoints


def _append_checkpoint(path: Path, checkpoint: BaselineGenerationCheckpoint) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(_checkpoint_json(checkpoint) + "\n")
        handle.flush()


def _generate_items(
    *,
    suite_id: str,
    prompts: Sequence[tuple[str, str]],
    generator: BaselineGenerator,
    system_prompt: str,
    generation_contract: str,
    output_dir: Path,
) -> tuple[BaselineGeneratedResponse, ...]:
    checkpoint_file = _checkpoint_path(output_dir, suite_id)
    cached = _load_checkpoints(checkpoint_file)
    prompt_by_id = dict(prompts)
    if len(prompt_by_id) != len(prompts):
        raise PythonBaselineError(f"{suite_id} prompt IDs must be unique")
    unknown = sorted(set(cached) - set(prompt_by_id))
    if unknown:
        raise PythonBaselineError(f"{suite_id} checkpoint contains unknown item IDs: {unknown!r}")

    responses: list[BaselineGeneratedResponse] = []
    for item_id, user_prompt in prompts:
        digest = prompt_sha256(item_id=item_id, user_prompt=user_prompt)
        checkpoint = cached.get(item_id)
        if checkpoint is not None:
            if checkpoint.prompt_sha256 != digest:
                raise PythonBaselineError(
                    f"{suite_id} checkpoint prompt drift detected for {item_id!r}"
                )
            if checkpoint.generation_contract_sha256 != generation_contract:
                raise PythonBaselineError(
                    f"{suite_id} checkpoint generation-contract drift detected for {item_id!r}"
                )
            responses.append(checkpoint.response)
            continue

        response = generator.generate(system_prompt=system_prompt, user_prompt=user_prompt)
        checkpoint = BaselineGenerationCheckpoint(
            item_id=item_id,
            prompt_sha256=digest,
            generation_contract_sha256=generation_contract,
            response=response,
        )
        _append_checkpoint(checkpoint_file, checkpoint)
        responses.append(response)
    return tuple(responses)


def _suite_performance(
    suite_id: str,
    responses: Iterable[BaselineGeneratedResponse],
) -> BaselineSuitePerformance:
    items = tuple(responses)
    prompt_tokens = sum(item.generation.prompt_tokens or 0 for item in items)
    generated_tokens = sum(item.generation.generated_tokens for item in items)
    latency = sum(item.generation.latency_seconds for item in items)
    return BaselineSuitePerformance(
        suite_id=suite_id,
        requests=len(items),
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        generation_latency_seconds=latency,
        tokens_per_second=generated_tokens / latency if latency > 0 else 0.0,
    )


def _validate_runtime_metadata(
    metadata: BaselineRuntimeMetadata,
    suites: tuple[BaselineSuitePerformance, ...],
) -> None:
    if metadata.suites != suites:
        raise PythonBaselineError(
            "baseline runtime metadata suite measurements do not match generated responses"
        )


def _validate_baseline_contract(
    evaluation: EvaluationConfig,
    settings: FrozenEvaluationSettings,
    base_model: BaseModelIdentity,
) -> tuple[str, str]:
    if evaluation.language != "python":
        raise PythonBaselineError("P6-005 baseline requires language='python'")
    if evaluation.adapter_id is not None:
        raise PythonBaselineError("P6-005 must evaluate the unchanged base without an adapter")
    if evaluation.suites != _EXPECTED_SUITES:
        raise PythonBaselineError(
            f"P6-005 suite order must be exactly {_EXPECTED_SUITES!r}; got {evaluation.suites!r}"
        )
    expected_base = load_baseline_base_model_identity(Path(evaluation.base_config))
    if expected_base != base_model:
        raise PythonBaselineError(
            "P6-005 base-model identity does not match evaluation base_config"
        )
    settings_digest = validate_evaluation_config_settings(evaluation, settings)
    plugin = load_python_plugin()
    system_prompt = plugin.spec.config.system_prompt
    if system_prompt.version != _EXPECTED_SYSTEM_PROMPT_VERSION:
        raise PythonBaselineError(
            "Python system-prompt version drifted; explicitly version the P6-005 baseline first"
        )
    digest = system_prompt_sha256(system_prompt.text)
    if digest != _EXPECTED_SYSTEM_PROMPT_SHA256:
        raise PythonBaselineError(
            "Python system-prompt content drifted; explicitly version the P6-005 baseline first"
        )
    if settings_digest != evaluation_settings_sha256(settings):
        raise PythonBaselineError("P6-005 evaluation-settings fingerprint is inconsistent")
    return system_prompt.version, system_prompt.text


def _regression_results(
    suite: RegressionSuite,
    responses: Sequence[BaselineGeneratedResponse],
) -> tuple[RegressionBaselineCaseResult, ...]:
    if len(responses) != len(suite.cases):
        raise PythonBaselineError("regression response count does not match frozen suite")
    output: list[RegressionBaselineCaseResult] = []
    for case, response in zip(suite.cases, responses, strict=True):
        scored = evaluate_regression_response(case, response.generated_text)
        output.append(
            RegressionBaselineCaseResult(
                case_id=case.id,
                category=case.category,
                generated_text=response.generated_text,
                passed=scored.passed,
                detail=scored.detail,
                generation=response.generation,
            )
        )
    return tuple(output)


def _regression_aggregate(
    *,
    suite: RegressionSuite,
    results: tuple[RegressionBaselineCaseResult, ...],
    settings: FrozenEvaluationSettings,
    system_prompt_version: str,
    system_prompt: str,
    base_model: BaseModelIdentity,
) -> RegressionBaselineAggregate:
    categories = tuple(
        RegressionCategoryScore(
            category=category,
            passed=sum(item.passed for item in results if item.category is category),
            total=sum(1 for item in results if item.category is category),
        )
        for category in RegressionCategory
    )
    passed = sum(item.passed for item in results)
    return RegressionBaselineAggregate(
        schema_version=1,
        suite_id=suite.suite_id,
        suite_version=suite.suite_version,
        suite_sha256=regression_suite_sha256(suite),
        evaluation_settings_sha256=evaluation_settings_sha256(settings),
        system_prompt_version=system_prompt_version,
        system_prompt_sha256=system_prompt_sha256(system_prompt),
        base_model=base_model,
        adapter=AdapterIdentity(family=None, adapter_id=None),
        total_cases=len(results),
        passed=passed,
        failed=len(results) - passed,
        pass_rate=passed / len(results),
        categories=categories,
        results_sha256=regression_baseline_results_sha256(results),
    )


def _preflight_source_tree(repo_root: Path) -> tuple[str, bool]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PythonBaselineError("could not inspect source Git tree") from exc
    if len(sha) != 40 or any(character not in "0123456789abcdef" for character in sha):
        raise PythonBaselineError("source Git commit must be a full lowercase SHA")
    dirty = bool(status.strip())
    if dirty:
        raise PythonBaselineError(
            "P6-005 baseline must run from a clean source tree so its artifacts can be frozen"
        )
    return sha, dirty


def _preflight_execution_images(
    runtime: OciRuntimeSpec,
    images: Iterable[str],
) -> None:
    unique_images = tuple(dict.fromkeys(images))
    if not unique_images:
        raise PythonBaselineError("P6-005 has no constrained-execution images to preflight")
    for image in unique_images:
        try:
            completed = subprocess.run(
                [str(runtime.executable), "image", "inspect", image],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30.0,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise PythonBaselineError(
                f"could not inspect required constrained-execution image {image!r}"
            ) from exc
        if completed.returncode != 0:
            detail = completed.stderr.strip()
            suffix = f": {detail}" if detail else ""
            raise PythonBaselineError(
                "required constrained-execution image is not available locally: "
                f"{image!r}{suffix}; pull the pinned image before running P6-005"
            )


def run_python_base_baseline(
    *,
    evaluation: EvaluationConfig,
    generator: BaselineGenerator,
    base_model: BaseModelIdentity,
    settings: FrozenEvaluationSettings | None = None,
    humaneval_problems: tuple[HumanEvalProblem, ...] | None = None,
    mbpp_problems: tuple[MBPPProblem, ...] | None = None,
    harness: ConstrainedExecutionHarness | None = None,
    repo_root: Path = Path("."),
    runtime_metadata: BaselineRuntimeMetadata | None = None,
) -> PythonBaselineManifest:
    """Generate, score, write, and freeze all required unchanged-base Python artifacts."""

    resolved_settings = settings or load_frozen_evaluation_settings()
    system_prompt_version, system_prompt = _validate_baseline_contract(
        evaluation,
        resolved_settings,
        base_model,
    )
    output_dir = Path(evaluation.output_dir)
    frozen_manifest_path = output_dir / "baseline-manifest.json"
    if frozen_manifest_path.exists():
        return validate_python_baseline_artifacts(output_dir)
    _preflight_source_tree(repo_root)
    shared_harness = harness or ConstrainedExecutionHarness()
    generation_contract = generation_contract_sha256(
        base_model=base_model,
        settings=resolved_settings,
        system_prompt_version=system_prompt_version,
        system_prompt=system_prompt,
    )

    humaneval = HumanEvalEvaluator(
        evaluation,
        base_model=base_model,
        adapter=AdapterIdentity(family=None, adapter_id=None),
        settings=resolved_settings,
        harness=shared_harness,
    )
    resolved_humaneval = humaneval_problems or humaneval.load_problems()
    humaneval_prompts = tuple(
        (problem.task_id, humaneval.prompt_for(problem).user_content)
        for problem in resolved_humaneval
    )
    humaneval_responses = _generate_items(
        suite_id="humaneval",
        prompts=humaneval_prompts,
        generator=generator,
        system_prompt=system_prompt,
        generation_contract=generation_contract,
        output_dir=output_dir,
    )
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

    mbpp = MBPPEvaluator(
        evaluation,
        base_model=base_model,
        adapter=AdapterIdentity(family=None, adapter_id=None),
        settings=resolved_settings,
        harness=shared_harness,
    )
    resolved_mbpp = mbpp_problems or mbpp.load_problems()
    mbpp_prompts = tuple(
        (problem.task_id, mbpp.prompt_for(problem).user_content) for problem in resolved_mbpp
    )
    mbpp_responses = _generate_items(
        suite_id="mbpp",
        prompts=mbpp_prompts,
        generator=generator,
        system_prompt=system_prompt,
        generation_contract=generation_contract,
        output_dir=output_dir,
    )
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

    holdout = RepositoryHoldoutEvaluator(
        evaluation,
        base_model=base_model,
        adapter=AdapterIdentity(family=None, adapter_id=None),
        settings=resolved_settings,
        harness=shared_harness,
    )
    holdout_prompts = tuple(
        (task.problem_id, holdout.prompt_for(task).user_content) for task in holdout.suite.tasks
    )
    holdout_responses = _generate_items(
        suite_id="repository-holdout",
        prompts=holdout_prompts,
        generator=generator,
        system_prompt=system_prompt,
        generation_contract=generation_contract,
        output_dir=output_dir,
    )
    holdout_completions = tuple(
        RepositoryHoldoutCompletion(
            problem_id=task.problem_id,
            generated_text=response.generated_text,
            generation=response.generation,
        )
        for task, response in zip(holdout.suite.tasks, holdout_responses, strict=True)
    )
    holdout_result = holdout.evaluate_suite(holdout_completions)
    holdout.write_artifacts(holdout_result, output_dir / "repository-holdout")

    regression_suite = load_frozen_general_tool_regression_suite()
    regression_prompts = tuple((case.id, case.prompt) for case in regression_suite.cases)
    regression_responses = _generate_items(
        suite_id="general-tool-regression",
        prompts=regression_prompts,
        generator=generator,
        system_prompt=system_prompt,
        generation_contract=generation_contract,
        output_dir=output_dir,
    )
    regression_results = _regression_results(regression_suite, regression_responses)
    regression_aggregate = _regression_aggregate(
        suite=regression_suite,
        results=regression_results,
        settings=resolved_settings,
        system_prompt_version=system_prompt_version,
        system_prompt=system_prompt,
        base_model=base_model,
    )
    write_regression_baseline_artifacts(
        results=regression_results,
        aggregate=regression_aggregate,
        output_dir=output_dir / "general-tool-regression",
    )

    suite_performance = (
        _suite_performance("humaneval", humaneval_responses),
        _suite_performance("mbpp", mbpp_responses),
        _suite_performance("repository-holdout", holdout_responses),
        _suite_performance("general-tool-regression", regression_responses),
    )
    if runtime_metadata is None:
        metadata_method = getattr(generator, "runtime_metadata", None)
        if metadata_method is None or not callable(metadata_method):
            raise PythonBaselineError(
                "baseline generator must expose runtime_metadata() or explicit metadata "
                "must be supplied"
            )
        runtime_metadata = cast(BaselineRuntimeMetadata, metadata_method(suite_performance))
    _validate_runtime_metadata(runtime_metadata, suite_performance)
    write_runtime_metadata(runtime_metadata, output_dir)

    provenance = collect_baseline_provenance(
        base_model=base_model,
        seed=evaluation.seed,
        repo_root=repo_root,
    )
    write_baseline_provenance(provenance, output_dir)
    manifest = freeze_python_baseline(
        output_dir=output_dir,
        evaluation=evaluation,
        settings=resolved_settings,
        system_prompt_version=system_prompt_version,
        system_prompt=system_prompt,
        generation_contract_sha256=generation_contract,
        provenance=provenance,
    )
    work_dir = output_dir / _WORK_DIR_NAME
    if work_dir.exists():
        shutil.rmtree(work_dir)
    return manifest


def run_canonical_python_base_baseline(
    *,
    config_path: Path = _CANONICAL_BASELINE_CONFIG_PATH,
    device_index: int = 0,
    repo_root: Path = Path("."),
) -> PythonBaselineManifest:
    """Run the complete canonical P6-005 baseline on one CUDA device."""

    evaluation = load_evaluation_config(config_path)
    settings = load_frozen_evaluation_settings()
    base_model = load_baseline_base_model_identity(Path(evaluation.base_config))
    _validate_baseline_contract(evaluation, settings, base_model)
    _preflight_source_tree(repo_root)
    runtime = discover_oci_runtime()
    harness = ConstrainedExecutionHarness(runtime=runtime)
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
    _preflight_execution_images(
        runtime,
        (
            humaneval.runner.execution_image,
            mbpp.runner.execution_image,
            holdout.suite.execution_image,
        ),
    )
    humaneval_problems = humaneval.load_problems()
    mbpp_problems = mbpp.load_problems()

    generator = HuggingFaceBaselineGenerator(
        base_model=base_model,
        settings=settings,
        device_index=device_index,
    )
    return run_python_base_baseline(
        evaluation=evaluation,
        generator=generator,
        base_model=base_model,
        settings=settings,
        humaneval_problems=humaneval_problems,
        mbpp_problems=mbpp_problems,
        harness=harness,
        repo_root=repo_root,
    )
