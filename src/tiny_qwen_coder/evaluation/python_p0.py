"""P8-001 canonical Python P0 adapter evaluation orchestration."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from tiny_qwen_coder.config import EvaluationConfig, load_evaluation_config
from tiny_qwen_coder.evaluation._baseline_artifacts import write_runtime_metadata
from tiny_qwen_coder.evaluation._baseline_generation import BaselineGenerator
from tiny_qwen_coder.evaluation._baseline_provenance import load_baseline_base_model_identity
from tiny_qwen_coder.evaluation._baseline_runner import (
    _generate_items,
    _preflight_execution_images,
    _preflight_source_tree,
    _suite_performance,
)
from tiny_qwen_coder.evaluation._baseline_types import BaselineGeneratedResponse
from tiny_qwen_coder.evaluation._python_p0_comparison import (
    aggregate,
    comparison,
    load_baseline,
    verify_python_p0_evaluation,
    write_evaluation_manifest,
)
from tiny_qwen_coder.evaluation._python_p0_contract import (
    COMPARISON,
    DEFAULT_CONFIG,
    EVALUATION_MANIFEST,
    STAGE_MANIFEST,
    PythonP0EvaluationError,
    adapter_from_stage,
    adapter_generation_contract,
    read_json,
    stage_payload,
    validate_contract,
    validate_stage,
    write_json,
)
from tiny_qwen_coder.evaluation._python_p0_generation import (
    EXPECTED_ADAPTER_FAMILY,
    EXPECTED_ADAPTER_ID,
    HuggingFacePythonP0Generator,
    validate_python_p0_adapter,
)
from tiny_qwen_coder.evaluation.execution import ConstrainedExecutionHarness, discover_oci_runtime
from tiny_qwen_coder.evaluation.humaneval import HumanEvalCompletion, HumanEvalEvaluator, HumanEvalProblem
from tiny_qwen_coder.evaluation.mbpp import MBPPCompletion, MBPPEvaluator, MBPPProblem
from tiny_qwen_coder.evaluation.repository_holdout import (
    RepositoryHoldoutCompletion,
    RepositoryHoldoutEvaluator,
)
from tiny_qwen_coder.evaluation.settings import (
    FrozenEvaluationSettings,
    load_frozen_evaluation_settings,
)
from tiny_qwen_coder.identities import AdapterIdentity, BaseModelIdentity


class _CheckpointOnlyGenerator(BaselineGenerator):
    """Refuse any scoring-stage attempt to regenerate missing GPU responses."""

    def generate(self, *, system_prompt: str, user_prompt: str) -> BaselineGeneratedResponse:
        del system_prompt, user_prompt
        raise PythonP0EvaluationError(
            "scoring is missing a required GPU checkpoint; refusing silent regeneration"
        )


def _load_inputs(
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
    adapter = AdapterIdentity(family=EXPECTED_ADAPTER_FAMILY, adapter_id=EXPECTED_ADAPTER_ID)
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


def _generate_suites(
    *,
    evaluation: EvaluationConfig,
    settings: FrozenEvaluationSettings,
    base_model: BaseModelIdentity,
    generator: BaselineGenerator,
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
]:
    humaneval, humaneval_problems, mbpp, mbpp_problems, holdout = _load_inputs(
        evaluation=evaluation,
        settings=settings,
        base_model=base_model,
        harness=harness,
    )
    he = _generate_items(
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
    mb = _generate_items(
        suite_id="mbpp",
        prompts=tuple(
            (problem.task_id, mbpp.prompt_for(problem).user_content) for problem in mbpp_problems
        ),
        generator=generator,
        system_prompt=system_prompt,
        generation_contract=generation_contract,
        output_dir=output_dir,
    )
    rh = _generate_items(
        suite_id="repository-holdout",
        prompts=tuple(
            (task.problem_id, holdout.prompt_for(task).user_content) for task in holdout.suite.tasks
        ),
        generator=generator,
        system_prompt=system_prompt,
        generation_contract=generation_contract,
        output_dir=output_dir,
    )
    if (len(he), len(mb), len(rh)) != (164, 500, 11):
        raise PythonP0EvaluationError("P8-001 protected benchmark cardinality drift detected")
    return humaneval_problems, he, mbpp_problems, mb, holdout, rh


def generate_python_p0_stage(
    *,
    training_output: Path,
    config_path: Path = DEFAULT_CONFIG,
    device_index: int = 0,
    repo_root: Path = Path("."),
) -> Path:
    """Generate exactly 675 adapted responses on CUDA; execute no generated code."""

    evaluation = load_evaluation_config(config_path)
    settings = load_frozen_evaluation_settings()
    base_model = load_baseline_base_model_identity(Path(evaluation.base_config))
    system_version, system_prompt = validate_contract(evaluation, settings, base_model)
    source_git_sha, _ = _preflight_source_tree(repo_root)
    adapter = validate_python_p0_adapter(training_output, base_model)
    contract = adapter_generation_contract(
        base_model=base_model,
        settings=settings,
        system_prompt_version=system_version,
        system_prompt=system_prompt,
        adapter=adapter,
    )
    output_dir = Path(evaluation.output_dir)
    if (output_dir / EVALUATION_MANIFEST).exists() or (output_dir / COMPARISON).exists():
        raise PythonP0EvaluationError("P8-001 generation refuses an already-scored output directory")
    if (output_dir / STAGE_MANIFEST).exists():
        validate_stage(
            output_dir=output_dir,
            source_git_sha=source_git_sha,
            evaluation=evaluation,
            settings=settings,
            system_prompt=system_prompt,
            generation_contract=contract,
            adapter=adapter,
        )
        return output_dir / STAGE_MANIFEST

    generator = HuggingFacePythonP0Generator(
        training_output=training_output,
        base_model=base_model,
        settings=settings,
        device_index=device_index,
    )
    _he_problems, he, _mbpp_problems, mb, _holdout, rh = _generate_suites(
        evaluation=evaluation,
        settings=settings,
        base_model=base_model,
        generator=generator,
        system_prompt=system_prompt,
        generation_contract=contract,
        output_dir=output_dir,
    )
    suites = (
        _suite_performance("humaneval", he),
        _suite_performance("mbpp", mb),
        _suite_performance("repository-holdout", rh),
    )
    runtime = generator.runtime_metadata(suites)
    requests = tuple(item.requests for item in runtime.suites)
    if runtime.total_requests != 675 or requests != (164, 500, 11):
        raise PythonP0EvaluationError("P8-001 runtime evidence does not cover all 675 requests")
    write_runtime_metadata(runtime, output_dir)
    write_json(
        output_dir / STAGE_MANIFEST,
        stage_payload(
            output_dir=output_dir,
            source_git_sha=source_git_sha,
            evaluation=evaluation,
            settings=settings,
            system_prompt=system_prompt,
            generation_contract=contract,
            adapter=adapter,
        ),
    )
    return output_dir / STAGE_MANIFEST


def score_python_p0_stage(
    *,
    baseline_dir: Path,
    config_path: Path = DEFAULT_CONFIG,
    repo_root: Path = Path("."),
) -> Path:
    """Score GPU responses in OCI and write the direct P6-vs-P0 comparison."""

    evaluation = load_evaluation_config(config_path)
    settings = load_frozen_evaluation_settings()
    base_model = load_baseline_base_model_identity(Path(evaluation.base_config))
    system_version, system_prompt = validate_contract(evaluation, settings, base_model)
    source_git_sha, _ = _preflight_source_tree(repo_root)
    output_dir = Path(evaluation.output_dir)
    stage = read_json(output_dir / STAGE_MANIFEST, context="P8-001 generation stage")
    adapter = adapter_from_stage(stage)
    contract = adapter_generation_contract(
        base_model=base_model,
        settings=settings,
        system_prompt_version=system_version,
        system_prompt=system_prompt,
        adapter=adapter,
    )
    validate_stage(
        output_dir=output_dir,
        source_git_sha=source_git_sha,
        evaluation=evaluation,
        settings=settings,
        system_prompt=system_prompt,
        generation_contract=contract,
        adapter=adapter,
    )
    baseline_manifest, baseline = load_baseline(baseline_dir, base_model)

    runtime = discover_oci_runtime()
    harness = ConstrainedExecutionHarness(runtime=runtime)
    humaneval, he_problems, mbpp, mbpp_problems, holdout = _load_inputs(
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
    resolved_he, he, resolved_mbpp, mb, resolved_holdout, rh = _generate_suites(
        evaluation=evaluation,
        settings=settings,
        base_model=base_model,
        generator=_CheckpointOnlyGenerator(),
        system_prompt=system_prompt,
        generation_contract=contract,
        output_dir=output_dir,
        harness=harness,
    )
    if (
        resolved_he != he_problems
        or resolved_mbpp != mbpp_problems
        or resolved_holdout.suite != holdout.suite
    ):
        raise PythonP0EvaluationError("protected benchmark inputs changed during scoring setup")

    he_result = humaneval.evaluate_suite(
        resolved_he,
        tuple(
            HumanEvalCompletion(
                task_id=problem.task_id,
                generated_text=response.generated_text,
                generation=response.generation,
            )
            for problem, response in zip(resolved_he, he, strict=True)
        ),
    )
    humaneval.write_artifacts(he_result, output_dir / "humaneval")
    mbpp_result = mbpp.evaluate_suite(
        resolved_mbpp,
        tuple(
            MBPPCompletion(
                task_id=problem.task_id,
                generated_text=response.generated_text,
                generation=response.generation,
            )
            for problem, response in zip(resolved_mbpp, mb, strict=True)
        ),
    )
    mbpp.write_artifacts(mbpp_result, output_dir / "mbpp")
    holdout_result = holdout.evaluate_suite(
        tuple(
            RepositoryHoldoutCompletion(
                problem_id=task.problem_id,
                generated_text=response.generated_text,
                generation=response.generation,
            )
            for task, response in zip(resolved_holdout.suite.tasks, rh, strict=True)
        )
    )
    holdout.write_artifacts(holdout_result, output_dir / "repository-holdout")

    adapted = {
        "humaneval": aggregate(
            output_dir / "humaneval/humaneval-aggregate.json",
            suite_id="humaneval",
            base_model=base_model,
            adapter_id=EXPECTED_ADAPTER_ID,
        ),
        "mbpp": aggregate(
            output_dir / "mbpp/mbpp-aggregate.json",
            suite_id="mbpp",
            base_model=base_model,
            adapter_id=EXPECTED_ADAPTER_ID,
        ),
        "repository-holdout": aggregate(
            output_dir / "repository-holdout/repository-holdout-aggregate.json",
            suite_id="repository-holdout",
            base_model=base_model,
            adapter_id=EXPECTED_ADAPTER_ID,
        ),
    }
    write_json(
        output_dir / COMPARISON,
        comparison(
            source_git_sha=source_git_sha,
            base_model=base_model,
            adapter=adapter,
            settings=settings,
            generation_contract=contract,
            baseline_manifest=baseline_manifest,
            baseline=baseline,
            adapted=adapted,
        ),
    )
    return write_evaluation_manifest(output_dir, source_git_sha)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run/verify P8-001 Python P0 evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("generate", "score", "verify"):
        item = sub.add_parser(name)
        item.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        item.add_argument("--repo-root", type=Path, default=Path("."))
        if name == "generate":
            item.add_argument("--training-output", type=Path, required=True)
            item.add_argument("--device-index", type=int, default=0)
        if name in {"score", "verify"}:
            item.add_argument("--baseline-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    command = cast(str, args.command)
    config = cast(Path, args.config)
    repo_root = cast(Path, args.repo_root)
    if command == "generate":
        path = generate_python_p0_stage(
            training_output=cast(Path, args.training_output),
            config_path=config,
            device_index=cast(int, args.device_index),
            repo_root=repo_root,
        )
        print(path)
    elif command == "score":
        path = score_python_p0_stage(
            baseline_dir=cast(Path, args.baseline_dir),
            config_path=config,
            repo_root=repo_root,
        )
        print(path)
    else:
        print(
            json.dumps(
                verify_python_p0_evaluation(
                    baseline_dir=cast(Path, args.baseline_dir),
                    config_path=config,
                    repo_root=repo_root,
                ),
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
