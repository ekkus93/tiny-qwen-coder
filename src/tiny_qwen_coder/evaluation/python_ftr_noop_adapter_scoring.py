"""Score the FTR-102 no-op control without weakening benchmark identity gates.

The shared HumanEval, MBPP, and repository-holdout evaluators intentionally
accept only the normal base-only or language-adapter evaluation contracts.
FTR-102 uses a synthetic ``family='control'`` adapter, so its already-generated
completions are scored under the unchanged-base scoring contract and then the
persisted result provenance is rebound to the exact control adapter identity.
Result hashes are recomputed after rebinding before exact comparison to FTR-101.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import cast

from tiny_qwen_coder.evaluation import python_ftr_noop_adapter_equivalence as ftr
from tiny_qwen_coder.evaluation._baseline_artifacts import (
    file_sha256,
    write_regression_baseline_artifacts,
)
from tiny_qwen_coder.evaluation._baseline_runner import (
    _generate_items,
    _preflight_execution_images,
    _preflight_source_tree,
    _regression_aggregate,
    _regression_results,
)
from tiny_qwen_coder.evaluation.execution import (
    ConstrainedExecutionHarness,
    OciRuntime,
    OciRuntimeSpec,
)
from tiny_qwen_coder.evaluation.humaneval import (
    HumanEvalCompletion,
    HumanEvalEvaluator,
    HumanEvalSuiteResult,
    humaneval_results_sha256,
)
from tiny_qwen_coder.evaluation.mbpp import (
    MBPPCompletion,
    MBPPEvaluator,
    MBPPSuiteResult,
    mbpp_results_sha256,
)
from tiny_qwen_coder.evaluation.regression import load_frozen_general_tool_regression_suite
from tiny_qwen_coder.evaluation.repository_holdout import (
    RepositoryHoldoutCompletion,
    RepositoryHoldoutEvaluator,
    RepositoryHoldoutSuiteResult,
    repository_holdout_results_sha256,
)
from tiny_qwen_coder.identities import AdapterIdentity

_BASE_SCORING_ADAPTER = AdapterIdentity(family=None, adapter_id=None)


def _rebind_humaneval(result: HumanEvalSuiteResult) -> HumanEvalSuiteResult:
    """Attach the exact control provenance and restore aggregate hash integrity."""

    results = tuple(replace(item, adapter=ftr._CONTROL_ADAPTER) for item in result.results)
    aggregate = replace(
        result.aggregate,
        adapter=ftr._CONTROL_ADAPTER,
        results_sha256=humaneval_results_sha256(results),
    )
    return HumanEvalSuiteResult(results=results, aggregate=aggregate)


def _rebind_mbpp(result: MBPPSuiteResult) -> MBPPSuiteResult:
    """Attach the exact control provenance and restore aggregate hash integrity."""

    results = tuple(replace(item, adapter=ftr._CONTROL_ADAPTER) for item in result.results)
    aggregate = replace(
        result.aggregate,
        adapter=ftr._CONTROL_ADAPTER,
        results_sha256=mbpp_results_sha256(results),
    )
    return MBPPSuiteResult(results=results, aggregate=aggregate)


def _rebind_holdout(result: RepositoryHoldoutSuiteResult) -> RepositoryHoldoutSuiteResult:
    """Attach the exact control provenance and restore aggregate hash integrity."""

    results = tuple(replace(item, adapter=ftr._CONTROL_ADAPTER) for item in result.results)
    aggregate = replace(
        result.aggregate,
        adapter=ftr._CONTROL_ADAPTER,
        results_sha256=repository_holdout_results_sha256(results),
    )
    return RepositoryHoldoutSuiteResult(results=results, aggregate=aggregate)


def _score_control(*, output_dir: Path, runtime: OciRuntimeSpec) -> None:
    """Score transported control generations and persist control-bound artifacts."""

    (
        evaluation,
        settings,
        base_model,
        system_prompt_version,
        system_prompt,
        generation_contract,
    ) = ftr._evaluation_context()
    harness = ConstrainedExecutionHarness(runtime=runtime)
    humaneval = HumanEvalEvaluator(
        evaluation,
        base_model=base_model,
        adapter=_BASE_SCORING_ADAPTER,
        settings=settings,
        harness=harness,
    )
    mbpp = MBPPEvaluator(
        evaluation,
        base_model=base_model,
        adapter=_BASE_SCORING_ADAPTER,
        settings=settings,
        harness=harness,
    )
    holdout = RepositoryHoldoutEvaluator(
        evaluation,
        base_model=base_model,
        adapter=_BASE_SCORING_ADAPTER,
        settings=settings,
        harness=harness,
    )

    humaneval_problems = humaneval.load_problems()
    mbpp_problems = mbpp.load_problems()
    _preflight_execution_images(
        runtime,
        (
            humaneval.runner.execution_image,
            mbpp.runner.execution_image,
            holdout.suite.execution_image,
        ),
    )
    checkpoint = ftr._CheckpointOnly()
    humaneval_responses = _generate_items(
        suite_id="humaneval",
        prompts=tuple(
            (problem.task_id, humaneval.prompt_for(problem).user_content)
            for problem in humaneval_problems
        ),
        generator=checkpoint,
        system_prompt=system_prompt,
        generation_contract=generation_contract,
        output_dir=output_dir,
    )
    mbpp_responses = _generate_items(
        suite_id="mbpp",
        prompts=tuple(
            (problem.task_id, mbpp.prompt_for(problem).user_content) for problem in mbpp_problems
        ),
        generator=checkpoint,
        system_prompt=system_prompt,
        generation_contract=generation_contract,
        output_dir=output_dir,
    )
    holdout_responses = _generate_items(
        suite_id="repository-holdout",
        prompts=tuple(
            (task.problem_id, holdout.prompt_for(task).user_content) for task in holdout.suite.tasks
        ),
        generator=checkpoint,
        system_prompt=system_prompt,
        generation_contract=generation_contract,
        output_dir=output_dir,
    )
    regression_suite = load_frozen_general_tool_regression_suite()
    regression_responses = _generate_items(
        suite_id="general-tool-regression",
        prompts=tuple((case.id, case.prompt) for case in regression_suite.cases),
        generator=checkpoint,
        system_prompt=system_prompt,
        generation_contract=generation_contract,
        output_dir=output_dir,
    )

    humaneval_result = humaneval.evaluate_suite(
        humaneval_problems,
        tuple(
            HumanEvalCompletion(
                task_id=problem.task_id,
                generated_text=response.generated_text,
                generation=response.generation,
            )
            for problem, response in zip(humaneval_problems, humaneval_responses, strict=True)
        ),
    )
    humaneval.write_artifacts(_rebind_humaneval(humaneval_result), output_dir / "humaneval")

    mbpp_result = mbpp.evaluate_suite(
        mbpp_problems,
        tuple(
            MBPPCompletion(
                task_id=problem.task_id,
                generated_text=response.generated_text,
                generation=response.generation,
            )
            for problem, response in zip(mbpp_problems, mbpp_responses, strict=True)
        ),
    )
    mbpp.write_artifacts(_rebind_mbpp(mbpp_result), output_dir / "mbpp")

    holdout_result = holdout.evaluate_suite(
        tuple(
            RepositoryHoldoutCompletion(
                problem_id=task.problem_id,
                generated_text=response.generated_text,
                generation=response.generation,
            )
            for task, response in zip(holdout.suite.tasks, holdout_responses, strict=True)
        )
    )
    holdout.write_artifacts(_rebind_holdout(holdout_result), output_dir / "repository-holdout")

    regression_results = _regression_results(regression_suite, regression_responses)
    regression_aggregate = replace(
        _regression_aggregate(
            suite=regression_suite,
            results=regression_results,
            settings=settings,
            system_prompt_version=system_prompt_version,
            system_prompt=system_prompt,
            base_model=base_model,
        ),
        adapter=ftr._CONTROL_ADAPTER,
    )
    write_regression_baseline_artifacts(
        results=regression_results,
        aggregate=regression_aggregate,
        output_dir=output_dir / "general-tool-regression",
    )


def score_and_compare(
    *,
    frozen_dir: Path,
    frozen_identity_path: Path,
    adapter_manifest_path: Path,
    output_dir: Path = ftr._DEFAULT_OUTPUT_DIR,
    oci_runtime: Path,
    report_path: Path,
    repo_root: Path = Path("."),
    protocol_path: Path = ftr._PROTOCOL_CONFIG,
) -> dict[str, object]:
    """Score FTR-102 and fail closed unless it is exactly equivalent to FTR-101."""

    source_git_sha, _ = _preflight_source_tree(repo_root)
    protocol = ftr._protocol_config(protocol_path)
    adapter_manifest = ftr._read_json(adapter_manifest_path, context="FTR-102 adapter manifest")
    if adapter_manifest.get("source_git_sha") != source_git_sha:
        raise ftr.FTRNoopAdapterError("FTR-102 transported adapter source SHA drift detected")
    if adapter_manifest.get("protocol_config_sha256") != file_sha256(protocol_path):
        raise ftr.FTRNoopAdapterError("FTR-102 transported adapter protocol drift detected")
    (
        _evaluation,
        _settings,
        base_model,
        _system_prompt_version,
        _system_prompt,
        generation_contract,
    ) = ftr._evaluation_context()
    if base_model != ftr._protocol_base(protocol):
        raise ftr.FTRNoopAdapterError("FTR-102 score-stage base identity drift detected")
    ftr._validate_generation_manifest(
        output_dir=output_dir,
        source_git_sha=source_git_sha,
        protocol_path=protocol_path,
        generation_contract=generation_contract,
    )
    if not oci_runtime.is_absolute() or not oci_runtime.is_file():
        raise ftr.FTRNoopAdapterError("FTR-102 OCI runtime must be an existing absolute path")
    runtime = OciRuntimeSpec(kind=OciRuntime.DOCKER, executable=oci_runtime)
    _score_control(output_dir=output_dir, runtime=runtime)
    report = ftr.compare_control_to_frozen(
        frozen_dir=frozen_dir,
        frozen_identity_path=frozen_identity_path,
        current_dir=output_dir,
        current_identity_path=output_dir / ftr._CONTROL_IDENTITY_FILE,
    )
    report.update(
        {
            "source_git_sha": source_git_sha,
            "protocol_config_sha256": file_sha256(protocol_path),
            "adapter_manifest_sha256": file_sha256(adapter_manifest_path),
            "frozen_reference": ftr._strict_mapping(
                protocol.get("ftr_101_reference"), context="FTR-102 ftr_101_reference"
            ),
        }
    )
    ftr._write_json(report_path, report)
    if report["exact_noop_adapter_equivalence"] is not True:
        raise ftr.FTRNoopAdapterError(
            "FTR-102 no-op adapter changed canonical evaluation behavior; see report"
        )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score the FTR-102 no-op adapter control")
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument("--frozen-identity", type=Path, required=True)
    parser.add_argument("--adapter-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ftr._DEFAULT_OUTPUT_DIR)
    parser.add_argument("--oci-runtime", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the corrected FTR-102 scoring stage."""

    args = _parser().parse_args(argv)
    payload = score_and_compare(
        frozen_dir=cast(Path, args.frozen_dir),
        frozen_identity_path=cast(Path, args.frozen_identity),
        adapter_manifest_path=cast(Path, args.adapter_manifest),
        output_dir=cast(Path, args.output_dir),
        oci_runtime=cast(Path, args.oci_runtime),
        report_path=cast(Path, args.report),
        repo_root=cast(Path, args.repo_root),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
