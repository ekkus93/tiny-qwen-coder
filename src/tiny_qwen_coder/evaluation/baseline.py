"""Canonical unchanged-base Python baseline orchestration and artifact freezing."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from tiny_qwen_coder.evaluation._baseline_artifacts import (
    evaluation_config_sha256,
    file_sha256,
    freeze_python_baseline,
    load_python_baseline_manifest,
    python_baseline_manifest_json,
    regression_baseline_aggregate_json,
    regression_baseline_results_sha256,
    runtime_metadata_json,
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
from tiny_qwen_coder.evaluation._baseline_runner import (
    run_canonical_python_base_baseline,
    run_python_base_baseline,
)
from tiny_qwen_coder.evaluation._baseline_types import (
    BaselineArtifactDigest,
    BaselineGeneratedResponse,
    BaselineGenerationCheckpoint,
    BaselineRuntimeMetadata,
    BaselineSuitePerformance,
    PythonBaselineError,
    PythonBaselineManifest,
    RegressionBaselineAggregate,
    RegressionBaselineCaseResult,
)

__all__ = [
    "BaselineArtifactDigest",
    "BaselineGeneratedResponse",
    "BaselineGenerationCheckpoint",
    "BaselineGenerator",
    "BaselineRuntimeMetadata",
    "BaselineSuitePerformance",
    "HuggingFaceBaselineGenerator",
    "PythonBaselineError",
    "PythonBaselineManifest",
    "RegressionBaselineAggregate",
    "RegressionBaselineCaseResult",
    "evaluation_config_sha256",
    "file_sha256",
    "freeze_python_baseline",
    "generation_contract_sha256",
    "load_python_baseline_manifest",
    "prompt_sha256",
    "python_baseline_manifest_json",
    "regression_baseline_aggregate_json",
    "regression_baseline_results_sha256",
    "run_canonical_python_base_baseline",
    "run_python_base_baseline",
    "runtime_metadata_json",
    "system_prompt_sha256",
    "validate_python_baseline_artifacts",
    "write_regression_baseline_artifacts",
    "write_runtime_metadata",
    "main",
]


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run and freeze the canonical unchanged Qwen3.5-4B Python baseline "
            "for HumanEval, MBPP, the repository holdout, and general/tool regression."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/eval/python/base_baseline_v1.yaml"),
        help="Canonical P6-005 evaluation configuration.",
    )
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify an already-frozen artifact set instead of running model inference.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Artifact directory for --verify-only; defaults to the config output_dir.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point for P6-005 run or frozen-artifact verification."""

    args = _argument_parser().parse_args(argv)
    config_path = cast(Path, args.config)
    if cast(bool, args.verify_only):
        from tiny_qwen_coder.config import load_evaluation_config

        config = load_evaluation_config(config_path)
        output_dir = cast(Path | None, args.output_dir) or Path(config.output_dir)
        manifest = validate_python_baseline_artifacts(output_dir)
    else:
        if cast(Path | None, args.output_dir) is not None:
            raise SystemExit("--output-dir is only valid with --verify-only")
        manifest = run_canonical_python_base_baseline(
            config_path=config_path,
            device_index=cast(int, args.device_index),
            repo_root=cast(Path, args.repo_root),
        )
    print(python_baseline_manifest_json(manifest), end="")


if __name__ == "__main__":
    main()
