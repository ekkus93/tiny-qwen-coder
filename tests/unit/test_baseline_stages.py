"""Focused tests for the split GPU-generation and OCI-scoring baseline workflow."""

from __future__ import annotations

from pathlib import Path

import pytest

from tiny_qwen_coder.config import EvaluationConfig, load_evaluation_config
from tiny_qwen_coder.evaluation._baseline_generation import generation_contract_sha256
from tiny_qwen_coder.evaluation._baseline_provenance import load_baseline_base_model_identity
from tiny_qwen_coder.evaluation._baseline_runner import _validate_baseline_contract
from tiny_qwen_coder.evaluation._baseline_stages import (
    _STAGE_ARTIFACTS,
    _CheckpointOnlyGenerator,
    _validate_generation_stage_manifest,
    _write_generation_stage_manifest,
)
from tiny_qwen_coder.evaluation._baseline_types import PythonBaselineError
from tiny_qwen_coder.evaluation.settings import (
    FrozenEvaluationSettings,
    load_frozen_evaluation_settings,
)

_BASELINE_CONFIG = Path("configs/eval/python/base_baseline_v1.yaml")
_WORKFLOW = Path(".github/workflows/python-base-baseline.yml")
_SOURCE_SHA = "1" * 40


def _stage_contract() -> tuple[EvaluationConfig, FrozenEvaluationSettings, str, str]:
    evaluation = load_evaluation_config(_BASELINE_CONFIG)
    settings = load_frozen_evaluation_settings()
    base_model = load_baseline_base_model_identity(Path(evaluation.base_config))
    system_prompt_version, system_prompt = _validate_baseline_contract(
        evaluation,
        settings,
        base_model,
    )
    contract = generation_contract_sha256(
        base_model=base_model,
        settings=settings,
        system_prompt_version=system_prompt_version,
        system_prompt=system_prompt,
    )
    return evaluation, settings, system_prompt, contract


def test_generation_stage_manifest_detects_transferred_artifact_tampering(tmp_path: Path) -> None:
    evaluation, settings, system_prompt, contract = _stage_contract()
    for relative_path in _STAGE_ARTIFACTS:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"stage artifact: {relative_path}\n", encoding="utf-8")

    _write_generation_stage_manifest(
        output_dir=tmp_path,
        source_git_sha=_SOURCE_SHA,
        evaluation=evaluation,
        settings=settings,
        system_prompt=system_prompt,
        generation_contract=contract,
    )
    _validate_generation_stage_manifest(
        output_dir=tmp_path,
        source_git_sha=_SOURCE_SHA,
        evaluation=evaluation,
        settings=settings,
        system_prompt=system_prompt,
        generation_contract=contract,
    )

    checkpoint = tmp_path / _STAGE_ARTIFACTS[0]
    checkpoint.write_text(checkpoint.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
    with pytest.raises(PythonBaselineError, match="artifact digest mismatch"):
        _validate_generation_stage_manifest(
            output_dir=tmp_path,
            source_git_sha=_SOURCE_SHA,
            evaluation=evaluation,
            settings=settings,
            system_prompt=system_prompt,
            generation_contract=contract,
        )


def test_scoring_stage_refuses_missing_gpu_checkpoint() -> None:
    generator = _CheckpointOnlyGenerator()
    with pytest.raises(PythonBaselineError, match="missing a required GPU generation checkpoint"):
        generator.generate(system_prompt="system", user_prompt="prompt")


def test_gpu_workflow_keeps_oci_runtime_off_self_hosted_runner() -> None:
    workflow = _WORKFLOW.read_text(encoding="utf-8")
    generation_job, scoring_job = workflow.split("\n  score:\n", maxsplit=1)

    assert "self-hosted" in generation_job
    assert "Generate canonical unchanged-base responses" in generation_job
    assert "docker" not in generation_job.lower()
    assert "podman" not in generation_job.lower()
    assert "include-hidden-files: true" in generation_job

    assert "runs-on: ubuntu-latest" in scoring_job
    assert "docker pull \"${EXECUTION_IMAGE}\"" in scoring_job
    assert "Download GPU generation evidence" in scoring_job
    assert "Score GPU responses under constrained execution" in scoring_job
