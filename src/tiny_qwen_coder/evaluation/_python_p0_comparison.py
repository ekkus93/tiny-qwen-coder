"""P8-001 aggregate validation, direct comparison, and final evidence verification."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import cast

from tiny_qwen_coder.config import load_evaluation_config
from tiny_qwen_coder.evaluation._baseline_artifacts import (
    file_sha256,
    validate_python_baseline_artifacts,
)
from tiny_qwen_coder.evaluation._baseline_provenance import load_baseline_base_model_identity
from tiny_qwen_coder.evaluation._baseline_runner import _preflight_source_tree
from tiny_qwen_coder.evaluation._baseline_types import PythonBaselineManifest
from tiny_qwen_coder.evaluation._python_p0_contract import (
    COMPARISON,
    DEFAULT_CONFIG,
    EVALUATION_MANIFEST,
    EXPECTED_BASELINE_ARTIFACT_SET_SHA256,
    EXPECTED_BASELINE_ID,
    EXPECTED_BASELINE_SOURCE_SHA,
    EXPECTED_SUITES,
    EXPECTED_TOTALS,
    FINAL_ARTIFACTS,
    SCHEMA_VERSION,
    STAGE_MANIFEST,
    TASK_ID,
    AdapterEvidence,
    PythonP0EvaluationError,
    adapter_from_stage,
    adapter_generation_contract,
    artifact_set_sha256,
    read_json,
    strict_mapping,
    validate_contract,
    validate_stage,
    write_json,
)
from tiny_qwen_coder.evaluation._python_p0_generation import (
    EXPECTED_ADAPTER_FAMILY,
    EXPECTED_ADAPTER_ID,
    EXPECTED_ADAPTER_SHA256,
    EXPECTED_ADAPTER_SIZE_BYTES,
    EXPECTED_TRAINING_GIT_SHA,
    EXPECTED_TRAINING_RUN_ID,
)
from tiny_qwen_coder.evaluation.settings import (
    FrozenEvaluationSettings,
    evaluation_settings_sha256,
    load_frozen_evaluation_settings,
)
from tiny_qwen_coder.identities import BaseModelIdentity


def aggregate(
    path: Path,
    *,
    suite_id: str,
    base_model: BaseModelIdentity,
    adapter_id: str | None,
) -> dict[str, object]:
    """Load one coding aggregate and reject invalid or harness-contaminated metrics."""

    value = read_json(path, context=f"{suite_id} aggregate")
    total = value.get("total_problems")
    passed = value.get("passed")
    failed = value.get("failed")
    harness_errors = value.get("harness_errors")
    pass_at_1 = value.get("pass_at_1")
    if suite_id not in EXPECTED_TOTALS:
        raise PythonP0EvaluationError(f"unexpected P8-001 suite: {suite_id!r}")
    expected_total = EXPECTED_TOTALS[suite_id]
    if isinstance(total, bool) or not isinstance(total, int) or total != expected_total:
        raise PythonP0EvaluationError(f"{suite_id} total problem count drifted")
    if (
        isinstance(passed, bool)
        or not isinstance(passed, int)
        or isinstance(failed, bool)
        or not isinstance(failed, int)
        or isinstance(harness_errors, bool)
        or not isinstance(harness_errors, int)
    ):
        raise PythonP0EvaluationError(f"{suite_id} aggregate integer fields are invalid")
    if passed + failed != total:
        raise PythonP0EvaluationError(f"{suite_id} aggregate pass/fail counts are inconsistent")
    if harness_errors != 0:
        raise PythonP0EvaluationError(
            f"{suite_id} has execution-harness errors; metric is not accepted"
        )
    if isinstance(pass_at_1, bool) or not isinstance(pass_at_1, int | float):
        raise PythonP0EvaluationError(f"{suite_id} pass_at_1 is missing/non-numeric")
    if not 0.0 <= float(pass_at_1) <= 1.0:
        raise PythonP0EvaluationError(f"{suite_id} pass_at_1 is outside [0, 1]")
    if value.get("base_model") != asdict(base_model):
        raise PythonP0EvaluationError(f"{suite_id} aggregate base-model identity drifted")
    expected_adapter = {
        "family": EXPECTED_ADAPTER_FAMILY if adapter_id is not None else None,
        "adapter_id": adapter_id,
    }
    if value.get("adapter") != expected_adapter:
        raise PythonP0EvaluationError(f"{suite_id} aggregate adapter identity drifted")
    return value


def load_baseline(
    baseline_dir: Path,
    base_model: BaseModelIdentity,
) -> tuple[PythonBaselineManifest, dict[str, dict[str, object]]]:
    """Revalidate and load the exact accepted P6-005 coding aggregates."""

    manifest = validate_python_baseline_artifacts(baseline_dir)
    if manifest.baseline_id != EXPECTED_BASELINE_ID:
        raise PythonP0EvaluationError("baseline ID is not the accepted P6-005 baseline")
    if manifest.source_git_sha != EXPECTED_BASELINE_SOURCE_SHA:
        raise PythonP0EvaluationError("baseline source SHA is not the accepted P6-005 run")
    if manifest.artifact_set_sha256 != EXPECTED_BASELINE_ARTIFACT_SET_SHA256:
        raise PythonP0EvaluationError("baseline artifact-set hash is not the accepted P6-005 run")
    if manifest.base_model != base_model:
        raise PythonP0EvaluationError("baseline base model differs from P8-001 base model")
    aggregates = {
        "humaneval": aggregate(
            baseline_dir / "humaneval/humaneval-aggregate.json",
            suite_id="humaneval",
            base_model=base_model,
            adapter_id=None,
        ),
        "mbpp": aggregate(
            baseline_dir / "mbpp/mbpp-aggregate.json",
            suite_id="mbpp",
            base_model=base_model,
            adapter_id=None,
        ),
        "repository-holdout": aggregate(
            baseline_dir / "repository-holdout/repository-holdout-aggregate.json",
            suite_id="repository-holdout",
            base_model=base_model,
            adapter_id=None,
        ),
    }
    return manifest, aggregates


def comparison(
    *,
    source_git_sha: str,
    base_model: BaseModelIdentity,
    adapter: AdapterEvidence,
    settings: FrozenEvaluationSettings,
    generation_contract: str,
    baseline_manifest: PythonBaselineManifest,
    baseline: Mapping[str, Mapping[str, object]],
    adapted: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Create the direct persisted base-vs-adapter coding comparison."""

    suites: list[dict[str, object]] = []
    base_passed_total = 0
    adapted_passed_total = 0
    for suite_id in EXPECTED_SUITES:
        base_item = baseline[suite_id]
        adapted_item = adapted[suite_id]
        base_passed = cast(int, base_item["passed"])
        adapted_passed = cast(int, adapted_item["passed"])
        base_pass = float(cast(int | float, base_item["pass_at_1"]))
        adapted_pass = float(cast(int | float, adapted_item["pass_at_1"]))
        base_passed_total += base_passed
        adapted_passed_total += adapted_passed
        suites.append(
            {
                "suite_id": suite_id,
                "total_problems": EXPECTED_TOTALS[suite_id],
                "base_passed": base_passed,
                "base_failed": cast(int, base_item["failed"]),
                "base_pass_at_1": base_pass,
                "adapter_passed": adapted_passed,
                "adapter_failed": cast(int, adapted_item["failed"]),
                "adapter_pass_at_1": adapted_pass,
                "delta_passed": adapted_passed - base_passed,
                "delta_pass_at_1": adapted_pass - base_pass,
            }
        )
    total = sum(EXPECTED_TOTALS.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "evaluation_complete": True,
        "source_git_sha": source_git_sha,
        "base_model": asdict(base_model),
        "adapter": {
            "family": adapter.family,
            "adapter_id": adapter.adapter_id,
            "adapter_model_sha256": adapter.adapter_model_sha256,
            "adapter_model_size_bytes": adapter.adapter_model_size_bytes,
            "training_run_id": adapter.training_run_id,
            "training_git_sha": adapter.training_git_sha,
        },
        "baseline": {
            "baseline_id": baseline_manifest.baseline_id,
            "source_git_sha": baseline_manifest.source_git_sha,
            "artifact_set_sha256": baseline_manifest.artifact_set_sha256,
        },
        "evaluation_settings_sha256": evaluation_settings_sha256(settings),
        "generation_contract_sha256": generation_contract,
        "suites": suites,
        "overall": {
            "total_problems": total,
            "base_passed": base_passed_total,
            "base_micro_pass_rate": base_passed_total / total,
            "adapter_passed": adapted_passed_total,
            "adapter_micro_pass_rate": adapted_passed_total / total,
            "delta_passed": adapted_passed_total - base_passed_total,
            "delta_micro_pass_rate": (adapted_passed_total - base_passed_total) / total,
        },
    }


def write_evaluation_manifest(output_dir: Path, source_git_sha: str) -> Path:
    """Hash the complete final P8-001 evidence set and write its manifest."""

    artifacts: list[dict[str, str]] = []
    for relative in FINAL_ARTIFACTS:
        path = output_dir / relative
        if not path.is_file():
            raise PythonP0EvaluationError(f"P8-001 final evidence is missing {relative!r}")
        artifacts.append({"path": relative, "sha256": file_sha256(path)})
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "source_git_sha": source_git_sha,
        "artifacts": artifacts,
        "artifact_set_sha256": artifact_set_sha256(artifacts),
    }
    path = output_dir / EVALUATION_MANIFEST
    write_json(path, payload)
    return path


def _adapted_aggregates(
    output_dir: Path,
    base_model: BaseModelIdentity,
) -> dict[str, dict[str, object]]:
    return {
        suite: aggregate(
            output_dir
            / (
                "repository-holdout/repository-holdout-aggregate.json"
                if suite == "repository-holdout"
                else f"{suite}/{suite}-aggregate.json"
            ),
            suite_id=suite,
            base_model=base_model,
            adapter_id=EXPECTED_ADAPTER_ID,
        )
        for suite in EXPECTED_SUITES
    }


def verify_python_p0_evaluation(
    *,
    baseline_dir: Path,
    config_path: Path = DEFAULT_CONFIG,
    repo_root: Path = Path("."),
) -> dict[str, object]:
    """Independently recompute all persisted P8-001 comparison evidence."""

    evaluation = load_evaluation_config(config_path)
    settings = load_frozen_evaluation_settings()
    base_model = load_baseline_base_model_identity(Path(evaluation.base_config))
    system_version, system_prompt = validate_contract(evaluation, settings, base_model)
    source_git_sha, _ = _preflight_source_tree(repo_root)
    output_dir = Path(evaluation.output_dir)

    manifest = read_json(output_dir / EVALUATION_MANIFEST, context="P8-001 evaluation manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("task_id") != TASK_ID:
        raise PythonP0EvaluationError("P8-001 evaluation manifest identity is invalid")
    if manifest.get("source_git_sha") != source_git_sha:
        raise PythonP0EvaluationError("P8-001 evaluation source SHA drifted")
    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, list) or len(raw_artifacts) != len(FINAL_ARTIFACTS):
        raise PythonP0EvaluationError("P8-001 final artifact inventory is invalid")
    observed: list[dict[str, str]] = []
    for index, raw in enumerate(raw_artifacts):
        item = strict_mapping(raw, context=f"evaluation_manifest.artifacts[{index}]")
        path = item.get("path")
        digest = item.get("sha256")
        if not isinstance(path, str) or not isinstance(digest, str):
            raise PythonP0EvaluationError("P8-001 final artifact entry is invalid")
        if path != FINAL_ARTIFACTS[index] or file_sha256(output_dir / path) != digest:
            raise PythonP0EvaluationError(f"P8-001 final artifact drift detected: {path!r}")
        observed.append({"path": path, "sha256": digest})
    if manifest.get("artifact_set_sha256") != artifact_set_sha256(observed):
        raise PythonP0EvaluationError("P8-001 final artifact-set fingerprint mismatch")

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
    persisted = read_json(output_dir / COMPARISON, context="P8-001 comparison")
    if persisted.get("schema_version") != SCHEMA_VERSION or persisted.get("task_id") != TASK_ID:
        raise PythonP0EvaluationError("P8-001 comparison identity is invalid")
    if persisted.get("evaluation_complete") is not True:
        raise PythonP0EvaluationError("P8-001 comparison is not complete")
    if persisted.get("source_git_sha") != source_git_sha:
        raise PythonP0EvaluationError("P8-001 comparison source SHA drifted")
    expected_adapter = {
        "family": EXPECTED_ADAPTER_FAMILY,
        "adapter_id": EXPECTED_ADAPTER_ID,
        "adapter_model_sha256": EXPECTED_ADAPTER_SHA256,
        "adapter_model_size_bytes": EXPECTED_ADAPTER_SIZE_BYTES,
        "training_run_id": EXPECTED_TRAINING_RUN_ID,
        "training_git_sha": EXPECTED_TRAINING_GIT_SHA,
    }
    if persisted.get("adapter") != expected_adapter:
        raise PythonP0EvaluationError("P8-001 comparison adapter identity is invalid")
    if persisted.get("generation_contract_sha256") != contract:
        raise PythonP0EvaluationError("P8-001 comparison generation contract is invalid")
    if persisted.get("baseline") != {
        "baseline_id": EXPECTED_BASELINE_ID,
        "source_git_sha": EXPECTED_BASELINE_SOURCE_SHA,
        "artifact_set_sha256": EXPECTED_BASELINE_ARTIFACT_SET_SHA256,
    }:
        raise PythonP0EvaluationError("P8-001 comparison baseline identity is invalid")

    expected = comparison(
        source_git_sha=source_git_sha,
        base_model=base_model,
        adapter=adapter,
        settings=settings,
        generation_contract=contract,
        baseline_manifest=baseline_manifest,
        baseline=baseline,
        adapted=_adapted_aggregates(output_dir, base_model),
    )
    if persisted != expected:
        raise PythonP0EvaluationError("P8-001 comparison values do not match persisted aggregates")
    return persisted
