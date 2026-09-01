"""P8-002 general/tool regression evaluation for the accepted Python P0 adapter."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

from tiny_qwen_coder.config import EvaluationConfig, load_evaluation_config
from tiny_qwen_coder.evaluation._baseline_artifacts import (
    file_sha256,
    validate_python_baseline_artifacts,
)
from tiny_qwen_coder.evaluation._baseline_generation import BaselineGenerator
from tiny_qwen_coder.evaluation._baseline_provenance import load_baseline_base_model_identity
from tiny_qwen_coder.evaluation._baseline_runner import (
    _generate_items,
    _preflight_source_tree,
)
from tiny_qwen_coder.evaluation._baseline_types import BaselineGeneratedResponse
from tiny_qwen_coder.evaluation._python_p0_contract import (
    DEFAULT_CONFIG,
    EXPECTED_BASELINE_ARTIFACT_SET_SHA256,
    EXPECTED_BASELINE_ID,
    EXPECTED_BASELINE_SOURCE_SHA,
    PersistedAdapterEvidence,
    adapter_generation_contract,
    artifact_set_sha256,
    validate_contract,
)
from tiny_qwen_coder.evaluation._python_p0_generation import (
    EXPECTED_ADAPTER_FAMILY,
    EXPECTED_ADAPTER_ID,
    EXPECTED_ADAPTER_SHA256,
    EXPECTED_ADAPTER_SIZE_BYTES,
    EXPECTED_TRAINING_GIT_SHA,
    EXPECTED_TRAINING_RUN_ID,
    HuggingFacePythonP0Generator,
    validate_python_p0_adapter,
)
from tiny_qwen_coder.evaluation.regression import (
    RegressionCategory,
    RegressionSuite,
    load_frozen_general_tool_regression_suite,
    regression_suite_sha256,
    score_regression_suite,
)
from tiny_qwen_coder.evaluation.settings import (
    FrozenEvaluationSettings,
    evaluation_settings_sha256,
    load_frozen_evaluation_settings,
)
from tiny_qwen_coder.identities import BaseModelIdentity

TASK_ID = "P8-002"
SCHEMA_VERSION = 1
OUTPUT_DIR = Path("artifacts/eval/python/p0-general-tool-regression-v1")
CHECKPOINT = Path(".baseline-work/general-tool-regression-generation.jsonl")
STAGE_MANIFEST = Path("generation-stage.json")
COMPARISON = Path("comparison.json")
EVIDENCE_MANIFEST = Path("evaluation-manifest.json")
EXPECTED_SUITE_SHA256 = "9de462c05a05455b2cc5af8c0246d897fe7991510d470e837d205540922239f9"
EXPECTED_CASES = 12


class PythonP0RegressionError(RuntimeError):
    """Raised when P8-002 evidence is incomplete, stale, or inconsistent."""


class _NoGenerate(BaselineGenerator):
    def generate(self, *, system_prompt: str, user_prompt: str) -> BaselineGeneratedResponse:
        del system_prompt, user_prompt
        raise PythonP0RegressionError(
            "P8-002 scoring is missing a required GPU generation checkpoint; refusing regeneration"
        )


def _mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise PythonP0RegressionError(f"{context} must be a JSON object")
    output: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise PythonP0RegressionError(f"{context} keys must be strings")
        output[key] = item
    return output


def _read_json(path: Path, *, context: str) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PythonP0RegressionError(f"could not read {context}: {path}") from exc
    return _mapping(value, context=context)


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _context() -> tuple[
    EvaluationConfig, FrozenEvaluationSettings, BaseModelIdentity, str, str, RegressionSuite
]:
    evaluation = load_evaluation_config(DEFAULT_CONFIG)
    settings = load_frozen_evaluation_settings()
    base_model = load_baseline_base_model_identity(Path(evaluation.base_config))
    system_version, system_prompt = validate_contract(evaluation, settings, base_model)
    suite = load_frozen_general_tool_regression_suite()
    digest = regression_suite_sha256(suite)
    if digest != EXPECTED_SUITE_SHA256 or len(suite.cases) != EXPECTED_CASES:
        raise PythonP0RegressionError("frozen general/tool regression suite identity drifted")
    return evaluation, settings, base_model, system_version, system_prompt, suite


def _adapter_identity() -> dict[str, object]:
    return {
        "family": EXPECTED_ADAPTER_FAMILY,
        "adapter_id": EXPECTED_ADAPTER_ID,
        "adapter_model_sha256": EXPECTED_ADAPTER_SHA256,
        "adapter_model_size_bytes": EXPECTED_ADAPTER_SIZE_BYTES,
        "training_run_id": EXPECTED_TRAINING_RUN_ID,
        "training_git_sha": EXPECTED_TRAINING_GIT_SHA,
    }


def _validate_stage(
    *, source_git_sha: str, generation_contract: str, settings_sha256: str
) -> dict[str, object]:
    stage = _read_json(OUTPUT_DIR / STAGE_MANIFEST, context="P8-002 generation stage")
    expected_keys = {
        "schema_version",
        "task_id",
        "source_git_sha",
        "suite_id",
        "suite_version",
        "suite_sha256",
        "case_count",
        "evaluation_settings_sha256",
        "generation_contract_sha256",
        "adapter",
        "checkpoint",
    }
    if set(stage) != expected_keys:
        raise PythonP0RegressionError("P8-002 generation-stage schema drifted")
    if stage.get("schema_version") != SCHEMA_VERSION or stage.get("task_id") != TASK_ID:
        raise PythonP0RegressionError("P8-002 generation-stage identity is invalid")
    if stage.get("source_git_sha") != source_git_sha:
        raise PythonP0RegressionError("P8-002 generation-stage source SHA drifted")
    if stage.get("suite_id") != "general_tool_regression" or stage.get("suite_version") != 1:
        raise PythonP0RegressionError("P8-002 frozen suite version drifted")
    if stage.get("suite_sha256") != EXPECTED_SUITE_SHA256 or stage.get("case_count") != 12:
        raise PythonP0RegressionError("P8-002 frozen suite fingerprint/count drifted")
    if stage.get("evaluation_settings_sha256") != settings_sha256:
        raise PythonP0RegressionError("P8-002 frozen evaluation-settings hash drifted")
    if stage.get("generation_contract_sha256") != generation_contract:
        raise PythonP0RegressionError("P8-002 generation contract drifted")
    if stage.get("adapter") != _adapter_identity():
        raise PythonP0RegressionError("P8-002 adapter identity drifted")
    checkpoint = _mapping(stage.get("checkpoint"), context="generation_stage.checkpoint")
    if checkpoint.get("path") != CHECKPOINT.as_posix():
        raise PythonP0RegressionError("P8-002 checkpoint path drifted")
    digest = checkpoint.get("sha256")
    if not isinstance(digest, str) or file_sha256(OUTPUT_DIR / CHECKPOINT) != digest:
        raise PythonP0RegressionError("P8-002 checkpoint hash drifted")
    return stage


def generate(training_output: Path) -> dict[str, object]:
    """Generate all 12 frozen P8-002 responses on CUDA with the exact accepted P0 adapter."""

    evaluation, settings, base_model, system_version, system_prompt, suite = _context()
    source_git_sha, _ = _preflight_source_tree(Path("."))
    adapter = validate_python_p0_adapter(training_output, base_model)
    contract = adapter_generation_contract(
        base_model=base_model,
        settings=settings,
        system_prompt_version=system_version,
        system_prompt=system_prompt,
        adapter=adapter,
    )
    generator = HuggingFacePythonP0Generator(
        training_output=training_output,
        base_model=base_model,
        settings=settings,
    )
    prompts = tuple((case.id, case.prompt) for case in suite.cases)
    responses = _generate_items(
        suite_id="general-tool-regression",
        prompts=prompts,
        generator=generator,
        system_prompt=system_prompt,
        generation_contract=contract,
        output_dir=OUTPUT_DIR,
    )
    if len(responses) != EXPECTED_CASES:
        raise PythonP0RegressionError("P8-002 did not generate exactly 12 responses")
    checkpoint_path = OUTPUT_DIR / CHECKPOINT
    stage: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "source_git_sha": source_git_sha,
        "suite_id": suite.suite_id,
        "suite_version": suite.suite_version,
        "suite_sha256": regression_suite_sha256(suite),
        "case_count": len(suite.cases),
        "evaluation_settings_sha256": evaluation_settings_sha256(settings),
        "generation_contract_sha256": contract,
        "adapter": _adapter_identity(),
        "checkpoint": {
            "path": CHECKPOINT.as_posix(),
            "sha256": file_sha256(checkpoint_path),
        },
    }
    _write_json(OUTPUT_DIR / STAGE_MANIFEST, stage)
    return stage


def _baseline_cases(baseline_dir: Path) -> dict[str, dict[str, object]]:
    path = baseline_dir / "general-tool-regression/general-tool-regression-results.jsonl"
    output: dict[str, dict[str, object]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PythonP0RegressionError("could not read P6 regression case results") from exc
    if len(lines) != EXPECTED_CASES:
        raise PythonP0RegressionError("P6 regression case count drifted")
    for index, line in enumerate(lines, start=1):
        try:
            raw: object = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PythonP0RegressionError(f"invalid P6 regression result line {index}") from exc
        item = _mapping(raw, context=f"P6 regression result {index}")
        case_id = item.get("case_id")
        category = item.get("category")
        passed = item.get("passed")
        generated_text = item.get("generated_text")
        detail = item.get("detail")
        if not isinstance(case_id, str) or not isinstance(category, str):
            raise PythonP0RegressionError("P6 regression result identity is invalid")
        if not isinstance(passed, bool) or not isinstance(generated_text, str):
            raise PythonP0RegressionError("P6 regression result score/text is invalid")
        if detail is not None and not isinstance(detail, str):
            raise PythonP0RegressionError("P6 regression result detail is invalid")
        if case_id in output:
            raise PythonP0RegressionError(f"duplicate P6 regression case: {case_id}")
        output[case_id] = {
            "category": category,
            "passed": passed,
            "generated_text": generated_text,
            "detail": detail,
        }
    return output


def _build_comparison(*, baseline_dir: Path) -> dict[str, object]:
    evaluation, settings, base_model, system_version, system_prompt, suite = _context()
    source_git_sha, _ = _preflight_source_tree(Path("."))
    persisted_adapter = PersistedAdapterEvidence(
        adapter_id=EXPECTED_ADAPTER_ID,
        family=EXPECTED_ADAPTER_FAMILY,
        adapter_model_sha256=EXPECTED_ADAPTER_SHA256,
        adapter_model_size_bytes=EXPECTED_ADAPTER_SIZE_BYTES,
        training_run_id=EXPECTED_TRAINING_RUN_ID,
        training_git_sha=EXPECTED_TRAINING_GIT_SHA,
    )
    contract = adapter_generation_contract(
        base_model=base_model,
        settings=settings,
        system_prompt_version=system_version,
        system_prompt=system_prompt,
        adapter=persisted_adapter,
    )
    _validate_stage(
        source_git_sha=source_git_sha,
        generation_contract=contract,
        settings_sha256=evaluation_settings_sha256(settings),
    )

    prompts = tuple((case.id, case.prompt) for case in suite.cases)
    responses = _generate_items(
        suite_id="general-tool-regression",
        prompts=prompts,
        generator=_NoGenerate(),
        system_prompt=system_prompt,
        generation_contract=contract,
        output_dir=OUTPUT_DIR,
    )
    response_map = {
        case.id: response.generated_text
        for case, response in zip(suite.cases, responses, strict=True)
    }
    adapter_score = score_regression_suite(suite, response_map)

    baseline_manifest = validate_python_baseline_artifacts(baseline_dir)
    if (
        baseline_manifest.baseline_id != EXPECTED_BASELINE_ID
        or baseline_manifest.source_git_sha != EXPECTED_BASELINE_SOURCE_SHA
        or baseline_manifest.artifact_set_sha256 != EXPECTED_BASELINE_ARTIFACT_SET_SHA256
        or baseline_manifest.base_model != base_model
    ):
        raise PythonP0RegressionError("P8-002 baseline is not the accepted P6-005 artifact")
    baseline = _baseline_cases(baseline_dir)
    expected_ids = {case.id for case in suite.cases}
    if set(baseline) != expected_ids:
        raise PythonP0RegressionError("P6 regression case IDs drifted from frozen suite")

    cases: list[dict[str, object]] = []
    category_rows: list[dict[str, object]] = []
    for category in RegressionCategory:
        category_cases = [case for case in suite.cases if case.category is category]
        base_passed = sum(bool(baseline[case.id]["passed"]) for case in category_cases)
        adapter_passed = sum(
            result.passed for result in adapter_score.cases if result.category is category
        )
        category_rows.append(
            {
                "category": category.value,
                "total": len(category_cases),
                "base_passed": base_passed,
                "adapter_passed": adapter_passed,
                "delta_passed": adapter_passed - base_passed,
            }
        )

    adapter_by_id = {result.case_id: result for result in adapter_score.cases}
    regressions = 0
    improvements = 0
    for case in suite.cases:
        base_item = baseline[case.id]
        adapter_item = adapter_by_id[case.id]
        base_passed = bool(base_item["passed"])
        if base_passed and not adapter_item.passed:
            transition = "regression"
            regressions += 1
        elif not base_passed and adapter_item.passed:
            transition = "improvement"
            improvements += 1
        elif base_passed:
            transition = "preserved_pass"
        else:
            transition = "preserved_fail"
        cases.append(
            {
                "case_id": case.id,
                "category": case.category.value,
                "base_passed": base_passed,
                "adapter_passed": adapter_item.passed,
                "transition": transition,
                "base_generated_text": base_item["generated_text"],
                "adapter_generated_text": response_map[case.id],
                "base_detail": base_item["detail"],
                "adapter_detail": adapter_item.detail,
            }
        )

    base_passed_total = sum(bool(item["passed"]) for item in baseline.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "source_git_sha": source_git_sha,
        "suite": {
            "suite_id": suite.suite_id,
            "suite_version": suite.suite_version,
            "suite_sha256": regression_suite_sha256(suite),
            "total_cases": len(suite.cases),
        },
        "base_model": asdict(base_model),
        "adapter": _adapter_identity(),
        "baseline": {
            "baseline_id": baseline_manifest.baseline_id,
            "source_git_sha": baseline_manifest.source_git_sha,
            "artifact_set_sha256": baseline_manifest.artifact_set_sha256,
        },
        "evaluation_settings_sha256": evaluation_settings_sha256(settings),
        "generation_contract_sha256": contract,
        "overall": {
            "total_cases": EXPECTED_CASES,
            "base_passed": base_passed_total,
            "adapter_passed": adapter_score.passed,
            "delta_passed": adapter_score.passed - base_passed_total,
            "regressions": regressions,
            "improvements": improvements,
        },
        "categories": category_rows,
        "cases": cases,
    }


def score(baseline_dir: Path) -> dict[str, object]:
    """Score transported responses and persist the direct P6-vs-P0 regression comparison."""

    comparison = _build_comparison(baseline_dir=baseline_dir)
    _write_json(OUTPUT_DIR / COMPARISON, comparison)
    artifacts = [
        {"path": STAGE_MANIFEST.as_posix(), "sha256": file_sha256(OUTPUT_DIR / STAGE_MANIFEST)},
        {"path": CHECKPOINT.as_posix(), "sha256": file_sha256(OUTPUT_DIR / CHECKPOINT)},
        {"path": COMPARISON.as_posix(), "sha256": file_sha256(OUTPUT_DIR / COMPARISON)},
    ]
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "source_git_sha": comparison["source_git_sha"],
        "artifacts": artifacts,
        "artifact_set_sha256": artifact_set_sha256(artifacts),
    }
    _write_json(OUTPUT_DIR / EVIDENCE_MANIFEST, manifest)
    return comparison


def verify(baseline_dir: Path) -> dict[str, object]:
    """Rehash all P8-002 evidence and independently recompute the comparison."""

    manifest = _read_json(OUTPUT_DIR / EVIDENCE_MANIFEST, context="P8-002 evidence manifest")
    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, Sequence) or isinstance(raw_artifacts, str):
        raise PythonP0RegressionError("P8-002 evidence artifact list is invalid")
    observed: list[dict[str, str]] = []
    expected_paths = (STAGE_MANIFEST.as_posix(), CHECKPOINT.as_posix(), COMPARISON.as_posix())
    if len(raw_artifacts) != len(expected_paths):
        raise PythonP0RegressionError("P8-002 evidence artifact count is invalid")
    for index, raw in enumerate(raw_artifacts):
        item = _mapping(raw, context=f"P8-002 artifact {index}")
        path = item.get("path")
        digest = item.get("sha256")
        if path != expected_paths[index] or not isinstance(digest, str):
            raise PythonP0RegressionError("P8-002 evidence artifact identity is invalid")
        if file_sha256(OUTPUT_DIR / path) != digest:
            raise PythonP0RegressionError(f"P8-002 evidence hash drifted: {path}")
        observed.append({"path": path, "sha256": digest})
    if manifest.get("artifact_set_sha256") != artifact_set_sha256(observed):
        raise PythonP0RegressionError("P8-002 artifact-set hash drifted")
    persisted = _read_json(OUTPUT_DIR / COMPARISON, context="P8-002 comparison")
    expected = _build_comparison(baseline_dir=baseline_dir)
    if persisted != expected:
        raise PythonP0RegressionError("P8-002 comparison does not match recomputed evidence")
    if manifest.get("source_git_sha") != expected["source_git_sha"]:
        raise PythonP0RegressionError("P8-002 evidence source SHA drifted")
    return persisted


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run/verify P8-002 general/tool regression")
    sub = parser.add_subparsers(dest="command", required=True)
    generate_parser = sub.add_parser("generate")
    generate_parser.add_argument("--training-output", type=Path, required=True)
    score_parser = sub.add_parser("score")
    score_parser.add_argument("--baseline-dir", type=Path, required=True)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--baseline-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "generate":
        payload = generate(args.training_output)
    elif args.command == "score":
        payload = score(args.baseline_dir)
    else:
        payload = verify(args.baseline_dir)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
