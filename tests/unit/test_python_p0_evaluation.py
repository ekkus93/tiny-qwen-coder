from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import cast

import pytest

from tiny_qwen_coder.evaluation._baseline_provenance import load_baseline_base_model_identity
from tiny_qwen_coder.evaluation._baseline_types import (
    BaselineArtifactDigest,
    PythonBaselineManifest,
)
from tiny_qwen_coder.evaluation._python_p0_comparison import aggregate, comparison
from tiny_qwen_coder.evaluation._python_p0_contract import (
    PersistedAdapterEvidence,
    PythonP0EvaluationError,
    adapter_from_stage,
    adapter_generation_contract,
)
from tiny_qwen_coder.evaluation._python_p0_generation import (
    EXPECTED_ADAPTER_FAMILY,
    EXPECTED_ADAPTER_ID,
    EXPECTED_ADAPTER_SHA256,
    EXPECTED_ADAPTER_SIZE_BYTES,
    EXPECTED_TRAINING_GIT_SHA,
    EXPECTED_TRAINING_RUN_ID,
)
from tiny_qwen_coder.evaluation.settings import load_frozen_evaluation_settings
from tiny_qwen_coder.identities import AdapterIdentity
from tiny_qwen_coder.languages.python import load_python_plugin

_BASE_CONFIG = Path("configs/base/qwen35-4b.yaml")


def _stage_adapter(*, digest: str = EXPECTED_ADAPTER_SHA256) -> dict[str, object]:
    return {
        "adapter": {
            "adapter_id": EXPECTED_ADAPTER_ID,
            "family": EXPECTED_ADAPTER_FAMILY,
            "adapter_model_sha256": digest,
            "adapter_model_size_bytes": EXPECTED_ADAPTER_SIZE_BYTES,
            "training_run_id": EXPECTED_TRAINING_RUN_ID,
            "training_git_sha": EXPECTED_TRAINING_GIT_SHA,
        }
    }


def _aggregate_payload(*, passed: int, total: int, adapter_id: str | None) -> dict[str, object]:
    return {
        "total_problems": total,
        "passed": passed,
        "failed": total - passed,
        "harness_errors": 0,
        "pass_at_1": passed / total,
        "base_model": asdict(load_baseline_base_model_identity(_BASE_CONFIG)),
        "adapter": {
            "family": EXPECTED_ADAPTER_FAMILY if adapter_id is not None else None,
            "adapter_id": adapter_id,
        },
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _baseline_manifest() -> PythonBaselineManifest:
    base = load_baseline_base_model_identity(_BASE_CONFIG)
    return PythonBaselineManifest(
        schema_version=1,
        baseline_id="python-unchanged-base",
        baseline_version=1,
        frozen=True,
        base_model=base,
        adapter=AdapterIdentity(family=None, adapter_id=None),
        source_git_sha="da537443ab80b1380bee0fc3c7d9d01ca0574f35",
        evaluation_config_sha256="1" * 64,
        evaluation_settings_sha256="2" * 64,
        system_prompt_version="python-v1",
        system_prompt_sha256="3" * 64,
        generation_contract_sha256="4" * 64,
        artifacts=(
            BaselineArtifactDigest(
                artifact_id="dummy",
                path="dummy.json",
                sha256="5" * 64,
            ),
        ),
        artifact_set_sha256="4bf616c3e84bdd74f8cf6467fc2d9d760f04d3b1b44660a81e365ff6f99a72fc",
    )


def test_adapter_generation_contract_is_bound_to_exact_adapter_weights() -> None:
    base = load_baseline_base_model_identity(_BASE_CONFIG)
    settings = load_frozen_evaluation_settings()
    prompt = load_python_plugin().spec.config.system_prompt
    canonical = adapter_from_stage(_stage_adapter())
    first = adapter_generation_contract(
        base_model=base,
        settings=settings,
        system_prompt_version=prompt.version,
        system_prompt=prompt.text,
        adapter=canonical,
    )

    altered_evidence = PersistedAdapterEvidence(
        adapter_id=canonical.adapter_id,
        family=canonical.family,
        adapter_model_sha256="0" * 64,
        adapter_model_size_bytes=canonical.adapter_model_size_bytes,
        training_run_id=canonical.training_run_id,
        training_git_sha=canonical.training_git_sha,
    )
    second = adapter_generation_contract(
        base_model=base,
        settings=settings,
        system_prompt_version=prompt.version,
        system_prompt=prompt.text,
        adapter=altered_evidence,
    )

    assert len(first) == 64
    assert first != second


def test_generation_stage_rejects_noncanonical_adapter_hash() -> None:
    raw_adapter = _stage_adapter()["adapter"]
    assert isinstance(raw_adapter, dict)
    altered = dict(raw_adapter)
    altered["adapter_model_sha256"] = "0" * 64
    with pytest.raises(PythonP0EvaluationError, match="adapter identity"):
        adapter_from_stage({"adapter": altered})


def test_aggregate_rejects_harness_errors(tmp_path: Path) -> None:
    base = load_baseline_base_model_identity(_BASE_CONFIG)
    payload = _aggregate_payload(passed=100, total=164, adapter_id=EXPECTED_ADAPTER_ID)
    payload["harness_errors"] = 1
    path = tmp_path / "aggregate.json"
    _write_json(path, payload)

    with pytest.raises(PythonP0EvaluationError, match="harness errors"):
        aggregate(path, suite_id="humaneval", base_model=base, adapter_id=EXPECTED_ADAPTER_ID)


def test_aggregate_rejects_wrong_adapter_identity(tmp_path: Path) -> None:
    base = load_baseline_base_model_identity(_BASE_CONFIG)
    payload = _aggregate_payload(passed=100, total=164, adapter_id="language/python/not-p0")
    path = tmp_path / "aggregate.json"
    _write_json(path, payload)

    with pytest.raises(PythonP0EvaluationError, match="adapter identity"):
        aggregate(path, suite_id="humaneval", base_model=base, adapter_id=EXPECTED_ADAPTER_ID)


def test_comparison_uses_exact_three_suite_micro_totals() -> None:
    base = load_baseline_base_model_identity(_BASE_CONFIG)
    settings = load_frozen_evaluation_settings()
    adapter = adapter_from_stage(_stage_adapter())
    baseline = {
        "humaneval": _aggregate_payload(passed=128, total=164, adapter_id=None),
        "mbpp": _aggregate_payload(passed=290, total=500, adapter_id=None),
        "repository-holdout": _aggregate_payload(passed=6, total=11, adapter_id=None),
    }
    adapted = {
        "humaneval": _aggregate_payload(passed=130, total=164, adapter_id=EXPECTED_ADAPTER_ID),
        "mbpp": _aggregate_payload(passed=300, total=500, adapter_id=EXPECTED_ADAPTER_ID),
        "repository-holdout": _aggregate_payload(
            passed=7, total=11, adapter_id=EXPECTED_ADAPTER_ID
        ),
    }

    result = comparison(
        source_git_sha="a" * 40,
        base_model=base,
        adapter=adapter,
        settings=settings,
        generation_contract="b" * 64,
        baseline_manifest=_baseline_manifest(),
        baseline=baseline,
        adapted=adapted,
    )

    suites = cast(list[dict[str, object]], result["suites"])
    assert suites[0]["suite_id"] == "humaneval"
    overall = cast(dict[str, object], result["overall"])
    assert overall == {
        "total_problems": 675,
        "base_passed": 424,
        "base_micro_pass_rate": 424 / 675,
        "adapter_passed": 437,
        "adapter_micro_pass_rate": 437 / 675,
        "delta_passed": 13,
        "delta_micro_pass_rate": 13 / 675,
    }
