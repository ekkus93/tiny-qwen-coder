from __future__ import annotations

import json
from pathlib import Path

import pytest

from tiny_qwen_coder.evaluation import python_p0_regression as p8
from tiny_qwen_coder.evaluation._baseline_artifacts import file_sha256
from tiny_qwen_coder.evaluation._python_p0_contract import artifact_set_sha256
from tiny_qwen_coder.evaluation.regression import (
    load_frozen_general_tool_regression_suite,
    regression_suite_sha256,
)


def test_p8_002_pins_exact_frozen_suite_and_adapter() -> None:
    suite = load_frozen_general_tool_regression_suite()
    assert len(suite.cases) == 12
    assert regression_suite_sha256(suite) == p8.EXPECTED_SUITE_SHA256
    assert p8._adapter_identity() == {
        "family": "language",
        "adapter_id": "language/python/p0",
        "adapter_model_sha256": "c94606250e112f72362eb883a55f7b2c8af854d445f6bb6194352c2806a8f276",
        "adapter_model_size_bytes": 65_004_840,
        "training_run_id": "training-python-20260831T180916446466Z-02df92a9-eafc119d",
        "training_git_sha": "02df92a9c2d347b9fb013dc25714fe066c6bcafe",
    }


def test_p8_002_scoring_refuses_missing_generation() -> None:
    with pytest.raises(p8.PythonP0RegressionError, match="refusing regeneration"):
        p8._NoGenerate().generate(system_prompt="system", user_prompt="prompt")


def test_p8_002_baseline_case_reader_requires_exact_frozen_ids(tmp_path: Path) -> None:
    suite = load_frozen_general_tool_regression_suite()
    path = tmp_path / "general-tool-regression"
    path.mkdir()
    lines = []
    for case in suite.cases:
        lines.append(
            json.dumps(
                {
                    "case_id": case.id,
                    "category": case.category.value,
                    "generated_text": "candidate",
                    "passed": False,
                    "detail": "test",
                    "generation": {
                        "prompt_tokens": 1,
                        "generated_tokens": 1,
                        "latency_seconds": 0.1,
                        "tokens_per_second": 10.0,
                    },
                },
                sort_keys=True,
            )
        )
    (path / "general-tool-regression-results.jsonl").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    observed = p8._baseline_cases(tmp_path)
    assert set(observed) == {case.id for case in suite.cases}


def test_p8_002_stage_rehashes_checkpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(p8, "OUTPUT_DIR", tmp_path)
    checkpoint = tmp_path / p8.CHECKPOINT
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("canonical\n", encoding="utf-8")
    stage = {
        "schema_version": 1,
        "task_id": "P8-002",
        "source_git_sha": "a" * 40,
        "suite_id": "general_tool_regression",
        "suite_version": 1,
        "suite_sha256": p8.EXPECTED_SUITE_SHA256,
        "case_count": 12,
        "evaluation_settings_sha256": "settings",
        "generation_contract_sha256": "contract",
        "adapter": p8._adapter_identity(),
        "checkpoint": {
            "path": p8.CHECKPOINT.as_posix(),
            "sha256": file_sha256(checkpoint),
        },
    }
    p8._write_json(tmp_path / p8.STAGE_MANIFEST, stage)
    p8._validate_stage(
        source_git_sha="a" * 40, generation_contract="contract", settings_sha256="settings"
    )
    checkpoint.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(p8.PythonP0RegressionError, match="checkpoint hash drifted"):
        p8._validate_stage(
            source_git_sha="a" * 40, generation_contract="contract", settings_sha256="settings"
        )


def test_p8_002_evidence_manifest_rejects_tampering(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(p8, "OUTPUT_DIR", tmp_path)
    for relative in (p8.STAGE_MANIFEST, p8.CHECKPOINT, p8.COMPARISON):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    artifacts = [
        {"path": relative.as_posix(), "sha256": file_sha256(tmp_path / relative)}
        for relative in (p8.STAGE_MANIFEST, p8.CHECKPOINT, p8.COMPARISON)
    ]
    p8._write_json(
        tmp_path / p8.EVIDENCE_MANIFEST,
        {
            "schema_version": 1,
            "task_id": "P8-002",
            "source_git_sha": "a" * 40,
            "artifacts": artifacts,
            "artifact_set_sha256": artifact_set_sha256(artifacts),
        },
    )
    (tmp_path / p8.COMPARISON).write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(p8.PythonP0RegressionError, match="evidence hash drifted"):
        p8.verify(tmp_path)
