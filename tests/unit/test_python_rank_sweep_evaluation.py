"""P9-001 protected rank-evaluation contract tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tiny_qwen_coder.config import load_evaluation_config
from tiny_qwen_coder.evaluation.python_rank_sweep import (
    REGISTRY_PATH,
    PythonRankSweepEvaluationError,
    load_rank_candidate_registry,
)


def test_rank_candidate_registry_freezes_completed_training_identities() -> None:
    registry = load_rank_candidate_registry()

    assert registry.training_workflow_run_id == 33587975474
    assert registry.training_source_git_sha == "1aca2f2a778ade4cc300d66353a26206900fc84b"
    assert registry.dataset_manifest_id == "dataset/python/p0"
    assert (
        registry.dataset_config_sha256
        == "4f9663e72b22d81ce8975e6f6ed87ee7457d3bef0a08fe211e700dd5ea12fbff"
    )
    assert (
        registry.dataset_split_membership_sha256
        == "78559765eac305528e5ba96ae3dae04feffda39fc4727d450d602f6e68697428"
    )
    assert registry.contamination_status == "not_run"
    assert tuple(item.rank for item in registry.candidates) == (8, 16, 32, 64)

    expected = {
        8: (
            "language/python/p9-rank-r8",
            "31774b09bfdf638c5d6af23388dcb2426c7647684643dd3200f68f2d964fd808",
            32_539_280,
            9857352234,
        ),
        16: (
            "language/python/p0",
            "c94606250e112f72362eb883a55f7b2c8af854d445f6bb6194352c2806a8f276",
            65_004_840,
            9789946698,
        ),
        32: (
            "language/python/p9-rank-r32",
            "332886a5e3d02dc2c1a0c7c3d8f2d2765fb5327663f7a146b5d73c7c07937753",
            129_934_944,
            9880833842,
        ),
        64: (
            "language/python/p9-rank-r64",
            "603a0ac7c00ceb8d428d0254330a9ec329eb598ef332be48d16a071d43e69918",
            259_794_944,
            9910556769,
        ),
    }
    for rank, values in expected.items():
        candidate = registry.candidate(rank)
        assert (
            candidate.adapter_id,
            candidate.adapter_model_sha256,
            candidate.adapter_model_size_bytes,
            candidate.training_artifact_id,
        ) == values

    rank16 = registry.candidate(16)
    assert rank16.baseline_control is True
    assert rank16.evaluation_config is None
    assert rank16.canonical_evaluation_run_id == 33538724658


def test_new_rank_evaluation_configs_change_only_adapter_identity_and_output() -> None:
    baseline = load_evaluation_config(Path("configs/eval/python/p0_v1.yaml"))
    registry = load_rank_candidate_registry()

    for rank in (8, 32, 64):
        candidate = registry.candidate(rank)
        assert candidate.evaluation_config is not None
        config = load_evaluation_config(Path(candidate.evaluation_config))
        assert config.base_config == baseline.base_config
        assert config.language == baseline.language == "python"
        assert config.suites == baseline.suites
        assert config.seed == baseline.seed == 1729
        assert config.generation == baseline.generation
        assert config.execution == baseline.execution
        assert config.adapter_id == candidate.adapter_id
        assert config.output_dir == f"artifacts/eval/python/p9-rank-r{rank}-v1"


def test_rank_registry_rejects_unknown_fields(tmp_path: Path) -> None:
    payload = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    candidates = payload["candidates"]
    assert isinstance(candidates, list)
    first = candidates[0]
    assert isinstance(first, dict)
    first["unreviewed_field"] = True
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(PythonRankSweepEvaluationError, match="unknown fields"):
        load_rank_candidate_registry(path)


def test_rank_registry_rejects_training_sha_drift(tmp_path: Path) -> None:
    payload = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    candidates = payload["candidates"]
    assert isinstance(candidates, list)
    rank32 = candidates[2]
    assert isinstance(rank32, dict)
    rank32["training_git_sha"] = "0" * 64
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(PythonRankSweepEvaluationError, match="training SHA drifted"):
        load_rank_candidate_registry(path)
