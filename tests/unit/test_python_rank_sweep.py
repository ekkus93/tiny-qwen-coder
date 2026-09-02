"""CPU-only contract tests for the P9-001 Python LoRA rank sweep."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from tiny_qwen_coder.training.rank_sweep import (
    EXPECTED_FIXED_PAYLOAD_SHA256,
    EXPECTED_PROTOCOL_SHA256,
    RankSweepError,
    validate_rank_sweep,
)

_PROTOCOL = Path("configs/train/python/p9_rank_sweep_v1.yaml")
_FILES = (
    Path("configs/train/python/p0.yaml"),
    _PROTOCOL,
    Path("configs/train/python/p9_rank_r8.yaml"),
    Path("configs/train/python/p9_rank_r8_smoke.yaml"),
    Path("configs/train/python/p9_rank_r32.yaml"),
    Path("configs/train/python/p9_rank_r32_smoke.yaml"),
    Path("configs/train/python/p9_rank_r64.yaml"),
    Path("configs/train/python/p9_rank_r64_smoke.yaml"),
)


def _copy_protocol_tree(root: Path) -> None:
    for source in _FILES:
        target = root / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def test_rank_sweep_freezes_rank_as_the_only_training_variable() -> None:
    report = validate_rank_sweep()

    assert report.sweep_id == "python-p9-rank-v1"
    assert report.baseline_rank == 16
    assert report.baseline_config_path == "configs/train/python/p0.yaml"
    assert report.protocol_sha256 == EXPECTED_PROTOCOL_SHA256
    assert report.fixed_payload_sha256 == EXPECTED_FIXED_PAYLOAD_SHA256
    assert tuple(candidate.rank for candidate in report.candidates) == (8, 16, 32, 64)
    assert tuple(candidate.baseline for candidate in report.candidates) == (
        False,
        True,
        False,
        False,
    )
    assert tuple(candidate.adapter_id for candidate in report.candidates) == (
        "language/python/p9-rank-r8",
        "language/python/p0",
        "language/python/p9-rank-r32",
        "language/python/p9-rank-r64",
    )
    assert tuple(candidate.output_dir for candidate in report.candidates) == (
        "artifacts/train/python/p9-rank-r8",
        "artifacts/train/python/p0",
        "artifacts/train/python/p9-rank-r32",
        "artifacts/train/python/p9-rank-r64",
    )
    assert all(len(candidate.config_sha256) == 64 for candidate in report.candidates)


def test_rank_sweep_keeps_alpha_and_every_non_rank_hyperparameter_fixed() -> None:
    payloads: list[dict[str, object]] = []
    for path in (
        Path("configs/train/python/p9_rank_r8.yaml"),
        Path("configs/train/python/p0.yaml"),
        Path("configs/train/python/p9_rank_r32.yaml"),
        Path("configs/train/python/p9_rank_r64.yaml"),
    ):
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(loaded, dict)
        payloads.append(loaded)

    for rank, payload in zip((8, 16, 32, 64), payloads, strict=True):
        lora = payload["lora"]
        assert isinstance(lora, dict)
        assert lora["rank"] == rank
        assert lora["alpha"] == 32
        assert payload["learning_rate"] == 0.0002
        assert payload["epochs"] == 1.0
        assert payload["sequence_length"] == 2048
        assert payload["micro_batch_size"] == 1
        assert payload["gradient_accumulation_steps"] == 8
        assert payload["dataset_manifest"] == "data/python/p0/dataset-manifest.json"
        assert payload["train_records"] == "data/python/p0/train.jsonl"
        assert payload["validation_records"] == "data/python/p0/validation.jsonl"


def test_rank_sweep_rejects_learning_rate_drift(tmp_path: Path) -> None:
    _copy_protocol_tree(tmp_path)
    path = tmp_path / "configs/train/python/p9_rank_r8.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    payload["learning_rate"] = 0.0001
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(RankSweepError, match="other than LoRA rank"):
        validate_rank_sweep(repo_root=tmp_path)


def test_rank_sweep_rejects_alpha_scaling(tmp_path: Path) -> None:
    _copy_protocol_tree(tmp_path)
    path = tmp_path / "configs/train/python/p9_rank_r64.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    lora = payload["lora"]
    assert isinstance(lora, dict)
    lora["alpha"] = 128
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(RankSweepError, match="other than LoRA rank"):
        validate_rank_sweep(repo_root=tmp_path)


def test_rank_sweep_smoke_configs_are_one_step_and_candidate_bound() -> None:
    report = validate_rank_sweep()

    baseline = next(candidate for candidate in report.candidates if candidate.rank == 16)
    assert baseline.smoke_config_path is None
    for candidate in report.candidates:
        if candidate.rank == 16:
            continue
        assert candidate.smoke_config_path == (
            f"configs/train/python/p9_rank_r{candidate.rank}_smoke.yaml"
        )


def test_rank_sweep_workflows_are_manual_and_do_not_retrain_rank_16() -> None:
    for path in (
        Path(".github/workflows/python-p9-rank-sweep-smoke.yml"),
        Path(".github/workflows/python-p9-rank-sweep-training.yml"),
    ):
        text = path.read_text(encoding="utf-8")
        assert "workflow_dispatch:" in text
        assert "push:" not in text
        assert "rank: [8, 32, 64]" in text
        assert "max-parallel: 1" in text
        assert "rank: [8, 16, 32, 64]" not in text
        assert "configs/train/python/p9_rank_r${RANK}.yaml" in text


def test_full_training_workflow_verifies_rank_only_evidence() -> None:
    text = Path(".github/workflows/python-p9-rank-sweep-training.yml").read_text(encoding="utf-8")

    assert "report.get(\"global_steps\") != 4750" in text
    assert "lora.get(\"rank\") != rank" in text
    assert "lora.get(\"alpha\") != 32" in text
    assert "32_464_896 * rank // 16" in text
    assert "persisted artifact SHA mismatch" in text
