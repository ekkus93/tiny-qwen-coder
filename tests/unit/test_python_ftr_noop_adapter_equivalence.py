"""CPU-only tests for FTR-102 no-op adapter equivalence controls."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from tiny_qwen_coder.evaluation.python_ftr_noop_adapter_equivalence import (
    FTRNoopAdapterError,
    _aggregate_comparison,
    _protocol_config,
    _result_comparison,
    inspect_zero_adapter,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _benchmark_row(*, text: str = "answer", passed: int = 1) -> dict[str, object]:
    return {
        "problem_id": "task/1",
        "generated_text": text,
        "generated_code": "def f(): return 1",
        "generation": {
            "prompt_tokens": 10,
            "generated_tokens": 4,
            "latency_seconds": 1.0,
            "tokens_per_second": 4.0,
        },
        "parse_status": "passed",
        "compile_status": "passed",
        "error_category": "none",
        "tests": {"passed": passed, "total": 1},
        "adapter": {"family": None, "adapter_id": None},
    }


def _write_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_protocol_freezes_expected_noop_loader_and_ftr101_reference() -> None:
    config = _protocol_config()
    adapter = config["noop_adapter"]
    reference = config["ftr_101_reference"]

    assert isinstance(adapter, dict)
    assert adapter["loader"] == "peft.PeftModel.from_pretrained"
    assert adapter["rank"] == 8
    assert adapter["alpha"] == 32
    assert adapter["require_all_saved_adapter_tensors_zero"] is True
    assert isinstance(reference, dict)
    assert reference["run_id"] == 34336465327
    assert reference["scores"]["combined"] == [424, 675]


def test_zero_adapter_inspection_accepts_only_exact_zero_tensors(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    _write_json(adapter_dir / "adapter_config.json", {"peft_type": "LORA"})
    save_file(
        {
            "base_model.model.layer.lora_A.weight": torch.zeros((2, 3)),
            "base_model.model.layer.lora_B.weight": torch.zeros((4, 2)),
        },
        adapter_dir / "adapter_model.safetensors",
    )

    identity = inspect_zero_adapter(adapter_dir)

    assert identity["tensor_count"] == 2
    assert identity["nonzero_tensor_count"] == 0
    assert identity["all_saved_adapter_tensors_zero"] is True


def test_zero_adapter_inspection_rejects_nonzero_tensor(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    _write_json(adapter_dir / "adapter_config.json", {"peft_type": "LORA"})
    save_file(
        {"base_model.model.layer.lora_A.weight": torch.ones((1, 1))},
        adapter_dir / "adapter_model.safetensors",
    )

    with pytest.raises(FTRNoopAdapterError, match="non-zero tensors"):
        inspect_zero_adapter(adapter_dir)


def test_task_comparison_ignores_timing_and_adapter_identity(tmp_path: Path) -> None:
    frozen = tmp_path / "frozen.jsonl"
    current = tmp_path / "current.jsonl"
    frozen_row = _benchmark_row()
    current_row = _benchmark_row()
    current_row["adapter"] = {"family": "control", "adapter_id": "noop"}
    current_generation = current_row["generation"]
    assert isinstance(current_generation, dict)
    current_generation["latency_seconds"] = 9.0
    current_generation["tokens_per_second"] = 0.4
    _write_jsonl(frozen, frozen_row)
    _write_jsonl(current, current_row)

    comparison = _result_comparison(
        frozen_path=frozen,
        current_path=current,
        suite_id="fixture",
    )

    assert comparison["exact_task_records"] is True
    assert comparison["mismatches"] == []


def test_task_comparison_detects_generated_text_drift(tmp_path: Path) -> None:
    frozen = tmp_path / "frozen.jsonl"
    current = tmp_path / "current.jsonl"
    _write_jsonl(frozen, _benchmark_row(text="correct"))
    _write_jsonl(current, _benchmark_row(text="changed"))

    comparison = _result_comparison(
        frozen_path=frozen,
        current_path=current,
        suite_id="fixture",
    )

    assert comparison["exact_task_records"] is False
    mismatches = comparison["mismatches"]
    assert isinstance(mismatches, list)
    assert mismatches[0]["changed_fields"] == ["generated_text"]


def test_aggregate_comparison_allows_only_adapter_and_result_hash_drift(tmp_path: Path) -> None:
    frozen = tmp_path / "frozen.json"
    current = tmp_path / "current.json"
    _write_json(
        frozen,
        {
            "adapter": {"family": None, "adapter_id": None},
            "results_sha256": "a",
            "passed": 5,
            "failed": 1,
            "dataset_revision": "fixed",
        },
    )
    _write_json(
        current,
        {
            "adapter": {"family": "control", "adapter_id": "noop"},
            "results_sha256": "b",
            "passed": 5,
            "failed": 1,
            "dataset_revision": "fixed",
        },
    )

    comparison = _aggregate_comparison(frozen, current, context="fixture")

    assert comparison == {"exact": True, "changed_fields": []}


def test_aggregate_comparison_rejects_score_drift(tmp_path: Path) -> None:
    frozen = tmp_path / "frozen.json"
    current = tmp_path / "current.json"
    _write_json(frozen, {"adapter": {}, "results_sha256": "a", "passed": 5})
    _write_json(current, {"adapter": {}, "results_sha256": "b", "passed": 4})

    comparison = _aggregate_comparison(frozen, current, context="fixture")

    assert comparison["exact"] is False
    assert comparison["changed_fields"] == ["passed"]
