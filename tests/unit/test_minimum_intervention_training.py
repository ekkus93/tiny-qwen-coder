"""CPU regression tests for P9-004B trajectory training contracts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tiny_qwen_coder.training.minimum_intervention import MinimumInterventionError
from tiny_qwen_coder.training.minimum_intervention_training import (
    _AdapterSnapshotCallback,
    _snapshot_evidence,
    resolve_minimum_intervention_training_candidate,
)
from tiny_qwen_coder.training.plan import AdapterTrainingError


def test_resolves_each_frozen_low_lr_candidate() -> None:
    expected = {
        "lr-1e-5": (0.00001, "language/python/p9-min-lr-1e5"),
        "lr-2e-5": (0.00002, "language/python/p9-min-lr-2e5"),
        "lr-5e-5": (0.00005, "language/python/p9-min-lr-5e5"),
        "lr-1e-4": (0.0001, "language/python/p9-min-lr-1e4"),
        "lr-2e-4": (0.0002, "language/python/p9-min-lr-2e4"),
    }

    for label, (learning_rate, adapter_id) in expected.items():
        validation, candidate, plan = resolve_minimum_intervention_training_candidate(label)
        assert validation.trajectory_max_steps == 1000
        assert validation.checkpoint_steps == (50, 100, 250, 500, 1000)
        assert candidate.learning_rate == learning_rate
        assert candidate.adapter_id == adapter_id
        assert plan.config.learning_rate == learning_rate
        assert plan.config.lora.rank == 8


def test_rejects_non_frozen_candidate() -> None:
    with pytest.raises(MinimumInterventionError, match="is not frozen"):
        resolve_minimum_intervention_training_candidate("lr-3e-5")


class _FakeAdapterModel:
    def __init__(self) -> None:
        self.saved: list[Path] = []

    def save_pretrained(self, path: str, *, safe_serialization: bool) -> None:
        assert safe_serialization is True
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "adapter_config.json").write_text('{"peft_type":"LORA"}\n', encoding="utf-8")
        (destination / "adapter_model.safetensors").write_bytes(b"adapter")
        self.saved.append(destination)


def test_snapshot_callback_saves_only_precommitted_steps(tmp_path: Path) -> None:
    root = tmp_path / "snapshots"
    root.mkdir()
    callback = _AdapterSnapshotCallback(
        snapshot_root=root,
        steps=(50, 100, 250, 500, 1000),
    )
    model = _FakeAdapterModel()
    control = object()

    for step in (1, 49, 50, 51, 100, 100, 249, 250, 500, 999, 1000):
        returned = callback.on_step_end(
            object(), SimpleNamespace(global_step=step), control, model=model
        )
        assert returned is control

    assert callback.saved_steps == (50, 100, 250, 500, 1000)
    assert tuple(path.name for path in model.saved) == (
        "step-0050",
        "step-0100",
        "step-0250",
        "step-0500",
        "step-1000",
    )


def test_snapshot_evidence_fails_closed_on_full_model_weight(tmp_path: Path) -> None:
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    for step in (50, 100, 250, 500, 1000):
        directory = snapshots / f"step-{step:04d}"
        directory.mkdir()
        (directory / "adapter_config.json").write_text('{"peft_type":"LORA"}\n', encoding="utf-8")
        (directory / "adapter_model.safetensors").write_bytes(b"adapter")
    (snapshots / "step-0250" / "model.safetensors").write_bytes(b"forbidden")

    with pytest.raises(AdapterTrainingError, match="merged/full-model"):
        _snapshot_evidence(tmp_path)
