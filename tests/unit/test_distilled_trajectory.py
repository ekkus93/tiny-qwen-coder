"""CPU regression tests for the P9-007C distilled trajectory contract."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from transformers import TrainerCallback

from tiny_qwen_coder.training.distilled_trajectory import validate_distilled_trajectory
from tiny_qwen_coder.training.distilled_trajectory_training import (
    _DistilledAdapterSnapshotCallback,
    _snapshot_evidence,
)
from tiny_qwen_coder.training.plan import AdapterTrainingError


def test_frozen_distilled_trajectory_is_exactly_one_data_pass() -> None:
    validation = validate_distilled_trajectory()

    assert validation.task_id == "P9-007C"
    assert validation.train_records == 1479
    assert validation.validation_records == 78
    assert validation.accepted_records == 1557
    assert validation.effective_batch_size == 8
    assert validation.derived_one_pass_steps == 185
    assert validation.trajectory_max_steps == 185
    assert validation.checkpoint_steps == (25, 50, 100, 185)
    assert validation.selection.minimum_combined_passed == 104
    assert validation.selection.minimum_humaneval_passed == 33
    assert validation.selection.minimum_mbpp_passed == 70
    assert validation.qualification.one_shot is True
    assert validation.qualification.repository_holdout_qualification_only is True


class _FakeAdapterModel:
    def __init__(self) -> None:
        self.saved: list[Path] = []

    def save_pretrained(self, path: str, *, safe_serialization: bool) -> None:
        assert safe_serialization is True
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "adapter_config.json").write_text(
            '{"peft_type":"LORA"}\n', encoding="utf-8"
        )
        (destination / "adapter_model.safetensors").write_bytes(b"adapter")
        self.saved.append(destination)


def test_distilled_snapshot_callback_implements_transformers_contract(tmp_path: Path) -> None:
    root = tmp_path / "snapshots"
    root.mkdir()
    callback = _DistilledAdapterSnapshotCallback(
        snapshot_root=root,
        steps=(25, 50, 100, 185),
    )

    assert isinstance(callback, TrainerCallback)
    for event_name in (
        "on_train_begin",
        "on_epoch_begin",
        "on_step_begin",
        "on_step_end",
        "on_epoch_end",
        "on_train_end",
    ):
        assert callable(getattr(callback, event_name))


def test_distilled_snapshot_callback_saves_only_frozen_steps(tmp_path: Path) -> None:
    root = tmp_path / "snapshots"
    root.mkdir()
    callback = _DistilledAdapterSnapshotCallback(
        snapshot_root=root,
        steps=(25, 50, 100, 185),
    )
    model = _FakeAdapterModel()
    control = object()

    for step in (1, 24, 25, 26, 50, 50, 99, 100, 184, 185):
        returned = callback.on_step_end(
            object(), SimpleNamespace(global_step=step), control, model=model
        )
        assert returned is control

    assert callback.saved_steps == (25, 50, 100, 185)
    assert tuple(path.name for path in model.saved) == (
        "step-0025",
        "step-0050",
        "step-0100",
        "step-0185",
    )


def test_distilled_snapshot_evidence_fails_closed_on_full_model_weight(tmp_path: Path) -> None:
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    for step in (25, 50, 100, 185):
        directory = snapshots / f"step-{step:04d}"
        directory.mkdir()
        (directory / "adapter_config.json").write_text(
            '{"peft_type":"LORA"}\n', encoding="utf-8"
        )
        (directory / "adapter_model.safetensors").write_bytes(b"adapter")
    (snapshots / "step-0100" / "model.safetensors").write_bytes(b"forbidden")

    with pytest.raises(AdapterTrainingError, match="merged/full-model"):
        _snapshot_evidence(tmp_path)
