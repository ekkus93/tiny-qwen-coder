from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

import tiny_qwen_coder.training.runtime as training_runtime
from tiny_qwen_coder.training.plan import AdapterTrainingError, AdapterTrainingPlan


def test_training_runtime_runs_preflight_before_creating_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = cast(AdapterTrainingPlan, object())
    output_prepared = False

    monkeypatch.setattr(training_runtime, "resolve_adapter_training_plan", lambda path: plan)

    def fail_preflight(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AdapterTrainingError("preflight failed")

    def prepare_output(*args: object, **kwargs: object) -> None:
        nonlocal output_prepared
        del args, kwargs
        output_prepared = True

    monkeypatch.setattr(training_runtime, "run_training_preflight", fail_preflight)
    monkeypatch.setattr(training_runtime, "_prepare_output", prepare_output)

    with pytest.raises(AdapterTrainingError, match="preflight failed"):
        training_runtime.run_adapter_training(tmp_path / "training.yaml", repo_root=tmp_path)

    assert output_prepared is False
