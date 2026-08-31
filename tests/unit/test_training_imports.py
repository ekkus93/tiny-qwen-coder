"""Regression coverage for training-package and pinned trainer API boundaries."""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path


def test_training_smoke_imports_in_clean_interpreter() -> None:
    """The training smoke CLI must import without dataset/reporting cycles."""

    completed = subprocess.run(
        [sys.executable, "-c", "import tiny_qwen_coder.training.smoke"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_pinned_sft_config_uses_unified_warmup_steps_api() -> None:
    """Keep the generic trainer aligned with the pinned Transformers v5 API."""

    from trl.trainer.sft_config import SFTConfig

    parameters = inspect.signature(SFTConfig).parameters
    assert "warmup_steps" in parameters
    assert "warmup_ratio" not in parameters

    runtime_source = Path("src/tiny_qwen_coder/training/runtime.py").read_text(encoding="utf-8")
    assert "warmup_steps=plan.config.warmup_ratio" in runtime_source
    assert "warmup_ratio=plan.config.warmup_ratio" not in runtime_source
