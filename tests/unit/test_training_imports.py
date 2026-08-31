"""Regression coverage for training-package import boundaries."""

from __future__ import annotations

import subprocess
import sys


def test_training_smoke_imports_in_clean_interpreter() -> None:
    """The training smoke CLI must import without dataset/reporting cycles."""

    completed = subprocess.run(
        [sys.executable, "-c", "import tiny_qwen_coder.training.smoke"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
