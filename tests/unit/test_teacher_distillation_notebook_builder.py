from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_v4_2000_salvage_notebook_builder_matches_checked_in_artifact(
    tmp_path: Path,
) -> None:
    builder = Path("scripts/teacher_distillation/build_v4_2000_salvage_colab_notebook.py")
    committed = Path(
        "scripts/teacher_distillation/qwen38_teacher_distillation_v4_2000_salvage_colab.ipynb"
    )
    generated = tmp_path / committed.name

    subprocess.run(
        [sys.executable, str(builder), "--output", str(generated)],
        check=True,
    )

    assert generated.read_bytes() == committed.read_bytes()
