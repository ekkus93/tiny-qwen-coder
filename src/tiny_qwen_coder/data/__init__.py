"""Language-neutral dataset preparation services."""

from __future__ import annotations

from typing import NoReturn

from tiny_qwen_coder.data.splitting import (
    TrainValidationSplit,
    deterministic_train_validation_split,
)

__all__ = ["TrainValidationSplit", "deterministic_train_validation_split", "prepare_data"]


def prepare_data() -> NoReturn:
    """Run generic data preparation once the Phase 3 pipeline is implemented."""
    raise SystemExit("Data preparation is scaffolded; implementation is tracked by Phase 3.")
