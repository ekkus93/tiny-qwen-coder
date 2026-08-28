"""Inference, serving, and adapter-selection runtime services."""

from __future__ import annotations

from typing import NoReturn


def infer() -> NoReturn:
    """Run inference once the runtime implementation is available."""
    raise SystemExit("Inference is scaffolded; runtime implementation is tracked by Phase 10.")


def serve() -> NoReturn:
    """Run the local serving endpoint once the serving implementation is available."""
    raise SystemExit("Serving is scaffolded; implementation is tracked by Phase 16.")
