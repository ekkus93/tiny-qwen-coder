"""Language-neutral evaluation and execution services."""

from __future__ import annotations

from typing import NoReturn

from tiny_qwen_coder.evaluation.protected_benchmarks import (
    ProtectedBenchmark,
    ProtectedBenchmarkRegistrationError,
    ProtectedBenchmarkRegistry,
    ProtectedBenchmarkRegistryError,
    ProtectedBenchmarkTrainingSelectionError,
    UnknownProtectedBenchmarkError,
)

__all__ = [
    "ProtectedBenchmark",
    "ProtectedBenchmarkRegistrationError",
    "ProtectedBenchmarkRegistry",
    "ProtectedBenchmarkRegistryError",
    "ProtectedBenchmarkTrainingSelectionError",
    "UnknownProtectedBenchmarkError",
    "evaluate",
]


def evaluate() -> NoReturn:
    """Run generic evaluation once the remaining Phase 4 framework is implemented."""
    raise SystemExit("Evaluation is scaffolded; implementation is tracked by Phase 4.")
