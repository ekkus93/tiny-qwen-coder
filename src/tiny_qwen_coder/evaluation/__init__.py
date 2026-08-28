"""Language-neutral evaluation and execution services."""

from __future__ import annotations

from typing import NoReturn

from tiny_qwen_coder.evaluation.contamination import (
    EXACT_PROMPT_CHECK_ID,
    EXACT_SOLUTION_CHECK_ID,
    HIGH_OVERLAP_CHECK_ID,
    ContaminationCheckError,
    HighOverlapConfig,
    ProtectedBenchmarkExample,
    check_training_contamination,
)
from tiny_qwen_coder.evaluation.execution import (
    ConstrainedExecutionHarness,
    ExecutionCleanupError,
    ExecutionFile,
    ExecutionHarnessError,
    ExecutionHarnessUnavailableError,
    ExecutionLimits,
    ExecutionRequest,
    ExecutionRequestError,
    ExecutionResult,
    ExecutionStatus,
    OciRuntime,
    OciRuntimeSpec,
    discover_oci_runtime,
)
from tiny_qwen_coder.evaluation.protected_benchmarks import (
    ProtectedBenchmark,
    ProtectedBenchmarkRegistrationError,
    ProtectedBenchmarkRegistry,
    ProtectedBenchmarkRegistryError,
    ProtectedBenchmarkTrainingSelectionError,
    UnknownProtectedBenchmarkError,
)

__all__ = [
    "ConstrainedExecutionHarness",
    "ExecutionCleanupError",
    "ExecutionFile",
    "ExecutionHarnessError",
    "ExecutionHarnessUnavailableError",
    "ExecutionLimits",
    "ExecutionRequest",
    "ExecutionRequestError",
    "ExecutionResult",
    "ExecutionStatus",
    "OciRuntime",
    "OciRuntimeSpec",
    "discover_oci_runtime",
    "EXACT_PROMPT_CHECK_ID",
    "EXACT_SOLUTION_CHECK_ID",
    "HIGH_OVERLAP_CHECK_ID",
    "ContaminationCheckError",
    "HighOverlapConfig",
    "ProtectedBenchmarkExample",
    "ProtectedBenchmark",
    "ProtectedBenchmarkRegistrationError",
    "ProtectedBenchmarkRegistry",
    "ProtectedBenchmarkRegistryError",
    "ProtectedBenchmarkTrainingSelectionError",
    "UnknownProtectedBenchmarkError",
    "check_training_contamination",
    "evaluate",
]


def evaluate() -> NoReturn:
    """Run generic evaluation once the remaining Phase 4 framework is implemented."""
    raise SystemExit("Evaluation is scaffolded; implementation is tracked by Phase 4.")
