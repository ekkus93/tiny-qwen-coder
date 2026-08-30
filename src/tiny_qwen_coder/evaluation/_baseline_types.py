"""Immutable value objects for unchanged-base Python baseline evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from tiny_qwen_coder.evaluation.regression import RegressionCategory, RegressionCategoryScore
from tiny_qwen_coder.evaluation.results import GenerationStats
from tiny_qwen_coder.identities import AdapterIdentity, BaseModelIdentity


class PythonBaselineError(RuntimeError):
    """Raised when the canonical Python baseline cannot be produced safely."""


def _require_non_empty(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise PythonBaselineError(f"{field_name} must be a non-empty string")
    if value != value.strip():
        raise PythonBaselineError(f"{field_name} must not contain outer whitespace")


def _require_sha256(value: str, *, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise PythonBaselineError(f"{field_name} must be a lowercase SHA-256 hex digest")


@dataclass(frozen=True, slots=True)
class BaselineGeneratedResponse:
    """One deterministic generated response and its observed generation metrics."""

    generated_text: str
    generation: GenerationStats

    def __post_init__(self) -> None:
        if not isinstance(self.generated_text, str):
            raise PythonBaselineError("generated_text must be a string")
        if not isinstance(self.generation, GenerationStats):
            raise PythonBaselineError("generation must be GenerationStats")


@dataclass(frozen=True, slots=True)
class BaselineGenerationCheckpoint:
    """Resume-safe generated response bound to an exact prompt and generation contract."""

    item_id: str
    prompt_sha256: str
    generation_contract_sha256: str
    response: BaselineGeneratedResponse

    def __post_init__(self) -> None:
        _require_non_empty(self.item_id, field_name="checkpoint.item_id")
        _require_sha256(self.prompt_sha256, field_name="checkpoint.prompt_sha256")
        _require_sha256(
            self.generation_contract_sha256,
            field_name="checkpoint.generation_contract_sha256",
        )
        if not isinstance(self.response, BaselineGeneratedResponse):
            raise PythonBaselineError("checkpoint.response must be BaselineGeneratedResponse")


@dataclass(frozen=True, slots=True)
class BaselineSuitePerformance:
    """Aggregated generation performance for one evaluation suite."""

    suite_id: str
    requests: int
    prompt_tokens: int
    generated_tokens: int
    generation_latency_seconds: float
    tokens_per_second: float

    def __post_init__(self) -> None:
        _require_non_empty(self.suite_id, field_name="suite_performance.suite_id")
        for field_name, value in (
            ("requests", self.requests),
            ("prompt_tokens", self.prompt_tokens),
            ("generated_tokens", self.generated_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise PythonBaselineError(f"suite_performance.{field_name} must be non-negative")
        for field_name, numeric_value in (
            ("generation_latency_seconds", self.generation_latency_seconds),
            ("tokens_per_second", self.tokens_per_second),
        ):
            if isinstance(numeric_value, bool) or not isinstance(numeric_value, int | float):
                raise PythonBaselineError(f"suite_performance.{field_name} must be numeric")
            if not math.isfinite(float(numeric_value)) or float(numeric_value) < 0:
                raise PythonBaselineError(
                    f"suite_performance.{field_name} must be finite and non-negative"
                )


@dataclass(frozen=True, slots=True)
class BaselineRuntimeMetadata:
    """Measured CUDA memory and generation throughput for one complete baseline run."""

    schema_version: int
    device: str
    gpu_name: str
    gpu_compute_capability: str
    torch_version: str
    transformers_version: str
    model_class: str
    parameter_dtypes: tuple[str, ...]
    resolved_model_revision: str
    cuda_total_bytes: int
    cuda_free_before_load_bytes: int
    cuda_free_after_load_bytes: int
    torch_allocated_after_load_bytes: int
    torch_reserved_after_load_bytes: int
    load_peak_allocated_bytes: int
    load_peak_reserved_bytes: int
    generation_peak_allocated_bytes: int
    generation_peak_reserved_bytes: int
    model_load_seconds: float
    total_wall_seconds: float
    total_requests: int
    total_prompt_tokens: int
    total_generated_tokens: int
    total_generation_latency_seconds: float
    overall_tokens_per_second: float
    suites: tuple[BaselineSuitePerformance, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise PythonBaselineError("unsupported baseline runtime metadata schema version")
        for field_name, value in (
            ("device", self.device),
            ("gpu_name", self.gpu_name),
            ("gpu_compute_capability", self.gpu_compute_capability),
            ("torch_version", self.torch_version),
            ("transformers_version", self.transformers_version),
            ("model_class", self.model_class),
            ("resolved_model_revision", self.resolved_model_revision),
        ):
            _require_non_empty(value, field_name=f"runtime.{field_name}")
        if not self.device.startswith("cuda:"):
            raise PythonBaselineError("baseline runtime requires a CUDA device")
        if not self.parameter_dtypes:
            raise PythonBaselineError("runtime.parameter_dtypes must not be empty")
        integer_fields = (
            self.cuda_total_bytes,
            self.cuda_free_before_load_bytes,
            self.cuda_free_after_load_bytes,
            self.torch_allocated_after_load_bytes,
            self.torch_reserved_after_load_bytes,
            self.load_peak_allocated_bytes,
            self.load_peak_reserved_bytes,
            self.generation_peak_allocated_bytes,
            self.generation_peak_reserved_bytes,
            self.total_requests,
            self.total_prompt_tokens,
            self.total_generated_tokens,
        )
        if self.cuda_total_bytes <= 0 or any(value < 0 for value in integer_fields):
            raise PythonBaselineError("baseline runtime integer measurements are invalid")
        float_fields = (
            self.model_load_seconds,
            self.total_wall_seconds,
            self.total_generation_latency_seconds,
            self.overall_tokens_per_second,
        )
        if any(not math.isfinite(value) or value < 0 for value in float_fields):
            raise PythonBaselineError("baseline runtime timing measurements are invalid")
        if self.torch_reserved_after_load_bytes < self.torch_allocated_after_load_bytes:
            raise PythonBaselineError("reserved CUDA memory is smaller than allocated memory")
        if self.load_peak_allocated_bytes < self.torch_allocated_after_load_bytes:
            raise PythonBaselineError("load peak allocated memory is smaller than post-load memory")
        if self.load_peak_reserved_bytes < self.torch_reserved_after_load_bytes:
            raise PythonBaselineError("load peak reserved memory is smaller than post-load memory")
        if self.total_requests != sum(item.requests for item in self.suites):
            raise PythonBaselineError("runtime total_requests does not match suite totals")
        if self.total_prompt_tokens != sum(item.prompt_tokens for item in self.suites):
            raise PythonBaselineError("runtime total_prompt_tokens does not match suite totals")
        if self.total_generated_tokens != sum(item.generated_tokens for item in self.suites):
            raise PythonBaselineError("runtime total_generated_tokens does not match suite totals")


@dataclass(frozen=True, slots=True)
class RegressionBaselineCaseResult:
    """One generated general/tool regression response plus deterministic score."""

    case_id: str
    category: RegressionCategory
    generated_text: str
    passed: bool
    detail: str | None
    generation: GenerationStats

    def __post_init__(self) -> None:
        _require_non_empty(self.case_id, field_name="regression_result.case_id")
        if not isinstance(self.category, RegressionCategory):
            raise PythonBaselineError("regression_result.category must be RegressionCategory")
        if not isinstance(self.generated_text, str):
            raise PythonBaselineError("regression_result.generated_text must be a string")
        if not isinstance(self.passed, bool):
            raise PythonBaselineError("regression_result.passed must be a boolean")
        if self.detail is not None:
            _require_non_empty(self.detail, field_name="regression_result.detail")
        if not isinstance(self.generation, GenerationStats):
            raise PythonBaselineError("regression_result.generation must be GenerationStats")


@dataclass(frozen=True, slots=True)
class RegressionBaselineAggregate:
    """Frozen aggregate for the general/tool unchanged-base regression suite."""

    schema_version: int
    suite_id: str
    suite_version: int
    suite_sha256: str
    evaluation_settings_sha256: str
    system_prompt_version: str
    system_prompt_sha256: str
    base_model: BaseModelIdentity
    adapter: AdapterIdentity
    total_cases: int
    passed: int
    failed: int
    pass_rate: float
    categories: tuple[RegressionCategoryScore, ...]
    results_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise PythonBaselineError("unsupported regression baseline aggregate schema version")
        for field_name, value in (
            ("suite_id", self.suite_id),
            ("system_prompt_version", self.system_prompt_version),
        ):
            _require_non_empty(value, field_name=f"regression_aggregate.{field_name}")
        for field_name, value in (
            ("suite_sha256", self.suite_sha256),
            ("evaluation_settings_sha256", self.evaluation_settings_sha256),
            ("system_prompt_sha256", self.system_prompt_sha256),
            ("results_sha256", self.results_sha256),
        ):
            _require_sha256(value, field_name=f"regression_aggregate.{field_name}")
        if not isinstance(self.base_model, BaseModelIdentity):
            raise PythonBaselineError("regression_aggregate.base_model is invalid")
        if not isinstance(self.adapter, AdapterIdentity):
            raise PythonBaselineError("regression_aggregate.adapter is invalid")
        if self.adapter.adapter_id is not None:
            raise PythonBaselineError("unchanged-base regression aggregate must not define adapter")
        if self.total_cases <= 0:
            raise PythonBaselineError("regression_aggregate.total_cases must be positive")
        if self.passed < 0 or self.failed < 0 or self.passed + self.failed != self.total_cases:
            raise PythonBaselineError("regression aggregate pass/fail counts are inconsistent")
        if not 0.0 <= self.pass_rate <= 1.0:
            raise PythonBaselineError("regression_aggregate.pass_rate must be between 0 and 1")
        if not self.categories or any(
            not isinstance(item, RegressionCategoryScore) for item in self.categories
        ):
            raise PythonBaselineError("regression_aggregate.categories are invalid")


@dataclass(frozen=True, slots=True)
class BaselineArtifactDigest:
    """One required baseline artifact path and content digest."""

    artifact_id: str
    path: str
    sha256: str

    def __post_init__(self) -> None:
        _require_non_empty(self.artifact_id, field_name="artifact.artifact_id")
        _require_non_empty(self.path, field_name="artifact.path")
        _require_sha256(self.sha256, field_name="artifact.sha256")
        candidate = Path(self.path)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise PythonBaselineError("artifact.path must be a safe relative path")


@dataclass(frozen=True, slots=True)
class PythonBaselineManifest:
    """Complete frozen artifact inventory for one canonical unchanged-base run."""

    schema_version: int
    baseline_id: str
    baseline_version: int
    frozen: bool
    base_model: BaseModelIdentity
    adapter: AdapterIdentity
    source_git_sha: str
    evaluation_config_sha256: str
    evaluation_settings_sha256: str
    system_prompt_version: str
    system_prompt_sha256: str
    generation_contract_sha256: str
    artifacts: tuple[BaselineArtifactDigest, ...]
    artifact_set_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise PythonBaselineError("unsupported Python baseline manifest schema version")
        if self.baseline_version != 1:
            raise PythonBaselineError("unsupported Python baseline manifest version")
        if not self.frozen:
            raise PythonBaselineError("Python baseline manifest must be frozen")
        _require_non_empty(self.baseline_id, field_name="baseline_id")
        if not isinstance(self.base_model, BaseModelIdentity):
            raise PythonBaselineError("baseline base_model is invalid")
        if not isinstance(self.adapter, AdapterIdentity) or self.adapter.adapter_id is not None:
            raise PythonBaselineError("baseline manifest must represent unchanged base only")
        if len(self.source_git_sha) != 40 or any(
            character not in "0123456789abcdef" for character in self.source_git_sha
        ):
            raise PythonBaselineError("source_git_sha must be a lowercase 40-character Git SHA")
        for field_name, value in (
            ("evaluation_config_sha256", self.evaluation_config_sha256),
            ("evaluation_settings_sha256", self.evaluation_settings_sha256),
            ("system_prompt_sha256", self.system_prompt_sha256),
            ("generation_contract_sha256", self.generation_contract_sha256),
            ("artifact_set_sha256", self.artifact_set_sha256),
        ):
            _require_sha256(value, field_name=field_name)
        _require_non_empty(self.system_prompt_version, field_name="system_prompt_version")
        if not self.artifacts:
            raise PythonBaselineError("baseline artifacts must not be empty")
        artifact_ids = tuple(item.artifact_id for item in self.artifacts)
        paths = tuple(item.path for item in self.artifacts)
        if len(artifact_ids) != len(set(artifact_ids)):
            raise PythonBaselineError("baseline artifact IDs must be unique")
        if len(paths) != len(set(paths)):
            raise PythonBaselineError("baseline artifact paths must be unique")
