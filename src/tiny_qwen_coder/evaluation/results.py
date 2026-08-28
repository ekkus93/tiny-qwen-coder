"""Language-neutral machine-readable evaluation result schema."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from enum import StrEnum

from tiny_qwen_coder.identities import AdapterIdentity, BaseModelIdentity

_EVALUATION_RESULT_SCHEMA_VERSION = 1
_LANGUAGE_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")


class EvaluationResultError(ValueError):
    """Raised when an evaluation result violates the common schema."""


class EvaluationStageStatus(StrEnum):
    """Portable status for language-specific parse and compile stages."""

    NOT_RUN = "not_run"
    PASSED = "passed"
    FAILED = "failed"
    NOT_APPLICABLE = "not_applicable"


class EvaluationErrorCategory(StrEnum):
    """Portable primary failure category for one evaluation problem."""

    NONE = "none"
    GENERATION = "generation"
    PARSE = "parse"
    COMPILE = "compile"
    TEST = "test"
    RUNTIME = "runtime"
    TIMEOUT = "timeout"
    HARNESS = "harness"


@dataclass(frozen=True, slots=True)
class EvaluationTestSummary:
    """Language-neutral test outcome counts for one generated candidate."""

    passed: int
    total: int

    def __post_init__(self) -> None:
        if isinstance(self.passed, bool) or not isinstance(self.passed, int):
            raise EvaluationResultError("tests.passed must be an integer")
        if isinstance(self.total, bool) or not isinstance(self.total, int):
            raise EvaluationResultError("tests.total must be an integer")
        if self.passed < 0 or self.total < 0:
            raise EvaluationResultError("test counts must not be negative")
        if self.passed > self.total:
            raise EvaluationResultError("tests.passed must not exceed tests.total")

    @property
    def pass_rate(self) -> float | None:
        """Return the observed test pass fraction, or null when no tests exist."""

        if self.total == 0:
            return None
        return self.passed / self.total


@dataclass(frozen=True, slots=True)
class GenerationStats:
    """Common generation metrics recorded before language-specific evaluation."""

    prompt_tokens: int | None
    generated_tokens: int
    latency_seconds: float
    tokens_per_second: float

    def __post_init__(self) -> None:
        if self.prompt_tokens is not None:
            if isinstance(self.prompt_tokens, bool) or not isinstance(self.prompt_tokens, int):
                raise EvaluationResultError("generation.prompt_tokens must be an integer or null")
            if self.prompt_tokens < 0:
                raise EvaluationResultError("generation.prompt_tokens must not be negative")
        if isinstance(self.generated_tokens, bool) or not isinstance(self.generated_tokens, int):
            raise EvaluationResultError("generation.generated_tokens must be an integer")
        if self.generated_tokens < 0:
            raise EvaluationResultError("generation.generated_tokens must not be negative")
        for field_name, value in (
            ("latency_seconds", self.latency_seconds),
            ("tokens_per_second", self.tokens_per_second),
        ):
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise EvaluationResultError(f"generation.{field_name} must be numeric")
            if not math.isfinite(value) or value < 0:
                raise EvaluationResultError(
                    f"generation.{field_name} must be finite and non-negative"
                )


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """One language-independent evaluation record for a generated candidate."""

    schema_version: int
    problem_id: str
    language: str
    generated_text: str
    generated_code: str | None
    parse_status: EvaluationStageStatus
    compile_status: EvaluationStageStatus
    tests: EvaluationTestSummary
    error_category: EvaluationErrorCategory
    error_message: str | None
    generation: GenerationStats
    base_model: BaseModelIdentity
    adapter: AdapterIdentity

    def __post_init__(self) -> None:
        if self.schema_version != _EVALUATION_RESULT_SCHEMA_VERSION:
            raise EvaluationResultError(
                f"unsupported evaluation result schema_version {self.schema_version}; "
                f"expected {_EVALUATION_RESULT_SCHEMA_VERSION}"
            )
        _require_exact_non_empty(self.problem_id, field_name="problem_id")
        if not _LANGUAGE_PATTERN.fullmatch(self.language):
            raise EvaluationResultError("language must match ^[a-z][a-z0-9_-]*$")
        if not isinstance(self.generated_text, str):
            raise EvaluationResultError("generated_text must be a string")
        if self.generated_code is not None and not isinstance(self.generated_code, str):
            raise EvaluationResultError("generated_code must be a string or null")
        if not isinstance(self.parse_status, EvaluationStageStatus):
            raise EvaluationResultError("parse_status must be an EvaluationStageStatus")
        if not isinstance(self.compile_status, EvaluationStageStatus):
            raise EvaluationResultError("compile_status must be an EvaluationStageStatus")
        if not isinstance(self.tests, EvaluationTestSummary):
            raise EvaluationResultError("tests must be an EvaluationTestSummary")
        if not isinstance(self.error_category, EvaluationErrorCategory):
            raise EvaluationResultError("error_category must be an EvaluationErrorCategory")
        if self.error_message is not None:
            _require_exact_non_empty(self.error_message, field_name="error_message")
        if not isinstance(self.generation, GenerationStats):
            raise EvaluationResultError("generation must be GenerationStats")
        if not isinstance(self.base_model, BaseModelIdentity):
            raise EvaluationResultError("base_model must be BaseModelIdentity")
        if not isinstance(self.adapter, AdapterIdentity):
            raise EvaluationResultError("adapter must be AdapterIdentity")
        self._validate_outcome_consistency()

    def _validate_outcome_consistency(self) -> None:
        if self.error_category is EvaluationErrorCategory.NONE:
            if self.error_message is not None:
                raise EvaluationResultError(
                    "error_message must be null when error_category is none"
                )
            if self.parse_status is EvaluationStageStatus.FAILED:
                raise EvaluationResultError(
                    "failed parse status requires a non-none error category"
                )
            if self.compile_status is EvaluationStageStatus.FAILED:
                raise EvaluationResultError(
                    "failed compile status requires a non-none error category"
                )
            if self.tests.passed != self.tests.total:
                raise EvaluationResultError("failed tests require a non-none error category")
        elif self.error_category is EvaluationErrorCategory.PARSE:
            if self.parse_status is not EvaluationStageStatus.FAILED:
                raise EvaluationResultError("parse error category requires failed parse_status")
        elif self.error_category is EvaluationErrorCategory.COMPILE:
            if self.compile_status is not EvaluationStageStatus.FAILED:
                raise EvaluationResultError("compile error category requires failed compile_status")
        elif self.error_category is EvaluationErrorCategory.TEST:
            if self.tests.passed == self.tests.total:
                raise EvaluationResultError("test error category requires at least one failed test")


def _require_exact_non_empty(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise EvaluationResultError(f"{field_name} must be a non-empty string")
    if value != value.strip():
        raise EvaluationResultError(f"{field_name} must not contain leading or trailing whitespace")


def create_evaluation_result(
    *,
    problem_id: str,
    language: str,
    generated_text: str,
    generated_code: str | None,
    parse_status: EvaluationStageStatus,
    compile_status: EvaluationStageStatus,
    tests: EvaluationTestSummary,
    error_category: EvaluationErrorCategory,
    error_message: str | None,
    generation: GenerationStats,
    base_model: BaseModelIdentity,
    adapter: AdapterIdentity,
) -> EvaluationResult:
    """Create a versioned common evaluation result."""

    return EvaluationResult(
        schema_version=_EVALUATION_RESULT_SCHEMA_VERSION,
        problem_id=problem_id,
        language=language,
        generated_text=generated_text,
        generated_code=generated_code,
        parse_status=parse_status,
        compile_status=compile_status,
        tests=tests,
        error_category=error_category,
        error_message=error_message,
        generation=generation,
        base_model=base_model,
        adapter=adapter,
    )


def evaluation_result_json(result: EvaluationResult) -> str:
    """Serialize one evaluation result deterministically."""

    return json.dumps(asdict(result), indent=2, sort_keys=True) + "\n"
