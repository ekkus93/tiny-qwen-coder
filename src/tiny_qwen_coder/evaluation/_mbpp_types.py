"""Immutable public value objects for the MBPP evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tiny_qwen_coder.evaluation._mbpp_common import (
    _GIT_SHA_PATTERN,
    _IMAGE_DIGEST_PATTERN,
    _MBPP_BENCHMARK_ID,
    _MBPP_DATASET_ID,
    _MBPP_RUNNER_ID,
    _MBPP_RUNNER_VERSION,
    _MBPP_SCHEMA_VERSION,
    _TASK_ID_PATTERN,
    MBPPError,
    _require_non_empty,
    _require_sha256,
    _require_text,
)
from tiny_qwen_coder.evaluation.execution import ExecutionLimits
from tiny_qwen_coder.evaluation.results import (
    EvaluationErrorCategory,
    EvaluationResult,
    GenerationStats,
)
from tiny_qwen_coder.identities import AdapterIdentity, BaseModelIdentity


@dataclass(frozen=True, slots=True)
class MBPPRunnerConfig:
    """Frozen MBPP protocol and constrained-runtime policy."""

    schema_version: int
    runner_id: str
    runner_version: int
    frozen: bool
    benchmark_id: str
    reference_repository: str
    reference_revision: str
    dataset_config: str
    dataset_split: str
    task_id_start: int
    task_id_end: int
    expected_problem_count: int
    tests_per_problem: int
    prompt_version: str
    completion_normalizer_version: str
    stop_words: tuple[str, ...]
    execution_image: str
    execution_timeout_seconds: float
    limits: ExecutionLimits

    def __post_init__(self) -> None:
        if self.schema_version != _MBPP_SCHEMA_VERSION:
            raise MBPPError("unsupported MBPP runner schema version")
        if self.runner_id != _MBPP_RUNNER_ID:
            raise MBPPError(f"MBPP runner_id must be {_MBPP_RUNNER_ID!r}")
        if self.runner_version != _MBPP_RUNNER_VERSION:
            raise MBPPError(f"MBPP runner_version must be {_MBPP_RUNNER_VERSION}")
        if not self.frozen:
            raise MBPPError("MBPP runner configuration must be frozen")
        if self.benchmark_id != _MBPP_BENCHMARK_ID:
            raise MBPPError(f"MBPP benchmark_id must be {_MBPP_BENCHMARK_ID!r}")
        _require_non_empty(self.reference_repository, field_name="reference_repository")
        if not _GIT_SHA_PATTERN.fullmatch(self.reference_revision):
            raise MBPPError("reference_revision must be a lowercase 40-character Git SHA")
        if self.dataset_config != "full":
            raise MBPPError("MBPP dataset_config must be 'full'")
        if self.dataset_split != "test":
            raise MBPPError("MBPP dataset_split must be 'test'")
        if self.task_id_start <= 0 or self.task_id_end < self.task_id_start:
            raise MBPPError("MBPP task ID bounds are invalid")
        if self.expected_problem_count != self.task_id_end - self.task_id_start + 1:
            raise MBPPError("expected_problem_count must match the inclusive task ID range")
        if self.tests_per_problem <= 0:
            raise MBPPError("tests_per_problem must be greater than zero")
        _require_non_empty(self.prompt_version, field_name="prompt_version")
        _require_non_empty(
            self.completion_normalizer_version,
            field_name="completion_normalizer_version",
        )
        if not self.stop_words:
            raise MBPPError("stop_words must not be empty")
        if len(set(self.stop_words)) != len(self.stop_words):
            raise MBPPError("stop_words must not contain duplicates")
        for index, stop_word in enumerate(self.stop_words):
            _require_text(stop_word, field_name=f"stop_words[{index}]")
        if not _IMAGE_DIGEST_PATTERN.fullmatch(self.execution_image):
            raise MBPPError("execution_image must be pinned by an exact sha256 digest")
        if self.execution_timeout_seconds <= 0:
            raise MBPPError("execution_timeout_seconds must be greater than zero")
        if not isinstance(self.limits, ExecutionLimits):
            raise MBPPError("limits must be ExecutionLimits")


@dataclass(frozen=True, slots=True)
class MBPPProblem:
    """One protected MBPP problem without its canonical solution or challenge tests."""

    task_id: str
    description: str
    tests: tuple[str, ...]
    test_setup_code: str

    def __post_init__(self) -> None:
        if _TASK_ID_PATTERN.fullmatch(self.task_id) is None:
            raise MBPPError("task_id must match MBPP/<positive integer>")
        _require_text(self.description, field_name="description")
        if not self.tests:
            raise MBPPError("tests must not be empty")
        for index, test in enumerate(self.tests):
            _require_text(test, field_name=f"tests[{index}]")
        if not isinstance(self.test_setup_code, str):
            raise MBPPError("test_setup_code must be a string")
        if "\x00" in self.test_setup_code:
            raise MBPPError("test_setup_code must not contain NUL bytes")

    @property
    def task_number(self) -> int:
        return int(self.task_id.partition("/")[2])


@dataclass(frozen=True, slots=True)
class MBPPPrompt:
    """Normalized single-user-message prompt for one MBPP task."""

    task_id: str
    prompt_version: str
    user_content: str

    def __post_init__(self) -> None:
        if _TASK_ID_PATTERN.fullmatch(self.task_id) is None:
            raise MBPPError("prompt task_id must match MBPP/<positive integer>")
        _require_non_empty(self.prompt_version, field_name="prompt_version")
        _require_text(self.user_content, field_name="user_content")


@dataclass(frozen=True, slots=True)
class MBPPCompletion:
    """One generated response plus common generation statistics."""

    task_id: str
    generated_text: str
    generation: GenerationStats
    generation_error: str | None = None

    def __post_init__(self) -> None:
        if _TASK_ID_PATTERN.fullmatch(self.task_id) is None:
            raise MBPPError("completion task_id must match MBPP/<positive integer>")
        if not isinstance(self.generated_text, str):
            raise MBPPError("generated_text must be a string")
        if not isinstance(self.generation, GenerationStats):
            raise MBPPError("generation must be GenerationStats")
        if self.generation_error is not None:
            _require_non_empty(self.generation_error, field_name="generation_error")


@dataclass(frozen=True, slots=True)
class MBPPAggregate:
    """Deterministic aggregate MBPP pass@1 artifact."""

    schema_version: int
    benchmark_id: str
    dataset_id: str
    dataset_revision: str
    dataset_config: str
    dataset_split: str
    reference_repository: str
    reference_revision: str
    runner_config_sha256: str
    runner_source_sha256: str
    evaluation_settings_sha256: str
    prompt_version: str
    completion_normalizer_version: str
    execution_image: str
    base_model: BaseModelIdentity
    adapter: AdapterIdentity
    total_problems: int
    passed: int
    failed: int
    timed_out: int
    harness_errors: int
    pass_at_1: float | None
    problem_set_sha256: str
    results_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != _MBPP_SCHEMA_VERSION:
            raise MBPPError("unsupported MBPP aggregate schema version")
        if self.benchmark_id != _MBPP_BENCHMARK_ID:
            raise MBPPError("MBPP aggregate benchmark_id is invalid")
        if self.dataset_id != _MBPP_DATASET_ID:
            raise MBPPError("MBPP aggregate dataset_id is invalid")
        if not _GIT_SHA_PATTERN.fullmatch(self.dataset_revision):
            raise MBPPError("MBPP aggregate dataset_revision must be an immutable Git SHA")
        if self.dataset_config != "full" or self.dataset_split != "test":
            raise MBPPError("MBPP aggregate dataset selection is invalid")
        if not _GIT_SHA_PATTERN.fullmatch(self.reference_revision):
            raise MBPPError("MBPP aggregate reference_revision must be a Git SHA")
        if not _IMAGE_DIGEST_PATTERN.fullmatch(self.execution_image):
            raise MBPPError("MBPP aggregate execution_image must be digest-pinned")
        if self.total_problems <= 0:
            raise MBPPError("MBPP aggregate total_problems must be positive")
        if self.passed < 0 or self.failed < 0 or self.passed + self.failed != self.total_problems:
            raise MBPPError("MBPP aggregate pass/fail counts are inconsistent")
        if not 0 <= self.timed_out <= self.failed:
            raise MBPPError("MBPP aggregate timed_out count is inconsistent")
        if not 0 <= self.harness_errors <= self.failed:
            raise MBPPError("MBPP aggregate harness_errors count is inconsistent")
        if self.harness_errors and self.pass_at_1 is not None:
            raise MBPPError("pass_at_1 must be null when harness errors occurred")
        if not self.harness_errors and self.pass_at_1 != self.passed / self.total_problems:
            raise MBPPError("MBPP aggregate pass_at_1 is inconsistent")
        for field_name, value in (
            ("runner_config_sha256", self.runner_config_sha256),
            ("runner_source_sha256", self.runner_source_sha256),
            ("evaluation_settings_sha256", self.evaluation_settings_sha256),
            ("problem_set_sha256", self.problem_set_sha256),
            ("results_sha256", self.results_sha256),
        ):
            _require_sha256(value, field_name=field_name)


@dataclass(frozen=True, slots=True)
class MBPPSuiteResult:
    """Per-problem common results plus one aggregate MBPP score."""

    results: tuple[EvaluationResult, ...]
    aggregate: MBPPAggregate

    def __post_init__(self) -> None:
        from tiny_qwen_coder.evaluation._mbpp_data import mbpp_results_sha256

        if len(self.results) != self.aggregate.total_problems:
            raise MBPPError("MBPP result count must match aggregate total_problems")
        if mbpp_results_sha256(self.results) != self.aggregate.results_sha256:
            raise MBPPError("MBPP aggregate results_sha256 does not match results")
        passed = sum(
            result.error_category is EvaluationErrorCategory.NONE for result in self.results
        )
        timed_out = sum(
            result.error_category is EvaluationErrorCategory.TIMEOUT for result in self.results
        )
        harness_errors = sum(
            result.error_category is EvaluationErrorCategory.HARNESS for result in self.results
        )
        if passed != self.aggregate.passed:
            raise MBPPError("MBPP aggregate passed count does not match results")
        if timed_out != self.aggregate.timed_out:
            raise MBPPError("MBPP aggregate timeout count does not match results")
        if harness_errors != self.aggregate.harness_errors:
            raise MBPPError("MBPP aggregate harness-error count does not match results")


@dataclass(frozen=True, slots=True)
class MBPPArtifactPaths:
    """Paths written for one MBPP suite evaluation."""

    per_problem: Path
    aggregate: Path
