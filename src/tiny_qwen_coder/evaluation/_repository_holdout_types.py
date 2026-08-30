"""Immutable value objects for the repository-owned Python holdout evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tiny_qwen_coder.evaluation._repository_holdout_common import (
    _EXPECTED_CATEGORIES,
    _IMAGE_DIGEST_PATTERN,
    _REPOSITORY_HOLDOUT_DATASET_ID,
    _REPOSITORY_HOLDOUT_DATASET_REVISION,
    _REPOSITORY_HOLDOUT_SCHEMA_VERSION,
    _REPOSITORY_HOLDOUT_SUITE_ID,
    _REPOSITORY_HOLDOUT_SUITE_VERSION,
    _TASK_ID_PATTERN,
    RepositoryHoldoutError,
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
class RepositoryHoldoutTask:
    """One repo-owned holdout problem plus its protected executable tests."""

    task_id: str
    category: str
    prompt: str
    test_path: str
    setup_path: str | None
    expected_tests: int
    test_source: str
    setup_source: str
    test_sha256: str
    setup_sha256: str | None

    def __post_init__(self) -> None:
        if not _TASK_ID_PATTERN.fullmatch(self.task_id):
            raise RepositoryHoldoutError("task_id must be a stable lowercase slug")
        if self.category not in _EXPECTED_CATEGORIES:
            raise RepositoryHoldoutError(
                f"unsupported repository holdout category {self.category!r}"
            )
        _require_text(self.prompt, field_name="prompt")
        _require_non_empty(self.test_path, field_name="test_path")
        if self.setup_path is not None:
            _require_non_empty(self.setup_path, field_name="setup_path")
        if isinstance(self.expected_tests, bool) or not isinstance(self.expected_tests, int):
            raise RepositoryHoldoutError("expected_tests must be an integer")
        if self.expected_tests <= 0:
            raise RepositoryHoldoutError("expected_tests must be greater than zero")
        _require_text(self.test_source, field_name="test_source")
        if not isinstance(self.setup_source, str):
            raise RepositoryHoldoutError("setup_source must be a string")
        _require_sha256(self.test_sha256, field_name="test_sha256")
        if self.setup_path is None:
            if self.setup_source or self.setup_sha256 is not None:
                raise RepositoryHoldoutError("task without setup_path must have empty setup source")
        else:
            if not self.setup_source or self.setup_sha256 is None:
                raise RepositoryHoldoutError("task setup path requires setup source and digest")
            _require_sha256(self.setup_sha256, field_name="setup_sha256")

    @property
    def problem_id(self) -> str:
        return f"repository-holdout/{self.task_id}"


@dataclass(frozen=True, slots=True)
class RepositoryHoldoutSuiteConfig:
    """Frozen repo-owned suite metadata and task definitions."""

    schema_version: int
    suite_id: str
    suite_version: int
    frozen: bool
    benchmark_id: str
    dataset_revision: str
    prompt_version: str
    completion_normalizer_version: str
    instruction: str
    execution_image: str
    execution_timeout_seconds: float
    limits: ExecutionLimits
    tasks: tuple[RepositoryHoldoutTask, ...]

    def __post_init__(self) -> None:
        if self.schema_version != _REPOSITORY_HOLDOUT_SCHEMA_VERSION:
            raise RepositoryHoldoutError("unsupported repository holdout schema version")
        if self.suite_id != _REPOSITORY_HOLDOUT_SUITE_ID:
            raise RepositoryHoldoutError("repository holdout suite_id is invalid")
        if self.suite_version != _REPOSITORY_HOLDOUT_SUITE_VERSION:
            raise RepositoryHoldoutError("repository holdout suite_version is invalid")
        if not self.frozen:
            raise RepositoryHoldoutError("repository holdout suite must be frozen")
        if self.benchmark_id != _REPOSITORY_HOLDOUT_SUITE_ID:
            raise RepositoryHoldoutError("repository holdout benchmark_id is invalid")
        if self.dataset_revision != _REPOSITORY_HOLDOUT_DATASET_REVISION:
            raise RepositoryHoldoutError("repository holdout dataset_revision is invalid")
        _require_non_empty(self.prompt_version, field_name="prompt_version")
        _require_non_empty(
            self.completion_normalizer_version,
            field_name="completion_normalizer_version",
        )
        _require_text(self.instruction, field_name="instruction")
        if not _IMAGE_DIGEST_PATTERN.fullmatch(self.execution_image):
            raise RepositoryHoldoutError("execution_image must be pinned by exact sha256 digest")
        if self.execution_timeout_seconds <= 0:
            raise RepositoryHoldoutError("execution_timeout_seconds must be greater than zero")
        if not self.tasks:
            raise RepositoryHoldoutError("repository holdout must contain tasks")
        task_ids = tuple(task.task_id for task in self.tasks)
        if len(task_ids) != len(set(task_ids)):
            raise RepositoryHoldoutError("repository holdout task IDs must be unique")
        categories = tuple(task.category for task in self.tasks)
        if len(categories) != len(set(categories)):
            raise RepositoryHoldoutError("repository holdout categories must be unique")
        if frozenset(categories) != _EXPECTED_CATEGORIES:
            raise RepositoryHoldoutError(
                "repository holdout must cover every frozen P6-004 category exactly once"
            )


@dataclass(frozen=True, slots=True)
class RepositoryHoldoutPrompt:
    """Normalized single-user prompt for one repo-owned task."""

    problem_id: str
    prompt_version: str
    user_content: str

    def __post_init__(self) -> None:
        _require_non_empty(self.problem_id, field_name="problem_id")
        _require_non_empty(self.prompt_version, field_name="prompt_version")
        _require_text(self.user_content, field_name="user_content")


@dataclass(frozen=True, slots=True)
class RepositoryHoldoutCompletion:
    """One generated module and its generation statistics."""

    problem_id: str
    generated_text: str
    generation: GenerationStats
    generation_error: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.problem_id, field_name="problem_id")
        if not isinstance(self.generated_text, str):
            raise RepositoryHoldoutError("generated_text must be a string")
        if not isinstance(self.generation, GenerationStats):
            raise RepositoryHoldoutError("generation must be GenerationStats")
        if self.generation_error is not None:
            _require_non_empty(self.generation_error, field_name="generation_error")


@dataclass(frozen=True, slots=True)
class RepositoryHoldoutAggregate:
    """Deterministic aggregate result for the repository-owned holdout."""

    schema_version: int
    benchmark_id: str
    dataset_id: str
    dataset_revision: str
    suite_version: int
    suite_sha256: str
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
    results_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != _REPOSITORY_HOLDOUT_SCHEMA_VERSION:
            raise RepositoryHoldoutError("unsupported repository holdout aggregate schema")
        if self.benchmark_id != _REPOSITORY_HOLDOUT_SUITE_ID:
            raise RepositoryHoldoutError("aggregate benchmark_id is invalid")
        if self.dataset_id != _REPOSITORY_HOLDOUT_DATASET_ID:
            raise RepositoryHoldoutError("aggregate dataset_id is invalid")
        if self.dataset_revision != _REPOSITORY_HOLDOUT_DATASET_REVISION:
            raise RepositoryHoldoutError("aggregate dataset_revision is invalid")
        if self.suite_version != _REPOSITORY_HOLDOUT_SUITE_VERSION:
            raise RepositoryHoldoutError("aggregate suite_version is invalid")
        for field_name, value in (
            ("suite_sha256", self.suite_sha256),
            ("runner_source_sha256", self.runner_source_sha256),
            ("evaluation_settings_sha256", self.evaluation_settings_sha256),
            ("results_sha256", self.results_sha256),
        ):
            _require_sha256(value, field_name=field_name)
        _require_non_empty(self.prompt_version, field_name="prompt_version")
        _require_non_empty(
            self.completion_normalizer_version,
            field_name="completion_normalizer_version",
        )
        if not _IMAGE_DIGEST_PATTERN.fullmatch(self.execution_image):
            raise RepositoryHoldoutError("aggregate execution_image must be digest-pinned")
        if not isinstance(self.base_model, BaseModelIdentity):
            raise RepositoryHoldoutError("aggregate base_model must be BaseModelIdentity")
        if not isinstance(self.adapter, AdapterIdentity):
            raise RepositoryHoldoutError("aggregate adapter must be AdapterIdentity")
        if self.total_problems <= 0:
            raise RepositoryHoldoutError("aggregate total_problems must be positive")
        if self.passed < 0 or self.failed < 0 or self.passed + self.failed != self.total_problems:
            raise RepositoryHoldoutError("aggregate pass/fail counts are inconsistent")
        if not 0 <= self.timed_out <= self.failed:
            raise RepositoryHoldoutError("aggregate timed_out count is inconsistent")
        if not 0 <= self.harness_errors <= self.failed:
            raise RepositoryHoldoutError("aggregate harness_errors count is inconsistent")
        if self.harness_errors and self.pass_at_1 is not None:
            raise RepositoryHoldoutError("pass_at_1 must be null when harness errors occurred")
        if not self.harness_errors and self.pass_at_1 != self.passed / self.total_problems:
            raise RepositoryHoldoutError("aggregate pass_at_1 is inconsistent")


@dataclass(frozen=True, slots=True)
class RepositoryHoldoutSuiteResult:
    """Per-task common results plus the deterministic aggregate."""

    results: tuple[EvaluationResult, ...]
    aggregate: RepositoryHoldoutAggregate

    def __post_init__(self) -> None:
        from tiny_qwen_coder.evaluation._repository_holdout_runtime import (
            repository_holdout_results_sha256,
        )

        if len(self.results) != self.aggregate.total_problems:
            raise RepositoryHoldoutError("result count must match aggregate total_problems")
        if repository_holdout_results_sha256(self.results) != self.aggregate.results_sha256:
            raise RepositoryHoldoutError("aggregate results_sha256 does not match results")
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
            raise RepositoryHoldoutError("aggregate passed count does not match results")
        if timed_out != self.aggregate.timed_out:
            raise RepositoryHoldoutError("aggregate timeout count does not match results")
        if harness_errors != self.aggregate.harness_errors:
            raise RepositoryHoldoutError("aggregate harness-error count does not match results")


@dataclass(frozen=True, slots=True)
class RepositoryHoldoutArtifactPaths:
    """Paths written for one repository holdout evaluation."""

    per_problem: Path
    aggregate: Path
