"""Deterministic protected-benchmark contamination checks for prepared SFT data.

P4-002 compares the deduplicated prepared training corpus against exact protected
benchmark examples. Exact checks reuse the same conservative P3 text
normalization and content hashing used by dataset deduplication. A separate,
explicitly heuristic high-overlap check reports suspicious lexical containment
without treating it as an exact match.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from tiny_qwen_coder.evaluation.protected_benchmarks import ProtectedBenchmarkRegistry

if TYPE_CHECKING:
    from tiny_qwen_coder.data.records import NormalizedTrainingRecord, TrainingMessage
    from tiny_qwen_coder.reporting.dataset_manifest import (
        ContaminationFinding,
        ContaminationSummary,
    )

EXACT_PROMPT_CHECK_ID = "exact_prompt"
EXACT_SOLUTION_CHECK_ID = "exact_solution"
HIGH_OVERLAP_CHECK_ID = "high_overlap"

_EXACT_PROMPT_FINDING_TYPE = "exact_prompt_match"
_EXACT_SOLUTION_FINDING_TYPE = "exact_solution_match"
_HIGH_PROMPT_FINDING_TYPE = "high_prompt_overlap"
_HIGH_SOLUTION_FINDING_TYPE = "high_solution_overlap"

_LANGUAGE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")
_BENCHMARK_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]*$")
_OVERLAP_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)

_Shingle = tuple[str, ...]


class ContaminationCheckError(ValueError):
    """Raised when contamination checks cannot produce trustworthy evidence."""


@dataclass(frozen=True, slots=True)
class HighOverlapConfig:
    """Stable lexical-shingle policy for suspicious-overlap reporting.

    ``threshold`` uses the overlap coefficient: shared shingles divided by the
    smaller shingle-set size. This intentionally catches a benchmark item that
    has been embedded inside a longer training example. Exact matches are
    reported by the exact checkers instead of duplicated as high-overlap hits.
    """

    threshold: float = 0.8
    shingle_size: int = 5
    min_tokens: int = 16

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError("high-overlap threshold must be greater than zero and at most one")
        if self.shingle_size < 1:
            raise ValueError("high-overlap shingle_size must be at least one")
        if self.min_tokens < self.shingle_size:
            raise ValueError("high-overlap min_tokens must be at least shingle_size")


@dataclass(frozen=True, slots=True)
class ProtectedBenchmarkExample:
    """One exact example loaded from a registered protected benchmark revision."""

    language: str
    benchmark_id: str
    dataset_id: str
    dataset_revision: str
    record_id: str
    prompt_messages: tuple[TrainingMessage, ...]
    solution: str | None = None

    def __post_init__(self) -> None:
        if not _LANGUAGE_ID_PATTERN.fullmatch(self.language):
            raise ValueError("protected example language must be a stable language ID")
        if not _BENCHMARK_ID_PATTERN.fullmatch(self.benchmark_id):
            raise ValueError("protected example benchmark_id must be a stable component ID")
        for field_name, value in (
            ("dataset_id", self.dataset_id),
            ("dataset_revision", self.dataset_revision),
            ("record_id", self.record_id),
        ):
            if not value or value != value.strip():
                raise ValueError(f"protected example {field_name} must be exact and non-empty")
        if not self.prompt_messages:
            raise ValueError("protected example prompt_messages must not be empty")
        if self.prompt_messages[-1].role != "user":
            raise ValueError("protected example prompt_messages must end with a user message")
        if self.solution is not None and not self.solution.strip():
            raise ValueError("protected example solution must not be empty when provided")


@dataclass(frozen=True, slots=True)
class _PreparedProtectedExample:
    example: ProtectedBenchmarkExample
    prompt_sha256: str
    solution_sha256: str | None
    prompt_shingles: frozenset[_Shingle]
    solution_shingles: frozenset[_Shingle]


def _normalize_text(text: str) -> str:
    from tiny_qwen_coder.data.filtering import normalize_training_text

    return normalize_training_text(text)


def _overlap_tokens(text: str) -> tuple[str, ...]:
    normalized = _normalize_text(text)
    return tuple(_OVERLAP_TOKEN_PATTERN.findall(normalized))


def _message_overlap_text(messages: Sequence[TrainingMessage]) -> str:
    return "\n".join(
        f"<{message.role}>\n{_normalize_text(message.content)}" for message in messages
    )


def _shingles(text: str, config: HighOverlapConfig) -> frozenset[_Shingle]:
    tokens = _overlap_tokens(text)
    if len(tokens) < config.min_tokens:
        return frozenset()
    return frozenset(
        tuple(tokens[index : index + config.shingle_size])
        for index in range(len(tokens) - config.shingle_size + 1)
    )


def _overlap_score(left: frozenset[_Shingle], right: frozenset[_Shingle]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def _prepare_protected_examples(
    examples: Sequence[ProtectedBenchmarkExample],
    *,
    language: str,
    registry: ProtectedBenchmarkRegistry,
    overlap: HighOverlapConfig,
) -> tuple[_PreparedProtectedExample, ...]:
    from tiny_qwen_coder.data.deduplication import (
        normalized_prompt_sha256,
        normalized_response_sha256,
    )

    expected_benchmarks = registry.list_benchmarks(language=language)
    expected_ids = {benchmark.id for benchmark in expected_benchmarks}
    if not expected_ids:
        raise ContaminationCheckError(
            f"no protected benchmarks are registered for contamination language {language!r}"
        )
    if not examples:
        raise ContaminationCheckError(
            f"protected benchmark examples are required for contamination language {language!r}"
        )

    provided_ids: set[str] = set()
    identities: set[tuple[str, str]] = set()
    prepared: list[_PreparedProtectedExample] = []
    for example in examples:
        if example.language != language:
            raise ContaminationCheckError(
                f"protected example {example.record_id!r} uses language {example.language!r}; "
                f"expected {language!r}"
            )
        try:
            benchmark = registry.resolve(language, example.benchmark_id)
        except ValueError as exc:
            raise ContaminationCheckError(
                f"protected example {example.record_id!r} references unregistered benchmark "
                f"{language}/{example.benchmark_id}"
            ) from exc
        if (
            example.dataset_id != benchmark.dataset_id
            or example.dataset_revision != benchmark.dataset_revision
        ):
            raise ContaminationCheckError(
                f"protected example {example.record_id!r} does not match registered dataset "
                f"identity for {benchmark.qualified_id!r}"
            )

        identity = (example.benchmark_id, example.record_id)
        if identity in identities:
            raise ContaminationCheckError(
                f"duplicate protected example identity {example.benchmark_id}/{example.record_id}"
            )
        identities.add(identity)
        provided_ids.add(example.benchmark_id)

        solution_sha256 = (
            normalized_response_sha256(example.solution) if example.solution is not None else None
        )
        prepared.append(
            _PreparedProtectedExample(
                example=example,
                prompt_sha256=normalized_prompt_sha256(example.prompt_messages),
                solution_sha256=solution_sha256,
                prompt_shingles=_shingles(_message_overlap_text(example.prompt_messages), overlap),
                solution_shingles=(
                    _shingles(example.solution, overlap)
                    if example.solution is not None
                    else frozenset()
                ),
            )
        )

    missing = tuple(sorted(expected_ids - provided_ids))
    unexpected = tuple(sorted(provided_ids - expected_ids))
    if missing or unexpected:
        raise ContaminationCheckError(
            f"protected example coverage does not match registered benchmarks for {language!r}; "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )

    return tuple(
        sorted(
            prepared,
            key=lambda item: (item.example.benchmark_id, item.example.record_id),
        )
    )


def _index_exact(
    protected: Sequence[_PreparedProtectedExample],
    *,
    solution: bool,
) -> dict[str, tuple[int, ...]]:
    staged: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(protected):
        digest = item.solution_sha256 if solution else item.prompt_sha256
        if digest is not None:
            staged[digest].append(index)
    return {digest: tuple(indexes) for digest, indexes in staged.items()}


def _index_shingles(
    protected: Sequence[_PreparedProtectedExample],
    *,
    solution: bool,
) -> dict[_Shingle, tuple[int, ...]]:
    staged: dict[_Shingle, list[int]] = defaultdict(list)
    for index, item in enumerate(protected):
        values = item.solution_shingles if solution else item.prompt_shingles
        for shingle in values:
            staged[shingle].append(index)
    return {shingle: tuple(indexes) for shingle, indexes in staged.items()}


def _candidate_indexes(
    shingles: frozenset[_Shingle],
    index: dict[_Shingle, tuple[int, ...]],
) -> tuple[int, ...]:
    candidates: set[int] = set()
    for shingle in shingles:
        candidates.update(index.get(shingle, ()))
    return tuple(sorted(candidates))


def _finding(
    *,
    checker_id: str,
    finding_type: str,
    training_record_sha256: str,
    protected: _PreparedProtectedExample,
    score: float,
    detail: str,
) -> ContaminationFinding:
    from tiny_qwen_coder.reporting.dataset_manifest import ContaminationFinding

    return ContaminationFinding(
        checker_id=checker_id,
        protected_dataset_id=protected.example.benchmark_id,
        finding_type=finding_type,
        training_record_sha256=training_record_sha256,
        protected_record_id=protected.example.record_id,
        score=round(score, 6),
        detail=detail,
    )


def check_training_contamination(
    training_records: Iterable[NormalizedTrainingRecord],
    protected_examples: Sequence[ProtectedBenchmarkExample],
    *,
    language: str,
    registry: ProtectedBenchmarkRegistry,
    overlap: HighOverlapConfig | None = None,
) -> ContaminationSummary:
    """Check retained training records against all protected examples for a language.

    Exact prompt/solution checks use P3's content fingerprints. Suspicious
    high-overlap uses token shingles and the overlap coefficient, and only runs
    for text long enough to satisfy ``HighOverlapConfig.min_tokens``. The
    protected example set must cover every registered benchmark for ``language``
    before a clean result can be emitted.
    """

    from tiny_qwen_coder.data.deduplication import normalized_record_fingerprint
    from tiny_qwen_coder.reporting.dataset_manifest import (
        ContaminationStatus,
        ContaminationSummary,
    )

    if not _LANGUAGE_ID_PATTERN.fullmatch(language):
        raise ValueError("contamination language must be a stable language ID")
    selected_overlap = overlap or HighOverlapConfig()

    protected = _prepare_protected_examples(
        protected_examples,
        language=language,
        registry=registry,
        overlap=selected_overlap,
    )
    has_solutions = any(item.solution_sha256 is not None for item in protected)
    check_ids = tuple(
        sorted(
            (
                EXACT_PROMPT_CHECK_ID,
                HIGH_OVERLAP_CHECK_ID,
                *((EXACT_SOLUTION_CHECK_ID,) if has_solutions else ()),
            )
        )
    )

    prompt_exact_index = _index_exact(protected, solution=False)
    solution_exact_index = _index_exact(protected, solution=True)
    prompt_shingle_index = _index_shingles(protected, solution=False)
    solution_shingle_index = _index_shingles(protected, solution=True)

    findings: list[ContaminationFinding] = []
    for record in training_records:
        if record.language != language:
            raise ContaminationCheckError(
                f"training record language {record.language!r} does not match contamination "
                f"language {language!r}"
            )
        fingerprint = normalized_record_fingerprint(record)
        prompt_messages = record.messages[:-1]
        response = record.messages[-1].content

        exact_prompt_indexes = set(prompt_exact_index.get(fingerprint.prompt_sha256, ()))
        for index in sorted(exact_prompt_indexes):
            findings.append(
                _finding(
                    checker_id=EXACT_PROMPT_CHECK_ID,
                    finding_type=_EXACT_PROMPT_FINDING_TYPE,
                    training_record_sha256=fingerprint.record_sha256,
                    protected=protected[index],
                    score=1.0,
                    detail="exact prompt match after P3 text normalization",
                )
            )

        exact_solution_indexes = set(solution_exact_index.get(fingerprint.response_sha256, ()))
        for index in sorted(exact_solution_indexes):
            findings.append(
                _finding(
                    checker_id=EXACT_SOLUTION_CHECK_ID,
                    finding_type=_EXACT_SOLUTION_FINDING_TYPE,
                    training_record_sha256=fingerprint.record_sha256,
                    protected=protected[index],
                    score=1.0,
                    detail="exact solution/code match after P3 text normalization",
                )
            )

        prompt_shingles = _shingles(_message_overlap_text(prompt_messages), selected_overlap)
        for index in _candidate_indexes(prompt_shingles, prompt_shingle_index):
            if index in exact_prompt_indexes:
                continue
            score = _overlap_score(prompt_shingles, protected[index].prompt_shingles)
            if score >= selected_overlap.threshold:
                findings.append(
                    _finding(
                        checker_id=HIGH_OVERLAP_CHECK_ID,
                        finding_type=_HIGH_PROMPT_FINDING_TYPE,
                        training_record_sha256=fingerprint.record_sha256,
                        protected=protected[index],
                        score=score,
                        detail=(
                            f"prompt {selected_overlap.shingle_size}-token shingle overlap "
                            f"coefficient >= {selected_overlap.threshold:.3f}"
                        ),
                    )
                )

        response_shingles = _shingles(response, selected_overlap)
        for index in _candidate_indexes(response_shingles, solution_shingle_index):
            if index in exact_solution_indexes:
                continue
            score = _overlap_score(response_shingles, protected[index].solution_shingles)
            if score >= selected_overlap.threshold:
                findings.append(
                    _finding(
                        checker_id=HIGH_OVERLAP_CHECK_ID,
                        finding_type=_HIGH_SOLUTION_FINDING_TYPE,
                        training_record_sha256=fingerprint.record_sha256,
                        protected=protected[index],
                        score=score,
                        detail=(
                            f"solution/code {selected_overlap.shingle_size}-token shingle overlap "
                            f"coefficient >= {selected_overlap.threshold:.3f}"
                        ),
                    )
                )

    ordered_findings = tuple(
        sorted(
            findings,
            key=lambda finding: (
                finding.training_record_sha256,
                finding.protected_dataset_id,
                finding.protected_record_id or "",
                finding.checker_id,
                finding.finding_type,
            ),
        )
    )
    status = ContaminationStatus.FINDINGS if ordered_findings else ContaminationStatus.CLEAN
    return ContaminationSummary(status=status, check_ids=check_ids, findings=ordered_findings)
