"""Language-neutral normalized training-record schema.

The schema captures stable structure and provenance only. Required-content and
conversation-shape filtering intentionally belongs to P3-004 so malformed or
empty examples can be rejected with explicit reason accounting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

MessageRole = Literal["system", "user", "assistant"]

_LANGUAGE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")
_COMPONENT_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]*$")
_ALLOWED_MESSAGE_ROLES = frozenset({"system", "user", "assistant"})


def _require_non_empty(value: str, *, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")


def _require_optional_non_empty(value: str | None, *, field_name: str) -> None:
    if value is not None:
        _require_non_empty(value, field_name=field_name)


@dataclass(frozen=True, slots=True)
class TrainingMessage:
    """One normalized chat message used by supervised fine-tuning."""

    role: MessageRole
    content: str

    def __post_init__(self) -> None:
        if self.role not in _ALLOWED_MESSAGE_ROLES:
            raise ValueError(f"unsupported training message role: {self.role!r}")
        if not isinstance(self.content, str):
            raise TypeError("training message content must be a string")


@dataclass(frozen=True, slots=True)
class LicenseMetadata:
    """License information retained from the upstream source."""

    name: str
    url: str | None = None
    attribution: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.name, field_name="license name")
        _require_optional_non_empty(self.url, field_name="license url")
        _require_optional_non_empty(self.attribution, field_name="license attribution")


@dataclass(frozen=True, slots=True)
class SourceProvenance:
    """Immutable identity and record-level provenance for one upstream source."""

    source_id: str
    revision: str
    license: LicenseMetadata
    split: str | None = None
    record_id: str | None = None
    url: str | None = None
    source_metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty(self.source_id, field_name="source id")
        _require_non_empty(self.revision, field_name="source revision")
        _require_optional_non_empty(self.split, field_name="source split")
        _require_optional_non_empty(self.record_id, field_name="source record id")
        _require_optional_non_empty(self.url, field_name="source url")
        metadata_keys: list[str] = []
        for key, value in self.source_metadata:
            _require_non_empty(key, field_name="source metadata key")
            _require_non_empty(value, field_name="source metadata value")
            metadata_keys.append(key)
        if len(metadata_keys) != len(set(metadata_keys)):
            raise ValueError("source metadata must not repeat keys")
        if tuple(sorted(self.source_metadata)) != self.source_metadata:
            raise ValueError("source metadata must be sorted by key")


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Result emitted by one language-specific or generic validator."""

    validator_id: str
    passed: bool
    detail: str | None = None

    def __post_init__(self) -> None:
        if not _COMPONENT_ID_PATTERN.fullmatch(self.validator_id):
            raise ValueError("validator_id must be a stable lowercase component ID")
        if not isinstance(self.passed, bool):
            raise TypeError("validation passed flag must be boolean")
        _require_optional_non_empty(self.detail, field_name="validation detail")


@dataclass(frozen=True, slots=True)
class ValidationMetadata:
    """Optional validation evidence attached to a normalized record."""

    results: tuple[ValidationResult, ...]

    def __post_init__(self) -> None:
        if not self.results:
            raise ValueError("validation metadata must contain at least one result")
        validator_ids = tuple(result.validator_id for result in self.results)
        if len(validator_ids) != len(set(validator_ids)):
            raise ValueError("validation metadata must not repeat validator IDs")

    @property
    def passed(self) -> bool:
        """Return whether every recorded validator passed."""

        return all(result.passed for result in self.results)


@dataclass(frozen=True, slots=True)
class NormalizedTrainingRecord:
    """Canonical language-neutral representation consumed by the data pipeline."""

    schema_version: int
    messages: tuple[TrainingMessage, ...]
    language: str
    provenance: SourceProvenance
    validation: ValidationMetadata | None = None

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported normalized training-record schema version")
        if not _LANGUAGE_ID_PATTERN.fullmatch(self.language):
            raise ValueError("language must be a stable lowercase language ID")


def single_turn_messages(
    *, system: str | None, user: str, assistant: str
) -> tuple[TrainingMessage, ...]:
    """Build normalized messages for a single-turn upstream example.

    Content is deliberately not required to be non-empty here. P3-004 owns
    required-content filtering and rejection-reason accounting.
    """

    messages: list[TrainingMessage] = []
    if system is not None:
        messages.append(TrainingMessage(role="system", content=system))
    messages.extend(
        (
            TrainingMessage(role="user", content=user),
            TrainingMessage(role="assistant", content=assistant),
        )
    )
    return tuple(messages)
