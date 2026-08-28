"""Typed schema for programming-language configuration.

This module defines the in-memory contract only. Config-file parsing and strict
unknown-field handling are introduced separately in P1-003.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_LANGUAGE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")
_HOOK_REFERENCE_PATTERN = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_.]*$"
)


def _require_non_empty(value: str, *, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")


def _require_unique(values: tuple[str, ...], *, field_name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicates")


@dataclass(frozen=True, slots=True)
class RepositoryDetectionSignals:
    """Repository markers used to identify a programming language."""

    files: tuple[str, ...] = ()
    directories: tuple[str, ...] = ()
    globs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name, values in (
            ("files", self.files),
            ("directories", self.directories),
            ("globs", self.globs),
        ):
            _require_unique(values, field_name=field_name)
            for value in values:
                _require_non_empty(value, field_name=field_name)

        if not (self.files or self.directories or self.globs):
            raise ValueError("repository detection must define at least one signal")


@dataclass(frozen=True, slots=True)
class SystemPromptSpec:
    """Versioned system prompt associated with a language adapter."""

    version: str
    text: str

    def __post_init__(self) -> None:
        _require_non_empty(self.version, field_name="system prompt version")
        _require_non_empty(self.text, field_name="system prompt text")


@dataclass(frozen=True, slots=True)
class ConfigReferences:
    """References to language-specific data and evaluation configuration."""

    data_sources: tuple[str, ...]
    evaluation: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name, values in (
            ("data_sources", self.data_sources),
            ("evaluation", self.evaluation),
        ):
            _require_unique(values, field_name=field_name)
            if not values:
                raise ValueError(f"{field_name} must contain at least one reference")
            for value in values:
                _require_non_empty(value, field_name=field_name)


@dataclass(frozen=True, slots=True)
class LanguageHookReferences:
    """Import references for language-specific validation and execution hooks."""

    validator: str
    executor: str

    def __post_init__(self) -> None:
        for field_name, value in (
            ("validator", self.validator),
            ("executor", self.executor),
        ):
            if not _HOOK_REFERENCE_PATTERN.fullmatch(value):
                raise ValueError(
                    f"{field_name} must use 'package.module:attribute' syntax"
                )


@dataclass(frozen=True, slots=True)
class LanguageConfig:
    """Language-neutral configuration contract for one programming language."""

    schema_version: int
    id: str
    aliases: tuple[str, ...]
    extensions: tuple[str, ...]
    repository_detection: RepositoryDetectionSignals
    system_prompt: SystemPromptSpec
    config_refs: ConfigReferences
    hooks: LanguageHookReferences

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported language config schema version")
        if not _LANGUAGE_ID_PATTERN.fullmatch(self.id):
            raise ValueError(
                "language id must match ^[a-z][a-z0-9_-]*$ for stable lookup"
            )

        _require_unique(self.aliases, field_name="aliases")
        _require_unique(self.extensions, field_name="extensions")

        for alias in self.aliases:
            if not _LANGUAGE_ID_PATTERN.fullmatch(alias):
                raise ValueError(f"invalid language alias: {alias!r}")
        if self.id in self.aliases:
            raise ValueError("aliases must not repeat the canonical language id")

        if not self.extensions:
            raise ValueError("extensions must contain at least one file extension")
        for extension in self.extensions:
            if not extension.startswith(".") or len(extension) == 1:
                raise ValueError(
                    f"file extension must start with '.' and include a suffix: {extension!r}"
                )
