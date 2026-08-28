"""Runtime programming-language plugin contract.

The declarative :class:`LanguageConfig` remains the source of truth for stable
language identity and repository-detection metadata.  This module layers the
runtime plugin surface on top of that configuration without defining the later
registry, normalized training-record schema, benchmark registry, or evaluation
result schema.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from tiny_qwen_coder.languages.schema import LanguageConfig, RepositoryDetectionSignals

_COMPONENT_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]*$")
_IMPORT_REFERENCE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_.]*$")


def _require_component_id(value: str, *, field_name: str) -> None:
    if not _COMPONENT_ID_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must be a stable lowercase component ID")


def _require_unique_component_ids(
    values: tuple[LanguageComponentRef, ...], *, field_name: str
) -> None:
    ids = tuple(value.id for value in values)
    if len(ids) != len(set(ids)):
        raise ValueError(f"{field_name} must not contain duplicate component IDs")


@dataclass(frozen=True, slots=True)
class LanguageComponentRef:
    """Stable identity plus import reference for one language-plugin component."""

    id: str
    import_ref: str

    def __post_init__(self) -> None:
        _require_component_id(self.id, field_name="component id")
        if not _IMPORT_REFERENCE_PATTERN.fullmatch(self.import_ref):
            raise ValueError("component import_ref must use 'package.module:attribute' syntax")


@dataclass(frozen=True, slots=True)
class ProtectedBenchmarkRef:
    """Language-owned protected benchmark identity.

    P4-001 binds this stable plugin-owned ID to an exact protected dataset
    registration. Normal SFT data preparation must fail closed when that
    registration is missing or selected as training input.
    """

    id: str

    def __post_init__(self) -> None:
        _require_component_id(self.id, field_name="protected benchmark id")


@dataclass(frozen=True, slots=True)
class LanguageSpec:
    """Language-neutral runtime specification for one programming language."""

    config: LanguageConfig
    execution_hook: LanguageComponentRef
    data_adapters: tuple[LanguageComponentRef, ...] = ()
    validators: tuple[LanguageComponentRef, ...] = ()
    protected_benchmarks: tuple[ProtectedBenchmarkRef, ...] = ()
    evaluation_hooks: tuple[LanguageComponentRef, ...] = ()

    def __post_init__(self) -> None:
        for field_name, values in (
            ("data_adapters", self.data_adapters),
            ("validators", self.validators),
            ("evaluation_hooks", self.evaluation_hooks),
        ):
            _require_unique_component_ids(values, field_name=field_name)

        benchmark_ids = tuple(benchmark.id for benchmark in self.protected_benchmarks)
        if len(benchmark_ids) != len(set(benchmark_ids)):
            raise ValueError("protected_benchmarks must not contain duplicate benchmark IDs")

        if self.execution_hook.import_ref != self.config.hooks.executor:
            raise ValueError("execution_hook must match the executor declared by LanguageConfig")

        if self.validators and self.config.hooks.validator not in {
            validator.import_ref for validator in self.validators
        }:
            raise ValueError(
                "validators must include the primary validator declared by LanguageConfig"
            )

    @property
    def id(self) -> str:
        """Return the canonical stable language ID."""

        return self.config.id

    @property
    def aliases(self) -> tuple[str, ...]:
        """Return stable aliases accepted for language selection."""

        return self.config.aliases

    @property
    def extensions(self) -> tuple[str, ...]:
        """Return file extensions associated with the language."""

        return self.config.extensions

    @property
    def repository_detection(self) -> RepositoryDetectionSignals:
        """Return repository-level language-detection signals."""

        return self.config.repository_detection


@runtime_checkable
class LanguagePlugin(Protocol):
    """Minimal interface consumed by the later language registry and pipelines."""

    @property
    def spec(self) -> LanguageSpec:
        """Return this plugin's immutable language specification."""

        ...


@dataclass(frozen=True, slots=True)
class StaticLanguagePlugin:
    """Simple declarative implementation suitable for normal language plugins."""

    spec: LanguageSpec
