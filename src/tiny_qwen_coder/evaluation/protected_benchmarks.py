"""Protected evaluation-dataset registration and SFT access control.

Protected benchmarks are declared by language plugins using stable IDs. This
module binds those IDs to exact upstream dataset identities and source-config
selectors, then provides the fail-closed guard used by normal dataset
preparation. Evaluation-only data must never be selected as SFT input.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from tiny_qwen_coder.config import DataPreparationConfig
from tiny_qwen_coder.languages.spec import LanguagePlugin

_BENCHMARK_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]*$")
_LANGUAGE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")


class ProtectedBenchmarkRegistryError(ValueError):
    """Base class for protected-benchmark registry failures."""


class ProtectedBenchmarkRegistrationError(ProtectedBenchmarkRegistryError):
    """Raised when protected benchmark registration is invalid or ambiguous."""


class UnknownProtectedBenchmarkError(ProtectedBenchmarkRegistryError):
    """Raised when a protected benchmark selection is not registered."""


class ProtectedBenchmarkTrainingSelectionError(ProtectedBenchmarkRegistryError):
    """Raised when SFT input selects an evaluation-only protected benchmark."""


def _require_exact_non_empty(value: str, *, field_name: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    if value != value.strip():
        raise ValueError(f"{field_name} must not contain leading or trailing whitespace")


@dataclass(frozen=True, slots=True)
class ProtectedBenchmark:
    """One language-owned evaluation-only dataset registration.

    ``dataset_id`` plus ``dataset_revision`` identify the exact upstream
    dataset used for audit and future contamination checks. ``source_configs``
    are exact configuration selectors that must never appear in a normal
    ``DataPreparationConfig``.
    The dataset identity itself is also treated as a protected SFT selector so
    a direct dataset reference cannot bypass the config-path guard.
    """

    language: str
    id: str
    dataset_id: str
    dataset_revision: str
    source_configs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not _LANGUAGE_ID_PATTERN.fullmatch(self.language):
            raise ValueError("protected benchmark language must be a stable language ID")
        if not _BENCHMARK_ID_PATTERN.fullmatch(self.id):
            raise ValueError("protected benchmark id must be a stable lowercase component ID")
        _require_exact_non_empty(self.dataset_id, field_name="dataset_id")
        _require_exact_non_empty(self.dataset_revision, field_name="dataset_revision")
        if not self.source_configs:
            raise ValueError("source_configs must contain at least one protected selector")
        if len(self.source_configs) != len(set(self.source_configs)):
            raise ValueError("source_configs must not contain duplicates")
        for source_config in self.source_configs:
            _require_exact_non_empty(source_config, field_name="source_config")

    @property
    def qualified_id(self) -> str:
        """Return the deterministic language-qualified benchmark identity."""

        return f"{self.language}/{self.id}"

    @property
    def sft_selectors(self) -> tuple[str, ...]:
        """Return all exact selectors forbidden in normal SFT source configs."""

        return (self.dataset_id, *self.source_configs)


def _strict_mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ProtectedBenchmarkRegistrationError(f"{context} must be a mapping")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ProtectedBenchmarkRegistrationError(f"{context} keys must be strings")
        result[key] = item
    return result


def _validate_keys(
    mapping: Mapping[str, object],
    *,
    required: frozenset[str],
    context: str,
) -> None:
    unknown = sorted(set(mapping) - required)
    missing = sorted(required - set(mapping))
    if unknown:
        raise ProtectedBenchmarkRegistrationError(
            f"{context} contains unknown field(s): {', '.join(unknown)}"
        )
    if missing:
        raise ProtectedBenchmarkRegistrationError(
            f"{context} is missing required field(s): {', '.join(missing)}"
        )


def _expect_str(mapping: Mapping[str, object], key: str, *, context: str) -> str:
    value = mapping[key]
    if not isinstance(value, str):
        raise ProtectedBenchmarkRegistrationError(f"{context}.{key} must be a string")
    try:
        _require_exact_non_empty(value, field_name=f"{context}.{key}")
    except ValueError as exc:
        raise ProtectedBenchmarkRegistrationError(str(exc)) from exc
    return value


def _expect_int(mapping: Mapping[str, object], key: str, *, context: str) -> int:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtectedBenchmarkRegistrationError(f"{context}.{key} must be an integer")
    return value


def _expect_str_tuple(
    mapping: Mapping[str, object], key: str, *, context: str
) -> tuple[str, ...]:
    value = mapping[key]
    if not isinstance(value, list):
        raise ProtectedBenchmarkRegistrationError(f"{context}.{key} must be a YAML sequence")
    output: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ProtectedBenchmarkRegistrationError(
                f"{context}.{key}[{index}] must be a string"
            )
        try:
            _require_exact_non_empty(item, field_name=f"{context}.{key}[{index}]")
        except ValueError as exc:
            raise ProtectedBenchmarkRegistrationError(str(exc)) from exc
        output.append(item)
    return tuple(output)


def parse_protected_benchmark_config(value: object) -> ProtectedBenchmark:
    """Parse one strict protected-benchmark registration mapping."""

    context = "protected_benchmark"
    mapping = _strict_mapping(value, context=context)
    _validate_keys(
        mapping,
        required=frozenset(
            {
                "schema_version",
                "language",
                "id",
                "dataset_id",
                "dataset_revision",
                "source_configs",
            }
        ),
        context=context,
    )
    if _expect_int(mapping, "schema_version", context=context) != 1:
        raise ProtectedBenchmarkRegistrationError(
            "unsupported protected-benchmark config schema version"
        )
    try:
        return ProtectedBenchmark(
            language=_expect_str(mapping, "language", context=context),
            id=_expect_str(mapping, "id", context=context),
            dataset_id=_expect_str(mapping, "dataset_id", context=context),
            dataset_revision=_expect_str(mapping, "dataset_revision", context=context),
            source_configs=_expect_str_tuple(mapping, "source_configs", context=context),
        )
    except ValueError as exc:
        raise ProtectedBenchmarkRegistrationError(str(exc)) from exc


def load_protected_benchmark_config(path: Path) -> ProtectedBenchmark:
    """Load one strict protected-benchmark registration from YAML."""

    try:
        loaded: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ProtectedBenchmarkRegistrationError(
            f"could not read protected benchmark config {path}: {exc}"
        ) from exc
    except yaml.YAMLError as exc:
        raise ProtectedBenchmarkRegistrationError(
            f"invalid YAML in protected benchmark config {path}: {exc}"
        ) from exc
    return parse_protected_benchmark_config(loaded)


class ProtectedBenchmarkRegistry:
    """Register protected datasets per language and guard SFT configuration.

    Language registration is atomic. The benchmark IDs registered for one
    plugin must exactly match ``LanguageSpec.protected_benchmarks`` so a plugin
    cannot declare protection without binding it to concrete protected source
    selectors. Protected SFT selectors occupy one global namespace; this keeps
    a training config from bypassing protection by claiming another language.
    """

    def __init__(self) -> None:
        self._benchmarks_by_key: dict[tuple[str, str], ProtectedBenchmark] = {}
        self._keys_by_language: dict[str, tuple[tuple[str, str], ...]] = {}
        self._key_by_sft_selector: dict[str, tuple[str, str]] = {}

    def register_language(
        self,
        plugin: LanguagePlugin,
        benchmarks: Iterable[ProtectedBenchmark],
    ) -> None:
        """Atomically register all protected benchmarks declared by a plugin."""

        language = plugin.spec.id
        if language in self._keys_by_language:
            raise ProtectedBenchmarkRegistrationError(
                f"protected benchmarks are already registered for language {language!r}"
            )

        registrations = tuple(benchmarks)
        declared_ids = tuple(sorted(ref.id for ref in plugin.spec.protected_benchmarks))
        registered_ids = tuple(sorted(benchmark.id for benchmark in registrations))
        if declared_ids != registered_ids:
            raise ProtectedBenchmarkRegistrationError(
                f"protected benchmark declarations for language {language!r} do not match "
                f"registrations; declared={declared_ids!r}, registered={registered_ids!r}"
            )

        keys: list[tuple[str, str]] = []
        staged_owners: dict[str, ProtectedBenchmark] = {}
        for benchmark in registrations:
            if benchmark.language != language:
                raise ProtectedBenchmarkRegistrationError(
                    f"benchmark {benchmark.qualified_id!r} cannot be registered for "
                    f"language {language!r}"
                )

            key = (benchmark.language, benchmark.id)
            if key in self._benchmarks_by_key or key in keys:
                raise ProtectedBenchmarkRegistrationError(
                    f"protected benchmark {benchmark.qualified_id!r} is already registered"
                )
            keys.append(key)

            for selector in benchmark.sft_selectors:
                existing_key = self._key_by_sft_selector.get(selector)
                existing_owner = (
                    self._benchmarks_by_key[existing_key]
                    if existing_key is not None
                    else staged_owners.get(selector)
                )
                if existing_owner is not None:
                    raise ProtectedBenchmarkRegistrationError(
                        f"protected SFT selector {selector!r} is already owned by "
                        f"benchmark {existing_owner.qualified_id!r}"
                    )
                staged_owners[selector] = benchmark

        ordered_keys = tuple(sorted(keys))
        self._keys_by_language[language] = ordered_keys
        for benchmark in registrations:
            key = (benchmark.language, benchmark.id)
            self._benchmarks_by_key[key] = benchmark
        self._key_by_sft_selector.update(
            {selector: (owner.language, owner.id) for selector, owner in staged_owners.items()}
        )

    def assert_plugin_registration_matches(self, plugin: LanguagePlugin) -> None:
        """Fail if a plugin's protected declarations are not fully registered."""

        language = plugin.spec.id
        declared_ids = tuple(sorted(ref.id for ref in plugin.spec.protected_benchmarks))
        registered_ids = tuple(
            benchmark.id for benchmark in self.list_benchmarks(language=language)
        )
        if declared_ids != registered_ids:
            raise ProtectedBenchmarkRegistrationError(
                f"protected benchmark registry mismatch for language {language!r}; "
                f"declared={declared_ids!r}, registered={registered_ids!r}"
            )

    def assert_sft_config_allowed(self, config: DataPreparationConfig) -> None:
        """Reject any SFT source selector owned by a protected benchmark."""

        conflicts: list[tuple[str, ProtectedBenchmark]] = []
        for source_config in config.source_configs:
            key = self._key_by_sft_selector.get(source_config)
            if key is not None:
                conflicts.append((source_config, self._benchmarks_by_key[key]))

        if not conflicts:
            return

        details = ", ".join(
            f"{selector!r} -> {benchmark.qualified_id!r}"
            for selector, benchmark in sorted(
                conflicts,
                key=lambda item: (item[0], item[1].qualified_id),
            )
        )
        raise ProtectedBenchmarkTrainingSelectionError(
            f"SFT data config for language {config.language!r} selects protected "
            f"evaluation-only benchmark source(s): {details}; protected benchmarks "
            "cannot be used for training"
        )

    def resolve(self, language: str, benchmark_id: str) -> ProtectedBenchmark:
        """Resolve one exact canonical language/benchmark ID pair."""

        key = (language, benchmark_id)
        benchmark = self._benchmarks_by_key.get(key)
        if benchmark is None:
            available = ", ".join(item.qualified_id for item in self.list_benchmarks()) or "<none>"
            raise UnknownProtectedBenchmarkError(
                f"unknown protected benchmark {language}/{benchmark_id}; "
                f"registered benchmarks: {available}"
            )
        return benchmark

    def list_benchmarks(self, *, language: str | None = None) -> tuple[ProtectedBenchmark, ...]:
        """List protected benchmarks deterministically by language then ID."""

        if language is None:
            keys = tuple(sorted(self._benchmarks_by_key))
        else:
            keys = self._keys_by_language.get(language, ())
        return tuple(self._benchmarks_by_key[key] for key in keys)

    def list_languages(self) -> tuple[str, ...]:
        """Return languages with explicit protected-benchmark registration."""

        return tuple(sorted(self._keys_by_language))
