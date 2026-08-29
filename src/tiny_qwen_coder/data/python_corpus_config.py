"""Strict composition configuration for the canonical Python P0 corpus."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from tiny_qwen_coder.reproducibility import SeedError, validate_seed

_SCHEMA_VERSION = 1
_DEFAULT_P0_SEED = 1729
_DEFAULT_VALIDATION_FRACTION = 0.05
DEFAULT_PYTHON_P0_CONFIG = Path("configs/data/python/p0.yaml")


class PythonP0CorpusError(ValueError):
    """Raised when the canonical Python P0 corpus cannot be constructed safely."""


@dataclass(frozen=True, slots=True)
class PythonP0SourceBudget:
    """Configured accepted-record target for one pinned Python source."""

    id: str
    source_config: str
    target_accepted: int

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise PythonP0CorpusError("source budget id must not be empty")
        if not self.source_config.strip():
            raise PythonP0CorpusError("source budget config path must not be empty")
        if self.target_accepted <= 0:
            raise PythonP0CorpusError("source target_accepted must be greater than zero")


@dataclass(frozen=True, slots=True)
class PythonP0CorpusConfig:
    """Frozen composition and filtering policy for the Python P0 corpus."""

    schema_version: int
    id: str
    language: str
    target_total: int
    min_tokens: int
    max_tokens: int
    sources: tuple[PythonP0SourceBudget, ...]
    fill_shortfall_from: str | None
    output_jsonl: str
    seed: int = _DEFAULT_P0_SEED
    validation_fraction: float = _DEFAULT_VALIDATION_FRACTION

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise PythonP0CorpusError("unsupported Python P0 corpus schema version")
        if self.id != "python-p0":
            raise PythonP0CorpusError("Python P0 corpus id must be 'python-p0'")
        if self.language != "python":
            raise PythonP0CorpusError("Python P0 corpus language must be 'python'")
        if self.target_total <= 0:
            raise PythonP0CorpusError("target_total must be greater than zero")
        if self.min_tokens < 1:
            raise PythonP0CorpusError("min_tokens must be at least 1")
        if self.max_tokens < self.min_tokens:
            raise PythonP0CorpusError("max_tokens must be >= min_tokens")
        if not self.sources:
            raise PythonP0CorpusError("sources must contain at least one source budget")
        source_ids = tuple(source.id for source in self.sources)
        if len(source_ids) != len(set(source_ids)):
            raise PythonP0CorpusError("sources must not repeat source IDs")
        source_paths = tuple(source.source_config for source in self.sources)
        if len(source_paths) != len(set(source_paths)):
            raise PythonP0CorpusError("sources must not repeat source config paths")
        if sum(source.target_accepted for source in self.sources) != self.target_total:
            raise PythonP0CorpusError("source targets must sum exactly to target_total")
        if self.fill_shortfall_from is not None:
            if self.fill_shortfall_from not in source_ids:
                raise PythonP0CorpusError("fill_shortfall_from must name one configured source ID")
            if self.fill_shortfall_from != source_ids[0]:
                raise PythonP0CorpusError(
                    "fill_shortfall_from must name the primary (first) configured source"
                )
        if not self.output_jsonl.strip():
            raise PythonP0CorpusError("output_jsonl must not be empty")
        try:
            validate_seed(self.seed)
        except SeedError as exc:
            raise PythonP0CorpusError(str(exc)) from exc
        if not 0.0 < self.validation_fraction < 1.0:
            raise PythonP0CorpusError("validation_fraction must be greater than 0 and less than 1")


def _strict_mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise PythonP0CorpusError(f"{context} must be a mapping")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise PythonP0CorpusError(f"{context} keys must be strings")
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
        raise PythonP0CorpusError(f"{context} contains unknown field(s): {', '.join(unknown)}")
    if missing:
        raise PythonP0CorpusError(f"{context} is missing required field(s): {', '.join(missing)}")


def _expect_str(mapping: Mapping[str, object], key: str, *, context: str) -> str:
    value = mapping[key]
    if not isinstance(value, str) or not value.strip():
        raise PythonP0CorpusError(f"{context}.{key} must be a non-empty string")
    return value


def _expect_int(mapping: Mapping[str, object], key: str, *, context: str) -> int:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise PythonP0CorpusError(f"{context}.{key} must be an integer")
    return value


def _expect_float(mapping: Mapping[str, object], key: str, *, context: str) -> float:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PythonP0CorpusError(f"{context}.{key} must be a number")
    return float(value)


def _parse_source_budget(value: object, *, index: int) -> PythonP0SourceBudget:
    context = f"python_p0.sources[{index}]"
    mapping = _strict_mapping(value, context=context)
    _validate_keys(
        mapping,
        required=frozenset({"id", "source_config", "target_accepted"}),
        context=context,
    )
    return PythonP0SourceBudget(
        id=_expect_str(mapping, "id", context=context),
        source_config=_expect_str(mapping, "source_config", context=context),
        target_accepted=_expect_int(mapping, "target_accepted", context=context),
    )


def parse_python_p0_config(value: object) -> PythonP0CorpusConfig:
    """Parse one strict canonical Python P0 composition config."""

    context = "python_p0"
    mapping = _strict_mapping(value, context=context)
    _validate_keys(
        mapping,
        required=frozenset(
            {
                "schema_version",
                "id",
                "language",
                "target_total",
                "min_tokens",
                "max_tokens",
                "sources",
                "fill_shortfall_from",
                "output_jsonl",
                "seed",
                "validation_fraction",
            }
        ),
        context=context,
    )
    raw_sources = mapping["sources"]
    if not isinstance(raw_sources, list):
        raise PythonP0CorpusError("python_p0.sources must be a YAML sequence")
    fill_value = mapping["fill_shortfall_from"]
    fill_source: str | None
    if fill_value is None:
        fill_source = None
    elif isinstance(fill_value, str) and fill_value.strip():
        fill_source = fill_value
    else:
        raise PythonP0CorpusError("python_p0.fill_shortfall_from must be a string or null")
    return PythonP0CorpusConfig(
        schema_version=_expect_int(mapping, "schema_version", context=context),
        id=_expect_str(mapping, "id", context=context),
        language=_expect_str(mapping, "language", context=context),
        target_total=_expect_int(mapping, "target_total", context=context),
        min_tokens=_expect_int(mapping, "min_tokens", context=context),
        max_tokens=_expect_int(mapping, "max_tokens", context=context),
        sources=tuple(
            _parse_source_budget(item, index=index) for index, item in enumerate(raw_sources)
        ),
        fill_shortfall_from=fill_source,
        output_jsonl=_expect_str(mapping, "output_jsonl", context=context),
        seed=_expect_int(mapping, "seed", context=context),
        validation_fraction=_expect_float(mapping, "validation_fraction", context=context),
    )


def load_python_p0_config(
    path: Path = DEFAULT_PYTHON_P0_CONFIG,
) -> PythonP0CorpusConfig:
    """Load the strict canonical Python P0 composition config from YAML."""

    try:
        loaded: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PythonP0CorpusError(f"could not read Python P0 config {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise PythonP0CorpusError(f"invalid YAML in {path}: {exc}") from exc
    return parse_python_p0_config(loaded)
