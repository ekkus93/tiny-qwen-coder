"""Frozen generation/evaluation settings shared by base and adapter comparisons."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from tiny_qwen_coder.config import (
    EvaluationConfig,
    GenerationConfig,
    load_yaml_mapping,
    parse_generation_config,
)
from tiny_qwen_coder.reproducibility import SeedError, validate_seed

_EVALUATION_SETTINGS_SCHEMA_VERSION = 1
_CANONICAL_SETTINGS_ID = "canonical_evaluation"
_CANONICAL_SETTINGS_VERSION = 1
_CANONICAL_SETTINGS_PATH = Path("configs/eval/canonical_generation_v1.yaml")
_FROZEN_CANONICAL_SETTINGS_SHA256 = (
    "8660c83a561dc5d9896de2fbf5471ecec5c9763d46dcc16891b24f55e6262591"
)


class EvaluationSettingsError(ValueError):
    """Raised when frozen evaluation settings are malformed or drift unexpectedly."""


@dataclass(frozen=True, slots=True)
class FrozenEvaluationSettings:
    """One immutable generation protocol used for base/adapter comparisons."""

    schema_version: int
    settings_id: str
    settings_version: int
    frozen: bool
    seed: int
    generation: GenerationConfig

    def __post_init__(self) -> None:
        if self.schema_version != _EVALUATION_SETTINGS_SCHEMA_VERSION:
            raise EvaluationSettingsError(
                f"unsupported evaluation settings schema_version {self.schema_version}; "
                f"expected {_EVALUATION_SETTINGS_SCHEMA_VERSION}"
            )
        if not self.settings_id.strip():
            raise EvaluationSettingsError("settings_id must not be empty")
        if isinstance(self.settings_version, bool) or not isinstance(self.settings_version, int):
            raise EvaluationSettingsError("settings_version must be an integer")
        if self.settings_version <= 0:
            raise EvaluationSettingsError("settings_version must be greater than zero")
        if not isinstance(self.frozen, bool):
            raise EvaluationSettingsError("frozen must be a boolean")
        try:
            validate_seed(self.seed)
        except SeedError as exc:
            raise EvaluationSettingsError(str(exc)) from exc
        if not isinstance(self.generation, GenerationConfig):
            raise EvaluationSettingsError("generation must be a GenerationConfig")


def _validate_keys(
    mapping: Mapping[str, object],
    *,
    required: frozenset[str],
    context: str,
) -> None:
    unknown = sorted(set(mapping) - required)
    missing = sorted(required - set(mapping))
    if unknown:
        raise EvaluationSettingsError(
            f"{context} contains unknown field(s): {', '.join(unknown)}"
        )
    if missing:
        raise EvaluationSettingsError(
            f"{context} is missing required field(s): {', '.join(missing)}"
        )


def _expect_int(mapping: Mapping[str, object], key: str, *, context: str) -> int:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvaluationSettingsError(f"{context}.{key} must be an integer")
    return value


def _expect_str(mapping: Mapping[str, object], key: str, *, context: str) -> str:
    value = mapping[key]
    if not isinstance(value, str) or not value.strip():
        raise EvaluationSettingsError(f"{context}.{key} must be a non-empty string")
    return value


def _expect_bool(mapping: Mapping[str, object], key: str, *, context: str) -> bool:
    value = mapping[key]
    if not isinstance(value, bool):
        raise EvaluationSettingsError(f"{context}.{key} must be a boolean")
    return value


def load_evaluation_settings(path: Path) -> FrozenEvaluationSettings:
    """Load one strict versioned evaluation-settings YAML document."""

    try:
        mapping = load_yaml_mapping(path)
    except ValueError as exc:
        raise EvaluationSettingsError(str(exc)) from exc
    context = "evaluation settings"
    _validate_keys(
        mapping,
        required=frozenset(
            {
                "schema_version",
                "settings_id",
                "settings_version",
                "frozen",
                "seed",
                "generation",
            }
        ),
        context=context,
    )
    try:
        generation = parse_generation_config(
            mapping["generation"],
            context="evaluation settings.generation",
        )
    except ValueError as exc:
        raise EvaluationSettingsError(str(exc)) from exc
    return FrozenEvaluationSettings(
        schema_version=_expect_int(mapping, "schema_version", context=context),
        settings_id=_expect_str(mapping, "settings_id", context=context),
        settings_version=_expect_int(mapping, "settings_version", context=context),
        frozen=_expect_bool(mapping, "frozen", context=context),
        seed=_expect_int(mapping, "seed", context=context),
        generation=generation,
    )


def evaluation_settings_json(settings: FrozenEvaluationSettings) -> str:
    """Serialize settings deterministically for audit artifacts and hashing."""

    return json.dumps(asdict(settings), indent=2, sort_keys=True) + "\n"


def evaluation_settings_sha256(settings: FrozenEvaluationSettings) -> str:
    """Return the semantic SHA-256 of one evaluation-settings definition."""

    payload = evaluation_settings_json(settings).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_frozen_evaluation_settings(
    path: Path = _CANONICAL_SETTINGS_PATH,
) -> FrozenEvaluationSettings:
    """Load the canonical P4-006 settings and fail closed on unversioned drift."""

    settings = load_evaluation_settings(path)
    if settings.settings_id != _CANONICAL_SETTINGS_ID:
        raise EvaluationSettingsError(
            f"canonical settings_id must be {_CANONICAL_SETTINGS_ID!r}; "
            f"got {settings.settings_id!r}"
        )
    if settings.settings_version != _CANONICAL_SETTINGS_VERSION:
        raise EvaluationSettingsError(
            f"canonical settings_version must be {_CANONICAL_SETTINGS_VERSION}; "
            f"got {settings.settings_version}"
        )
    if not settings.frozen:
        raise EvaluationSettingsError("canonical evaluation settings must be frozen")
    fingerprint = evaluation_settings_sha256(settings)
    if fingerprint != _FROZEN_CANONICAL_SETTINGS_SHA256:
        raise EvaluationSettingsError(
            "canonical evaluation settings fingerprint mismatch; increment settings_version "
            "and explicitly update the frozen fingerprint before evaluation"
        )
    return settings


def validate_evaluation_config_settings(
    config: EvaluationConfig,
    settings: FrozenEvaluationSettings,
) -> str:
    """Require an evaluation config to use one exact frozen generation protocol."""

    if not settings.frozen:
        raise EvaluationSettingsError("comparison settings must be frozen")
    if config.seed != settings.seed:
        raise EvaluationSettingsError(
            f"evaluation seed {config.seed} does not match frozen seed {settings.seed}"
        )
    if config.generation != settings.generation:
        raise EvaluationSettingsError(
            "evaluation generation settings do not match the frozen comparison protocol"
        )
    return evaluation_settings_sha256(settings)
