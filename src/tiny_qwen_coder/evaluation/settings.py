"""Frozen generation/evaluation settings shared by base and adapter comparisons."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from tiny_qwen_coder.config import EvaluationConfig
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
class FrozenGenerationSettings:
    """Exact decoding and prompt/template settings used for comparisons."""

    decoding_strategy: str
    temperature: float
    top_p: float
    top_k: int
    max_new_tokens: int
    stop_policy: str
    prompt_version: str
    chat_template_version: str

    def __post_init__(self) -> None:
        if self.decoding_strategy != "greedy":
            raise EvaluationSettingsError("decoding_strategy must be greedy")
        if self.temperature != 0.0:
            raise EvaluationSettingsError("greedy generation requires temperature=0")
        if self.top_p != 1.0:
            raise EvaluationSettingsError("greedy generation requires top_p=1")
        if self.top_k != 0:
            raise EvaluationSettingsError("greedy generation requires top_k=0")
        if self.max_new_tokens <= 0:
            raise EvaluationSettingsError("max_new_tokens must be greater than zero")
        if self.stop_policy != "eos_or_max_new_tokens":
            raise EvaluationSettingsError("stop_policy must be eos_or_max_new_tokens")
        if not self.prompt_version.strip():
            raise EvaluationSettingsError("prompt_version must not be empty")
        if not self.chat_template_version.strip():
            raise EvaluationSettingsError("chat_template_version must not be empty")


@dataclass(frozen=True, slots=True)
class FrozenEvaluationSettings:
    """One immutable generation protocol used for base/adapter comparisons."""

    schema_version: int
    settings_id: str
    settings_version: int
    frozen: bool
    seed: int
    generation: FrozenGenerationSettings

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


def _strict_mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise EvaluationSettingsError(f"{context} must be a mapping")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise EvaluationSettingsError(f"{context} keys must be strings")
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


def _expect_float(mapping: Mapping[str, object], key: str, *, context: str) -> float:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationSettingsError(f"{context}.{key} must be a number")
    return float(value)


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


def _parse_generation(value: object) -> FrozenGenerationSettings:
    context = "evaluation settings.generation"
    mapping = _strict_mapping(value, context=context)
    _validate_keys(
        mapping,
        required=frozenset(
            {
                "decoding_strategy",
                "temperature",
                "top_p",
                "top_k",
                "max_new_tokens",
                "stop_policy",
                "prompt_version",
                "chat_template_version",
            }
        ),
        context=context,
    )
    return FrozenGenerationSettings(
        decoding_strategy=_expect_str(mapping, "decoding_strategy", context=context),
        temperature=_expect_float(mapping, "temperature", context=context),
        top_p=_expect_float(mapping, "top_p", context=context),
        top_k=_expect_int(mapping, "top_k", context=context),
        max_new_tokens=_expect_int(mapping, "max_new_tokens", context=context),
        stop_policy=_expect_str(mapping, "stop_policy", context=context),
        prompt_version=_expect_str(mapping, "prompt_version", context=context),
        chat_template_version=_expect_str(
            mapping,
            "chat_template_version",
            context=context,
        ),
    )


def load_evaluation_settings(path: Path) -> FrozenEvaluationSettings:
    """Load one strict versioned evaluation-settings YAML document."""

    try:
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise EvaluationSettingsError(f"could not read evaluation settings {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise EvaluationSettingsError(f"invalid YAML in evaluation settings {path}: {exc}") from exc
    mapping = _strict_mapping(raw, context="evaluation settings")
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
    return FrozenEvaluationSettings(
        schema_version=_expect_int(mapping, "schema_version", context=context),
        settings_id=_expect_str(mapping, "settings_id", context=context),
        settings_version=_expect_int(mapping, "settings_version", context=context),
        frozen=_expect_bool(mapping, "frozen", context=context),
        seed=_expect_int(mapping, "seed", context=context),
        generation=_parse_generation(mapping["generation"]),
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
    generation = config.generation
    frozen_generation = settings.generation
    compared_fields = (
        ("temperature", generation.temperature, frozen_generation.temperature),
        ("top_p", generation.top_p, frozen_generation.top_p),
        ("top_k", generation.top_k, frozen_generation.top_k),
        ("max_new_tokens", generation.max_new_tokens, frozen_generation.max_new_tokens),
        ("prompt_version", generation.prompt_version, frozen_generation.prompt_version),
    )
    drift = [name for name, actual, expected in compared_fields if actual != expected]
    if drift:
        raise EvaluationSettingsError(
            "evaluation generation settings do not match the frozen comparison protocol: "
            + ", ".join(drift)
        )
    return evaluation_settings_sha256(settings)
