"""Frozen selective LoRA target profile for the canonical Qwen3.5 base."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import yaml

from tiny_qwen_coder.adapters.targets import PeftTargetDiscoveryReport

SelectiveTargetCategory: TypeAlias = Literal[
    "full_attention",
    "mlp",
    "gated_deltanet",
]

_SCHEMA_VERSION = 1
_CATEGORY_ORDER: tuple[SelectiveTargetCategory, ...] = (
    "full_attention",
    "mlp",
    "gated_deltanet",
)
_FROZEN_PROFILE_PATH = Path("configs/base/qwen35-4b-selective-lora-v1.yaml")
_FROZEN_PROFILE_SHA256 = "edc61481737903c729eb6671bee846879004b91ba0644175beb9fe5e0be05dc6"
_GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class SelectiveLoraTargetProfileError(ValueError):
    """Raised when the frozen selective LoRA target profile is invalid or drifts."""


@dataclass(frozen=True, slots=True)
class SelectiveTargetGroup:
    """One architecture category and its exact selected PEFT leaf names."""

    category: SelectiveTargetCategory
    target_modules: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.target_modules:
            raise SelectiveLoraTargetProfileError(
                f"selective target group {self.category!r} must contain at least one module"
            )
        if self.target_modules != tuple(sorted(set(self.target_modules))):
            raise SelectiveLoraTargetProfileError(
                f"selective target group {self.category!r} modules must be sorted and unique"
            )


@dataclass(frozen=True, slots=True)
class SelectiveLoraTargetProfile:
    """Revision-bound P7-001 selective language-LoRA architecture contract."""

    schema_version: int
    profile_id: str
    base_repository: str
    base_revision: str
    strategy: str
    groups: tuple[SelectiveTargetGroup, ...]
    target_modules: tuple[str, ...]
    measurement_source_task: str
    measurement_rank: int
    measured_trainable_parameters: int
    source_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise SelectiveLoraTargetProfileError(
                f"unsupported selective target schema_version {self.schema_version}; "
                f"expected {_SCHEMA_VERSION}"
            )
        if not self.profile_id.strip():
            raise SelectiveLoraTargetProfileError("selective target profile id must not be empty")
        if not self.base_repository.strip():
            raise SelectiveLoraTargetProfileError("base repository must not be empty")
        if not _GIT_SHA_PATTERN.fullmatch(self.base_revision):
            raise SelectiveLoraTargetProfileError(
                "base revision must be an immutable lowercase 40-character Git SHA"
            )
        if self.strategy != "selective":
            raise SelectiveLoraTargetProfileError(
                "canonical language-LoRA target strategy must be selective"
            )
        if tuple(group.category for group in self.groups) != _CATEGORY_ORDER:
            raise SelectiveLoraTargetProfileError(
                "selective target groups must use canonical architecture-category order"
            )
        if self.target_modules != tuple(sorted(set(self.target_modules))):
            raise SelectiveLoraTargetProfileError("target_modules must be sorted and unique")
        grouped_modules = tuple(
            sorted({module for group in self.groups for module in group.target_modules})
        )
        if self.target_modules != grouped_modules:
            raise SelectiveLoraTargetProfileError(
                "target_modules must exactly equal the union of architecture target groups"
            )
        if not self.measurement_source_task.strip():
            raise SelectiveLoraTargetProfileError("measurement source task must not be empty")
        if self.measurement_rank <= 0:
            raise SelectiveLoraTargetProfileError("measurement rank must be greater than zero")
        if self.measured_trainable_parameters <= 0:
            raise SelectiveLoraTargetProfileError(
                "measured trainable parameter count must be greater than zero"
            )
        if not _SHA256_PATTERN.fullmatch(self.source_sha256):
            raise SelectiveLoraTargetProfileError(
                "selective target profile source_sha256 must be a lowercase SHA-256 digest"
            )


def _strict_mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise SelectiveLoraTargetProfileError(f"{context} must be a mapping")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise SelectiveLoraTargetProfileError(f"{context} keys must be strings")
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
        raise SelectiveLoraTargetProfileError(
            f"{context} contains unknown field(s): {', '.join(unknown)}"
        )
    if missing:
        raise SelectiveLoraTargetProfileError(
            f"{context} is missing required field(s): {', '.join(missing)}"
        )


def _expect_str(mapping: Mapping[str, object], key: str, *, context: str) -> str:
    value = mapping[key]
    if not isinstance(value, str) or not value.strip():
        raise SelectiveLoraTargetProfileError(f"{context}.{key} must be a non-empty string")
    return value


def _expect_int(mapping: Mapping[str, object], key: str, *, context: str) -> int:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise SelectiveLoraTargetProfileError(f"{context}.{key} must be an integer")
    return value


def _expect_str_tuple(mapping: Mapping[str, object], key: str, *, context: str) -> tuple[str, ...]:
    value = mapping[key]
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise SelectiveLoraTargetProfileError(f"{context}.{key} must be a sequence of strings")
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise SelectiveLoraTargetProfileError(
                f"{context}.{key}[{index}] must be a non-empty string"
            )
        result.append(item)
    return tuple(result)


def parse_selective_lora_target_profile(
    text: str,
    *,
    source_sha256: str | None = None,
) -> SelectiveLoraTargetProfile:
    """Parse one strict selective-target YAML document."""

    try:
        raw: object = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SelectiveLoraTargetProfileError(
            f"could not parse selective target profile YAML: {exc}"
        ) from exc
    root = _strict_mapping(raw, context="selective target profile")
    _validate_keys(
        root,
        required=frozenset(
            {
                "schema_version",
                "id",
                "base",
                "strategy",
                "categories",
                "target_modules",
                "measurement",
            }
        ),
        context="selective target profile",
    )

    base = _strict_mapping(root["base"], context="selective target profile.base")
    _validate_keys(
        base,
        required=frozenset({"repository", "revision"}),
        context="selective target profile.base",
    )

    categories = _strict_mapping(
        root["categories"],
        context="selective target profile.categories",
    )
    _validate_keys(
        categories,
        required=frozenset(_CATEGORY_ORDER),
        context="selective target profile.categories",
    )
    groups = tuple(
        SelectiveTargetGroup(
            category=category,
            target_modules=_expect_str_tuple(
                categories,
                category,
                context="selective target profile.categories",
            ),
        )
        for category in _CATEGORY_ORDER
    )

    measurement = _strict_mapping(
        root["measurement"],
        context="selective target profile.measurement",
    )
    _validate_keys(
        measurement,
        required=frozenset({"source_task", "rank", "trainable_parameters"}),
        context="selective target profile.measurement",
    )

    digest = source_sha256 or hashlib.sha256(text.encode("utf-8")).hexdigest()
    return SelectiveLoraTargetProfile(
        schema_version=_expect_int(root, "schema_version", context="selective target profile"),
        profile_id=_expect_str(root, "id", context="selective target profile"),
        base_repository=_expect_str(
            base,
            "repository",
            context="selective target profile.base",
        ),
        base_revision=_expect_str(
            base,
            "revision",
            context="selective target profile.base",
        ),
        strategy=_expect_str(root, "strategy", context="selective target profile"),
        groups=groups,
        target_modules=_expect_str_tuple(
            root,
            "target_modules",
            context="selective target profile",
        ),
        measurement_source_task=_expect_str(
            measurement,
            "source_task",
            context="selective target profile.measurement",
        ),
        measurement_rank=_expect_int(
            measurement,
            "rank",
            context="selective target profile.measurement",
        ),
        measured_trainable_parameters=_expect_int(
            measurement,
            "trainable_parameters",
            context="selective target profile.measurement",
        ),
        source_sha256=digest,
    )


def load_selective_lora_target_profile(path: Path) -> SelectiveLoraTargetProfile:
    """Load and validate a selective-target profile without requiring the frozen fingerprint."""

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SelectiveLoraTargetProfileError(
            f"could not read selective target profile {path}: {exc}"
        ) from exc
    return parse_selective_lora_target_profile(text)


def load_frozen_selective_lora_target_profile(
    path: Path = _FROZEN_PROFILE_PATH,
) -> SelectiveLoraTargetProfile:
    """Load the canonical P7-001 target profile and fail closed on unreviewed drift."""

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SelectiveLoraTargetProfileError(
            f"could not read frozen selective target profile {path}: {exc}"
        ) from exc
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if digest != _FROZEN_PROFILE_SHA256:
        raise SelectiveLoraTargetProfileError(
            "frozen selective target profile fingerprint mismatch; "
            f"expected {_FROZEN_PROFILE_SHA256}, got {digest}"
        )
    return parse_selective_lora_target_profile(text, source_sha256=digest)


def require_profile_matches_discovery(
    profile: SelectiveLoraTargetProfile,
    discovery: PeftTargetDiscoveryReport,
) -> None:
    """Require a P2 architecture inspection to reproduce the frozen P7 target profile."""

    if discovery.model_repository != profile.base_repository:
        raise SelectiveLoraTargetProfileError(
            "P2 target discovery base repository does not match frozen P7 target profile"
        )
    if discovery.model_revision != profile.base_revision:
        raise SelectiveLoraTargetProfileError(
            "P2 target discovery base revision does not match frozen P7 target profile"
        )
    if discovery.unclassified_text_module_count:
        raise SelectiveLoraTargetProfileError(
            "P2 target discovery contains unclassified text linears"
        )
    if discovery.selective_target_modules != profile.target_modules:
        raise SelectiveLoraTargetProfileError(
            "P2 selective target modules do not match frozen P7 target profile"
        )

    summaries = {summary.category: summary for summary in discovery.categories}
    for group in profile.groups:
        summary = summaries.get(group.category)
        if summary is None or summary.leaf_names != group.target_modules:
            raise SelectiveLoraTargetProfileError(
                f"P2 {group.category} target modules do not match frozen P7 target profile"
            )


def require_measured_trainable_parameters(
    profile: SelectiveLoraTargetProfile,
    *,
    rank: int,
    trainable_parameters: int,
) -> None:
    """Require a measured PEFT attachment to reproduce the frozen P2-008 parameter count."""

    if rank != profile.measurement_rank:
        raise SelectiveLoraTargetProfileError(
            f"LoRA rank {rank} does not match measured rank {profile.measurement_rank}"
        )
    if trainable_parameters != profile.measured_trainable_parameters:
        raise SelectiveLoraTargetProfileError(
            "trainable parameter count does not match P2-008 measurement: "
            f"expected {profile.measured_trainable_parameters}, got {trainable_parameters}"
        )
