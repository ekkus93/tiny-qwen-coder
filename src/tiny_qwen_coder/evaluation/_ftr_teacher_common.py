"""Shared parsing helpers for FTR-202/FTR-203 teacher evaluation evidence."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path

import yaml


class FTRTeacherSuperiorityError(RuntimeError):
    """Raised when teacher benchmark evidence or gate configuration is invalid."""


def sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise FTRTeacherSuperiorityError(f"could not hash {path}") from exc


def strict_mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise FTRTeacherSuperiorityError(f"{context} must be a mapping")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise FTRTeacherSuperiorityError(f"{context} keys must be strings")
        result[key] = item
    return result


def read_json(path: Path, *, context: str) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FTRTeacherSuperiorityError(f"could not read {context}: {path}") from exc
    return strict_mapping(value, context=context)


def read_yaml(path: Path, *, context: str) -> dict[str, object]:
    try:
        value: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise FTRTeacherSuperiorityError(f"could not read {context}: {path}") from exc
    return strict_mapping(value, context=context)


def expect_str(mapping: Mapping[str, object], key: str, *, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise FTRTeacherSuperiorityError(f"{context}.{key} must be a non-empty string")
    return value


def expect_int(mapping: Mapping[str, object], key: str, *, context: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise FTRTeacherSuperiorityError(f"{context}.{key} must be an integer")
    return value


def expect_number(mapping: Mapping[str, object], key: str, *, context: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise FTRTeacherSuperiorityError(f"{context}.{key} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise FTRTeacherSuperiorityError(f"{context}.{key} must be finite")
    return numeric


def score_pair(value: object, *, context: str) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise FTRTeacherSuperiorityError(f"{context} must be [passed, total]")
    return value[0], value[1]
