"""Dependency-light model and adapter identity contracts shared across run artifacts."""

from __future__ import annotations

import re
from dataclasses import dataclass

_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class ManifestError(ValueError):
    """Raised when a run artifact identity or manifest cannot be created safely."""


def _require_non_empty(value: str, *, field_name: str) -> None:
    if not value.strip():
        raise ManifestError(f"{field_name} must not be empty")


def _require_sha(value: str, *, field_name: str) -> None:
    if not _SHA_PATTERN.fullmatch(value):
        raise ManifestError(f"{field_name} must be a lowercase 40-character Git SHA")


@dataclass(frozen=True, slots=True)
class BaseModelIdentity:
    """Exact model/tokenizer identity required by training and evaluation artifacts."""

    repository: str
    revision: str
    tokenizer_repository: str
    tokenizer_revision: str

    def __post_init__(self) -> None:
        _require_non_empty(self.repository, field_name="base_model.repository")
        _require_sha(self.revision, field_name="base_model.revision")
        _require_non_empty(self.tokenizer_repository, field_name="base_model.tokenizer_repository")
        _require_sha(self.tokenizer_revision, field_name="base_model.tokenizer_revision")


@dataclass(frozen=True, slots=True)
class AdapterIdentity:
    """Adapter identity; both fields are null only for base-only evaluation."""

    family: str | None
    adapter_id: str | None

    def __post_init__(self) -> None:
        if (self.family is None) != (self.adapter_id is None):
            raise ManifestError("adapter family and adapter_id must be defined together")
        if self.family is not None:
            _require_non_empty(self.family, field_name="adapter.family")
        if self.adapter_id is not None:
            _require_non_empty(self.adapter_id, field_name="adapter.adapter_id")
