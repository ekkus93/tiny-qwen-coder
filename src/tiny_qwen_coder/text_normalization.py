"""Lightweight semantics-preserving text normalization shared across data checks."""

from __future__ import annotations


def normalize_training_text(text: str) -> str:
    """Normalize only UTF-8 BOM and line endings without changing code semantics."""

    text.encode("utf-8", errors="strict")
    return text.removeprefix("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
