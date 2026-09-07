"""Normalize and split Qwen thinking-mode completions safely."""

from __future__ import annotations

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


class TeacherReasoningParseError(ValueError):
    """Raised when Qwen thinking markup cannot be separated safely."""


def normalize_qwen_thinking_completion(text: str) -> str:
    """Restore Qwen's opening marker when the chat template prefilled it.

    Qwen thinking templates can prefill ``<think>`` in the assistant prefix. In
    that mode vLLM may return generated text beginning with the reasoning body and
    containing only the generated closing ``</think>`` marker. Downstream
    generation code expects a self-contained completion, so restore the opening
    marker at the inference boundary.
    """

    stripped = text.strip()
    if not stripped:
        return stripped
    if stripped.startswith(THINK_OPEN):
        return stripped
    if THINK_CLOSE in stripped:
        if THINK_OPEN in stripped:
            raise TeacherReasoningParseError(
                "Qwen completion contains a non-leading <think> marker"
            )
        return f"{THINK_OPEN}{stripped}"
    if THINK_OPEN in stripped:
        raise TeacherReasoningParseError(
            "Qwen completion contains an opening <think> marker without a closing marker"
        )
    return stripped


def split_qwen_thinking_completion(text: str) -> tuple[str | None, str]:
    """Return hidden reasoning and the trainable final answer.

    Both fully wrapped ``<think>...</think>answer`` output and vLLM's
    closing-only ``reasoning</think>answer`` output are supported. The returned
    final answer is guaranteed not to contain thinking markers.
    """

    normalized = normalize_qwen_thinking_completion(text)
    if not normalized.startswith(THINK_OPEN):
        return None, normalized
    closing = normalized.find(THINK_CLOSE, len(THINK_OPEN))
    if closing < 0:
        raise TeacherReasoningParseError(
            "Qwen completion opened <think> without closing </think>"
        )
    reasoning = normalized[len(THINK_OPEN) : closing].strip()
    final = normalized[closing + len(THINK_CLOSE) :].strip()
    if not final:
        raise TeacherReasoningParseError(
            "Qwen completion contained no final answer after </think>"
        )
    if THINK_OPEN in final or THINK_CLOSE in final:
        raise TeacherReasoningParseError(
            "Qwen final answer still contains thinking markup"
        )
    return reasoning or None, final


__all__ = [
    "THINK_CLOSE",
    "THINK_OPEN",
    "TeacherReasoningParseError",
    "normalize_qwen_thinking_completion",
    "split_qwen_thinking_completion",
]
