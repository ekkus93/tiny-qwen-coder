"""Conservative Python-specific quality validation for normalized training records."""

from __future__ import annotations

import ast
import re
from enum import StrEnum

from tiny_qwen_coder.data.records import NormalizedTrainingRecord, ValidationResult

PYTHON_QUALITY_VALIDATOR_ID = "python.quality"


class PythonQualityReason(StrEnum):
    """Stable reasons emitted by the Python quality validator."""

    VALID_STANDALONE = "validated_standalone_python"
    NOT_APPLICABLE_DIFF = "not_applicable_diff"
    NOT_APPLICABLE_FRAGMENT = "not_applicable_fragment"
    NOT_APPLICABLE_MIXED_CONTENT = "not_applicable_mixed_content"
    NOT_APPLICABLE_UNLABELED_FENCE = "not_applicable_unlabeled_fence"
    NOT_APPLICABLE_NON_PYTHON_FENCE = "not_applicable_non_python_fence"
    NOT_APPLICABLE_NON_STANDALONE = "not_applicable_non_standalone"
    WRONG_LANGUAGE = "wrong_language"
    MISSING_ASSISTANT_RESPONSE = "missing_assistant_response"
    PYTHON2_SOURCE_METADATA = "python2_source_metadata"
    PYTHON2_PRINT_STATEMENT = "python2_print_statement"
    PYTHON2_EXCEPT_SYNTAX = "python2_except_syntax"
    PYTHON2_RAISE_SYNTAX = "python2_raise_syntax"
    PYTHON2_EXEC_STATEMENT = "python2_exec_statement"
    PYTHON2_XRANGE = "python2_xrange"
    PYTHON2_RAW_INPUT = "python2_raw_input"
    PYTHON2_DICT_ITERATION = "python2_dict_iteration_api"
    PYTHON2_NOT_EQUAL = "python2_not_equal_operator"
    PYTHON2_LONG_LITERAL = "python2_long_literal"
    SYNTAX_ERROR = "syntax_error"


_FULL_FENCE_RE = re.compile(
    r"\A\s*```(?P<label>[A-Za-z0-9_+-]*)[ \t]*\n(?P<code>.*?)\n```[ \t]*\s*\Z",
    re.DOTALL,
)
_FRAGMENT_PREFIX_RE = re.compile(
    r"^(?:return\b|yield\b|await\b|elif\b|else\s*:|except\b|finally\s*:|"
    r"break\b|continue\b|case\b|[\)\]\}])"
)
_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\[\]'\"-]*\s*(?::[^=]+)?=(?!=)")
_CALL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*\s*\(")
_TOP_LEVEL_PREFIXES = (
    "def ",
    "async def ",
    "class ",
    "import ",
    "from ",
    "@",
    "if ",
    "for ",
    "while ",
    "with ",
    "try:",
    "match ",
    "assert ",
    "raise ",
    "del ",
    "global ",
    "nonlocal ",
)
_PYTHON_FENCE_LABELS = frozenset({"python", "python3", "py"})
_PYTHON2_METADATA_VALUES = frozenset({"python2", "python-2", "python 2", "py2"})
_PYTHON2_PATTERNS: tuple[tuple[PythonQualityReason, re.Pattern[str]], ...] = (
    (
        PythonQualityReason.PYTHON2_PRINT_STATEMENT,
        re.compile(r"(?m)^[ \t]*print(?:[ \t]+>>[^,\n]+,|[ \t]+)(?!\()"),
    ),
    (
        PythonQualityReason.PYTHON2_EXCEPT_SYNTAX,
        re.compile(r"(?m)^[ \t]*except[ \t]+[^:\n]+,[ \t]*[A-Za-z_][A-Za-z0-9_]*[ \t]*:"),
    ),
    (
        PythonQualityReason.PYTHON2_RAISE_SYNTAX,
        re.compile(r"(?m)^[ \t]*raise[ \t]+[A-Za-z_][A-Za-z0-9_.]*[ \t]*,"),
    ),
    (
        PythonQualityReason.PYTHON2_EXEC_STATEMENT,
        re.compile(r"(?m)^[ \t]*exec[ \t]+(?!\()"),
    ),
    (PythonQualityReason.PYTHON2_XRANGE, re.compile(r"\bxrange\s*\(")),
    (PythonQualityReason.PYTHON2_RAW_INPUT, re.compile(r"\braw_input\s*\(")),
    (
        PythonQualityReason.PYTHON2_DICT_ITERATION,
        re.compile(r"\.(?:iteritems|iterkeys|itervalues)\s*\("),
    ),
    (PythonQualityReason.PYTHON2_NOT_EQUAL, re.compile(r"<>")),
    (PythonQualityReason.PYTHON2_LONG_LITERAL, re.compile(r"\b\d+[lL]\b")),
)


def _result(
    passed: bool,
    reason: PythonQualityReason,
    *,
    line: int | None = None,
    offset: int | None = None,
) -> ValidationResult:
    detail = f"reason={reason.value}"
    if reason is PythonQualityReason.SYNTAX_ERROR:
        detail += f";line={line or 0};offset={offset or 0}"
    return ValidationResult(
        validator_id=PYTHON_QUALITY_VALIDATOR_ID,
        passed=passed,
        detail=detail,
    )


def _source_declares_python2(record: NormalizedTrainingRecord) -> bool:
    metadata = dict(record.provenance.source_metadata)
    values = (metadata.get(key) for key in ("metadata.extension", "language", "lang"))
    return any(
        value is not None and value.strip().lower() in _PYTHON2_METADATA_VALUES for value in values
    )


def _assistant_response(record: NormalizedTrainingRecord) -> str | None:
    responses = tuple(message.content for message in record.messages if message.role == "assistant")
    return responses[-1] if responses else None


def _looks_like_diff(text: str) -> bool:
    stripped = text.lstrip()
    return stripped.startswith(
        ("diff --git ", "--- a/", "*** Begin Patch", "```diff", "```patch")
    ) or ("\n@@ " in stripped and "\n+++ " in stripped and "\n--- " in stripped)


def _unwrap_full_fence(
    text: str,
) -> tuple[str | None, bool, PythonQualityReason | None]:
    match = _FULL_FENCE_RE.fullmatch(text)
    if match is not None:
        label = match.group("label").lower()
        if label in _PYTHON_FENCE_LABELS:
            return match.group("code"), True, None
        if not label:
            return None, False, PythonQualityReason.NOT_APPLICABLE_UNLABELED_FENCE
        return None, False, PythonQualityReason.NOT_APPLICABLE_NON_PYTHON_FENCE
    if "```" in text:
        return None, False, PythonQualityReason.NOT_APPLICABLE_MIXED_CONTENT
    return text, False, None


def _first_code_line(code: str) -> str:
    for line in code.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        return line
    return ""


def _looks_like_fragment(code: str) -> bool:
    first_line = _first_code_line(code)
    if not first_line:
        return False
    if first_line[:1].isspace():
        return True
    stripped = first_line.strip()
    return stripped.startswith((">>> ", "... ")) or _FRAGMENT_PREFIX_RE.match(stripped) is not None


def _python2_reason(code: str) -> PythonQualityReason | None:
    for reason, pattern in _PYTHON2_PATTERNS:
        if pattern.search(code) is not None:
            return reason
    return None


def _looks_like_standalone(code: str, *, explicit_python_fence: bool) -> bool:
    if explicit_python_fence:
        return True
    first_line = _first_code_line(code)
    if not first_line:
        return False
    stripped = first_line.strip()
    if stripped.startswith("#!"):
        return "python" in stripped.lower()
    if stripped.startswith(_TOP_LEVEL_PREFIXES):
        return True
    if _ASSIGNMENT_RE.match(stripped) is not None or _CALL_RE.match(stripped) is not None:
        return True
    return _python2_reason(code) is not None


def validate_python_quality(record: NormalizedTrainingRecord) -> ValidationResult:
    """Detect clearly invalid standalone Python without rejecting ambiguous fragments."""

    if record.language != "python":
        return _result(False, PythonQualityReason.WRONG_LANGUAGE)
    if _source_declares_python2(record):
        return _result(False, PythonQualityReason.PYTHON2_SOURCE_METADATA)

    response = _assistant_response(record)
    if response is None:
        return _result(False, PythonQualityReason.MISSING_ASSISTANT_RESPONSE)
    if _looks_like_diff(response):
        return _result(True, PythonQualityReason.NOT_APPLICABLE_DIFF)

    code, explicit_python_fence, skipped_reason = _unwrap_full_fence(response)
    if skipped_reason is not None:
        return _result(True, skipped_reason)
    assert code is not None

    if _looks_like_fragment(code):
        return _result(True, PythonQualityReason.NOT_APPLICABLE_FRAGMENT)
    if not _looks_like_standalone(code, explicit_python_fence=explicit_python_fence):
        return _result(True, PythonQualityReason.NOT_APPLICABLE_NON_STANDALONE)

    python2_reason = _python2_reason(code)
    if python2_reason is not None:
        return _result(False, python2_reason)

    try:
        ast.parse(code)
    except SyntaxError as exc:
        return _result(
            False,
            PythonQualityReason.SYNTAX_ERROR,
            line=exc.lineno,
            offset=exc.offset,
        )
    return _result(True, PythonQualityReason.VALID_STANDALONE)
