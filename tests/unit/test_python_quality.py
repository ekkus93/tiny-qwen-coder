from __future__ import annotations

import pytest

from tiny_qwen_coder.data.pipeline import apply_language_validators
from tiny_qwen_coder.data.records import (
    LicenseMetadata,
    NormalizedTrainingRecord,
    SourceProvenance,
    ValidationResult,
    single_turn_messages,
)
from tiny_qwen_coder.languages import PRIMARY_VALIDATOR_ID
from tiny_qwen_coder.languages.python import load_python_plugin
from tiny_qwen_coder.languages.python_quality import (
    PYTHON_QUALITY_VALIDATOR_ID,
    PythonQualityReason,
    validate_python_quality,
)

_REVISION = "a" * 40


def _record(
    assistant: str,
    *,
    language: str = "python",
    source_metadata: tuple[tuple[str, str], ...] = (),
) -> NormalizedTrainingRecord:
    return NormalizedTrainingRecord(
        schema_version=1,
        language=language,
        messages=single_turn_messages(
            system=None,
            user="Implement the requested Python behavior.",
            assistant=assistant,
        ),
        provenance=SourceProvenance(
            source_id="fixture/python-quality",
            revision=_REVISION,
            license=LicenseMetadata(name="MIT"),
            split="train",
            record_id="quality-1",
            source_metadata=source_metadata,
        ),
    )


def _reason(result: ValidationResult) -> str:
    assert result.detail is not None
    return result.detail.split(";", maxsplit=1)[0]


def test_concrete_python_plugin_registers_quality_validator_after_identity() -> None:
    plugin = load_python_plugin()

    assert tuple(component.id for component in plugin.spec.validators) == (
        PRIMARY_VALIDATOR_ID,
        PYTHON_QUALITY_VALIDATOR_ID,
    )
    assert plugin.spec.validators[1].import_ref.endswith(":validate_python_quality")


@pytest.mark.parametrize(
    "assistant",
    (
        "def answer(value: int) -> int:\n    return value + 1\n",
        "```python\ndef answer(value: int) -> int:\n    return value + 1\n```",
        "result = sum(values)\n",
        "print('ready')\n",
    ),
)
def test_complete_valid_standalone_python_passes_ast_validation(assistant: str) -> None:
    result = validate_python_quality(_record(assistant))

    assert result.passed is True
    assert _reason(result) == f"reason={PythonQualityReason.VALID_STANDALONE.value}"


def test_complete_invalid_standalone_python_records_stable_syntax_rejection() -> None:
    result = validate_python_quality(_record("def broken(value)\n    return value\n"))

    assert result.passed is False
    assert _reason(result) == f"reason={PythonQualityReason.SYNTAX_ERROR.value}"
    assert result.detail is not None
    assert ";line=1;offset=" in result.detail


@pytest.mark.parametrize(
    ("assistant", "expected_reason"),
    (
        ("print 'legacy'\n", PythonQualityReason.PYTHON2_PRINT_STATEMENT),
        (
            "try:\n    run()\nexcept ValueError, exc:\n    handle(exc)\n",
            PythonQualityReason.PYTHON2_EXCEPT_SYNTAX,
        ),
        ("raise ValueError, 'legacy'\n", PythonQualityReason.PYTHON2_RAISE_SYNTAX),
        ("exec source\n", PythonQualityReason.PYTHON2_EXEC_STATEMENT),
        ("for index in xrange(3):\n    print(index)\n", PythonQualityReason.PYTHON2_XRANGE),
        ("name = raw_input('Name: ')\n", PythonQualityReason.PYTHON2_RAW_INPUT),
        ("items = mapping.iteritems()\n", PythonQualityReason.PYTHON2_DICT_ITERATION),
        ("if left <> right:\n    pass\n", PythonQualityReason.PYTHON2_NOT_EQUAL),
        ("timeout = 1000L\n", PythonQualityReason.PYTHON2_LONG_LITERAL),
    ),
)
def test_explicit_python2_constructs_are_rejected(
    assistant: str,
    expected_reason: PythonQualityReason,
) -> None:
    result = validate_python_quality(_record(assistant))

    assert result.passed is False
    assert _reason(result) == f"reason={expected_reason.value}"


def test_python2_source_metadata_is_rejected_before_code_heuristics() -> None:
    result = validate_python_quality(
        _record(
            "print('syntax itself is modern')\n",
            source_metadata=(("metadata.extension", "python2"),),
        )
    )

    assert result.passed is False
    assert _reason(result) == f"reason={PythonQualityReason.PYTHON2_SOURCE_METADATA.value}"


@pytest.mark.parametrize(
    ("assistant", "expected_reason"),
    (
        (
            "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n"
            "@@ -1 +1 @@\n-print 'old'\n+print('new')\n",
            PythonQualityReason.NOT_APPLICABLE_DIFF,
        ),
        ("elif ready:\n    run()\n", PythonQualityReason.NOT_APPLICABLE_FRAGMENT),
        ("    return partial_value\n", PythonQualityReason.NOT_APPLICABLE_FRAGMENT),
        (
            "```python\nreturn partial_value\n```",
            PythonQualityReason.NOT_APPLICABLE_FRAGMENT,
        ),
        (
            "Explanation first.\n```python\ndef broken(value)\n    return value\n```",
            PythonQualityReason.NOT_APPLICABLE_MIXED_CONTENT,
        ),
        (
            "```\ndef broken(value)\n    return value\n```",
            PythonQualityReason.NOT_APPLICABLE_UNLABELED_FENCE,
        ),
        (
            '```json\n{"language": "python"}\n```',
            PythonQualityReason.NOT_APPLICABLE_NON_PYTHON_FENCE,
        ),
        (
            "Use a small helper and keep the existing public API unchanged.",
            PythonQualityReason.NOT_APPLICABLE_NON_STANDALONE,
        ),
    ),
)
def test_ambiguous_fragments_and_nonstandalone_outputs_are_preserved(
    assistant: str,
    expected_reason: PythonQualityReason,
) -> None:
    result = validate_python_quality(_record(assistant))

    assert result.passed is True
    assert _reason(result) == f"reason={expected_reason.value}"


def test_generic_validation_attaches_rejection_metadata_without_deleting_record() -> None:
    original = _record("def broken(value)\n    return value\n")

    validated = apply_language_validators((original,), load_python_plugin())

    assert len(validated) == 1
    assert validated[0].messages == original.messages
    assert validated[0].provenance == original.provenance
    validation = validated[0].validation
    assert validation is not None
    assert validation.results[0] == ValidationResult(
        validator_id=PRIMARY_VALIDATOR_ID,
        passed=True,
    )
    quality = validation.results[1]
    assert quality.validator_id == PYTHON_QUALITY_VALIDATOR_ID
    assert quality.passed is False
    assert _reason(quality) == f"reason={PythonQualityReason.SYNTAX_ERROR.value}"
    assert validation.passed is False
