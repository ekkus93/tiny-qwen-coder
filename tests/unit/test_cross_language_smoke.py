"""CPU-only contract tests for P8-003 cross-language smoke evaluation."""

from __future__ import annotations

from pathlib import Path

from tiny_qwen_coder.evaluation.cross_language_smoke import (
    CASES,
    EXPECTED_ADAPTER_SHA256,
    EXPECTED_SUITE_SHA256,
    _language_summary,
    _overall_summary,
    score_text,
    suite_sha256,
)

_WORKFLOW = Path(".github/workflows/python-p0-cross-language-smoke.yml")


def _rows(*, ts_base: int, ts_adapter: int, rust_base: int, rust_adapter: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for language, base_count, adapter_count in (
        ("typescript", ts_base, ts_adapter),
        ("rust", rust_base, rust_adapter),
    ):
        for index in range(3):
            base = index < base_count
            adapter = index < adapter_count
            if base and not adapter:
                transition = "regression"
            elif not base and adapter:
                transition = "improvement"
            elif base:
                transition = "preserved_pass"
            else:
                transition = "preserved_fail"
            rows.append(
                {
                    "language": language,
                    "base_passed": base,
                    "adapter_passed": adapter,
                    "transition": transition,
                }
            )
    return rows


def test_suite_is_frozen_and_balanced() -> None:
    assert len(CASES) == 6
    assert [case.language for case in CASES].count("typescript") == 3
    assert [case.language for case in CASES].count("rust") == 3
    assert len({case.case_id for case in CASES}) == 6
    assert suite_sha256() == EXPECTED_SUITE_SHA256


def test_structural_score_accepts_representative_typescript_and_rust() -> None:
    typescript = CASES[0]
    rust = CASES[3]

    ts_score = score_text(
        typescript,
        "export function add(a: number, b: number): number { return a + b; }",
        generated_tokens=20,
        max_new_tokens=128,
    )
    rust_score = score_text(
        rust,
        "pub fn add(a: i32, b: i32) -> i32 { a + b }",
        generated_tokens=18,
        max_new_tokens=128,
    )

    assert ts_score.passed is True
    assert rust_score.passed is True


def test_structural_score_rejects_wrong_language_fences_missing_contract_and_truncation() -> None:
    typescript = CASES[0]

    assert not score_text(
        typescript,
        "```ts\nexport function add(a: number, b: number): number { return a + b; }\n```",
        generated_tokens=25,
        max_new_tokens=128,
    ).passed
    assert not score_text(
        typescript,
        "pub fn add(a: i32, b: i32) -> i32 { a + b }",
        generated_tokens=18,
        max_new_tokens=128,
    ).passed
    assert not score_text(
        typescript,
        "export function add(a, b) { return a + b; }",
        generated_tokens=15,
        max_new_tokens=128,
    ).passed
    assert not score_text(
        typescript,
        "export function add(a: number, b: number): number { return a + b; }",
        generated_tokens=128,
        max_new_tokens=128,
    ).passed


def test_catastrophic_rule_is_frozen_before_gpu_measurement() -> None:
    no_collapse_rows = _rows(ts_base=3, ts_adapter=2, rust_base=3, rust_adapter=2)
    no_collapse_languages = _language_summary(no_collapse_rows)
    no_collapse = _overall_summary(no_collapse_rows, no_collapse_languages)
    assert no_collapse["catastrophic_non_python_collapse_detected"] is False

    language_collapse_rows = _rows(ts_base=2, ts_adapter=0, rust_base=1, rust_adapter=1)
    language_collapse_languages = _language_summary(language_collapse_rows)
    language_collapse = _overall_summary(language_collapse_rows, language_collapse_languages)
    assert language_collapse["catastrophic_non_python_collapse_detected"] is True

    overall_collapse_rows = _rows(ts_base=3, ts_adapter=1, rust_base=3, rust_adapter=2)
    overall_collapse_languages = _language_summary(overall_collapse_rows)
    overall_collapse = _overall_summary(overall_collapse_rows, overall_collapse_languages)
    assert overall_collapse["catastrophic_non_python_collapse_detected"] is True


def test_p8_003_workflow_is_manual_only_and_pins_exact_p0_adapter() -> None:
    workflow = _WORKFLOW.read_text(encoding="utf-8")
    trigger_block = workflow.split("permissions:", 1)[0]

    assert "workflow_dispatch:" in trigger_block
    assert "\n  push:" not in trigger_block
    assert "P7_006_RUN_ID: \"33422910444\"" in workflow
    assert EXPECTED_ADAPTER_SHA256 in workflow
    assert "if-no-files-found: error" in workflow
    assert "retention-days: 7" in workflow
    assert "retention-days: 3" in workflow
