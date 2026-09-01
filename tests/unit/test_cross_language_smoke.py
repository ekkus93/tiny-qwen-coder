"""CPU-only contract tests for P8-003 cross-language smoke evaluation."""

from __future__ import annotations

from pathlib import Path

from tiny_qwen_coder.evaluation.cross_language_smoke import (
    CASES,
    EXPECTED_ADAPTER_SHA256,
    EXPECTED_SCORING_CONTRACT_SHA256,
    EXPECTED_SUITE_SHA256,
    FORMAT_DIMENSION,
    SEMANTIC_DIMENSION,
    _dimension_summary,
    score_semantic_text,
    score_text,
    scoring_contract_sha256,
    suite_sha256,
)

_WORKFLOW = Path(".github/workflows/python-p0-cross-language-smoke.yml")


def _score_block(base: bool, adapter: bool) -> dict[str, object]:
    if base and not adapter:
        transition = "regression"
    elif not base and adapter:
        transition = "improvement"
    elif base:
        transition = "preserved_pass"
    else:
        transition = "preserved_fail"
    return {
        "base_passed": base,
        "adapter_passed": adapter,
        "transition": transition,
        "base_detail": None,
        "adapter_detail": None,
    }


def _rows(
    *, ts_base: int, ts_adapter: int, rust_base: int, rust_adapter: int
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for language, base_count, adapter_count in (
        ("typescript", ts_base, ts_adapter),
        ("rust", rust_base, rust_adapter),
    ):
        for index in range(3):
            semantic = _score_block(index < base_count, index < adapter_count)
            rows.append(
                {
                    "language": language,
                    FORMAT_DIMENSION: semantic.copy(),
                    SEMANTIC_DIMENSION: semantic,
                }
            )
    return rows


def test_suite_and_v2_scoring_contract_are_frozen() -> None:
    assert len(CASES) == 6
    assert [case.language for case in CASES].count("typescript") == 3
    assert [case.language for case in CASES].count("rust") == 3
    assert len({case.case_id for case in CASES}) == 6
    assert suite_sha256() == EXPECTED_SUITE_SHA256
    assert scoring_contract_sha256() == EXPECTED_SCORING_CONTRACT_SHA256


def test_format_score_accepts_plain_typescript_and_rust() -> None:
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


def test_format_score_keeps_v1_strict_code_only_behavior() -> None:
    typescript = CASES[0]

    assert not score_text(
        typescript,
        "```typescript\nexport function add(a: number, b: number): number { return a + b; }\n```",
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


def test_semantic_score_accepts_plain_or_one_matching_whole_response_fence() -> None:
    typescript = CASES[0]
    rust = CASES[3]

    for text in (
        "export function add(a: number, b: number): number { return a + b; }",
        "```typescript\nexport function add(a: number, b: number): number { return a + b; }\n```",
    ):
        assert score_semantic_text(typescript, text, generated_tokens=25, max_new_tokens=128).passed

    assert score_semantic_text(
        rust,
        "```rust\npub fn add(a: i32, b: i32) -> i32 { a + b }\n```",
        generated_tokens=25,
        max_new_tokens=128,
    ).passed


def test_semantic_score_rejects_wrapper_drift_wrong_language_and_truncation() -> None:
    typescript = CASES[0]
    valid_code = "export function add(a: number, b: number): number { return a + b; }"

    bad_wrappers = (
        f"```rust\n{valid_code}\n```",
        f"```\n{valid_code}\n```",
        f"Here is the code:\n```typescript\n{valid_code}\n```",
        f"```typescript\n{valid_code}\n```\nextra prose",
        f"```typescript\n{valid_code}\n```\n```typescript\n{valid_code}\n```",
        f"```typescript\n{valid_code}",
    )
    for text in bad_wrappers:
        assert not score_semantic_text(
            typescript, text, generated_tokens=30, max_new_tokens=128
        ).passed

    assert not score_semantic_text(
        typescript,
        "pub fn add(a: i32, b: i32) -> i32 { a + b }",
        generated_tokens=18,
        max_new_tokens=128,
    ).passed
    assert not score_semantic_text(
        typescript,
        valid_code,
        generated_tokens=128,
        max_new_tokens=128,
    ).passed


def test_v1_observed_fenced_base_is_semantically_adequate_under_v2() -> None:
    base_texts = (
        (
            "```typescript\n"
            "export function add(a: number, b: number): number {\n"
            "  return a + b;\n"
            "}\n"
            "```\n"
        ),
        (
            "```typescript\n"
            "export function firstOrUndefined<T>(items: readonly T[]): T | undefined {\n"
            "  return items.length > 0 ? items[0] : undefined;\n"
            "}\n"
            "```\n"
        ),
        (
            "```typescript\n"
            "export async function doubleAsync(value: number): Promise<number> {\n"
            "  return value * 2;\n"
            "}\n"
            "```\n"
        ),
        ("```rust\npub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n```\n"),
        (
            "```rust\n"
            "pub fn first_or_none<T: Clone>(items: &[T]) -> Option<T> {\n"
            "    items.first().cloned()\n"
            "}\n"
            "```\n"
        ),
        (
            "```rust\n"
            "pub fn checked_div(a: i32, b: i32) -> Option<i32> {\n"
            "    if b == 0 {\n"
            "        None\n"
            "    } else {\n"
            "        Some(a / b)\n"
            "    }\n"
            "}\n"
            "```\n"
        ),
    )

    assert all(
        not score_text(case, text, generated_tokens=64, max_new_tokens=128).passed
        for case, text in zip(CASES, base_texts, strict=True)
    )
    assert all(
        score_semantic_text(case, text, generated_tokens=64, max_new_tokens=128).passed
        for case, text in zip(CASES, base_texts, strict=True)
    )


def test_catastrophic_rule_uses_semantic_dimension_only() -> None:
    no_collapse = _dimension_summary(
        _rows(ts_base=3, ts_adapter=2, rust_base=3, rust_adapter=2), SEMANTIC_DIMENSION
    )["overall"]
    assert no_collapse["baseline_adequate_for_collapse_detection"] is True
    assert no_collapse["catastrophic_non_python_collapse_detected"] is False
    assert no_collapse["conclusion"] == "no_catastrophic_regression"

    inconclusive = _dimension_summary(
        _rows(ts_base=2, ts_adapter=0, rust_base=1, rust_adapter=1), SEMANTIC_DIMENSION
    )["overall"]
    assert inconclusive["baseline_adequate_for_collapse_detection"] is False
    assert inconclusive["catastrophic_non_python_collapse_detected"] is False
    assert inconclusive["conclusion"] == "inconclusive_base"

    language_collapse = _dimension_summary(
        _rows(ts_base=2, ts_adapter=0, rust_base=2, rust_adapter=2), SEMANTIC_DIMENSION
    )["overall"]
    assert language_collapse["catastrophic_non_python_collapse_detected"] is True
    assert language_collapse["conclusion"] == "catastrophic_regression"

    format_only = _dimension_summary(
        _rows(ts_base=3, ts_adapter=0, rust_base=3, rust_adapter=0), FORMAT_DIMENSION
    )["overall"]
    assert format_only["catastrophic_non_python_collapse_detected"] is False
    assert format_only["conclusion"] == "supplemental_only"


def test_p8_003_workflow_is_manual_only_pins_adapter_and_writes_v2_evidence() -> None:
    workflow = _WORKFLOW.read_text(encoding="utf-8")
    trigger_block = workflow.split("permissions:", 1)[0]

    assert "workflow_dispatch:" in trigger_block
    assert "\n  push:" not in trigger_block
    assert 'P7_006_RUN_ID: "33422910444"' in workflow
    assert EXPECTED_ADAPTER_SHA256 in workflow
    assert "p0-cross-language-smoke-v2/report.json" in workflow
    assert "if-no-files-found: error" in workflow
    assert "retention-days: 7" in workflow
    assert "retention-days: 3" in workflow
