from __future__ import annotations

from pathlib import Path

import pytest

from tiny_qwen_coder.evaluation.regression import (
    RegressionCategory,
    RegressionExpectationKind,
    RegressionSuiteError,
    evaluate_regression_response,
    load_frozen_general_tool_regression_suite,
    load_regression_suite,
    regression_suite_json,
    regression_suite_sha256,
    score_regression_suite,
)

_SUITE_PATH = Path("configs/eval/general_tool_regression_v1.yaml")
_FROZEN_SHA256 = "9de462c05a05455b2cc5af8c0246d897fe7991510d470e837d205540922239f9"


def _passing_responses() -> dict[str, str]:
    return {
        "instruction_following.exact_ack": "ACK",
        "instruction_following.three_words": "red green blue",
        "json_structured_output.person_record": '{"active": true, "name": "Ada"}',
        "json_structured_output.number_list": '{"values": [2, 4, 6]}',
        "simple_reasoning.arithmetic": "20",
        "simple_reasoning.ordering": "Lin",
        "shell_reasoning.current_directory": "pwd",
        "shell_reasoning.pipeline_count": "A",
        "git_reasoning.stage_all": "git add -A",
        "git_reasoning.inspect_unstaged": "A",
        "tool_call.formatting": (
            '<tool_call>{"name":"calculator_add","arguments":{"a":17,"b":25}}</tool_call>'
        ),
        "tool_call.selection": (
            '<tool_call>{"name":"calculator_add","arguments":{"a":9,"b":14}}</tool_call>'
        ),
    }


def test_frozen_suite_identity_and_required_category_coverage() -> None:
    suite = load_frozen_general_tool_regression_suite()

    assert suite.schema_version == 1
    assert suite.suite_id == "general_tool_regression"
    assert suite.suite_version == 1
    assert suite.frozen is True
    assert len(suite.cases) == 12
    assert regression_suite_sha256(suite) == _FROZEN_SHA256

    categories = {case.category for case in suite.cases}
    assert categories == set(RegressionCategory)
    for category in RegressionCategory:
        assert sum(case.category is category for case in suite.cases) == 2


def test_suite_cases_are_deterministic_and_language_neutral() -> None:
    suite = load_frozen_general_tool_regression_suite()

    assert len({case.id for case in suite.cases}) == len(suite.cases)
    assert all(case.prompt == case.prompt.strip() for case in suite.cases)
    assert all("python" not in case.prompt.lower() for case in suite.cases)
    assert all("typescript" not in case.prompt.lower() for case in suite.cases)
    assert all("rust" not in case.prompt.lower() for case in suite.cases)
    assert regression_suite_json(suite) == regression_suite_json(
        load_regression_suite(_SUITE_PATH)
    )


def test_complete_passing_response_set_scores_perfectly() -> None:
    suite = load_frozen_general_tool_regression_suite()
    score = score_regression_suite(suite, _passing_responses())

    assert score.suite_sha256 == _FROZEN_SHA256
    assert score.passed == score.total == 12
    assert all(category.passed == category.total == 2 for category in score.categories)
    assert all(case.passed and case.detail is None for case in score.cases)


def test_missing_response_is_counted_as_regression() -> None:
    suite = load_frozen_general_tool_regression_suite()
    responses = _passing_responses()
    del responses["simple_reasoning.arithmetic"]

    score = score_regression_suite(suite, responses)

    assert score.passed == 11
    missing = next(case for case in score.cases if case.case_id == "simple_reasoning.arithmetic")
    assert missing.passed is False
    assert missing.detail == "response missing"


def test_unknown_response_case_fails_closed() -> None:
    suite = load_frozen_general_tool_regression_suite()
    responses = _passing_responses()
    responses["unknown.case"] = "anything"

    with pytest.raises(RegressionSuiteError, match="unknown case ID"):
        score_regression_suite(suite, responses)


def test_exact_text_rejects_extra_content() -> None:
    suite = load_frozen_general_tool_regression_suite()
    case = next(case for case in suite.cases if case.id == "instruction_following.exact_ack")

    assert case.expectation.kind is RegressionExpectationKind.EXACT_TEXT
    assert evaluate_regression_response(case, "ACK").passed is True
    assert evaluate_regression_response(case, "ACK\n").passed is False
    assert evaluate_regression_response(case, "Sure: ACK").passed is False


def test_json_scoring_is_semantic_but_requires_standalone_json() -> None:
    suite = load_frozen_general_tool_regression_suite()
    case = next(
        case for case in suite.cases if case.id == "json_structured_output.person_record"
    )

    assert evaluate_regression_response(case, '{"name":"Ada","active":true}').passed is True
    assert evaluate_regression_response(case, '{"active":true,"name":"Ada"}').passed is True
    fenced = '```json\n{"name":"Ada","active":true}\n```'
    assert evaluate_regression_response(case, fenced).passed is False
    assert evaluate_regression_response(
        case, '{"name":"Ada","active":true,"extra":1}'
    ).passed is False


def test_tool_call_scoring_requires_exact_wrapper_selection_and_arguments() -> None:
    suite = load_frozen_general_tool_regression_suite()
    case = next(case for case in suite.cases if case.id == "tool_call.selection")

    valid = '<tool_call>{"arguments":{"b":14,"a":9},"name":"calculator_add"}</tool_call>'
    wrong_tool = '<tool_call>{"name":"weather_lookup","arguments":{"city":"9"}}</tool_call>'
    extra_text = f"Calling a tool.\n{valid}"

    assert evaluate_regression_response(case, valid).passed is True
    assert evaluate_regression_response(case, wrong_tool).passed is False
    assert evaluate_regression_response(case, extra_text).passed is False


def test_loader_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "suite.yaml"
    path.write_text(
        _SUITE_PATH.read_text(encoding="utf-8").replace(
            "schema_version: 1", "schema_version: 1\nunknown_field: true", 1
        ),
        encoding="utf-8",
    )

    with pytest.raises(RegressionSuiteError, match="unknown field"):
        load_regression_suite(path)


def test_frozen_loader_rejects_suite_drift_without_version_bump(tmp_path: Path) -> None:
    path = tmp_path / "suite.yaml"
    path.write_text(
        _SUITE_PATH.read_text(encoding="utf-8").replace("value: ACK", "value: NACK", 1),
        encoding="utf-8",
    )

    with pytest.raises(RegressionSuiteError, match="fingerprint mismatch"):
        load_frozen_general_tool_regression_suite(path)
