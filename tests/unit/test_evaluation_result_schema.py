from __future__ import annotations

import json

import pytest

from tiny_qwen_coder.evaluation import (
    EvaluationErrorCategory,
    EvaluationResult,
    EvaluationResultError,
    EvaluationStageStatus,
    EvaluationTestSummary,
    GenerationStats,
    create_evaluation_result,
    evaluation_result_json,
)
from tiny_qwen_coder.identities import AdapterIdentity, BaseModelIdentity

_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def _base() -> BaseModelIdentity:
    return BaseModelIdentity(
        repository="Qwen/Qwen3.5-4B",
        revision=_REVISION,
        tokenizer_repository="Qwen/Qwen3.5-4B",
        tokenizer_revision=_REVISION,
    )


def _generation() -> GenerationStats:
    return GenerationStats(
        prompt_tokens=64,
        generated_tokens=32,
        latency_seconds=0.5,
        tokens_per_second=64.0,
    )


@pytest.mark.parametrize(
    ("language", "problem_id", "code"),
    (
        ("python", "humaneval/0", "def add(a, b):\n    return a + b\n"),
        ("typescript", "ts-eval/add", "export const add = (a: number, b: number) => a + b;\n"),
        ("rust", "rust-eval/add", "fn add(a: i32, b: i32) -> i32 { a + b }\n"),
    ),
)
def test_python_typescript_and_rust_emit_the_same_high_level_schema(
    language: str,
    problem_id: str,
    code: str,
) -> None:
    result = create_evaluation_result(
        problem_id=problem_id,
        language=language,
        generated_text=code,
        generated_code=code,
        parse_status=EvaluationStageStatus.PASSED,
        compile_status=EvaluationStageStatus.PASSED,
        tests=EvaluationTestSummary(passed=3, total=3),
        error_category=EvaluationErrorCategory.NONE,
        error_message=None,
        generation=_generation(),
        base_model=_base(),
        adapter=AdapterIdentity(family="language", adapter_id=f"language/{language}/p0"),
    )

    assert isinstance(result, EvaluationResult)
    payload = json.loads(evaluation_result_json(result))
    assert set(payload) == {
        "adapter",
        "base_model",
        "compile_status",
        "error_category",
        "error_message",
        "generated_code",
        "generated_text",
        "generation",
        "language",
        "parse_status",
        "problem_id",
        "schema_version",
        "tests",
    }
    assert payload["schema_version"] == 1
    assert payload["problem_id"] == problem_id
    assert payload["language"] == language
    assert payload["generated_text"] == code
    assert payload["generated_code"] == code
    assert payload["parse_status"] == "passed"
    assert payload["compile_status"] == "passed"
    assert payload["tests"] == {"passed": 3, "total": 3}
    assert payload["error_category"] == "none"
    assert payload["generation"]["generated_tokens"] == 32
    assert payload["base_model"]["revision"] == _REVISION
    assert payload["adapter"]["adapter_id"] == f"language/{language}/p0"


def test_common_schema_represents_timeout_and_base_only_evaluation() -> None:
    result = create_evaluation_result(
        problem_id="python/timeout",
        language="python",
        generated_text="while True:\n    pass\n",
        generated_code="while True:\n    pass\n",
        parse_status=EvaluationStageStatus.PASSED,
        compile_status=EvaluationStageStatus.PASSED,
        tests=EvaluationTestSummary(passed=0, total=1),
        error_category=EvaluationErrorCategory.TIMEOUT,
        error_message="candidate exceeded the evaluation wall-clock limit",
        generation=GenerationStats(
            prompt_tokens=None,
            generated_tokens=5,
            latency_seconds=0.25,
            tokens_per_second=20.0,
        ),
        base_model=_base(),
        adapter=AdapterIdentity(family=None, adapter_id=None),
    )

    payload = json.loads(evaluation_result_json(result))
    assert payload["error_category"] == "timeout"
    assert payload["adapter"] == {"adapter_id": None, "family": None}
    assert payload["generation"]["prompt_tokens"] is None


def _fixture_result(
    *,
    language: str = "rust",
    parse_status: EvaluationStageStatus = EvaluationStageStatus.PASSED,
    compile_status: EvaluationStageStatus = EvaluationStageStatus.PASSED,
    tests: EvaluationTestSummary | None = None,
    error_category: EvaluationErrorCategory = EvaluationErrorCategory.NONE,
    error_message: str | None = None,
) -> EvaluationResult:
    return create_evaluation_result(
        problem_id="fixture/problem",
        language=language,
        generated_text="fn main() {}",
        generated_code="fn main() {}",
        parse_status=parse_status,
        compile_status=compile_status,
        tests=tests if tests is not None else EvaluationTestSummary(passed=0, total=0),
        error_category=error_category,
        error_message=error_message,
        generation=_generation(),
        base_model=_base(),
        adapter=AdapterIdentity(family=None, adapter_id=None),
    )


def test_stage_and_test_failure_categories_are_consistent() -> None:
    with pytest.raises(EvaluationResultError, match="parse error category"):
        _fixture_result(
            compile_status=EvaluationStageStatus.NOT_RUN,
            error_category=EvaluationErrorCategory.PARSE,
            error_message="parser rejected candidate",
        )

    with pytest.raises(EvaluationResultError, match="compile error category"):
        _fixture_result(
            error_category=EvaluationErrorCategory.COMPILE,
            error_message="compiler rejected candidate",
        )

    with pytest.raises(EvaluationResultError, match="test error category"):
        _fixture_result(
            tests=EvaluationTestSummary(passed=2, total=2),
            error_category=EvaluationErrorCategory.TEST,
            error_message="test failed",
        )


def test_success_cannot_hide_parse_compile_or_test_failures() -> None:
    with pytest.raises(EvaluationResultError, match="failed parse status"):
        _fixture_result(
            language="typescript",
            parse_status=EvaluationStageStatus.FAILED,
            compile_status=EvaluationStageStatus.NOT_RUN,
        )

    with pytest.raises(EvaluationResultError, match="failed compile status"):
        _fixture_result(
            language="typescript",
            compile_status=EvaluationStageStatus.FAILED,
        )

    with pytest.raises(EvaluationResultError, match="failed tests"):
        _fixture_result(
            language="typescript",
            tests=EvaluationTestSummary(passed=1, total=2),
        )


def test_test_summary_and_generation_stats_reject_invalid_values() -> None:
    with pytest.raises(EvaluationResultError, match="must not exceed"):
        EvaluationTestSummary(passed=2, total=1)
    with pytest.raises(EvaluationResultError, match="must not be negative"):
        EvaluationTestSummary(passed=-1, total=1)
    with pytest.raises(EvaluationResultError, match="prompt_tokens must not be negative"):
        GenerationStats(
            prompt_tokens=-1,
            generated_tokens=1,
            latency_seconds=1.0,
            tokens_per_second=1.0,
        )
    with pytest.raises(EvaluationResultError, match="finite and non-negative"):
        GenerationStats(
            prompt_tokens=1,
            generated_tokens=1,
            latency_seconds=float("nan"),
            tokens_per_second=1.0,
        )


def test_problem_language_error_and_schema_fields_are_strict() -> None:
    with pytest.raises(EvaluationResultError, match="problem_id"):
        create_evaluation_result(
            problem_id=" problem ",
            language="python",
            generated_text="",
            generated_code=None,
            parse_status=EvaluationStageStatus.NOT_RUN,
            compile_status=EvaluationStageStatus.NOT_RUN,
            tests=EvaluationTestSummary(passed=0, total=0),
            error_category=EvaluationErrorCategory.GENERATION,
            error_message="generation failed",
            generation=GenerationStats(
                prompt_tokens=10,
                generated_tokens=0,
                latency_seconds=0.1,
                tokens_per_second=0.0,
            ),
            base_model=_base(),
            adapter=AdapterIdentity(family=None, adapter_id=None),
        )

    with pytest.raises(EvaluationResultError, match="language must match"):
        create_evaluation_result(
            problem_id="problem",
            language="TypeScript",
            generated_text="",
            generated_code=None,
            parse_status=EvaluationStageStatus.NOT_RUN,
            compile_status=EvaluationStageStatus.NOT_RUN,
            tests=EvaluationTestSummary(passed=0, total=0),
            error_category=EvaluationErrorCategory.GENERATION,
            error_message="generation failed",
            generation=_generation(),
            base_model=_base(),
            adapter=AdapterIdentity(family=None, adapter_id=None),
        )

    with pytest.raises(EvaluationResultError, match="unsupported evaluation result schema_version"):
        EvaluationResult(
            schema_version=2,
            problem_id="problem",
            language="python",
            generated_text="",
            generated_code=None,
            parse_status=EvaluationStageStatus.NOT_RUN,
            compile_status=EvaluationStageStatus.NOT_RUN,
            tests=EvaluationTestSummary(passed=0, total=0),
            error_category=EvaluationErrorCategory.GENERATION,
            error_message="generation failed",
            generation=_generation(),
            base_model=_base(),
            adapter=AdapterIdentity(family=None, adapter_id=None),
        )


def test_evaluation_result_json_is_deterministic() -> None:
    result = create_evaluation_result(
        problem_id="fixture/deterministic",
        language="python",
        generated_text="print(1)\n",
        generated_code="print(1)\n",
        parse_status=EvaluationStageStatus.PASSED,
        compile_status=EvaluationStageStatus.PASSED,
        tests=EvaluationTestSummary(passed=1, total=1),
        error_category=EvaluationErrorCategory.NONE,
        error_message=None,
        generation=_generation(),
        base_model=_base(),
        adapter=AdapterIdentity(family="language", adapter_id="language/python/p0"),
    )

    assert evaluation_result_json(result) == evaluation_result_json(result)
    assert EvaluationTestSummary(passed=1, total=4).pass_rate == 0.25
    assert EvaluationTestSummary(passed=0, total=0).pass_rate is None
