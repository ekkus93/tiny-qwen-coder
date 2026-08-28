"""Frozen non-language regression suite for general and tool-use behavior."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

import yaml

_REGRESSION_SUITE_SCHEMA_VERSION = 1
_GENERAL_TOOL_SUITE_ID = "general_tool_regression"
_GENERAL_TOOL_SUITE_VERSION = 1
_GENERAL_TOOL_SUITE_PATH = Path("configs/eval/general_tool_regression_v1.yaml")
_FROZEN_GENERAL_TOOL_SUITE_SHA256 = (
    "9de462c05a05455b2cc5af8c0246d897fe7991510d470e837d205540922239f9"
)
_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]*$")
_TOOL_CALL_PATTERN = re.compile(r"^<tool_call>(.+)</tool_call>$", re.DOTALL)


class RegressionSuiteError(ValueError):
    """Raised when a regression suite or response violates the frozen contract."""


class RegressionCategory(StrEnum):
    """Required capability categories in the general/tool regression suite."""

    INSTRUCTION_FOLLOWING = "instruction_following"
    JSON_STRUCTURED_OUTPUT = "json_structured_output"
    SIMPLE_REASONING = "simple_reasoning"
    SHELL_REASONING = "shell_reasoning"
    GIT_REASONING = "git_reasoning"
    TOOL_CALL = "tool_call"


class RegressionExpectationKind(StrEnum):
    """Deterministic scoring mode for one regression case."""

    EXACT_TEXT = "exact_text"
    JSON = "json"
    TOOL_CALL = "tool_call"


@dataclass(frozen=True, slots=True)
class RegressionExpectation:
    """Canonical expected response, stored as text or canonical JSON."""

    kind: RegressionExpectationKind
    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, RegressionExpectationKind):
            raise RegressionSuiteError("expectation.kind must be a RegressionExpectationKind")
        if not self.value:
            raise RegressionSuiteError("expectation value must not be empty")
        if self.kind is RegressionExpectationKind.EXACT_TEXT:
            if self.value != self.value.strip():
                raise RegressionSuiteError(
                    "exact-text expectation must not contain outer whitespace"
                )
            return
        _parse_json(self.value, context="expectation value")


@dataclass(frozen=True, slots=True)
class RegressionCase:
    """One deterministic, language-neutral regression prompt and expectation."""

    id: str
    category: RegressionCategory
    prompt: str
    expectation: RegressionExpectation

    def __post_init__(self) -> None:
        _require_id(self.id, field_name="case.id")
        if not isinstance(self.category, RegressionCategory):
            raise RegressionSuiteError("case.category must be a RegressionCategory")
        if not self.prompt.strip():
            raise RegressionSuiteError("case.prompt must not be empty")
        if self.prompt != self.prompt.strip():
            raise RegressionSuiteError("case.prompt must not contain outer whitespace")
        if not isinstance(self.expectation, RegressionExpectation):
            raise RegressionSuiteError("case.expectation must be a RegressionExpectation")


@dataclass(frozen=True, slots=True)
class RegressionSuite:
    """Versioned immutable collection of non-language regression cases."""

    schema_version: int
    suite_id: str
    suite_version: int
    frozen: bool
    cases: tuple[RegressionCase, ...]

    def __post_init__(self) -> None:
        if self.schema_version != _REGRESSION_SUITE_SCHEMA_VERSION:
            raise RegressionSuiteError(
                f"unsupported regression suite schema_version {self.schema_version}; "
                f"expected {_REGRESSION_SUITE_SCHEMA_VERSION}"
            )
        _require_id(self.suite_id, field_name="suite_id")
        if isinstance(self.suite_version, bool) or not isinstance(self.suite_version, int):
            raise RegressionSuiteError("suite_version must be an integer")
        if self.suite_version <= 0:
            raise RegressionSuiteError("suite_version must be greater than zero")
        if not isinstance(self.frozen, bool):
            raise RegressionSuiteError("frozen must be a boolean")
        if not self.cases:
            raise RegressionSuiteError("regression suite must contain at least one case")
        case_ids = tuple(case.id for case in self.cases)
        if len(case_ids) != len(set(case_ids)):
            raise RegressionSuiteError("regression suite case IDs must be unique")


@dataclass(frozen=True, slots=True)
class RegressionCaseResult:
    """Deterministic score for one regression response."""

    case_id: str
    category: RegressionCategory
    passed: bool
    detail: str | None


@dataclass(frozen=True, slots=True)
class RegressionCategoryScore:
    """Aggregate score for one required regression category."""

    category: RegressionCategory
    passed: int
    total: int


@dataclass(frozen=True, slots=True)
class RegressionSuiteScore:
    """Aggregate score tied to one exact frozen suite fingerprint."""

    suite_id: str
    suite_version: int
    suite_sha256: str
    passed: int
    total: int
    categories: tuple[RegressionCategoryScore, ...]
    cases: tuple[RegressionCaseResult, ...]


_REQUIRED_GENERAL_TOOL_CATEGORIES = frozenset(RegressionCategory)


def _require_id(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not _ID_PATTERN.fullmatch(value):
        raise RegressionSuiteError(f"{field_name} must match ^[a-z][a-z0-9_.-]*$")
    return value


def _strict_mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise RegressionSuiteError(f"{context} must be a mapping")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise RegressionSuiteError(f"{context} keys must be strings")
        result[key] = item
    return result


def _validate_keys(
    mapping: Mapping[str, object],
    *,
    required: frozenset[str],
    context: str,
) -> None:
    unknown = sorted(set(mapping) - required)
    missing = sorted(required - set(mapping))
    if unknown:
        raise RegressionSuiteError(f"{context} contains unknown field(s): {', '.join(unknown)}")
    if missing:
        raise RegressionSuiteError(f"{context} is missing required field(s): {', '.join(missing)}")


def _expect_int(mapping: Mapping[str, object], key: str, *, context: str) -> int:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise RegressionSuiteError(f"{context}.{key} must be an integer")
    return value


def _expect_str(mapping: Mapping[str, object], key: str, *, context: str) -> str:
    value = mapping[key]
    if not isinstance(value, str) or not value.strip():
        raise RegressionSuiteError(f"{context}.{key} must be a non-empty string")
    return value


def _canonical_json(value: object, *, context: str) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise RegressionSuiteError(f"{context} must be JSON-serializable") from exc
    _parse_json(encoded, context=context)
    return encoded


def _parse_json(value: str, *, context: str) -> object:
    try:
        parsed: object = json.loads(value)
    except json.JSONDecodeError as exc:
        raise RegressionSuiteError(f"{context} must contain valid JSON") from exc
    return parsed


def _parse_expectation(value: object, *, context: str) -> RegressionExpectation:
    mapping = _strict_mapping(value, context=context)
    _validate_keys(mapping, required=frozenset({"kind", "value"}), context=context)
    kind_text = _expect_str(mapping, "kind", context=context)
    try:
        kind = RegressionExpectationKind(kind_text)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in RegressionExpectationKind)
        raise RegressionSuiteError(f"{context}.kind must be one of: {allowed}") from exc
    raw_value = mapping["value"]
    if kind is RegressionExpectationKind.EXACT_TEXT:
        if not isinstance(raw_value, str):
            raise RegressionSuiteError(f"{context}.value must be a string for exact_text")
        return RegressionExpectation(kind=kind, value=raw_value)
    return RegressionExpectation(
        kind=kind,
        value=_canonical_json(raw_value, context=f"{context}.value"),
    )


def _parse_case(value: object, *, index: int) -> RegressionCase:
    context = f"cases[{index}]"
    mapping = _strict_mapping(value, context=context)
    _validate_keys(
        mapping,
        required=frozenset({"id", "category", "prompt", "expectation"}),
        context=context,
    )
    category_text = _expect_str(mapping, "category", context=context)
    try:
        category = RegressionCategory(category_text)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in RegressionCategory)
        raise RegressionSuiteError(f"{context}.category must be one of: {allowed}") from exc
    return RegressionCase(
        id=_require_id(mapping["id"], field_name=f"{context}.id"),
        category=category,
        prompt=_expect_str(mapping, "prompt", context=context),
        expectation=_parse_expectation(mapping["expectation"], context=f"{context}.expectation"),
    )


def load_regression_suite(path: Path) -> RegressionSuite:
    """Load one strict regression suite YAML document."""

    try:
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RegressionSuiteError(f"could not read regression suite {path}") from exc
    mapping = _strict_mapping(raw, context="regression suite")
    _validate_keys(
        mapping,
        required=frozenset({"schema_version", "suite_id", "suite_version", "frozen", "cases"}),
        context="regression suite",
    )
    raw_cases = mapping["cases"]
    if not isinstance(raw_cases, list):
        raise RegressionSuiteError("regression suite.cases must be a YAML sequence")
    frozen = mapping["frozen"]
    if not isinstance(frozen, bool):
        raise RegressionSuiteError("regression suite.frozen must be a boolean")
    return RegressionSuite(
        schema_version=_expect_int(mapping, "schema_version", context="regression suite"),
        suite_id=_expect_str(mapping, "suite_id", context="regression suite"),
        suite_version=_expect_int(mapping, "suite_version", context="regression suite"),
        frozen=frozen,
        cases=tuple(_parse_case(case, index=index) for index, case in enumerate(raw_cases)),
    )


def regression_suite_json(suite: RegressionSuite) -> str:
    """Serialize one regression suite deterministically for storage and hashing."""

    return json.dumps(asdict(suite), indent=2, sort_keys=True) + "\n"


def regression_suite_sha256(suite: RegressionSuite) -> str:
    """Hash the exact semantic suite definition, independent of YAML formatting."""

    return hashlib.sha256(regression_suite_json(suite).encode("utf-8")).hexdigest()


def load_frozen_general_tool_regression_suite(
    path: Path = _GENERAL_TOOL_SUITE_PATH,
) -> RegressionSuite:
    """Load the canonical P4-005 suite and fail closed if its frozen identity drifted."""

    suite = load_regression_suite(path)
    if suite.suite_id != _GENERAL_TOOL_SUITE_ID:
        raise RegressionSuiteError(
            f"canonical suite_id must be {_GENERAL_TOOL_SUITE_ID!r}; got {suite.suite_id!r}"
        )
    if suite.suite_version != _GENERAL_TOOL_SUITE_VERSION:
        raise RegressionSuiteError(
            f"canonical suite_version must be {_GENERAL_TOOL_SUITE_VERSION}; "
            f"got {suite.suite_version}"
        )
    if not suite.frozen:
        raise RegressionSuiteError("canonical general/tool regression suite must be frozen")
    categories = frozenset(case.category for case in suite.cases)
    if categories != _REQUIRED_GENERAL_TOOL_CATEGORIES:
        missing = sorted(
            category.value for category in _REQUIRED_GENERAL_TOOL_CATEGORIES - categories
        )
        extra = sorted(
            category.value for category in categories - _REQUIRED_GENERAL_TOOL_CATEGORIES
        )
        detail = f"missing={missing}, extra={extra}"
        raise RegressionSuiteError(f"canonical suite category coverage mismatch: {detail}")
    fingerprint = regression_suite_sha256(suite)
    if fingerprint != _FROZEN_GENERAL_TOOL_SUITE_SHA256:
        raise RegressionSuiteError(
            "canonical general/tool regression suite fingerprint mismatch; "
            "increment suite_version and explicitly update the frozen fingerprint before evaluation"
        )
    return suite


def evaluate_regression_response(case: RegressionCase, response: str) -> RegressionCaseResult:
    """Score one raw model response using the case's deterministic expectation."""

    if not isinstance(response, str):
        raise RegressionSuiteError("response must be a string")
    kind = case.expectation.kind
    if kind is RegressionExpectationKind.EXACT_TEXT:
        passed = response == case.expectation.value
        detail = None if passed else "response did not match expected exact text"
    elif kind is RegressionExpectationKind.JSON:
        try:
            actual: object = json.loads(response)
        except json.JSONDecodeError:
            passed = False
            detail = "response was not standalone valid JSON"
        else:
            expected = _parse_json(case.expectation.value, context="stored JSON expectation")
            passed = actual == expected
            detail = None if passed else "JSON value did not match expected structure"
    else:
        match = _TOOL_CALL_PATTERN.fullmatch(response)
        if match is None:
            passed = False
            detail = "response was not exactly one <tool_call> JSON block"
        else:
            try:
                actual: object = json.loads(match.group(1))
            except json.JSONDecodeError:
                passed = False
                detail = "tool-call payload was not valid JSON"
            else:
                expected = _parse_json(
                    case.expectation.value,
                    context="stored tool-call expectation",
                )
                passed = actual == expected
                detail = None if passed else "tool name or arguments did not match expectation"
    return RegressionCaseResult(
        case_id=case.id,
        category=case.category,
        passed=passed,
        detail=detail,
    )


def score_regression_suite(
    suite: RegressionSuite,
    responses: Mapping[str, str],
) -> RegressionSuiteScore:
    """Score a complete response mapping against one exact regression suite."""

    expected_ids = {case.id for case in suite.cases}
    unknown = sorted(set(responses) - expected_ids)
    if unknown:
        raise RegressionSuiteError(f"responses contain unknown case ID(s): {', '.join(unknown)}")

    case_results = tuple(
        evaluate_regression_response(case, responses[case.id])
        if case.id in responses
        else RegressionCaseResult(
            case_id=case.id,
            category=case.category,
            passed=False,
            detail="response missing",
        )
        for case in suite.cases
    )
    category_scores = tuple(
        RegressionCategoryScore(
            category=category,
            passed=sum(result.passed for result in case_results if result.category is category),
            total=sum(1 for result in case_results if result.category is category),
        )
        for category in RegressionCategory
    )
    return RegressionSuiteScore(
        suite_id=suite.suite_id,
        suite_version=suite.suite_version,
        suite_sha256=regression_suite_sha256(suite),
        passed=sum(result.passed for result in case_results),
        total=len(case_results),
        categories=category_scores,
        cases=case_results,
    )
