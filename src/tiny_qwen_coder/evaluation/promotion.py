"""Fail-closed promotion policy for language adapters."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import yaml

FROZEN_PYTHON_PROMOTION_POLICY_PATH = Path("configs/eval/python/promotion_v1.yaml")
FROZEN_PYTHON_PROMOTION_POLICY_SHA256 = (
    "2c884ca66b3b09071971e89777c9877eddd730c9d4cc59e7475b1cbce963b22e"
)

PromotionDisposition = Literal["promote", "reject"]


class PromotionPolicyError(ValueError):
    """Raised when promotion policy or evidence violates the frozen contract."""


def _strict_mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise PromotionPolicyError(f"{context} must be a mapping")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise PromotionPolicyError(f"{context} keys must be strings")
        result[key] = item
    return result


def _require_keys(
    mapping: Mapping[str, object], *, required: frozenset[str], context: str
) -> None:
    unknown = sorted(set(mapping) - required)
    missing = sorted(required - set(mapping))
    if unknown:
        raise PromotionPolicyError(f"{context} contains unknown field(s): {', '.join(unknown)}")
    if missing:
        raise PromotionPolicyError(f"{context} is missing field(s): {', '.join(missing)}")


def _expect_str(mapping: Mapping[str, object], key: str, *, context: str) -> str:
    value = mapping[key]
    if not isinstance(value, str) or not value.strip():
        raise PromotionPolicyError(f"{context}.{key} must be a non-empty string")
    return value


def _expect_int(mapping: Mapping[str, object], key: str, *, context: str) -> int:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise PromotionPolicyError(f"{context}.{key} must be an integer")
    return value


def _expect_float(mapping: Mapping[str, object], key: str, *, context: str) -> float:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PromotionPolicyError(f"{context}.{key} must be a number")
    return float(value)


def _expect_bool(mapping: Mapping[str, object], key: str, *, context: str) -> bool:
    value = mapping[key]
    if not isinstance(value, bool):
        raise PromotionPolicyError(f"{context}.{key} must be a boolean")
    return value


@dataclass(frozen=True, slots=True)
class SuiteBaseline:
    """Frozen unchanged-base result for one protected Python suite."""

    passed: int
    total: int

    def __post_init__(self) -> None:
        if self.total <= 0 or not 0 <= self.passed <= self.total:
            raise PromotionPolicyError("suite baseline must satisfy 0 <= passed <= total")


@dataclass(frozen=True, slots=True)
class PythonPromotionPolicy:
    """Frozen evidentiary meaning of a recommended Python adapter."""

    schema_version: int
    policy_id: str
    language: str
    baseline_source_git_sha: str
    baseline_artifact_set_sha256: str
    humaneval: SuiteBaseline
    mbpp: SuiteBaseline
    repository_holdout: SuiteBaseline
    combined: SuiteBaseline
    minimum_combined_absolute_gain: float
    minimum_combined_passed: int
    require_no_suite_below_baseline: bool
    general_tool_minimum_passed: int
    general_tool_total: int
    general_tool_maximum_regressions: int
    cross_language_required_conclusion: str
    minimum_typescript_semantic_passed: int
    minimum_rust_semantic_passed: int
    cross_language_cases_per_language: int
    required_contamination_status: str
    require_adapter_load_validation: bool

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise PromotionPolicyError("promotion policy schema_version must be 1")
        if self.policy_id != "python-promotion-v1" or self.language != "python":
            raise PromotionPolicyError(
                "promotion policy identity must be python-promotion-v1/python"
            )
        if not 0.0 < self.minimum_combined_absolute_gain < 1.0:
            raise PromotionPolicyError("minimum_combined_absolute_gain must be between 0 and 1")
        expected_minimum = self.combined.passed + math.ceil(
            self.minimum_combined_absolute_gain * self.combined.total
        )
        if self.minimum_combined_passed != expected_minimum:
            raise PromotionPolicyError(
                "minimum_combined_passed does not match the frozen baseline plus gain"
            )
        if self.general_tool_total <= 0:
            raise PromotionPolicyError("general_tool.total must be positive")
        if not 0 <= self.general_tool_minimum_passed <= self.general_tool_total:
            raise PromotionPolicyError("general_tool.minimum_passed is outside suite bounds")
        if self.general_tool_maximum_regressions < 0:
            raise PromotionPolicyError("general_tool.maximum_regressions must be non-negative")
        if self.cross_language_cases_per_language <= 0:
            raise PromotionPolicyError("cross_language.cases_per_language must be positive")
        for minimum in (
            self.minimum_typescript_semantic_passed,
            self.minimum_rust_semantic_passed,
        ):
            if not 0 <= minimum <= self.cross_language_cases_per_language:
                raise PromotionPolicyError("cross-language minimum is outside suite bounds")


@dataclass(frozen=True, slots=True)
class PythonPromotionEvidence:
    """Evidence required to evaluate one Python adapter against the policy."""

    adapter_id: str
    humaneval_passed: int
    humaneval_total: int
    mbpp_passed: int
    mbpp_total: int
    repository_holdout_passed: int
    repository_holdout_total: int
    combined_passed: int
    combined_total: int
    general_tool_passed: int
    general_tool_total: int
    general_tool_regressions: int
    cross_language_conclusion: str
    typescript_semantic_passed: int
    typescript_semantic_total: int
    rust_semantic_passed: int
    rust_semantic_total: int
    contamination_status: str
    adapter_load_validated: bool

    def __post_init__(self) -> None:
        if not self.adapter_id.strip():
            raise PromotionPolicyError("adapter_id must not be empty")
        for passed, total, name in (
            (self.humaneval_passed, self.humaneval_total, "humaneval"),
            (self.mbpp_passed, self.mbpp_total, "mbpp"),
            (
                self.repository_holdout_passed,
                self.repository_holdout_total,
                "repository_holdout",
            ),
            (self.combined_passed, self.combined_total, "combined"),
            (self.general_tool_passed, self.general_tool_total, "general_tool"),
            (
                self.typescript_semantic_passed,
                self.typescript_semantic_total,
                "typescript_semantic",
            ),
            (self.rust_semantic_passed, self.rust_semantic_total, "rust_semantic"),
        ):
            if total <= 0 or not 0 <= passed <= total:
                raise PromotionPolicyError(f"{name} must satisfy 0 <= passed <= total")
        if self.general_tool_regressions < 0:
            raise PromotionPolicyError("general_tool_regressions must be non-negative")
        if not self.cross_language_conclusion.strip():
            raise PromotionPolicyError("cross_language_conclusion must not be empty")
        if not self.contamination_status.strip():
            raise PromotionPolicyError("contamination_status must not be empty")


@dataclass(frozen=True, slots=True)
class PromotionCheck:
    """One auditable promotion gate."""

    check_id: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class PythonPromotionDecision:
    """Deterministic promotion decision for one Python adapter."""

    policy_id: str
    adapter_id: str
    disposition: PromotionDisposition
    recommended_adapter_id: str | None
    checks: tuple[PromotionCheck, ...]
    rejection_reasons: tuple[str, ...]


def _parse_suite(value: object, *, context: str) -> SuiteBaseline:
    mapping = _strict_mapping(value, context=context)
    _require_keys(mapping, required=frozenset({"passed", "total"}), context=context)
    return SuiteBaseline(
        passed=_expect_int(mapping, "passed", context=context),
        total=_expect_int(mapping, "total", context=context),
    )


def parse_python_promotion_policy(value: object) -> PythonPromotionPolicy:
    """Parse a strict Python promotion policy mapping."""

    root = _strict_mapping(value, context="promotion policy")
    _require_keys(
        root,
        required=frozenset(
            {
                "schema_version",
                "policy_id",
                "language",
                "baseline",
                "target_language",
                "general_tool",
                "cross_language",
                "eligibility",
            }
        ),
        context="promotion policy",
    )
    baseline = _strict_mapping(root["baseline"], context="baseline")
    _require_keys(
        baseline,
        required=frozenset({"source_git_sha", "artifact_set_sha256", "suites"}),
        context="baseline",
    )
    suites = _strict_mapping(baseline["suites"], context="baseline.suites")
    _require_keys(
        suites,
        required=frozenset({"humaneval", "mbpp", "repository_holdout", "combined"}),
        context="baseline.suites",
    )
    target = _strict_mapping(root["target_language"], context="target_language")
    _require_keys(
        target,
        required=frozenset(
            {
                "minimum_combined_absolute_gain",
                "minimum_combined_passed",
                "require_no_suite_below_baseline",
            }
        ),
        context="target_language",
    )
    general = _strict_mapping(root["general_tool"], context="general_tool")
    _require_keys(
        general,
        required=frozenset({"minimum_passed", "total", "maximum_regressions"}),
        context="general_tool",
    )
    cross = _strict_mapping(root["cross_language"], context="cross_language")
    _require_keys(
        cross,
        required=frozenset(
            {
                "required_conclusion",
                "minimum_typescript_semantic_passed",
                "minimum_rust_semantic_passed",
                "cases_per_language",
            }
        ),
        context="cross_language",
    )
    eligibility = _strict_mapping(root["eligibility"], context="eligibility")
    _require_keys(
        eligibility,
        required=frozenset(
            {"required_contamination_status", "require_adapter_load_validation"}
        ),
        context="eligibility",
    )
    return PythonPromotionPolicy(
        schema_version=_expect_int(root, "schema_version", context="promotion policy"),
        policy_id=_expect_str(root, "policy_id", context="promotion policy"),
        language=_expect_str(root, "language", context="promotion policy"),
        baseline_source_git_sha=_expect_str(baseline, "source_git_sha", context="baseline"),
        baseline_artifact_set_sha256=_expect_str(
            baseline, "artifact_set_sha256", context="baseline"
        ),
        humaneval=_parse_suite(suites["humaneval"], context="baseline.suites.humaneval"),
        mbpp=_parse_suite(suites["mbpp"], context="baseline.suites.mbpp"),
        repository_holdout=_parse_suite(
            suites["repository_holdout"], context="baseline.suites.repository_holdout"
        ),
        combined=_parse_suite(suites["combined"], context="baseline.suites.combined"),
        minimum_combined_absolute_gain=_expect_float(
            target, "minimum_combined_absolute_gain", context="target_language"
        ),
        minimum_combined_passed=_expect_int(
            target, "minimum_combined_passed", context="target_language"
        ),
        require_no_suite_below_baseline=_expect_bool(
            target, "require_no_suite_below_baseline", context="target_language"
        ),
        general_tool_minimum_passed=_expect_int(
            general, "minimum_passed", context="general_tool"
        ),
        general_tool_total=_expect_int(general, "total", context="general_tool"),
        general_tool_maximum_regressions=_expect_int(
            general, "maximum_regressions", context="general_tool"
        ),
        cross_language_required_conclusion=_expect_str(
            cross, "required_conclusion", context="cross_language"
        ),
        minimum_typescript_semantic_passed=_expect_int(
            cross, "minimum_typescript_semantic_passed", context="cross_language"
        ),
        minimum_rust_semantic_passed=_expect_int(
            cross, "minimum_rust_semantic_passed", context="cross_language"
        ),
        cross_language_cases_per_language=_expect_int(
            cross, "cases_per_language", context="cross_language"
        ),
        required_contamination_status=_expect_str(
            eligibility, "required_contamination_status", context="eligibility"
        ),
        require_adapter_load_validation=_expect_bool(
            eligibility, "require_adapter_load_validation", context="eligibility"
        ),
    )


def load_frozen_python_promotion_policy(*, repo_root: Path = Path(".")) -> PythonPromotionPolicy:
    """Load the frozen policy and reject any byte-level drift."""

    path = repo_root / FROZEN_PYTHON_PROMOTION_POLICY_PATH
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise PromotionPolicyError(f"could not read promotion policy {path}") from exc
    actual_sha256 = hashlib.sha256(data).hexdigest()
    if actual_sha256 != FROZEN_PYTHON_PROMOTION_POLICY_SHA256:
        raise PromotionPolicyError(
            "frozen Python promotion policy SHA-256 mismatch: "
            f"expected {FROZEN_PYTHON_PROMOTION_POLICY_SHA256}, got {actual_sha256}"
        )
    try:
        value: object = yaml.safe_load(data.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise PromotionPolicyError(f"could not parse promotion policy {path}") from exc
    return parse_python_promotion_policy(value)


def evaluate_python_adapter_promotion(
    policy: PythonPromotionPolicy, evidence: PythonPromotionEvidence
) -> PythonPromotionDecision:
    """Evaluate every frozen promotion gate without fallback or normalization."""

    checks: list[PromotionCheck] = []

    combined_ok = (
        evidence.combined_total == policy.combined.total
        and evidence.combined_passed >= policy.minimum_combined_passed
    )
    checks.append(
        PromotionCheck(
            check_id="python_combined_gain",
            passed=combined_ok,
            detail=(
                f"{evidence.combined_passed}/{evidence.combined_total}; "
                f"requires >= {policy.minimum_combined_passed}/{policy.combined.total}"
            ),
        )
    )

    for check_id, passed, total, baseline in (
        (
            "humaneval_preservation",
            evidence.humaneval_passed,
            evidence.humaneval_total,
            policy.humaneval,
        ),
        ("mbpp_preservation", evidence.mbpp_passed, evidence.mbpp_total, policy.mbpp),
        (
            "repository_holdout_preservation",
            evidence.repository_holdout_passed,
            evidence.repository_holdout_total,
            policy.repository_holdout,
        ),
    ):
        required_passed = baseline.passed if policy.require_no_suite_below_baseline else 0
        passed_gate = total == baseline.total and passed >= required_passed
        checks.append(
            PromotionCheck(
                check_id=check_id,
                passed=passed_gate,
                detail=(
                    f"{passed}/{total}; requires >= {required_passed}/{baseline.total}"
                ),
            )
        )

    general_ok = (
        evidence.general_tool_total == policy.general_tool_total
        and evidence.general_tool_passed >= policy.general_tool_minimum_passed
        and evidence.general_tool_regressions <= policy.general_tool_maximum_regressions
    )
    checks.append(
        PromotionCheck(
            check_id="general_tool_preservation",
            passed=general_ok,
            detail=(
                f"{evidence.general_tool_passed}/{evidence.general_tool_total}, "
                f"regressions={evidence.general_tool_regressions}; requires >= "
                f"{policy.general_tool_minimum_passed}/{policy.general_tool_total}, "
                f"regressions <= {policy.general_tool_maximum_regressions}"
            ),
        )
    )

    cross_ok = (
        evidence.cross_language_conclusion == policy.cross_language_required_conclusion
        and evidence.typescript_semantic_total == policy.cross_language_cases_per_language
        and evidence.rust_semantic_total == policy.cross_language_cases_per_language
        and evidence.typescript_semantic_passed
        >= policy.minimum_typescript_semantic_passed
        and evidence.rust_semantic_passed >= policy.minimum_rust_semantic_passed
    )
    checks.append(
        PromotionCheck(
            check_id="cross_language_preservation",
            passed=cross_ok,
            detail=(
                f"conclusion={evidence.cross_language_conclusion}, "
                f"typescript={evidence.typescript_semantic_passed}/"
                f"{evidence.typescript_semantic_total}, rust={evidence.rust_semantic_passed}/"
                f"{evidence.rust_semantic_total}"
            ),
        )
    )

    contamination_ok = evidence.contamination_status == policy.required_contamination_status
    checks.append(
        PromotionCheck(
            check_id="contamination_evidence",
            passed=contamination_ok,
            detail=(
                f"status={evidence.contamination_status}; "
                f"requires {policy.required_contamination_status}"
            ),
        )
    )

    load_ok = evidence.adapter_load_validated or not policy.require_adapter_load_validation
    checks.append(
        PromotionCheck(
            check_id="adapter_load_validation",
            passed=load_ok,
            detail=(
                f"validated={str(evidence.adapter_load_validated).lower()}; "
                f"required={str(policy.require_adapter_load_validation).lower()}"
            ),
        )
    )

    failed = tuple(check.check_id for check in checks if not check.passed)
    disposition: PromotionDisposition = "reject" if failed else "promote"
    recommended_adapter_id = evidence.adapter_id if disposition == "promote" else None
    return PythonPromotionDecision(
        policy_id=policy.policy_id,
        adapter_id=evidence.adapter_id,
        disposition=disposition,
        recommended_adapter_id=recommended_adapter_id,
        checks=tuple(checks),
        rejection_reasons=failed,
    )


def python_promotion_decision_json(decision: PythonPromotionDecision) -> str:
    """Serialize a promotion decision deterministically."""

    return json.dumps(asdict(decision), indent=2, sort_keys=True) + "\n"
