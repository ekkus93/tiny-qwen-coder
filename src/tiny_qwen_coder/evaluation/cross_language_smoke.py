"""P8-003 preliminary TypeScript/Rust smoke comparison for Python P0."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from tiny_qwen_coder.runtime.adapter_validation import (
        GenerationObservation,
        VerifiedAdapterArtifacts,
    )

TASK_ID = "P8-003"
SCHEMA_VERSION = 2
MEASUREMENT_VERSION = "cross-language-smoke-v2"
DEFAULT_BASE_CONFIG = Path("configs/base/qwen35-4b.yaml")
DEFAULT_OUTPUT = Path("artifacts/eval/python/p0-cross-language-smoke-v2/report.json")
DEFAULT_SEED = 1729
DEFAULT_MAX_NEW_TOKENS = 128
SYSTEM_PROMPT_VERSION = "cross-language-smoke-v1"
SCORING_CONTRACT_VERSION = "cross-language-smoke-scoring-v2"
FORMAT_DIMENSION = "format_adherence"
SEMANTIC_DIMENSION = "semantic_shape"
DECISION_DIMENSION = SEMANTIC_DIMENSION
_FENCE_TAGS = {"typescript": "typescript", "rust": "rust"}
SYSTEM_PROMPT = (
    "You are a coding assistant. Follow the user's requested programming language exactly. "
    "When the user requests only code, return only code with no Markdown fences or explanation."
)
EXPECTED_SUITE_SHA256 = "7f6a6329cabbdb62c1823b0f16cad5bf8067300164c5a3b6ade84bcaaf802f52"
EXPECTED_ADAPTER_ID = "language/python/p0"
EXPECTED_ADAPTER_SHA256 = "c94606250e112f72362eb883a55f7b2c8af854d445f6bb6194352c2806a8f276"
EXPECTED_ADAPTER_SIZE_BYTES = 65004840
EXPECTED_TRAINING_RUN_ID = "training-python-20260831T180916446466Z-02df92a9-eafc119d"
EXPECTED_TRAINING_GIT_SHA = "02df92a9c2d347b9fb013dc25714fe066c6bcafe"


class CrossLanguageSmokeError(RuntimeError):
    """Raised when P8-003 evidence is incomplete, stale, or inconsistent."""


@dataclass(frozen=True, slots=True)
class SmokeCase:
    """One frozen structural cross-language smoke case."""

    case_id: str
    language: str
    prompt: str
    required_patterns: tuple[str, ...]
    forbidden_patterns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TextScore:
    """Deterministic structural score for one generated response."""

    passed: bool
    detail: str | None


CASES = (
    SmokeCase(
        case_id="typescript.add_numbers",
        language="typescript",
        prompt=(
            "Return only TypeScript code. Implement `export function add(a: number, b: number): "
            "number` so it returns the sum of `a` and `b`."
        ),
        required_patterns=(
            r"\bexport\s+function\s+add\s*\(",
            r"\ba\s*:\s*number\b",
            r"\bb\s*:\s*number\b",
            r"\)\s*:\s*number\b",
            r"\breturn\s+a\s*\+\s*b\s*;?",
        ),
        forbidden_patterns=(r"```", r"\bpub\s+fn\b", r"->\s*i32\b"),
    ),
    SmokeCase(
        case_id="typescript.first_or_undefined",
        language="typescript",
        prompt=(
            "Return only TypeScript code. Implement `export function firstOrUndefined<T>(items: "
            "readonly T[]): T | undefined` so it returns the first item, or `undefined` for an "
            "empty array."
        ),
        required_patterns=(
            r"\bexport\s+function\s+firstOrUndefined\s*<\s*T\s*>\s*\(",
            r"(?:readonly\s+T\s*\[\s*\]|ReadonlyArray\s*<\s*T\s*>)",
            r"\)\s*:\s*T\s*\|\s*undefined\b",
            r"(?:items\s*\[\s*0\s*\]|items\.at\s*\(\s*0\s*\))",
        ),
        forbidden_patterns=(r"```", r"\bpub\s+fn\b", r"->\s*Option\b"),
    ),
    SmokeCase(
        case_id="typescript.async_double",
        language="typescript",
        prompt=(
            "Return only TypeScript code. Implement `export async function doubleAsync(value: "
            "number): Promise<number>` so it resolves to `value * 2`."
        ),
        required_patterns=(
            r"\bexport\s+async\s+function\s+doubleAsync\s*\(",
            r"\bvalue\s*:\s*number\b",
            r"\)\s*:\s*Promise\s*<\s*number\s*>",
            r"\breturn\s+value\s*\*\s*2\s*;?",
        ),
        forbidden_patterns=(r"```", r"\bpub\s+fn\b", r"->\s*i32\b"),
    ),
    SmokeCase(
        case_id="rust.add_i32",
        language="rust",
        prompt=(
            "Return only Rust code. Implement `pub fn add(a: i32, b: i32) -> i32` so it returns "
            "the sum of `a` and `b`."
        ),
        required_patterns=(
            r"\bpub\s+fn\s+add\s*\(",
            r"\ba\s*:\s*i32\b",
            r"\bb\s*:\s*i32\b",
            r"\)\s*->\s*i32\b",
            r"(?:\breturn\s+a\s*\+\s*b\s*;|(?:\{|;)\s*a\s*\+\s*b\s*\}?)",
        ),
        forbidden_patterns=(r"```", r"\bexport\s+function\b", r":\s*number\b"),
    ),
    SmokeCase(
        case_id="rust.first_or_none",
        language="rust",
        prompt=(
            "Return only Rust code. Implement `pub fn first_or_none<T: Clone>(items: &[T]) -> "
            "Option<T>` so it returns a cloned first item, or `None` for an empty slice."
        ),
        required_patterns=(
            r"\bpub\s+fn\s+first_or_none\s*<\s*T\s*:\s*Clone\s*>\s*\(",
            r"\bitems\s*:\s*&\s*\[\s*T\s*\]",
            r"\)\s*->\s*Option\s*<\s*T\s*>",
            r"(?:\.first\s*\(\s*\)|\.get\s*\(\s*0\s*\))",
            r"\.cloned\s*\(\s*\)",
        ),
        forbidden_patterns=(r"```", r"\bexport\s+function\b", r":\s*number\b"),
    ),
    SmokeCase(
        case_id="rust.checked_div",
        language="rust",
        prompt=(
            "Return only Rust code. Implement `pub fn checked_div(a: i32, b: i32) -> Option<i32>` "
            "so it returns `None` when `b == 0` and otherwise returns `Some(a / b)`."
        ),
        required_patterns=(
            r"\bpub\s+fn\s+checked_div\s*\(",
            r"\ba\s*:\s*i32\b",
            r"\bb\s*:\s*i32\b",
            r"\)\s*->\s*Option\s*<\s*i32\s*>",
            r"\bNone\b",
            r"\bSome\s*\(\s*a\s*/\s*b\s*\)",
        ),
        forbidden_patterns=(r"```", r"\bexport\s+function\b", r":\s*number\b"),
    ),
)


def _suite_payload() -> list[dict[str, object]]:
    return [asdict(case) for case in CASES]


def suite_sha256() -> str:
    """Return the frozen suite fingerprint."""

    payload = json.dumps(
        _suite_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _scoring_contract_payload() -> dict[str, object]:
    return {
        "version": SCORING_CONTRACT_VERSION,
        "dimensions": {
            FORMAT_DIMENSION: {
                "markdown_fences": "reject",
                "structural_patterns": "frozen_suite",
                "truncation": "generated_tokens_must_be_less_than_max_new_tokens",
            },
            SEMANTIC_DIMENSION: {
                "plain_code": "score_directly",
                "markdown_fences": "allow_exactly_one_whole_response_fence",
                "whole_response_match": "after_trimming_outer_whitespace",
                "required_language_tags": dict(sorted(_FENCE_TAGS.items())),
                "language_tag_comparison": "case_insensitive_exact",
                "prose_outside_fence": "reject",
                "malformed_or_multiple_fences": "reject",
                "structural_patterns": "frozen_suite_after_unwrap",
                "truncation": "generated_tokens_must_be_less_than_max_new_tokens",
            },
        },
        "collapse_decision_dimension": DECISION_DIMENSION,
    }


def scoring_contract_sha256() -> str:
    """Return the frozen v2 scoring-contract fingerprint."""

    payload = json.dumps(
        _scoring_contract_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


EXPECTED_SCORING_CONTRACT_SHA256 = (
    "33c2459c64631ee7cd8903c36a6fe6ecb81df6ce6e1848bad096b5803cc77dd2"
)


def score_text(
    case: SmokeCase,
    text: str,
    *,
    generated_tokens: int,
    max_new_tokens: int,
) -> TextScore:
    """Score strict code-only format adherence plus structural shape."""

    if not text.strip():
        return TextScore(False, "empty response")
    if generated_tokens >= max_new_tokens:
        return TextScore(False, "generation reached the token budget")
    for pattern in case.forbidden_patterns:
        if re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE):
            return TextScore(False, f"forbidden pattern matched: {pattern}")
    missing = tuple(
        pattern
        for pattern in case.required_patterns
        if re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL) is None
    )
    if missing:
        return TextScore(False, "missing required patterns: " + ", ".join(missing))
    return TextScore(True, None)


def _semantic_code(case: SmokeCase, text: str) -> tuple[str | None, str | None]:
    stripped = text.strip()
    if "```" not in stripped:
        return stripped, None
    match = re.fullmatch(
        r"```(?P<tag>[A-Za-z0-9_+-]+)[ \t]*\r?\n(?P<code>.*?)\r?\n```",
        stripped,
        flags=re.DOTALL,
    )
    if match is None:
        return None, "Markdown fence must be exactly one whole-response fenced block"
    tag = match.group("tag").casefold()
    expected_tag = _FENCE_TAGS[case.language]
    if tag != expected_tag:
        return None, f"Markdown fence language tag must be {expected_tag!r}, got {tag!r}"
    code = match.group("code")
    if "```" in code:
        return None, "nested or multiple Markdown fences are not allowed"
    if not code.strip():
        return None, "empty fenced code response"
    return code, None


def score_semantic_text(
    case: SmokeCase,
    text: str,
    *,
    generated_tokens: int,
    max_new_tokens: int,
) -> TextScore:
    """Score language semantics while tolerating one correctly tagged whole-response fence."""

    code, wrapper_error = _semantic_code(case, text)
    if wrapper_error is not None or code is None:
        return TextScore(False, wrapper_error or "invalid semantic response wrapper")
    return score_text(
        case,
        code,
        generated_tokens=generated_tokens,
        max_new_tokens=max_new_tokens,
    )


def _transition(base: bool, adapter: bool) -> str:
    if base and not adapter:
        return "regression"
    if not base and adapter:
        return "improvement"
    if base:
        return "preserved_pass"
    return "preserved_fail"


def _score_block(item: Mapping[str, object], dimension: str) -> Mapping[str, object]:
    value = item.get(dimension)
    if not isinstance(value, Mapping):
        raise CrossLanguageSmokeError(f"missing {dimension} score block")
    return value


def _language_summary(
    cases: Sequence[Mapping[str, object]], *, dimension: str = SEMANTIC_DIMENSION
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for language in ("typescript", "rust"):
        relevant = [item for item in cases if item.get("language") == language]
        if len(relevant) != 3:
            raise CrossLanguageSmokeError(f"expected exactly three {language} cases")
        score_blocks = [_score_block(item, dimension) for item in relevant]
        base_passed = sum(item.get("base_passed") is True for item in score_blocks)
        adapter_passed = sum(item.get("adapter_passed") is True for item in score_blocks)
        catastrophic = dimension == DECISION_DIMENSION and base_passed >= 2 and adapter_passed == 0
        rows.append(
            {
                "language": language,
                "total": 3,
                "base_passed": base_passed,
                "adapter_passed": adapter_passed,
                "delta_passed": adapter_passed - base_passed,
                "baseline_adequate": base_passed >= 2,
                "catastrophic_regression": catastrophic,
            }
        )
    return rows


def _overall_summary(
    cases: Sequence[Mapping[str, object]],
    languages: Sequence[Mapping[str, object]],
    *,
    dimension: str = SEMANTIC_DIMENSION,
) -> dict[str, object]:
    score_blocks = [_score_block(item, dimension) for item in cases]
    base_passed = sum(item.get("base_passed") is True for item in score_blocks)
    adapter_passed = sum(item.get("adapter_passed") is True for item in score_blocks)
    regressions = sum(item.get("transition") == "regression" for item in score_blocks)
    improvements = sum(item.get("transition") == "improvement" for item in score_blocks)
    baseline_adequate = all(item.get("baseline_adequate") is True for item in languages)
    language_collapse = any(item.get("catastrophic_regression") is True for item in languages)
    overall_collapse = base_passed >= 4 and adapter_passed * 2 <= base_passed
    if dimension == DECISION_DIMENSION:
        catastrophic = baseline_adequate and (language_collapse or overall_collapse)
        if not baseline_adequate:
            conclusion = "inconclusive_base"
        elif catastrophic:
            conclusion = "catastrophic_regression"
        else:
            conclusion = "no_catastrophic_regression"
    else:
        catastrophic = False
        conclusion = "supplemental_only"
    return {
        "total_cases": len(cases),
        "base_passed": base_passed,
        "adapter_passed": adapter_passed,
        "delta_passed": adapter_passed - base_passed,
        "regressions": regressions,
        "improvements": improvements,
        "baseline_adequate_for_collapse_detection": baseline_adequate,
        "catastrophic_non_python_collapse_detected": catastrophic,
        "conclusion": conclusion,
    }


def _dimension_summary(
    cases: Sequence[Mapping[str, object]], dimension: str
) -> dict[str, object]:
    languages = _language_summary(cases, dimension=dimension)
    return {
        "languages": languages,
        "overall": _overall_summary(cases, languages, dimension=dimension),
    }


def _adapter_identity(artifacts: "VerifiedAdapterArtifacts") -> dict[str, object]:
    manifest = artifacts.manifest
    identity = {
        "adapter_id": manifest.adapter_id,
        "adapter_model_sha256": artifacts.adapter_model_sha256,
        "adapter_model_size_bytes": artifacts.adapter_model_size_bytes,
        "training_run_id": artifacts.training_run_id,
        "training_git_sha": artifacts.training_git_sha,
    }
    expected = {
        "adapter_id": EXPECTED_ADAPTER_ID,
        "adapter_model_sha256": EXPECTED_ADAPTER_SHA256,
        "adapter_model_size_bytes": EXPECTED_ADAPTER_SIZE_BYTES,
        "training_run_id": EXPECTED_TRAINING_RUN_ID,
        "training_git_sha": EXPECTED_TRAINING_GIT_SHA,
    }
    if identity != expected:
        raise CrossLanguageSmokeError(
            "P8-003 adapter is not the accepted P7-006 Python P0 artifact"
        )
    return identity


def _observation_dict(observation: "GenerationObservation") -> dict[str, object]:
    return {
        "text": observation.text,
        "token_ids": list(observation.token_ids),
        "prompt_tokens": observation.prompt_tokens,
        "generated_tokens": observation.generated_tokens,
        "latency_seconds": observation.latency_seconds,
    }


def _score_pair(
    case: SmokeCase,
    base_observation: "GenerationObservation",
    adapter_observation: "GenerationObservation",
    *,
    max_new_tokens: int,
) -> dict[str, object]:
    base_format = score_text(
        case,
        base_observation.text,
        generated_tokens=base_observation.generated_tokens,
        max_new_tokens=max_new_tokens,
    )
    adapter_format = score_text(
        case,
        adapter_observation.text,
        generated_tokens=adapter_observation.generated_tokens,
        max_new_tokens=max_new_tokens,
    )
    base_semantic = score_semantic_text(
        case,
        base_observation.text,
        generated_tokens=base_observation.generated_tokens,
        max_new_tokens=max_new_tokens,
    )
    adapter_semantic = score_semantic_text(
        case,
        adapter_observation.text,
        generated_tokens=adapter_observation.generated_tokens,
        max_new_tokens=max_new_tokens,
    )
    return {
        "case_id": case.case_id,
        "language": case.language,
        "prompt": case.prompt,
        FORMAT_DIMENSION: {
            "base_passed": base_format.passed,
            "adapter_passed": adapter_format.passed,
            "transition": _transition(base_format.passed, adapter_format.passed),
            "base_detail": base_format.detail,
            "adapter_detail": adapter_format.detail,
        },
        SEMANTIC_DIMENSION: {
            "base_passed": base_semantic.passed,
            "adapter_passed": adapter_semantic.passed,
            "transition": _transition(base_semantic.passed, adapter_semantic.passed),
            "base_detail": base_semantic.detail,
            "adapter_detail": adapter_semantic.detail,
        },
        "base": _observation_dict(base_observation),
        "adapter": _observation_dict(adapter_observation),
    }


def _score_cases(
    base: Sequence["GenerationObservation"],
    adapter: Sequence["GenerationObservation"],
    *,
    max_new_tokens: int,
) -> list[dict[str, object]]:
    if len(base) != len(CASES) or len(adapter) != len(CASES):
        raise CrossLanguageSmokeError("P8-003 generation count drifted")
    return [
        _score_pair(
            case,
            base_observation,
            adapter_observation,
            max_new_tokens=max_new_tokens,
        )
        for case, base_observation, adapter_observation in zip(CASES, base, adapter, strict=True)
    ]


def run_cross_language_smoke(
    training_output: Path,
    *,
    base_config: Path = DEFAULT_BASE_CONFIG,
    device_index: int = 0,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> dict[str, object]:
    """Generate and score base/Python-P0 TypeScript and Rust smoke responses."""

    import torch
    from torch import nn

    from tiny_qwen_coder.evaluation._baseline_runner import _preflight_source_tree
    from tiny_qwen_coder.reporting import load_base_model_identity
    from tiny_qwen_coder.reproducibility import seed_everything
    from tiny_qwen_coder.runtime.adapter_validation import (
        _floating_parameter_dtypes,
        _freeze_inference_parameters,
        _generate,
        _require_enabled_status,
        _resolved_revision,
        _status_snapshot,
        validate_adapter_artifacts,
    )

    if suite_sha256() != EXPECTED_SUITE_SHA256:
        raise CrossLanguageSmokeError("P8-003 frozen suite fingerprint drifted")
    if scoring_contract_sha256() != EXPECTED_SCORING_CONTRACT_SHA256:
        raise CrossLanguageSmokeError("P8-003 v2 scoring-contract fingerprint drifted")
    if max_new_tokens <= 0:
        raise CrossLanguageSmokeError("max_new_tokens must be greater than zero")
    if not torch.cuda.is_available():
        raise CrossLanguageSmokeError("P8-003 requires CUDA")
    if not torch.cuda.is_bf16_supported():
        raise CrossLanguageSmokeError("P8-003 requires a BF16-capable CUDA device")
    if not 0 <= device_index < torch.cuda.device_count():
        raise CrossLanguageSmokeError(f"invalid CUDA device index: {device_index}")

    source_git_sha, _ = _preflight_source_tree(Path("."))
    base_model = load_base_model_identity(base_config)
    artifacts = validate_adapter_artifacts(training_output, base_model)
    adapter_identity = _adapter_identity(artifacts)

    from peft import PeftModel
    from transformers import AutoModelForMultimodalLM, AutoTokenizer, PreTrainedTokenizerBase

    device = torch.device("cuda", device_index)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    seed_everything(DEFAULT_SEED)

    tokenizer_obj: object = AutoTokenizer.from_pretrained(
        base_model.tokenizer_repository,
        revision=base_model.tokenizer_revision,
    )
    if not isinstance(tokenizer_obj, PreTrainedTokenizerBase):
        raise CrossLanguageSmokeError("Transformers returned an unexpected tokenizer object")
    chat_template = tokenizer_obj.chat_template
    if not isinstance(chat_template, str) or not chat_template:
        raise CrossLanguageSmokeError("canonical tokenizer does not expose a chat template")
    template_sha = hashlib.sha256(chat_template.encode("utf-8")).hexdigest()
    if template_sha != artifacts.inference_chat_template_sha256:
        raise CrossLanguageSmokeError("canonical inference chat-template hash drifted")

    model_factory = cast(Any, AutoModelForMultimodalLM)
    started = time.perf_counter()
    loaded: object = model_factory.from_pretrained(
        base_model.repository,
        revision=base_model.revision,
        dtype=torch.bfloat16,
        device_map={"": device_index},
        low_cpu_mem_usage=True,
    )
    base_load_seconds = time.perf_counter() - started
    if not isinstance(loaded, nn.Module):
        raise CrossLanguageSmokeError("Transformers returned an unexpected model object")
    base = loaded
    base.eval()
    resolved_revision = _resolved_revision(base)
    if resolved_revision != base_model.revision:
        raise CrossLanguageSmokeError("loaded base revision drifted")
    floating_dtypes = _floating_parameter_dtypes(base)
    if floating_dtypes != ("torch.bfloat16",):
        raise CrossLanguageSmokeError(
            f"P8-003 requires BF16 floating parameters; observed {floating_dtypes!r}"
        )

    tokenizer = cast(Any, tokenizer_obj)
    base_observations = tuple(
        _generate(
            base,
            tokenizer,
            device,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=case.prompt,
            max_new_tokens=max_new_tokens,
        )
        for case in CASES
    )

    base_object_id = id(base)
    peft_factory = cast(Any, PeftModel)
    adapted_obj: object = peft_factory.from_pretrained(
        base,
        str(artifacts.adapter_dir),
        adapter_name="default",
        is_trainable=False,
    )
    if not isinstance(adapted_obj, nn.Module):
        raise CrossLanguageSmokeError("PEFT returned an unexpected adapted model object")
    adapted = adapted_obj
    adapted.eval()
    _freeze_inference_parameters(adapted)
    get_base = getattr(adapted, "get_base_model", None)
    if not callable(get_base) or id(get_base()) != base_object_id:
        raise CrossLanguageSmokeError("PEFT attachment rebuilt or replaced the loaded base")
    status = _status_snapshot(adapted)
    _require_enabled_status(status)

    adapter_observations = tuple(
        _generate(
            adapted,
            tokenizer,
            device,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=case.prompt,
            max_new_tokens=max_new_tokens,
        )
        for case in CASES
    )
    rows = _score_cases(base_observations, adapter_observations, max_new_tokens=max_new_tokens)
    dimensions = {
        FORMAT_DIMENSION: _dimension_summary(rows, FORMAT_DIMENSION),
        SEMANTIC_DIMENSION: _dimension_summary(rows, SEMANTIC_DIMENSION),
    }
    torch.cuda.synchronize(device)
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "measurement_version": MEASUREMENT_VERSION,
        "measurement_complete": True,
        "source_git_sha": source_git_sha,
        "suite_sha256": EXPECTED_SUITE_SHA256,
        "scoring_contract_version": SCORING_CONTRACT_VERSION,
        "scoring_contract_sha256": EXPECTED_SCORING_CONTRACT_SHA256,
        "decision_dimension": DECISION_DIMENSION,
        "system_prompt_version": SYSTEM_PROMPT_VERSION,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "seed": DEFAULT_SEED,
        "max_new_tokens": max_new_tokens,
        "base_model": asdict(base_model),
        "resolved_model_revision": resolved_revision,
        "adapter": adapter_identity,
        "same_base_object_after_attach": True,
        "adapter_status": asdict(status),
        "gpu": {
            "name": torch.cuda.get_device_name(device),
            "total_bytes": torch.cuda.get_device_properties(device).total_memory,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "base_load_seconds": base_load_seconds,
        },
        "cases": rows,
        "dimensions": dimensions,
    }


def _mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise CrossLanguageSmokeError(f"{context} must be a JSON object")
    return cast(dict[str, object], dict(value))


def verify_report(path: Path, *, base_config: Path = DEFAULT_BASE_CONFIG) -> dict[str, object]:
    """Recompute deterministic P8-003 scoring and provenance from a persisted report."""

    from tiny_qwen_coder.evaluation._baseline_runner import _preflight_source_tree
    from tiny_qwen_coder.reporting import load_base_model_identity

    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CrossLanguageSmokeError(f"could not read P8-003 report: {path}") from exc
    report = _mapping(raw, context="P8-003 report")
    if (
        report.get("schema_version") != SCHEMA_VERSION
        or report.get("task_id") != TASK_ID
        or report.get("measurement_version") != MEASUREMENT_VERSION
    ):
        raise CrossLanguageSmokeError("P8-003 report identity drifted")
    if report.get("measurement_complete") is not True:
        raise CrossLanguageSmokeError("P8-003 report is not complete")
    source_git_sha, _ = _preflight_source_tree(Path("."))
    if report.get("source_git_sha") != source_git_sha:
        raise CrossLanguageSmokeError("P8-003 report source Git SHA drifted")
    if (
        report.get("suite_sha256") != EXPECTED_SUITE_SHA256
        or suite_sha256() != EXPECTED_SUITE_SHA256
    ):
        raise CrossLanguageSmokeError("P8-003 suite identity drifted")
    if (
        report.get("scoring_contract_version") != SCORING_CONTRACT_VERSION
        or report.get("scoring_contract_sha256") != EXPECTED_SCORING_CONTRACT_SHA256
        or scoring_contract_sha256() != EXPECTED_SCORING_CONTRACT_SHA256
        or report.get("decision_dimension") != DECISION_DIMENSION
    ):
        raise CrossLanguageSmokeError("P8-003 v2 scoring contract drifted")
    if report.get("system_prompt_version") != SYSTEM_PROMPT_VERSION:
        raise CrossLanguageSmokeError("P8-003 system prompt version drifted")
    if report.get("system_prompt_sha256") != hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest():
        raise CrossLanguageSmokeError("P8-003 system prompt hash drifted")
    if report.get("seed") != DEFAULT_SEED:
        raise CrossLanguageSmokeError("P8-003 seed drifted")
    max_new_tokens = report.get("max_new_tokens")
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens <= 0
    ):
        raise CrossLanguageSmokeError("P8-003 max_new_tokens is invalid")

    base_model = load_base_model_identity(base_config)
    if report.get("base_model") != asdict(base_model):
        raise CrossLanguageSmokeError("P8-003 base model identity drifted")
    expected_adapter = {
        "adapter_id": EXPECTED_ADAPTER_ID,
        "adapter_model_sha256": EXPECTED_ADAPTER_SHA256,
        "adapter_model_size_bytes": EXPECTED_ADAPTER_SIZE_BYTES,
        "training_run_id": EXPECTED_TRAINING_RUN_ID,
        "training_git_sha": EXPECTED_TRAINING_GIT_SHA,
    }
    if report.get("adapter") != expected_adapter:
        raise CrossLanguageSmokeError("P8-003 adapter identity drifted")
    if report.get("resolved_model_revision") != base_model.revision:
        raise CrossLanguageSmokeError("P8-003 resolved base revision drifted")
    if report.get("same_base_object_after_attach") is not True:
        raise CrossLanguageSmokeError("P8-003 did not preserve the loaded base object")

    raw_cases = report.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) != len(CASES):
        raise CrossLanguageSmokeError("P8-003 case count drifted")
    recomputed: list[dict[str, object]] = []
    for case, raw_case in zip(CASES, raw_cases, strict=True):
        item = _mapping(raw_case, context=f"P8-003 case {case.case_id}")
        if (
            item.get("case_id") != case.case_id
            or item.get("language") != case.language
            or item.get("prompt") != case.prompt
        ):
            raise CrossLanguageSmokeError(f"P8-003 case identity drifted: {case.case_id}")
        base_obs = _mapping(item.get("base"), context=f"{case.case_id}.base")
        adapter_obs = _mapping(item.get("adapter"), context=f"{case.case_id}.adapter")
        base_text = base_obs.get("text")
        adapter_text = adapter_obs.get("text")
        base_tokens = base_obs.get("generated_tokens")
        adapter_tokens = adapter_obs.get("generated_tokens")
        if not isinstance(base_text, str) or not isinstance(adapter_text, str):
            raise CrossLanguageSmokeError(f"P8-003 case text is invalid: {case.case_id}")
        if (
            isinstance(base_tokens, bool)
            or not isinstance(base_tokens, int)
            or isinstance(adapter_tokens, bool)
            or not isinstance(adapter_tokens, int)
        ):
            raise CrossLanguageSmokeError(
                f"P8-003 generated token count is invalid: {case.case_id}"
            )
        base_format = score_text(
            case, base_text, generated_tokens=base_tokens, max_new_tokens=max_new_tokens
        )
        adapter_format = score_text(
            case, adapter_text, generated_tokens=adapter_tokens, max_new_tokens=max_new_tokens
        )
        base_semantic = score_semantic_text(
            case, base_text, generated_tokens=base_tokens, max_new_tokens=max_new_tokens
        )
        adapter_semantic = score_semantic_text(
            case, adapter_text, generated_tokens=adapter_tokens, max_new_tokens=max_new_tokens
        )
        expected_scores = {
            FORMAT_DIMENSION: {
                "base_passed": base_format.passed,
                "adapter_passed": adapter_format.passed,
                "transition": _transition(base_format.passed, adapter_format.passed),
                "base_detail": base_format.detail,
                "adapter_detail": adapter_format.detail,
            },
            SEMANTIC_DIMENSION: {
                "base_passed": base_semantic.passed,
                "adapter_passed": adapter_semantic.passed,
                "transition": _transition(base_semantic.passed, adapter_semantic.passed),
                "base_detail": base_semantic.detail,
                "adapter_detail": adapter_semantic.detail,
            },
        }
        if any(item.get(dimension) != scores for dimension, scores in expected_scores.items()):
            raise CrossLanguageSmokeError(f"P8-003 stored score drifted: {case.case_id}")
        recomputed.append(item)

    dimensions = {
        FORMAT_DIMENSION: _dimension_summary(recomputed, FORMAT_DIMENSION),
        SEMANTIC_DIMENSION: _dimension_summary(recomputed, SEMANTIC_DIMENSION),
    }
    if report.get("dimensions") != dimensions:
        raise CrossLanguageSmokeError("P8-003 aggregate score drifted")
    return report


def report_json(report: Mapping[str, object]) -> str:
    return json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_report(report: Mapping[str, object], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(report_json(report), encoding="utf-8")
    temporary.replace(output)
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or verify the P8-003 cross-language smoke.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--training-output", type=Path, required=True)
    run.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    run.add_argument("--device-index", type=int, default=0)
    run.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    run.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    verify.add_argument("--report", type=Path, default=DEFAULT_OUTPUT)
    return parser


def cross_language_smoke_main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "run":
        report = run_cross_language_smoke(
            cast(Path, args.training_output),
            base_config=cast(Path, args.base_config),
            device_index=cast(int, args.device_index),
            max_new_tokens=cast(int, args.max_new_tokens),
        )
        output = write_report(report, cast(Path, args.output))
        print(report_json(report), end="")
        print(f"P8-003 report: {output}")
        return
    report = verify_report(cast(Path, args.report), base_config=cast(Path, args.base_config))
    print(report_json(report), end="")


if __name__ == "__main__":
    cross_language_smoke_main()
