"""FTR-101 unchanged-base reproduction identity capture and exact comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Protocol, cast

from tiny_qwen_coder.evaluation._baseline_provenance import load_baseline_base_model_identity
from tiny_qwen_coder.evaluation.settings import (
    FrozenEvaluationSettings,
    evaluation_settings_sha256,
    load_frozen_evaluation_settings,
)

_BASE_CONFIG = Path("configs/base/qwen35-4b.yaml")
_PROTOCOL_CONFIG = Path("configs/eval/python/ftr_101_base_reproduction_v1.json")
_SUITES: tuple[tuple[str, str, str], ...] = (
    (
        "humaneval",
        "humaneval/humaneval-results.jsonl",
        "humaneval/humaneval-aggregate.json",
    ),
    ("mbpp", "mbpp/mbpp-results.jsonl", "mbpp/mbpp-aggregate.json"),
    (
        "repository-holdout",
        "repository-holdout/repository-holdout-results.jsonl",
        "repository-holdout/repository-holdout-aggregate.json",
    ),
)
_AGGREGATE_DYNAMIC_FIELDS = frozenset(
    {
        "results_sha256",
        "passed",
        "failed",
        "pass_at_1",
        "harness_errors",
        "timed_out",
    }
)
_RUNTIME_IDENTITY_FIELDS = (
    "torch_version",
    "transformers_version",
    "model_class",
    "parameter_dtypes",
    "resolved_model_revision",
    "gpu_name",
    "gpu_compute_capability",
)
_PROVENANCE_IDENTITY_FIELDS = ("python_version", "cuda_runtime", "dependencies")


class FTRBaseReproductionError(RuntimeError):
    """Raised when FTR-101 reproduction evidence does not match the frozen baseline."""


class _TokenizerLike(Protocol):
    chat_template: str | None
    eos_token: str | None
    eos_token_id: int | list[int] | None
    pad_token: str | None
    pad_token_id: int | None


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _strict_mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise FTRBaseReproductionError(f"{context} must be a JSON object")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise FTRBaseReproductionError(f"{context} keys must be strings")
        result[key] = item
    return result


def _read_json(path: Path, *, context: str) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FTRBaseReproductionError(f"could not read {context}: {path}") from exc
    return _strict_mapping(value, context=context)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise FTRBaseReproductionError(f"could not hash file: {path}") from exc
    return digest.hexdigest()


def _expect_str(mapping: Mapping[str, object], key: str, *, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise FTRBaseReproductionError(f"{context}.{key} must be a non-empty string")
    return value


def _expect_int(mapping: Mapping[str, object], key: str, *, context: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise FTRBaseReproductionError(f"{context}.{key} must be an integer")
    return value


def _load_protocol_config(path: Path = _PROTOCOL_CONFIG) -> dict[str, object]:
    config = _read_json(path, context="FTR-101 protocol config")
    if _expect_int(config, "schema_version", context="FTR-101 protocol config") != 1:
        raise FTRBaseReproductionError("unsupported FTR-101 protocol config schema")
    if _expect_str(config, "id", context="FTR-101 protocol config") != (
        "python-ftr-101-base-reproduction-v1"
    ):
        raise FTRBaseReproductionError("unexpected FTR-101 protocol config id")
    if _expect_str(config, "task_id", context="FTR-101 protocol config") != "FTR-101":
        raise FTRBaseReproductionError("unexpected FTR-101 task id")
    return config


def _protocol_frozen_baseline(config: Mapping[str, object]) -> dict[str, object]:
    return _strict_mapping(config.get("frozen_baseline"), context="FTR-101 frozen_baseline")


def _score_pair(value: object, *, context: str) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise FTRBaseReproductionError(f"{context} must be [passed, total] integers")
    return cast(tuple[int, int], tuple(value))


def _load_jsonl(path: Path, *, context: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value: object = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise FTRBaseReproductionError(
                        f"invalid JSON in {context} line {line_number}: {path}"
                    ) from exc
                rows.append(_strict_mapping(value, context=f"{context}[{line_number}]"))
    except OSError as exc:
        raise FTRBaseReproductionError(f"could not read {context}: {path}") from exc
    if not rows:
        raise FTRBaseReproductionError(f"{context} is empty: {path}")
    return rows


def _generation_settings_payload(settings: FrozenEvaluationSettings) -> dict[str, object]:
    generation = settings.generation
    return {
        "decoding_strategy": generation.decoding_strategy,
        "temperature": generation.temperature,
        "top_p": generation.top_p,
        "top_k": generation.top_k,
        "max_new_tokens": generation.max_new_tokens,
        "stop_policy": generation.stop_policy,
        "prompt_version": generation.prompt_version,
        "chat_template_version": generation.chat_template_version,
        "enable_thinking": False,
    }


def tokenizer_identity_payload(
    tokenizer: _TokenizerLike,
    *,
    settings: FrozenEvaluationSettings,
    repository: str,
    revision: str,
) -> dict[str, object]:
    """Build the exact tokenizer/template/stop identity used by FTR-101 generation."""

    template = tokenizer.chat_template
    if not isinstance(template, str) or not template:
        raise FTRBaseReproductionError("loaded tokenizer does not expose a chat template")
    eos_id = tokenizer.eos_token_id
    if isinstance(eos_id, list):
        eos_ids: list[int] = []
        for item in eos_id:
            if isinstance(item, bool) or not isinstance(item, int):
                raise FTRBaseReproductionError("tokenizer EOS token IDs must be integers")
            eos_ids.append(item)
    elif isinstance(eos_id, int) and not isinstance(eos_id, bool):
        eos_ids = [eos_id]
    elif eos_id is None:
        eos_ids = []
    else:
        raise FTRBaseReproductionError("tokenizer EOS token ID has unexpected type")

    return {
        "schema_version": 1,
        "tokenizer_repository": repository,
        "tokenizer_revision": revision,
        "tokenizer_class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "chat_template_sha256": _sha256_text(template),
        "generation": _generation_settings_payload(settings),
        "stop_conditions": {
            "explicit_stop_strings": [],
            "explicit_stop_token_ids": [],
            "policy": settings.generation.stop_policy,
            "tokenizer_eos_token": tokenizer.eos_token,
            "tokenizer_eos_token_ids": eos_ids,
            "tokenizer_pad_token": tokenizer.pad_token,
            "tokenizer_pad_token_id": tokenizer.pad_token_id,
        },
        "seed": settings.seed,
        "evaluation_settings_sha256": evaluation_settings_sha256(settings),
    }


def capture_identity(*, output: Path) -> dict[str, object]:
    """Load the pinned tokenizer and persist FTR-101 generation identity."""

    from transformers import AutoTokenizer, PreTrainedTokenizerBase

    base_model = load_baseline_base_model_identity(_BASE_CONFIG)
    settings = load_frozen_evaluation_settings()
    tokenizer_obj: object = AutoTokenizer.from_pretrained(
        base_model.tokenizer_repository,
        revision=base_model.tokenizer_revision,
    )
    if not isinstance(tokenizer_obj, PreTrainedTokenizerBase):
        raise FTRBaseReproductionError("Transformers returned an unexpected tokenizer object")
    payload = tokenizer_identity_payload(
        cast(_TokenizerLike, tokenizer_obj),
        settings=settings,
        repository=base_model.tokenizer_repository,
        revision=base_model.tokenizer_revision,
    )
    payload["base_model"] = asdict(base_model)
    payload["ftr_101_protocol_config_sha256"] = _file_sha256(_PROTOCOL_CONFIG)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _canonical_result(row: Mapping[str, object], *, context: str) -> dict[str, object]:
    problem_id = _expect_str(row, "problem_id", context=context)
    generation = _strict_mapping(row.get("generation"), context=f"{context}.generation")
    tests = _strict_mapping(row.get("tests"), context=f"{context}.tests")
    return {
        "problem_id": problem_id,
        "generated_text": _expect_str(row, "generated_text", context=context),
        "generated_code": row.get("generated_code"),
        "prompt_tokens": _expect_int(generation, "prompt_tokens", context=f"{context}.generation"),
        "generated_tokens": _expect_int(
            generation, "generated_tokens", context=f"{context}.generation"
        ),
        "parse_status": row.get("parse_status"),
        "compile_status": row.get("compile_status"),
        "error_category": row.get("error_category"),
        "tests_passed": _expect_int(tests, "passed", context=f"{context}.tests"),
        "tests_total": _expect_int(tests, "total", context=f"{context}.tests"),
    }


def _index_results(path: Path, *, suite_id: str) -> dict[str, dict[str, object]]:
    rows = _load_jsonl(path, context=f"{suite_id} results")
    indexed: dict[str, dict[str, object]] = {}
    for index, row in enumerate(rows):
        canonical = _canonical_result(row, context=f"{suite_id}[{index}]")
        problem_id = cast(str, canonical["problem_id"])
        if problem_id in indexed:
            raise FTRBaseReproductionError(f"duplicate task ID in {suite_id}: {problem_id}")
        indexed[problem_id] = canonical
    return indexed


def _stable_aggregate(path: Path, *, context: str) -> dict[str, object]:
    aggregate = _read_json(path, context=context)
    return {
        key: value for key, value in aggregate.items() if key not in _AGGREGATE_DYNAMIC_FIELDS
    }


def _suite_comparison(
    *,
    frozen_path: Path,
    current_path: Path,
    frozen_aggregate_path: Path,
    current_aggregate_path: Path,
    suite_id: str,
) -> dict[str, object]:
    frozen = _index_results(frozen_path, suite_id=f"frozen {suite_id}")
    current = _index_results(current_path, suite_id=f"current {suite_id}")
    frozen_aggregate = _stable_aggregate(
        frozen_aggregate_path, context=f"frozen {suite_id} aggregate"
    )
    current_aggregate = _stable_aggregate(
        current_aggregate_path, context=f"current {suite_id} aggregate"
    )
    aggregate_changed_fields = sorted(
        key
        for key in set(frozen_aggregate) | set(current_aggregate)
        if frozen_aggregate.get(key) != current_aggregate.get(key)
    )
    frozen_ids = set(frozen)
    current_ids = set(current)
    missing = sorted(frozen_ids - current_ids)
    unexpected = sorted(current_ids - frozen_ids)
    mismatches: list[dict[str, object]] = []
    frozen_passed = 0
    current_passed = 0
    for problem_id in sorted(frozen_ids & current_ids):
        frozen_row = frozen[problem_id]
        current_row = current[problem_id]
        frozen_ok = frozen_row["tests_passed"] == frozen_row["tests_total"]
        current_ok = current_row["tests_passed"] == current_row["tests_total"]
        frozen_passed += int(frozen_ok)
        current_passed += int(current_ok)
        if frozen_row != current_row:
            changed_fields = sorted(
                key for key in frozen_row if frozen_row.get(key) != current_row.get(key)
            )
            mismatches.append(
                {
                    "problem_id": problem_id,
                    "changed_fields": changed_fields,
                    "frozen_generated_text_sha256": _sha256_text(
                        cast(str, frozen_row["generated_text"])
                    ),
                    "current_generated_text_sha256": _sha256_text(
                        cast(str, current_row["generated_text"])
                    ),
                    "frozen_passed": frozen_ok,
                    "current_passed": current_ok,
                }
            )
    records_exact = not missing and not unexpected and not mismatches
    harness_identity_exact = not aggregate_changed_fields
    exact = records_exact and harness_identity_exact
    return {
        "suite_id": suite_id,
        "frozen_tasks": len(frozen),
        "current_tasks": len(current),
        "frozen_passed": frozen_passed,
        "current_passed": current_passed,
        "missing_task_ids": missing,
        "unexpected_task_ids": unexpected,
        "mismatches": mismatches,
        "aggregate_identity_changed_fields": aggregate_changed_fields,
        "harness_identity_exact": harness_identity_exact,
        "exact_task_record_reproduction": records_exact,
        "exact_suite_reproduction": exact,
    }


def _runtime_identity(root: Path) -> dict[str, object]:
    provenance = _read_json(root / "provenance.json", context="baseline provenance")
    runtime = _read_json(root / "runtime-metadata.json", context="baseline runtime metadata")
    return {
        "provenance": {key: provenance.get(key) for key in _PROVENANCE_IDENTITY_FIELDS},
        "runtime": {key: runtime.get(key) for key in _RUNTIME_IDENTITY_FIELDS},
    }


def compare_reproduction(
    *,
    frozen_dir: Path,
    current_dir: Path,
    identity_path: Path,
    output: Path,
    expected_current_source_sha: str,
    protocol_config_path: Path = _PROTOCOL_CONFIG,
) -> dict[str, object]:
    """Compare one current unchanged-base run against the exact frozen baseline."""

    frozen_manifest = _read_json(
        frozen_dir / "baseline-manifest.json", context="frozen baseline manifest"
    )
    current_manifest = _read_json(
        current_dir / "baseline-manifest.json", context="current baseline manifest"
    )
    identity = _read_json(identity_path, context="FTR-101 generation identity")
    protocol_config = _load_protocol_config(protocol_config_path)
    protocol_config_sha256 = _file_sha256(protocol_config_path)
    frozen_protocol = _protocol_frozen_baseline(protocol_config)

    frozen_source = _expect_str(frozen_manifest, "source_git_sha", context="frozen manifest")
    current_source = _expect_str(current_manifest, "source_git_sha", context="current manifest")
    if current_source != expected_current_source_sha:
        raise FTRBaseReproductionError(
            "current baseline source SHA mismatch: "
            f"{current_source} != {expected_current_source_sha}"
        )
    expected_frozen_source = _expect_str(
        frozen_protocol, "source_git_sha", context="FTR-101 frozen_baseline"
    )
    if frozen_source != expected_frozen_source:
        raise FTRBaseReproductionError(
            "downloaded frozen baseline source SHA mismatch: "
            f"{frozen_source} != {expected_frozen_source}"
        )

    identity_keys = (
        "base_model",
        "evaluation_config_sha256",
        "evaluation_settings_sha256",
        "system_prompt_sha256",
        "system_prompt_version",
        "generation_contract_sha256",
    )
    identity_mismatches = [
        key for key in identity_keys if frozen_manifest.get(key) != current_manifest.get(key)
    ]

    expected_template_sha = _expect_str(
        frozen_protocol, "chat_template_sha256", context="FTR-101 frozen_baseline"
    )
    observed_template_sha = _expect_str(
        identity, "chat_template_sha256", context="FTR-101 generation identity"
    )
    if observed_template_sha != expected_template_sha:
        identity_mismatches.append("chat_template_sha256")

    frozen_runtime_identity = _runtime_identity(frozen_dir)
    current_runtime_identity = _runtime_identity(current_dir)
    runtime_identity_exact = frozen_runtime_identity == current_runtime_identity

    suites = [
        _suite_comparison(
            frozen_path=frozen_dir / results_relative,
            current_path=current_dir / results_relative,
            frozen_aggregate_path=frozen_dir / aggregate_relative,
            current_aggregate_path=current_dir / aggregate_relative,
            suite_id=suite_id,
        )
        for suite_id, results_relative, aggregate_relative in _SUITES
    ]
    all_suites_exact = all(bool(suite["exact_suite_reproduction"]) for suite in suites)
    score_totals = {
        "frozen": sum(cast(int, suite["frozen_passed"]) for suite in suites),
        "current": sum(cast(int, suite["current_passed"]) for suite in suites),
        "total": sum(cast(int, suite["frozen_tasks"]) for suite in suites),
    }
    expected_scores = _strict_mapping(
        frozen_protocol.get("scores"), context="FTR-101 frozen_baseline.scores"
    )
    suite_by_id = {cast(str, suite["suite_id"]): suite for suite in suites}
    score_mismatches: list[str] = []
    for suite_id in ("humaneval", "mbpp", "repository-holdout"):
        expected_passed, expected_total = _score_pair(
            expected_scores.get(suite_id),
            context=f"FTR-101 frozen_baseline.scores.{suite_id}",
        )
        suite = suite_by_id[suite_id]
        if (
            cast(int, suite["frozen_passed"]) != expected_passed
            or cast(int, suite["frozen_tasks"]) != expected_total
            or cast(int, suite["current_passed"]) != expected_passed
            or cast(int, suite["current_tasks"]) != expected_total
        ):
            score_mismatches.append(suite_id)
    expected_combined_passed, expected_combined_total = _score_pair(
        expected_scores.get("combined"), context="FTR-101 frozen_baseline.scores.combined"
    )
    if (
        score_totals["frozen"] != expected_combined_passed
        or score_totals["current"] != expected_combined_passed
        or score_totals["total"] != expected_combined_total
    ):
        score_mismatches.append("combined")

    protocol_identity_mismatches: list[str] = []
    frozen_workflow_run_id = _expect_int(
        frozen_protocol, "workflow_run_id", context="FTR-101 frozen_baseline"
    )
    if frozen_workflow_run_id != 33301242379:
        protocol_identity_mismatches.append("workflow_run_id")
    if _expect_int(frozen_protocol, "artifact_id", context="FTR-101 frozen_baseline") != 9729636096:
        protocol_identity_mismatches.append("artifact_id")
    frozen_artifact_set = _expect_str(
        frozen_manifest, "artifact_set_sha256", context="frozen manifest"
    )
    expected_artifact_set = _expect_str(
        frozen_protocol, "artifact_set_sha256", context="FTR-101 frozen_baseline"
    )
    if frozen_artifact_set != expected_artifact_set:
        protocol_identity_mismatches.append("artifact_set_sha256")
    captured_protocol_sha = _expect_str(
        identity, "ftr_101_protocol_config_sha256", context="FTR-101 generation identity"
    )
    if captured_protocol_sha != protocol_config_sha256:
        protocol_identity_mismatches.append("protocol_config_sha256")

    payload: dict[str, object] = {
        "schema_version": 1,
        "task_id": "FTR-101",
        "status": (
            "exact_reproduction"
            if (
                not identity_mismatches
                and not protocol_identity_mismatches
                and not score_mismatches
                and runtime_identity_exact
                and all_suites_exact
            )
            else "drift"
        ),
        "protocol": {
            "path": str(protocol_config_path),
            "sha256": protocol_config_sha256,
        },
        "frozen_baseline": {
            "workflow_run_id": frozen_workflow_run_id,
            "artifact_id": _expect_int(
                frozen_protocol, "artifact_id", context="FTR-101 frozen_baseline"
            ),
            "artifact_name": _expect_str(
                frozen_protocol, "artifact_name", context="FTR-101 frozen_baseline"
            ),
            "artifact_digest": _expect_str(
                frozen_protocol, "artifact_digest", context="FTR-101 frozen_baseline"
            ),
            "artifact_set_sha256": expected_artifact_set,
            "source_git_sha": frozen_source,
        },
        "current_source_git_sha": current_source,
        "identity": identity,
        "manifest_identity_mismatches": identity_mismatches,
        "protocol_identity_mismatches": protocol_identity_mismatches,
        "score_mismatches": score_mismatches,
        "frozen_runtime_identity": frozen_runtime_identity,
        "current_runtime_identity": current_runtime_identity,
        "runtime_identity_exact": runtime_identity_exact,
        "suites": suites,
        "target_language": score_totals,
        "exact_task_record_reproduction": all(
            bool(suite["exact_task_record_reproduction"]) for suite in suites
        ),
        "exact_harness_reproduction": (
            runtime_identity_exact
            and all(bool(suite["harness_identity_exact"]) for suite in suites)
        ),
        "exact_reproduction": (
            not identity_mismatches
            and not protocol_identity_mismatches
            and not score_mismatches
            and runtime_identity_exact
            and all_suites_exact
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if (
        identity_mismatches
        or protocol_identity_mismatches
        or score_mismatches
        or not runtime_identity_exact
        or not all_suites_exact
    ):
        details = []
        if identity_mismatches:
            details.append(f"identity drift: {', '.join(identity_mismatches)}")
        if protocol_identity_mismatches:
            details.append(
                "protocol identity drift: " + ", ".join(protocol_identity_mismatches)
            )
        if score_mismatches:
            details.append("score drift: " + ", ".join(score_mismatches))
        if not runtime_identity_exact:
            details.append("runtime/dependency identity drift")
        if not all_suites_exact:
            details.append("task-level output/scoring or harness-contract drift")
        raise FTRBaseReproductionError("FTR-101 failed closed: " + "; ".join(details))
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture-identity")
    capture.add_argument("--output", type=Path, required=True)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--frozen-dir", type=Path, required=True)
    compare.add_argument("--current-dir", type=Path, required=True)
    compare.add_argument("--identity", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--expected-current-source-sha", required=True)
    compare.add_argument("--protocol-config", type=Path, default=_PROTOCOL_CONFIG)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "capture-identity":
        payload = capture_identity(output=args.output)
    else:
        payload = compare_reproduction(
            frozen_dir=args.frozen_dir,
            current_dir=args.current_dir,
            identity_path=args.identity,
            output=args.output,
            expected_current_source_sha=args.expected_current_source_sha,
            protocol_config_path=args.protocol_config,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
