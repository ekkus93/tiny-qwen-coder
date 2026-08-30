"""Artifact serialization, hashing, completeness checks, and baseline freezing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from pathlib import Path

from tiny_qwen_coder.config import EvaluationConfig, canonical_config_json
from tiny_qwen_coder.evaluation._baseline_provenance import (
    BaselineProvenance,
    baseline_provenance_json,
)
from tiny_qwen_coder.evaluation._baseline_types import (
    BaselineArtifactDigest,
    BaselineRuntimeMetadata,
    PythonBaselineError,
    PythonBaselineManifest,
    RegressionBaselineAggregate,
    RegressionBaselineCaseResult,
)
from tiny_qwen_coder.evaluation.settings import FrozenEvaluationSettings, evaluation_settings_sha256
from tiny_qwen_coder.identities import AdapterIdentity, BaseModelIdentity

_BASELINE_ID = "python-unchanged-base"
_BASELINE_VERSION = 1
_BASELINE_MANIFEST_NAME = "baseline-manifest.json"
_RUNTIME_METADATA_NAME = "runtime-metadata.json"
_REGRESSION_RESULTS_NAME = "general-tool-regression-results.jsonl"
_REGRESSION_AGGREGATE_NAME = "general-tool-regression-aggregate.json"
_REQUIRED_ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("provenance", "provenance.json"),
    ("runtime_metadata", _RUNTIME_METADATA_NAME),
    ("humaneval_results", "humaneval/humaneval-results.jsonl"),
    ("humaneval_aggregate", "humaneval/humaneval-aggregate.json"),
    ("mbpp_results", "mbpp/mbpp-results.jsonl"),
    ("mbpp_aggregate", "mbpp/mbpp-aggregate.json"),
    (
        "repository_holdout_results",
        "repository-holdout/repository-holdout-results.jsonl",
    ),
    (
        "repository_holdout_aggregate",
        "repository-holdout/repository-holdout-aggregate.json",
    ),
    (
        "general_tool_regression_results",
        f"general-tool-regression/{_REGRESSION_RESULTS_NAME}",
    ),
    (
        "general_tool_regression_aggregate",
        f"general-tool-regression/{_REGRESSION_AGGREGATE_NAME}",
    ),
)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    """Hash one baseline artifact exactly as stored on disk."""

    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise PythonBaselineError(f"could not read baseline artifact {path}: {exc}") from exc


def evaluation_config_sha256(config: EvaluationConfig) -> str:
    """Hash the canonical parsed evaluation configuration."""

    return _sha256_bytes(canonical_config_json(config).encode("utf-8"))


def system_prompt_sha256(system_prompt: str) -> str:
    """Hash the exact language system prompt text used for generation."""

    return _sha256_bytes(system_prompt.encode("utf-8"))


def runtime_metadata_json(metadata: BaselineRuntimeMetadata) -> str:
    """Serialize measured runtime metadata deterministically."""

    return json.dumps(asdict(metadata), indent=2, sort_keys=True) + "\n"


def write_runtime_metadata(metadata: BaselineRuntimeMetadata, output_dir: Path) -> Path:
    """Atomically materialize runtime/memory/performance metadata."""

    destination = output_dir / _RUNTIME_METADATA_NAME
    _atomic_write(destination, runtime_metadata_json(metadata))
    return destination


def _regression_result_json_line(result: RegressionBaselineCaseResult) -> str:
    return json.dumps(asdict(result), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def regression_baseline_results_sha256(results: tuple[RegressionBaselineCaseResult, ...]) -> str:
    """Hash ordered general/tool baseline result records."""

    payload = "".join(_regression_result_json_line(result) + "\n" for result in results)
    return _sha256_bytes(payload.encode("utf-8"))


def regression_baseline_aggregate_json(aggregate: RegressionBaselineAggregate) -> str:
    """Serialize the general/tool regression aggregate deterministically."""

    return json.dumps(asdict(aggregate), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_regression_baseline_artifacts(
    *,
    results: tuple[RegressionBaselineCaseResult, ...],
    aggregate: RegressionBaselineAggregate,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write per-case and aggregate general/tool baseline artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / _REGRESSION_RESULTS_NAME
    aggregate_path = output_dir / _REGRESSION_AGGREGATE_NAME
    _atomic_write(
        results_path,
        "".join(_regression_result_json_line(result) + "\n" for result in results),
    )
    _atomic_write(aggregate_path, regression_baseline_aggregate_json(aggregate))
    return results_path, aggregate_path


def _artifact_set_sha256(artifacts: Iterable[BaselineArtifactDigest]) -> str:
    payload = json.dumps(
        [asdict(item) for item in artifacts],
        sort_keys=True,
        separators=(",", ":"),
    )
    return _sha256_bytes(payload.encode("utf-8"))


def python_baseline_manifest_json(manifest: PythonBaselineManifest) -> str:
    """Serialize the frozen baseline manifest deterministically."""

    return json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n"


def _read_json_mapping(path: Path, *, context: str) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PythonBaselineError(f"could not read {context} {path}") from exc
    if not isinstance(value, Mapping):
        raise PythonBaselineError(f"{context} must be a JSON object")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise PythonBaselineError(f"{context} keys must be strings")
        result[key] = item
    return result


def _require_clean_provenance(provenance: BaselineProvenance) -> None:
    if provenance.language != "python":
        raise PythonBaselineError("baseline provenance must use language='python'")
    if provenance.adapter.adapter_id is not None:
        raise PythonBaselineError("baseline provenance must represent unchanged base only")
    if provenance.source_git_dirty:
        raise PythonBaselineError(
            "refusing to freeze baseline artifacts from a dirty source tree; "
            "rerun from a clean commit"
        )
    if not provenance.cuda_available or not provenance.gpus:
        raise PythonBaselineError("baseline provenance does not record a CUDA GPU")


def freeze_python_baseline(
    *,
    output_dir: Path,
    evaluation: EvaluationConfig,
    settings: FrozenEvaluationSettings,
    system_prompt_version: str,
    system_prompt: str,
    generation_contract_sha256: str,
    provenance: BaselineProvenance,
) -> PythonBaselineManifest:
    """Validate the complete artifact set and write the immutable baseline inventory."""

    _require_clean_provenance(provenance)
    adapter = AdapterIdentity(family=None, adapter_id=None)
    if provenance.adapter != adapter:
        raise PythonBaselineError("baseline provenance adapter identity is not base-only")

    provenance_path = output_dir / "provenance.json"
    try:
        stored_provenance = provenance_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PythonBaselineError(
            f"could not read baseline provenance artifact {provenance_path}: {exc}"
        ) from exc
    if stored_provenance != baseline_provenance_json(provenance):
        raise PythonBaselineError(
            "baseline provenance artifact does not match the provenance being frozen"
        )

    artifacts: list[BaselineArtifactDigest] = []
    for artifact_id, relative_path in _REQUIRED_ARTIFACTS:
        path = output_dir / relative_path
        if not path.is_file():
            raise PythonBaselineError(
                f"baseline artifact set is incomplete; missing {relative_path!r}"
            )
        artifacts.append(
            BaselineArtifactDigest(
                artifact_id=artifact_id,
                path=relative_path,
                sha256=file_sha256(path),
            )
        )
    ordered = tuple(artifacts)
    manifest = PythonBaselineManifest(
        schema_version=1,
        baseline_id=_BASELINE_ID,
        baseline_version=_BASELINE_VERSION,
        frozen=True,
        base_model=provenance.base_model,
        adapter=adapter,
        source_git_sha=provenance.source_git_sha,
        evaluation_config_sha256=evaluation_config_sha256(evaluation),
        evaluation_settings_sha256=evaluation_settings_sha256(settings),
        system_prompt_version=system_prompt_version,
        system_prompt_sha256=system_prompt_sha256(system_prompt),
        generation_contract_sha256=generation_contract_sha256,
        artifacts=ordered,
        artifact_set_sha256=_artifact_set_sha256(ordered),
    )
    _atomic_write(output_dir / _BASELINE_MANIFEST_NAME, python_baseline_manifest_json(manifest))
    return manifest


def _expect_str(mapping: Mapping[str, object], key: str, *, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PythonBaselineError(f"{context}.{key} must be a non-empty string")
    return value


def _expect_int(mapping: Mapping[str, object], key: str, *, context: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise PythonBaselineError(f"{context}.{key} must be an integer")
    return value


def _parse_base_model(value: object) -> BaseModelIdentity:
    if not isinstance(value, Mapping):
        raise PythonBaselineError("baseline_manifest.base_model must be an object")
    mapping = dict(value)
    return BaseModelIdentity(
        repository=_expect_str(mapping, "repository", context="baseline_manifest.base_model"),
        revision=_expect_str(mapping, "revision", context="baseline_manifest.base_model"),
        tokenizer_repository=_expect_str(
            mapping,
            "tokenizer_repository",
            context="baseline_manifest.base_model",
        ),
        tokenizer_revision=_expect_str(
            mapping,
            "tokenizer_revision",
            context="baseline_manifest.base_model",
        ),
    )


def _parse_adapter(value: object) -> AdapterIdentity:
    if not isinstance(value, Mapping):
        raise PythonBaselineError("baseline_manifest.adapter must be an object")
    family = value.get("family")
    adapter_id = value.get("adapter_id")
    if family is not None and not isinstance(family, str):
        raise PythonBaselineError("baseline_manifest.adapter.family must be string or null")
    if adapter_id is not None and not isinstance(adapter_id, str):
        raise PythonBaselineError("baseline_manifest.adapter.adapter_id must be string or null")
    return AdapterIdentity(family=family, adapter_id=adapter_id)


def load_python_baseline_manifest(
    path: Path,
) -> PythonBaselineManifest:
    """Load one frozen baseline manifest from JSON for later adapter-comparison gates."""

    mapping = _read_json_mapping(path, context="baseline manifest")
    raw_artifacts = mapping.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise PythonBaselineError("baseline_manifest.artifacts must be a list")
    artifacts: list[BaselineArtifactDigest] = []
    for index, value in enumerate(raw_artifacts):
        if not isinstance(value, Mapping):
            raise PythonBaselineError(f"baseline_manifest.artifacts[{index}] must be an object")
        item = dict(value)
        artifacts.append(
            BaselineArtifactDigest(
                artifact_id=_expect_str(
                    item,
                    "artifact_id",
                    context=f"baseline_manifest.artifacts[{index}]",
                ),
                path=_expect_str(item, "path", context=f"baseline_manifest.artifacts[{index}]"),
                sha256=_expect_str(
                    item,
                    "sha256",
                    context=f"baseline_manifest.artifacts[{index}]",
                ),
            )
        )
    frozen = mapping.get("frozen")
    if not isinstance(frozen, bool):
        raise PythonBaselineError("baseline_manifest.frozen must be a boolean")
    return PythonBaselineManifest(
        schema_version=_expect_int(mapping, "schema_version", context="baseline_manifest"),
        baseline_id=_expect_str(mapping, "baseline_id", context="baseline_manifest"),
        baseline_version=_expect_int(mapping, "baseline_version", context="baseline_manifest"),
        frozen=frozen,
        base_model=_parse_base_model(mapping.get("base_model")),
        adapter=_parse_adapter(mapping.get("adapter")),
        source_git_sha=_expect_str(mapping, "source_git_sha", context="baseline_manifest"),
        evaluation_config_sha256=_expect_str(
            mapping,
            "evaluation_config_sha256",
            context="baseline_manifest",
        ),
        evaluation_settings_sha256=_expect_str(
            mapping,
            "evaluation_settings_sha256",
            context="baseline_manifest",
        ),
        system_prompt_version=_expect_str(
            mapping,
            "system_prompt_version",
            context="baseline_manifest",
        ),
        system_prompt_sha256=_expect_str(
            mapping,
            "system_prompt_sha256",
            context="baseline_manifest",
        ),
        generation_contract_sha256=_expect_str(
            mapping,
            "generation_contract_sha256",
            context="baseline_manifest",
        ),
        artifacts=tuple(artifacts),
        artifact_set_sha256=_expect_str(
            mapping,
            "artifact_set_sha256",
            context="baseline_manifest",
        ),
    )


def validate_python_baseline_artifacts(
    output_dir: Path,
    manifest: PythonBaselineManifest | None = None,
) -> PythonBaselineManifest:
    """Fail closed if any frozen P6-005 artifact is missing or changed."""

    resolved = manifest or load_python_baseline_manifest(output_dir / _BASELINE_MANIFEST_NAME)
    expected_artifacts = tuple(
        BaselineArtifactDigest(
            artifact_id=item.artifact_id,
            path=item.path,
            sha256=file_sha256(output_dir / item.path),
        )
        for item in resolved.artifacts
    )
    if expected_artifacts != resolved.artifacts:
        raise PythonBaselineError("frozen Python baseline artifact digest mismatch")
    if _artifact_set_sha256(expected_artifacts) != resolved.artifact_set_sha256:
        raise PythonBaselineError("frozen Python baseline artifact-set fingerprint mismatch")
    required_ids = tuple(artifact_id for artifact_id, _ in _REQUIRED_ARTIFACTS)
    actual_ids = tuple(item.artifact_id for item in resolved.artifacts)
    if actual_ids != required_ids:
        raise PythonBaselineError(
            "frozen Python baseline artifact inventory does not match the required P6-005 set"
        )
    return resolved
