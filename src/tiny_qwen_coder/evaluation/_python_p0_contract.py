"""Frozen P8-001 identities, generation contract, and transported-stage evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from tiny_qwen_coder.config import EvaluationConfig
from tiny_qwen_coder.evaluation._baseline_artifacts import (
    evaluation_config_sha256,
    file_sha256,
    system_prompt_sha256,
)
from tiny_qwen_coder.evaluation._baseline_generation import generation_contract_sha256
from tiny_qwen_coder.evaluation._baseline_provenance import load_baseline_base_model_identity
from tiny_qwen_coder.evaluation._python_p0_generation import (
    EXPECTED_ADAPTER_FAMILY,
    EXPECTED_ADAPTER_ID,
    EXPECTED_ADAPTER_SHA256,
    EXPECTED_ADAPTER_SIZE_BYTES,
    EXPECTED_TRAINING_GIT_SHA,
    EXPECTED_TRAINING_RUN_ID,
)
from tiny_qwen_coder.evaluation.settings import (
    FrozenEvaluationSettings,
    evaluation_settings_sha256,
    validate_evaluation_config_settings,
)
from tiny_qwen_coder.identities import BaseModelIdentity
from tiny_qwen_coder.languages.python import load_python_plugin

TASK_ID = "P8-001"
SCHEMA_VERSION = 1
DEFAULT_CONFIG = Path("configs/eval/python/p0_v1.yaml")
EXPECTED_SUITES = ("humaneval", "mbpp", "repository-holdout")
EXPECTED_TOTALS = {"humaneval": 164, "mbpp": 500, "repository-holdout": 11}
EXPECTED_SYSTEM_PROMPT_VERSION = "python-v1"
EXPECTED_SYSTEM_PROMPT_SHA256 = "ed10dcc67116e4b3633eed7413228bb083a797fc6917132df03890aa7e05497e"
EXPECTED_BASELINE_ID = "python-unchanged-base"
EXPECTED_BASELINE_SOURCE_SHA = "da537443ab80b1380bee0fc3c7d9d01ca0574f35"
EXPECTED_BASELINE_ARTIFACT_SET_SHA256 = (
    "4bf616c3e84bdd74f8cf6467fc2d9d760f04d3b1b44660a81e365ff6f99a72fc"
)
STAGE_MANIFEST = "generation-stage.json"
EVALUATION_MANIFEST = "evaluation-manifest.json"
COMPARISON = "comparison.json"
WORK_DIR = ".baseline-work"
STAGE_ARTIFACTS = (
    f"{WORK_DIR}/humaneval-generation.jsonl",
    f"{WORK_DIR}/mbpp-generation.jsonl",
    f"{WORK_DIR}/repository-holdout-generation.jsonl",
    "runtime-metadata.json",
)
FINAL_ARTIFACTS = (
    STAGE_MANIFEST,
    "runtime-metadata.json",
    f"{WORK_DIR}/humaneval-generation.jsonl",
    f"{WORK_DIR}/mbpp-generation.jsonl",
    f"{WORK_DIR}/repository-holdout-generation.jsonl",
    "humaneval/humaneval-results.jsonl",
    "humaneval/humaneval-aggregate.json",
    "mbpp/mbpp-results.jsonl",
    "mbpp/mbpp-aggregate.json",
    "repository-holdout/repository-holdout-results.jsonl",
    "repository-holdout/repository-holdout-aggregate.json",
    COMPARISON,
)


class PythonP0EvaluationError(RuntimeError):
    """Raised when P8-001 cannot produce trustworthy evaluation evidence."""


class AdapterEvidence(Protocol):
    adapter_id: str
    family: str
    adapter_model_sha256: str
    adapter_model_size_bytes: int
    training_run_id: str
    training_git_sha: str


@dataclass(frozen=True, slots=True)
class PersistedAdapterEvidence:
    """Exact accepted adapter identity copied into transported P8 evidence."""

    adapter_id: str
    family: str
    adapter_model_sha256: str
    adapter_model_size_bytes: int
    training_run_id: str
    training_git_sha: str


def strict_mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise PythonP0EvaluationError(f"{context} must be a JSON object")
    output: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise PythonP0EvaluationError(f"{context} keys must be strings")
        output[key] = item
    return output


def read_json(path: Path, *, context: str) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PythonP0EvaluationError(f"could not read {context}: {path}") from exc
    return strict_mapping(value, context=context)


def write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def artifact_set_sha256(items: Sequence[Mapping[str, str]]) -> str:
    payload = json.dumps(list(items), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_contract(
    evaluation: EvaluationConfig,
    settings: FrozenEvaluationSettings,
    base_model: BaseModelIdentity,
) -> tuple[str, str]:
    """Require the exact P6 coding-suite generation contract for the P0 adapter."""

    if evaluation.language != "python":
        raise PythonP0EvaluationError("P8-001 requires language='python'")
    if evaluation.adapter_id != EXPECTED_ADAPTER_ID:
        raise PythonP0EvaluationError("P8-001 must evaluate exactly language/python/p0")
    if evaluation.suites != EXPECTED_SUITES:
        raise PythonP0EvaluationError(
            f"P8-001 suite order must be exactly {EXPECTED_SUITES!r}; got {evaluation.suites!r}"
        )
    expected_base = load_baseline_base_model_identity(Path(evaluation.base_config))
    if expected_base != base_model:
        raise PythonP0EvaluationError("P8-001 base identity does not match evaluation base_config")
    settings_digest = validate_evaluation_config_settings(evaluation, settings)
    if settings_digest != evaluation_settings_sha256(settings):
        raise PythonP0EvaluationError("P8-001 frozen evaluation-settings hash is inconsistent")
    system_prompt = load_python_plugin().spec.config.system_prompt
    if system_prompt.version != EXPECTED_SYSTEM_PROMPT_VERSION:
        raise PythonP0EvaluationError("Python system-prompt version drifted from P6-005")
    if system_prompt_sha256(system_prompt.text) != EXPECTED_SYSTEM_PROMPT_SHA256:
        raise PythonP0EvaluationError("Python system-prompt content drifted from P6-005")
    return system_prompt.version, system_prompt.text


def adapter_generation_contract(
    *,
    base_model: BaseModelIdentity,
    settings: FrozenEvaluationSettings,
    system_prompt_version: str,
    system_prompt: str,
    adapter: AdapterEvidence,
) -> str:
    """Bind P6 generation identity to the exact accepted P7-006 adapter."""

    base_contract = generation_contract_sha256(
        base_model=base_model,
        settings=settings,
        system_prompt_version=system_prompt_version,
        system_prompt=system_prompt,
    )
    payload = json.dumps(
        {
            "base_generation_contract_sha256": base_contract,
            "adapter": {
                "adapter_id": adapter.adapter_id,
                "family": adapter.family,
                "adapter_model_sha256": adapter.adapter_model_sha256,
                "adapter_model_size_bytes": adapter.adapter_model_size_bytes,
                "training_run_id": adapter.training_run_id,
                "training_git_sha": adapter.training_git_sha,
            },
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stage_payload(
    *,
    output_dir: Path,
    source_git_sha: str,
    evaluation: EvaluationConfig,
    settings: FrozenEvaluationSettings,
    system_prompt: str,
    generation_contract: str,
    adapter: AdapterEvidence,
) -> dict[str, object]:
    """Build the exact transported GPU-stage manifest from persisted files."""

    artifacts: list[dict[str, str]] = []
    for relative in STAGE_ARTIFACTS:
        path = output_dir / relative
        if not path.is_file():
            raise PythonP0EvaluationError(f"GPU generation is incomplete; missing {relative!r}")
        artifacts.append({"path": relative, "sha256": file_sha256(path)})
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "source_git_sha": source_git_sha,
        "evaluation_config_sha256": evaluation_config_sha256(evaluation),
        "evaluation_settings_sha256": evaluation_settings_sha256(settings),
        "system_prompt_sha256": system_prompt_sha256(system_prompt),
        "generation_contract_sha256": generation_contract,
        "adapter": {
            "adapter_id": adapter.adapter_id,
            "family": adapter.family,
            "adapter_model_sha256": adapter.adapter_model_sha256,
            "adapter_model_size_bytes": adapter.adapter_model_size_bytes,
            "training_run_id": adapter.training_run_id,
            "training_git_sha": adapter.training_git_sha,
        },
        "artifacts": artifacts,
    }


def adapter_from_stage(stage: Mapping[str, object]) -> PersistedAdapterEvidence:
    """Extract only the one accepted P7-006 adapter identity; reject all drift."""

    raw = strict_mapping(stage.get("adapter"), context="generation_stage.adapter")
    expected = {
        "adapter_id": EXPECTED_ADAPTER_ID,
        "family": EXPECTED_ADAPTER_FAMILY,
        "adapter_model_sha256": EXPECTED_ADAPTER_SHA256,
        "adapter_model_size_bytes": EXPECTED_ADAPTER_SIZE_BYTES,
        "training_run_id": EXPECTED_TRAINING_RUN_ID,
        "training_git_sha": EXPECTED_TRAINING_GIT_SHA,
    }
    if raw != expected:
        raise PythonP0EvaluationError("P8-001 generation-stage adapter identity is invalid")
    return PersistedAdapterEvidence(
        adapter_id=EXPECTED_ADAPTER_ID,
        family=EXPECTED_ADAPTER_FAMILY,
        adapter_model_sha256=EXPECTED_ADAPTER_SHA256,
        adapter_model_size_bytes=EXPECTED_ADAPTER_SIZE_BYTES,
        training_run_id=EXPECTED_TRAINING_RUN_ID,
        training_git_sha=EXPECTED_TRAINING_GIT_SHA,
    )


def validate_stage(
    *,
    output_dir: Path,
    source_git_sha: str,
    evaluation: EvaluationConfig,
    settings: FrozenEvaluationSettings,
    system_prompt: str,
    generation_contract: str,
    adapter: AdapterEvidence,
) -> dict[str, object]:
    """Rehash and validate the complete transported generation-stage identity."""

    actual = read_json(output_dir / STAGE_MANIFEST, context="P8-001 generation stage")
    expected = stage_payload(
        output_dir=output_dir,
        source_git_sha=source_git_sha,
        evaluation=evaluation,
        settings=settings,
        system_prompt=system_prompt,
        generation_contract=generation_contract,
        adapter=adapter,
    )
    if actual != expected:
        raise PythonP0EvaluationError("P8-001 generation-stage identity or artifact hash drifted")
    return actual
