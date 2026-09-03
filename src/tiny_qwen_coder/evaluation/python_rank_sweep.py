"""P9-001 protected Python evaluation for frozen LoRA rank candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
import yaml
from torch import nn

from tiny_qwen_coder.adapters import load_adapter_manifest
from tiny_qwen_coder.config import EvaluationConfig, load_evaluation_config
from tiny_qwen_coder.evaluation._baseline_artifacts import (
    evaluation_config_sha256,
    file_sha256,
    system_prompt_sha256,
    write_runtime_metadata,
)
from tiny_qwen_coder.evaluation._baseline_generation import BaselineGenerator
from tiny_qwen_coder.evaluation._baseline_provenance import load_baseline_base_model_identity
from tiny_qwen_coder.evaluation._baseline_runner import (
    _generate_items,
    _preflight_execution_images,
    _preflight_source_tree,
    _suite_performance,
)
from tiny_qwen_coder.evaluation._baseline_types import (
    BaselineGeneratedResponse,
    BaselineRuntimeMetadata,
)
from tiny_qwen_coder.evaluation._python_p0_comparison import aggregate, load_baseline
from tiny_qwen_coder.evaluation._python_p0_contract import (
    EXPECTED_SUITES,
    EXPECTED_SYSTEM_PROMPT_SHA256,
    EXPECTED_SYSTEM_PROMPT_VERSION,
    EXPECTED_TOTALS,
    FINAL_ARTIFACTS,
    STAGE_ARTIFACTS,
    artifact_set_sha256,
    read_json,
    write_json,
)
from tiny_qwen_coder.evaluation._python_p0_generation import (
    HuggingFacePythonP0Generator,
    PythonP0GenerationError,
    VerifiedPythonP0Adapter,
    _parameter_dtypes,
    _resolved_revision,
    _Tokenizer,
    _validate_peft_status,
)
from tiny_qwen_coder.evaluation.execution import ConstrainedExecutionHarness, discover_oci_runtime
from tiny_qwen_coder.evaluation.humaneval import (
    HumanEvalCompletion,
    HumanEvalEvaluator,
    HumanEvalProblem,
)
from tiny_qwen_coder.evaluation.mbpp import MBPPCompletion, MBPPEvaluator, MBPPProblem
from tiny_qwen_coder.evaluation.repository_holdout import (
    RepositoryHoldoutCompletion,
    RepositoryHoldoutEvaluator,
)
from tiny_qwen_coder.evaluation.settings import (
    FrozenEvaluationSettings,
    evaluation_settings_sha256,
    load_frozen_evaluation_settings,
    validate_evaluation_config_settings,
)
from tiny_qwen_coder.identities import AdapterIdentity, BaseModelIdentity
from tiny_qwen_coder.languages.python import load_python_plugin
from tiny_qwen_coder.reproducibility import seed_everything
from tiny_qwen_coder.runtime.adapter_validation import (
    _FORBIDDEN_MODEL_PATTERNS,
    _REQUIRED_OUTPUT_FILES,
    VerifiedAdapterArtifacts,
    _load_json_object,
    _persisted_artifact,
    _relative_leaves,
    _require_mapping,
    _require_str,
)
from tiny_qwen_coder.runtime.adapter_validation import (
    _sha256 as _file_sha256,
)

TASK_ID = "P9-001"
SWEEP_ID = "python-p9-rank-v1"
REGISTRY_PATH = Path("configs/eval/python/p9_rank_sweep_candidates_v1.yaml")
STAGE_MANIFEST = "generation-stage.json"
COMPARISON = "comparison.json"
EVALUATION_MANIFEST = "evaluation-manifest.json"
_EXPECTED_RANKS = (8, 16, 32, 64)
_EVALUATED_RANKS = (8, 32, 64)
_EXPECTED_FAMILY = "language"
_EXPECTED_LANGUAGE = "python"
_EXPECTED_ALPHA = 32


class PythonRankSweepEvaluationError(RuntimeError):
    """Raised when P9-001 rank evaluation evidence is incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class RankCandidate:
    """Exact immutable post-training identity for one rank candidate."""

    rank: int
    adapter_id: str
    training_config: str
    training_config_sha256: str
    training_run_id: str
    training_git_sha: str
    training_artifact_id: int
    training_artifact_name: str
    training_artifact_digest: str
    adapter_model_sha256: str
    adapter_model_size_bytes: int
    artifact_set_sha256: str | None
    evaluation_config: str | None
    baseline_control: bool
    canonical_evaluation_run_id: int | None


@dataclass(frozen=True, slots=True)
class RankCandidateRegistry:
    """Frozen candidate registry bound to the completed P9-001 training run."""

    training_workflow_run_id: int
    training_source_git_sha: str
    dataset_manifest_id: str
    dataset_config_sha256: str
    dataset_split_membership_sha256: str
    contamination_status: str
    candidates: tuple[RankCandidate, ...]

    def candidate(self, rank: int) -> RankCandidate:
        matches = tuple(item for item in self.candidates if item.rank == rank)
        if len(matches) != 1:
            raise PythonRankSweepEvaluationError(f"rank {rank} is not uniquely frozen")
        return matches[0]


def _mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise PythonRankSweepEvaluationError(f"{context} must be a mapping")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise PythonRankSweepEvaluationError(f"{context} keys must be strings")
        result[key] = item
    return result


def _require_keys(mapping: Mapping[str, object], expected: set[str], *, context: str) -> None:
    actual = set(mapping)
    if actual != expected:
        raise PythonRankSweepEvaluationError(
            f"{context} fields drifted: expected {sorted(expected)!r}, got {sorted(actual)!r}"
        )


def _string(mapping: Mapping[str, object], key: str, *, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise PythonRankSweepEvaluationError(f"{context}.{key} must be a non-empty string")
    return value


def _integer(mapping: Mapping[str, object], key: str, *, context: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PythonRankSweepEvaluationError(f"{context}.{key} must be a positive integer")
    return value


def _sha256(value: str, *, context: str) -> str:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise PythonRankSweepEvaluationError(f"{context} must be a lowercase SHA-256")
    return value


def _artifact_digest(value: str, *, context: str) -> str:
    prefix = "sha256:"
    if not value.startswith(prefix):
        raise PythonRankSweepEvaluationError(f"{context} must use sha256:<digest> form")
    _sha256(value[len(prefix) :], context=context)
    return value


def _optional_string(mapping: Mapping[str, object], key: str, *, context: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise PythonRankSweepEvaluationError(f"{context}.{key} must be null or non-empty string")
    return value


def _optional_integer(mapping: Mapping[str, object], key: str, *, context: str) -> int | None:
    value = mapping.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PythonRankSweepEvaluationError(f"{context}.{key} must be null or positive integer")
    return value


def load_rank_candidate_registry(path: Path = REGISTRY_PATH) -> RankCandidateRegistry:
    """Load and fail closed on any P9-001 candidate-registry drift."""

    try:
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PythonRankSweepEvaluationError(
            f"could not load rank candidate registry: {path}"
        ) from exc
    root = _mapping(raw, context="rank candidate registry")
    _require_keys(
        root,
        {
            "schema_version",
            "sweep_id",
            "training_workflow_run_id",
            "training_source_git_sha",
            "dataset",
            "candidates",
        },
        context="rank candidate registry",
    )
    if root.get("schema_version") != 1 or root.get("sweep_id") != SWEEP_ID:
        raise PythonRankSweepEvaluationError("rank candidate registry identity drifted")
    training_run = _integer(root, "training_workflow_run_id", context="rank candidate registry")
    training_sha = _sha256(
        _string(root, "training_source_git_sha", context="rank candidate registry"),
        context="rank candidate registry.training_source_git_sha",
    )
    dataset = _mapping(root.get("dataset"), context="rank candidate registry.dataset")
    _require_keys(
        dataset,
        {"manifest_id", "config_sha256", "split_membership_sha256", "contamination_status"},
        context="rank candidate registry.dataset",
    )
    manifest_id = _string(dataset, "manifest_id", context="rank candidate registry.dataset")
    config_sha = _sha256(
        _string(dataset, "config_sha256", context="rank candidate registry.dataset"),
        context="dataset.config_sha256",
    )
    membership_sha = _sha256(
        _string(dataset, "split_membership_sha256", context="rank candidate registry.dataset"),
        context="dataset.split_membership_sha256",
    )
    contamination = _string(
        dataset, "contamination_status", context="rank candidate registry.dataset"
    )

    raw_candidates = root.get("candidates")
    if not isinstance(raw_candidates, list):
        raise PythonRankSweepEvaluationError("rank candidate registry.candidates must be a list")
    candidates: list[RankCandidate] = []
    for index, item in enumerate(raw_candidates):
        context = f"rank candidate registry.candidates[{index}]"
        candidate = _mapping(item, context=context)
        allowed = {
            "rank",
            "adapter_id",
            "baseline_control",
            "training_config",
            "training_config_sha256",
            "training_run_id",
            "training_git_sha",
            "training_artifact_id",
            "training_artifact_name",
            "training_artifact_digest",
            "adapter_model_sha256",
            "adapter_model_size_bytes",
            "artifact_set_sha256",
            "evaluation_config",
            "canonical_evaluation_run_id",
        }
        unknown = set(candidate) - allowed
        if unknown:
            raise PythonRankSweepEvaluationError(
                f"{context} has unknown fields: {sorted(unknown)!r}"
            )
        required = allowed - {
            "baseline_control",
            "evaluation_config",
            "canonical_evaluation_run_id",
        }
        missing = required - set(candidate)
        if missing:
            raise PythonRankSweepEvaluationError(
                f"{context} is missing fields: {sorted(missing)!r}"
            )
        rank = _integer(candidate, "rank", context=context)
        baseline_control = candidate.get("baseline_control", False)
        if not isinstance(baseline_control, bool):
            raise PythonRankSweepEvaluationError(f"{context}.baseline_control must be boolean")
        artifact_set_value = candidate.get("artifact_set_sha256")
        artifact_set: str | None
        if artifact_set_value is None:
            artifact_set = None
        elif isinstance(artifact_set_value, str):
            artifact_set = _sha256(artifact_set_value, context=f"{context}.artifact_set_sha256")
        else:
            raise PythonRankSweepEvaluationError(f"{context}.artifact_set_sha256 is invalid")
        candidates.append(
            RankCandidate(
                rank=rank,
                adapter_id=_string(candidate, "adapter_id", context=context),
                training_config=_string(candidate, "training_config", context=context),
                training_config_sha256=_sha256(
                    _string(candidate, "training_config_sha256", context=context),
                    context=f"{context}.training_config_sha256",
                ),
                training_run_id=_string(candidate, "training_run_id", context=context),
                training_git_sha=_sha256(
                    _string(candidate, "training_git_sha", context=context),
                    context=f"{context}.training_git_sha",
                ),
                training_artifact_id=_integer(candidate, "training_artifact_id", context=context),
                training_artifact_name=_string(
                    candidate, "training_artifact_name", context=context
                ),
                training_artifact_digest=_artifact_digest(
                    _string(candidate, "training_artifact_digest", context=context),
                    context=f"{context}.training_artifact_digest",
                ),
                adapter_model_sha256=_sha256(
                    _string(candidate, "adapter_model_sha256", context=context),
                    context=f"{context}.adapter_model_sha256",
                ),
                adapter_model_size_bytes=_integer(
                    candidate, "adapter_model_size_bytes", context=context
                ),
                artifact_set_sha256=artifact_set,
                evaluation_config=_optional_string(candidate, "evaluation_config", context=context),
                baseline_control=baseline_control,
                canonical_evaluation_run_id=_optional_integer(
                    candidate, "canonical_evaluation_run_id", context=context
                ),
            )
        )
    if tuple(item.rank for item in candidates) != _EXPECTED_RANKS:
        raise PythonRankSweepEvaluationError(
            f"rank registry must preserve exact order {_EXPECTED_RANKS!r}"
        )
    for item in candidates:
        if item.rank == 16:
            if not item.baseline_control or item.canonical_evaluation_run_id is None:
                raise PythonRankSweepEvaluationError(
                    "rank 16 must be the canonical evaluated P0 control"
                )
            if item.evaluation_config is not None:
                raise PythonRankSweepEvaluationError(
                    "rank 16 must reuse its canonical P8 evaluation"
                )
        else:
            if item.baseline_control or item.evaluation_config is None:
                raise PythonRankSweepEvaluationError(
                    f"rank {item.rank} evaluation identity is incomplete"
                )
            if item.training_git_sha != training_sha:
                raise PythonRankSweepEvaluationError(f"rank {item.rank} training SHA drifted")
    return RankCandidateRegistry(
        training_workflow_run_id=training_run,
        training_source_git_sha=training_sha,
        dataset_manifest_id=manifest_id,
        dataset_config_sha256=config_sha,
        dataset_split_membership_sha256=membership_sha,
        contamination_status=contamination,
        candidates=tuple(candidates),
    )


def _require_equal_local(actual: object, expected: object, *, field: str) -> None:
    if actual != expected:
        raise PythonRankSweepEvaluationError(
            f"{field} mismatch: observed {actual!r}; expected {expected!r}"
        )


def _validated_candidate_bundle(
    training_output: Path,
    base_model: BaseModelIdentity,
    candidate: RankCandidate,
) -> VerifiedAdapterArtifacts:
    """Run P7-007-equivalent artifact checks with a frozen P9 candidate identity."""

    root = training_output.resolve()
    if not root.is_dir():
        raise PythonRankSweepEvaluationError(f"training output directory does not exist: {root}")
    for relative in _REQUIRED_OUTPUT_FILES:
        path = root / relative
        if not path.is_file():
            raise PythonRankSweepEvaluationError(
                f"required training artifact is missing: {relative}"
            )
    for pattern in _FORBIDDEN_MODEL_PATTERNS:
        matches = tuple((root / "adapter").glob(pattern))
        if matches:
            raise PythonRankSweepEvaluationError(
                "adapter directory contains forbidden merged/full-model weights: "
                + ", ".join(sorted(path.name for path in matches))
            )

    manifest = load_adapter_manifest(root / "adapter-manifest.json")
    _require_equal_local(
        manifest.adapter_id, candidate.adapter_id, field="adapter_manifest.adapter_id"
    )
    _require_equal_local(manifest.family, _EXPECTED_FAMILY, field="adapter_manifest.family")
    _require_equal_local(manifest.language, _EXPECTED_LANGUAGE, field="adapter_manifest.language")
    _require_equal_local(
        manifest.base_model.repository,
        base_model.repository,
        field="adapter_manifest.base_model.repository",
    )
    _require_equal_local(
        manifest.base_model.revision,
        base_model.revision,
        field="adapter_manifest.base_model.revision",
    )
    _require_equal_local(
        manifest.tokenizer.repository,
        base_model.tokenizer_repository,
        field="adapter_manifest.tokenizer.repository",
    )
    _require_equal_local(
        manifest.tokenizer.revision,
        base_model.tokenizer_revision,
        field="adapter_manifest.tokenizer.revision",
    )
    _require_equal_local(manifest.lora.rank, candidate.rank, field="adapter_manifest.lora.rank")
    _require_equal_local(manifest.lora.alpha, _EXPECTED_ALPHA, field="adapter_manifest.lora.alpha")

    try:
        training_report = _load_json_object(root / "training-report.json", label="training report")
        training_config = _load_json_object(root / "training-config.json", label="training config")
        run_manifest = _load_json_object(root / "run-manifest.json", label="run manifest")
        adapter_config = _load_json_object(
            root / "adapter" / "adapter_config.json", label="PEFT config"
        )
    except RuntimeError as exc:
        raise PythonRankSweepEvaluationError(str(exc)) from exc

    _require_equal_local(
        training_report.get("adapter_id"), manifest.adapter_id, field="training_report.adapter_id"
    )
    _require_equal_local(
        training_report.get("language"), manifest.language, field="training_report.language"
    )
    _require_equal_local(
        training_report.get("global_steps"), 4750, field="training_report.global_steps"
    )
    _require_equal_local(
        training_report.get("source_training_config"),
        candidate.training_config,
        field="training_report.source_training_config",
    )
    _require_equal_local(
        training_report.get("source_training_config_sha256"),
        candidate.training_config_sha256,
        field="training_report.source_training_config_sha256",
    )
    _require_equal_local(
        manifest.training.config_sha256,
        candidate.training_config_sha256,
        field="adapter_manifest.training.config_sha256",
    )
    _require_equal_local(
        training_config.get("config_sha256"),
        candidate.training_config_sha256,
        field="training_config.config_sha256",
    )

    resolved_base = _require_mapping(training_config.get("base"), field="training_config.base")
    for key, expected in (
        ("model_repository", base_model.repository),
        ("model_revision", base_model.revision),
        ("tokenizer_repository", base_model.tokenizer_repository),
        ("tokenizer_revision", base_model.tokenizer_revision),
    ):
        _require_equal_local(resolved_base.get(key), expected, field=f"training_config.base.{key}")

    dataset = _require_mapping(training_config.get("dataset"), field="training_config.dataset")
    inference_template_sha = _require_str(
        dataset, "chat_template_sha256", field="training_config.dataset"
    )
    _require_equal_local(
        _file_sha256(root / "adapter" / "chat_template.jinja"),
        inference_template_sha,
        field="adapter.chat_template.jinja.sha256",
    )

    run_base = _require_mapping(run_manifest.get("base_model"), field="run_manifest.base_model")
    for key, expected in (
        ("repository", base_model.repository),
        ("revision", base_model.revision),
        ("tokenizer_repository", base_model.tokenizer_repository),
        ("tokenizer_revision", base_model.tokenizer_revision),
    ):
        _require_equal_local(run_base.get(key), expected, field=f"run_manifest.base_model.{key}")
    run_adapter = _require_mapping(run_manifest.get("adapter"), field="run_manifest.adapter")
    _require_equal_local(
        run_adapter.get("adapter_id"), manifest.adapter_id, field="run_manifest.adapter.adapter_id"
    )
    _require_equal_local(
        run_adapter.get("family"), manifest.family, field="run_manifest.adapter.family"
    )
    _require_equal_local(
        run_manifest.get("language"), manifest.language, field="run_manifest.language"
    )
    run_git = _require_mapping(run_manifest.get("git"), field="run_manifest.git")
    training_git_sha = _require_str(run_git, "sha", field="run_manifest.git")
    training_run_id = _require_str(run_manifest, "run_id", field="run_manifest")
    _require_equal_local(training_git_sha, candidate.training_git_sha, field="run_manifest.git.sha")
    _require_equal_local(training_run_id, candidate.training_run_id, field="run_manifest.run_id")

    _require_equal_local(
        str(adapter_config.get("peft_type", "")).upper(), "LORA", field="adapter_config.peft_type"
    )
    _require_equal_local(
        adapter_config.get("task_type"), "CAUSAL_LM", field="adapter_config.task_type"
    )
    _require_equal_local(
        adapter_config.get("inference_mode"), True, field="adapter_config.inference_mode"
    )
    _require_equal_local(
        adapter_config.get("base_model_name_or_path"),
        base_model.repository,
        field="adapter_config.base_model_name_or_path",
    )
    _require_equal_local(adapter_config.get("r"), candidate.rank, field="adapter_config.r")
    _require_equal_local(
        adapter_config.get("lora_alpha"), _EXPECTED_ALPHA, field="adapter_config.lora_alpha"
    )
    _require_equal_local(
        adapter_config.get("lora_dropout"),
        manifest.lora.dropout,
        field="adapter_config.lora_dropout",
    )
    _require_equal_local(
        adapter_config.get("bias"), manifest.lora.bias, field="adapter_config.bias"
    )
    _require_equal_local(
        adapter_config.get("peft_version"),
        manifest.training.peft_version,
        field="adapter_config.peft_version",
    )
    config_targets = adapter_config.get("target_modules")
    if not isinstance(config_targets, list) or any(
        not isinstance(item, str) for item in config_targets
    ):
        raise PythonRankSweepEvaluationError("adapter_config.target_modules must be a string list")
    _require_equal_local(
        tuple(sorted(cast(list[str], config_targets))),
        _relative_leaves(manifest.lora.target_modules),
        field="adapter_config.target_modules",
    )

    adapter_model = root / "adapter" / "adapter_model.safetensors"
    adapter_size = adapter_model.stat().st_size
    if adapter_size <= 0:
        raise PythonRankSweepEvaluationError("adapter_model.safetensors is empty")
    adapter_sha = _file_sha256(adapter_model)
    _require_equal_local(
        adapter_size, candidate.adapter_model_size_bytes, field="adapter weights size"
    )
    _require_equal_local(
        adapter_sha, candidate.adapter_model_sha256, field="adapter weights sha256"
    )
    persisted_weights = _persisted_artifact(training_report, "adapter/adapter_model.safetensors")
    _require_equal_local(
        persisted_weights.get("size_bytes"), adapter_size, field="persisted adapter weight size"
    )
    _require_equal_local(
        persisted_weights.get("sha256"), adapter_sha, field="persisted adapter weight sha256"
    )
    persisted_config = _persisted_artifact(training_report, "adapter/adapter_config.json")
    config_path = root / "adapter" / "adapter_config.json"
    _require_equal_local(
        persisted_config.get("size_bytes"),
        config_path.stat().st_size,
        field="persisted adapter config size",
    )
    _require_equal_local(
        persisted_config.get("sha256"),
        _file_sha256(config_path),
        field="persisted adapter config sha256",
    )
    if candidate.artifact_set_sha256 is not None:
        _require_equal_local(
            training_report.get("artifact_set_sha256"),
            candidate.artifact_set_sha256,
            field="training_report.artifact_set_sha256",
        )

    return VerifiedAdapterArtifacts(
        output_dir=root,
        adapter_dir=root / "adapter",
        manifest=manifest,
        adapter_model_sha256=adapter_sha,
        adapter_model_size_bytes=adapter_size,
        training_run_id=training_run_id,
        training_git_sha=training_git_sha,
        inference_chat_template_sha256=inference_template_sha,
    )


def _validate_candidate_artifacts(
    training_output: Path,
    base_model: BaseModelIdentity,
    candidate: RankCandidate,
) -> VerifiedPythonP0Adapter:
    artifacts = _validated_candidate_bundle(training_output, base_model, candidate)
    return VerifiedPythonP0Adapter(
        adapter_dir=artifacts.adapter_dir,
        adapter_id=artifacts.manifest.adapter_id,
        family=artifacts.manifest.family,
        adapter_model_sha256=artifacts.adapter_model_sha256,
        adapter_model_size_bytes=artifacts.adapter_model_size_bytes,
        training_run_id=artifacts.training_run_id,
        training_git_sha=artifacts.training_git_sha,
        inference_chat_template_sha256=artifacts.inference_chat_template_sha256,
    )


class RankCandidateGenerator(HuggingFacePythonP0Generator):
    """P8-equivalent deterministic CUDA generator for one pinned P9 rank adapter."""

    def __init__(
        self,
        *,
        training_output: Path,
        base_model: BaseModelIdentity,
        settings: FrozenEvaluationSettings,
        candidate: RankCandidate,
        device_index: int = 0,
    ) -> None:
        if not torch.cuda.is_available():
            raise PythonRankSweepEvaluationError("P9-001 generation requires CUDA")
        if not torch.cuda.is_bf16_supported():
            raise PythonRankSweepEvaluationError("P9-001 requires a BF16-capable CUDA device")
        if not 0 <= device_index < torch.cuda.device_count():
            raise PythonRankSweepEvaluationError(f"invalid CUDA device index: {device_index}")
        if settings.generation.decoding_strategy != "greedy":
            raise PythonRankSweepEvaluationError("P9-001 requires frozen greedy generation")

        self.adapter = _validate_candidate_artifacts(training_output, base_model, candidate)
        from peft import PeftModel
        from transformers import AutoModelForMultimodalLM, AutoTokenizer, PreTrainedTokenizerBase

        self._device = torch.device("cuda", device_index)
        self._settings = settings
        torch.cuda.set_device(self._device)
        torch.cuda.empty_cache()
        torch.cuda.synchronize(self._device)
        seed_everything(settings.seed)
        free_before, total_bytes = torch.cuda.mem_get_info(self._device)
        torch.cuda.reset_peak_memory_stats(self._device)

        tokenizer_obj: object = AutoTokenizer.from_pretrained(
            base_model.tokenizer_repository,
            revision=base_model.tokenizer_revision,
        )
        if not isinstance(tokenizer_obj, PreTrainedTokenizerBase):
            raise PythonRankSweepEvaluationError(
                "Transformers returned unexpected tokenizer object"
            )
        tokenizer = cast(_Tokenizer, tokenizer_obj)
        if not isinstance(tokenizer.chat_template, str) or not tokenizer.chat_template:
            raise PythonRankSweepEvaluationError(
                "canonical tokenizer does not expose chat template"
            )
        template_sha = hashlib.sha256(tokenizer.chat_template.encode("utf-8")).hexdigest()
        if template_sha != self.adapter.inference_chat_template_sha256:
            raise PythonRankSweepEvaluationError("inference chat template differs from training")
        self._tokenizer = tokenizer

        base_started = time.perf_counter()
        loaded: object = cast(Any, AutoModelForMultimodalLM).from_pretrained(
            base_model.repository,
            revision=base_model.revision,
            dtype=torch.bfloat16,
            device_map={"": device_index},
            low_cpu_mem_usage=True,
        )
        if not isinstance(loaded, nn.Module):
            raise PythonRankSweepEvaluationError("Transformers returned unexpected model object")
        base = loaded
        base.eval()
        if _resolved_revision(base) != base_model.revision:
            raise PythonRankSweepEvaluationError("loaded base revision is not canonical")
        base_dtypes = _parameter_dtypes(base)
        if base_dtypes != ("torch.bfloat16",):
            raise PythonRankSweepEvaluationError(
                f"canonical base evaluation requires BF16; observed {base_dtypes!r}"
            )
        self._base_load_seconds = time.perf_counter() - base_started

        adapter_started = time.perf_counter()
        adapted_obj: object = cast(Any, PeftModel).from_pretrained(
            base,
            str(self.adapter.adapter_dir),
            adapter_name="default",
            is_trainable=False,
        )
        if not isinstance(adapted_obj, nn.Module):
            raise PythonRankSweepEvaluationError("PEFT returned unexpected model object")
        self._model = adapted_obj
        self._model.eval()
        self._model.requires_grad_(False)
        try:
            _validate_peft_status(self._model)
        except PythonP0GenerationError as exc:
            raise PythonRankSweepEvaluationError(str(exc)) from exc
        self._adapter_load_seconds = time.perf_counter() - adapter_started
        torch.cuda.synchronize(self._device)

        free_after, total_after = torch.cuda.mem_get_info(self._device)
        if total_after != total_bytes:
            raise PythonRankSweepEvaluationError("CUDA total memory changed during load")
        self._cuda_total_bytes = total_bytes
        self._cuda_free_before_load_bytes = free_before
        self._cuda_free_after_load_bytes = free_after
        self._torch_allocated_after_load_bytes = torch.cuda.memory_allocated(self._device)
        self._torch_reserved_after_load_bytes = torch.cuda.memory_reserved(self._device)
        self._load_peak_allocated_bytes = torch.cuda.max_memory_allocated(self._device)
        self._load_peak_reserved_bytes = torch.cuda.max_memory_reserved(self._device)
        self._generation_peak_allocated_bytes = self._torch_allocated_after_load_bytes
        self._generation_peak_reserved_bytes = self._torch_reserved_after_load_bytes
        self._parameter_dtypes = _parameter_dtypes(self._model)
        self._started_wall = time.perf_counter()
        self._resolved_model_revision = base_model.revision


class _CheckpointOnlyGenerator(BaselineGenerator):
    def generate(self, *, system_prompt: str, user_prompt: str) -> BaselineGeneratedResponse:
        del system_prompt, user_prompt
        raise PythonRankSweepEvaluationError(
            "scoring is missing a transported GPU response; refusing silent regeneration"
        )


def _evaluation_context(
    candidate: RankCandidate,
) -> tuple[EvaluationConfig, FrozenEvaluationSettings, BaseModelIdentity, str, str]:
    if candidate.evaluation_config is None:
        raise PythonRankSweepEvaluationError(f"rank {candidate.rank} has no P9 evaluation config")
    evaluation = load_evaluation_config(Path(candidate.evaluation_config))
    settings = load_frozen_evaluation_settings()
    base_model = load_baseline_base_model_identity(Path(evaluation.base_config))
    if evaluation.language != _EXPECTED_LANGUAGE or evaluation.adapter_id != candidate.adapter_id:
        raise PythonRankSweepEvaluationError("rank evaluation config identity drifted")
    if evaluation.suites != EXPECTED_SUITES:
        raise PythonRankSweepEvaluationError("rank evaluation suite set/order drifted from P8")
    expected_base = load_baseline_base_model_identity(Path(evaluation.base_config))
    if expected_base != base_model:
        raise PythonRankSweepEvaluationError("rank evaluation base identity is inconsistent")
    settings_digest = validate_evaluation_config_settings(evaluation, settings)
    if settings_digest != evaluation_settings_sha256(settings):
        raise PythonRankSweepEvaluationError("frozen evaluation-settings hash is inconsistent")
    system_prompt = load_python_plugin().spec.config.system_prompt
    if system_prompt.version != EXPECTED_SYSTEM_PROMPT_VERSION:
        raise PythonRankSweepEvaluationError("Python system-prompt version drifted from P8")
    if system_prompt_sha256(system_prompt.text) != EXPECTED_SYSTEM_PROMPT_SHA256:
        raise PythonRankSweepEvaluationError("Python system-prompt content drifted from P8")
    return evaluation, settings, base_model, system_prompt.version, system_prompt.text


def _adapter_contract(
    *,
    base_model: BaseModelIdentity,
    settings: FrozenEvaluationSettings,
    system_prompt_version: str,
    system_prompt: str,
    adapter: VerifiedPythonP0Adapter,
) -> str:
    from tiny_qwen_coder.evaluation._python_p0_contract import adapter_generation_contract

    return adapter_generation_contract(
        base_model=base_model,
        settings=settings,
        system_prompt_version=system_prompt_version,
        system_prompt=system_prompt,
        adapter=adapter,
    )


def _adapter_payload(adapter: VerifiedPythonP0Adapter) -> dict[str, object]:
    return {
        "adapter_id": adapter.adapter_id,
        "family": adapter.family,
        "adapter_model_sha256": adapter.adapter_model_sha256,
        "adapter_model_size_bytes": adapter.adapter_model_size_bytes,
        "training_run_id": adapter.training_run_id,
        "training_git_sha": adapter.training_git_sha,
    }


def _expected_candidate_payload(candidate: RankCandidate) -> dict[str, object]:
    return {
        "adapter_id": candidate.adapter_id,
        "family": _EXPECTED_FAMILY,
        "adapter_model_sha256": candidate.adapter_model_sha256,
        "adapter_model_size_bytes": candidate.adapter_model_size_bytes,
        "training_run_id": candidate.training_run_id,
        "training_git_sha": candidate.training_git_sha,
    }


def _stage_payload(
    *,
    output_dir: Path,
    source_git_sha: str,
    candidate: RankCandidate,
    evaluation: EvaluationConfig,
    settings: FrozenEvaluationSettings,
    system_prompt: str,
    generation_contract: str,
    adapter: VerifiedPythonP0Adapter,
) -> dict[str, object]:
    artifacts: list[dict[str, str]] = []
    for relative in STAGE_ARTIFACTS:
        path = output_dir / relative
        if not path.is_file():
            raise PythonRankSweepEvaluationError(f"GPU generation missing {relative!r}")
        artifacts.append({"path": relative, "sha256": file_sha256(path)})
    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "sweep_id": SWEEP_ID,
        "rank": candidate.rank,
        "source_git_sha": source_git_sha,
        "evaluation_config_sha256": evaluation_config_sha256(evaluation),
        "evaluation_settings_sha256": evaluation_settings_sha256(settings),
        "system_prompt_sha256": system_prompt_sha256(system_prompt),
        "generation_contract_sha256": generation_contract,
        "adapter": _adapter_payload(adapter),
        "artifacts": artifacts,
    }


def _adapter_from_stage(
    stage: Mapping[str, object], candidate: RankCandidate
) -> VerifiedPythonP0Adapter:
    raw = _mapping(stage.get("adapter"), context="rank generation stage.adapter")
    expected = _expected_candidate_payload(candidate)
    if raw != expected:
        raise PythonRankSweepEvaluationError("transported rank adapter identity drifted")
    return VerifiedPythonP0Adapter(
        adapter_dir=Path("."),
        adapter_id=candidate.adapter_id,
        family=_EXPECTED_FAMILY,
        adapter_model_sha256=candidate.adapter_model_sha256,
        adapter_model_size_bytes=candidate.adapter_model_size_bytes,
        training_run_id=candidate.training_run_id,
        training_git_sha=candidate.training_git_sha,
        inference_chat_template_sha256="",
    )


def _validate_stage(
    *,
    output_dir: Path,
    source_git_sha: str,
    candidate: RankCandidate,
    evaluation: EvaluationConfig,
    settings: FrozenEvaluationSettings,
    system_prompt: str,
    generation_contract: str,
    adapter: VerifiedPythonP0Adapter,
) -> dict[str, object]:
    actual = read_json(output_dir / STAGE_MANIFEST, context="P9-001 generation stage")
    expected = _stage_payload(
        output_dir=output_dir,
        source_git_sha=source_git_sha,
        candidate=candidate,
        evaluation=evaluation,
        settings=settings,
        system_prompt=system_prompt,
        generation_contract=generation_contract,
        adapter=adapter,
    )
    if actual != expected:
        raise PythonRankSweepEvaluationError("rank generation-stage identity or hash drifted")
    return actual


def _load_inputs(
    *,
    evaluation: EvaluationConfig,
    settings: FrozenEvaluationSettings,
    base_model: BaseModelIdentity,
    candidate: RankCandidate,
    harness: ConstrainedExecutionHarness | None = None,
) -> tuple[
    HumanEvalEvaluator,
    tuple[HumanEvalProblem, ...],
    MBPPEvaluator,
    tuple[MBPPProblem, ...],
    RepositoryHoldoutEvaluator,
]:
    adapter = AdapterIdentity(family=_EXPECTED_FAMILY, adapter_id=candidate.adapter_id)
    humaneval = HumanEvalEvaluator(
        evaluation, base_model=base_model, adapter=adapter, settings=settings, harness=harness
    )
    mbpp = MBPPEvaluator(
        evaluation, base_model=base_model, adapter=adapter, settings=settings, harness=harness
    )
    holdout = RepositoryHoldoutEvaluator(
        evaluation, base_model=base_model, adapter=adapter, settings=settings, harness=harness
    )
    return humaneval, humaneval.load_problems(), mbpp, mbpp.load_problems(), holdout


def _generate_suites(
    *,
    evaluation: EvaluationConfig,
    settings: FrozenEvaluationSettings,
    base_model: BaseModelIdentity,
    candidate: RankCandidate,
    generator: BaselineGenerator,
    system_prompt: str,
    generation_contract: str,
    output_dir: Path,
    harness: ConstrainedExecutionHarness | None = None,
) -> tuple[
    tuple[HumanEvalProblem, ...],
    tuple[BaselineGeneratedResponse, ...],
    tuple[MBPPProblem, ...],
    tuple[BaselineGeneratedResponse, ...],
    RepositoryHoldoutEvaluator,
    tuple[BaselineGeneratedResponse, ...],
]:
    humaneval, humaneval_problems, mbpp, mbpp_problems, holdout = _load_inputs(
        evaluation=evaluation,
        settings=settings,
        base_model=base_model,
        candidate=candidate,
        harness=harness,
    )
    he = _generate_items(
        suite_id="humaneval",
        prompts=tuple(
            (problem.task_id, humaneval.prompt_for(problem).user_content)
            for problem in humaneval_problems
        ),
        generator=generator,
        system_prompt=system_prompt,
        generation_contract=generation_contract,
        output_dir=output_dir,
    )
    mb = _generate_items(
        suite_id="mbpp",
        prompts=tuple(
            (problem.task_id, mbpp.prompt_for(problem).user_content) for problem in mbpp_problems
        ),
        generator=generator,
        system_prompt=system_prompt,
        generation_contract=generation_contract,
        output_dir=output_dir,
    )
    rh = _generate_items(
        suite_id="repository-holdout",
        prompts=tuple(
            (task.problem_id, holdout.prompt_for(task).user_content) for task in holdout.suite.tasks
        ),
        generator=generator,
        system_prompt=system_prompt,
        generation_contract=generation_contract,
        output_dir=output_dir,
    )
    if (len(he), len(mb), len(rh)) != (164, 500, 11):
        raise PythonRankSweepEvaluationError("protected benchmark cardinality drift detected")
    return humaneval_problems, he, mbpp_problems, mb, holdout, rh


def _comparison(
    *,
    source_git_sha: str,
    candidate: RankCandidate,
    base_model: BaseModelIdentity,
    adapter: VerifiedPythonP0Adapter,
    settings: FrozenEvaluationSettings,
    generation_contract: str,
    baseline_manifest: object,
    baseline: Mapping[str, Mapping[str, object]],
    adapted: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    suites: list[dict[str, object]] = []
    base_passed_total = 0
    adapted_passed_total = 0
    for suite_id in EXPECTED_SUITES:
        base_item = baseline[suite_id]
        adapted_item = adapted[suite_id]
        base_passed = cast(int, base_item["passed"])
        adapted_passed = cast(int, adapted_item["passed"])
        base_rate = float(cast(int | float, base_item["pass_at_1"]))
        adapted_rate = float(cast(int | float, adapted_item["pass_at_1"]))
        base_passed_total += base_passed
        adapted_passed_total += adapted_passed
        suites.append(
            {
                "suite_id": suite_id,
                "total_problems": EXPECTED_TOTALS[suite_id],
                "base_passed": base_passed,
                "base_pass_at_1": base_rate,
                "adapter_passed": adapted_passed,
                "adapter_pass_at_1": adapted_rate,
                "delta_passed": adapted_passed - base_passed,
                "delta_pass_at_1": adapted_rate - base_rate,
            }
        )
    baseline_id = getattr(baseline_manifest, "baseline_id", None)
    baseline_source = getattr(baseline_manifest, "source_git_sha", None)
    baseline_artifact_set = getattr(baseline_manifest, "artifact_set_sha256", None)
    total = sum(EXPECTED_TOTALS.values())
    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "sweep_id": SWEEP_ID,
        "rank": candidate.rank,
        "evaluation_complete": True,
        "source_git_sha": source_git_sha,
        "base_model": asdict(base_model),
        "adapter": _adapter_payload(adapter),
        "baseline": {
            "baseline_id": baseline_id,
            "source_git_sha": baseline_source,
            "artifact_set_sha256": baseline_artifact_set,
        },
        "evaluation_settings_sha256": evaluation_settings_sha256(settings),
        "generation_contract_sha256": generation_contract,
        "suites": suites,
        "overall": {
            "total_problems": total,
            "base_passed": base_passed_total,
            "base_micro_pass_rate": base_passed_total / total,
            "adapter_passed": adapted_passed_total,
            "adapter_micro_pass_rate": adapted_passed_total / total,
            "delta_passed": adapted_passed_total - base_passed_total,
            "delta_micro_pass_rate": (adapted_passed_total - base_passed_total) / total,
        },
    }


def _evaluation_manifest_payload(
    output_dir: Path, source_git_sha: str, rank: int
) -> dict[str, object]:
    artifacts: list[dict[str, str]] = []
    for relative in FINAL_ARTIFACTS:
        path = output_dir / relative
        if not path.is_file():
            raise PythonRankSweepEvaluationError(f"final rank evidence missing {relative!r}")
        artifacts.append({"path": relative, "sha256": file_sha256(path)})
    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "sweep_id": SWEEP_ID,
        "rank": rank,
        "source_git_sha": source_git_sha,
        "artifacts": artifacts,
        "artifact_set_sha256": artifact_set_sha256(artifacts),
    }


def _write_evaluation_manifest(output_dir: Path, source_git_sha: str, rank: int) -> Path:
    path = output_dir / EVALUATION_MANIFEST
    write_json(path, _evaluation_manifest_payload(output_dir, source_git_sha, rank))
    return path


def generate_rank_stage(
    *,
    rank: int,
    training_output: Path,
    device_index: int = 0,
    repo_root: Path = Path("."),
) -> Path:
    registry = load_rank_candidate_registry()
    candidate = registry.candidate(rank)
    if rank not in _EVALUATED_RANKS:
        raise PythonRankSweepEvaluationError("rank 16 must reuse canonical P8 evaluation")
    evaluation, settings, base_model, system_version, system_prompt = _evaluation_context(candidate)
    source_git_sha, _ = _preflight_source_tree(repo_root)
    adapter = _validate_candidate_artifacts(training_output, base_model, candidate)
    generation_contract = _adapter_contract(
        base_model=base_model,
        settings=settings,
        system_prompt_version=system_version,
        system_prompt=system_prompt,
        adapter=adapter,
    )
    output_dir = Path(evaluation.output_dir)
    if (output_dir / EVALUATION_MANIFEST).exists() or (output_dir / COMPARISON).exists():
        raise PythonRankSweepEvaluationError(
            "generation refuses an already-scored output directory"
        )
    generator = RankCandidateGenerator(
        training_output=training_output,
        base_model=base_model,
        settings=settings,
        candidate=candidate,
        device_index=device_index,
    )
    _he, he, _mbp, mb, _holdout, rh = _generate_suites(
        evaluation=evaluation,
        settings=settings,
        base_model=base_model,
        candidate=candidate,
        generator=generator,
        system_prompt=system_prompt,
        generation_contract=generation_contract,
        output_dir=output_dir,
    )
    suites = (
        _suite_performance("humaneval", he),
        _suite_performance("mbpp", mb),
        _suite_performance("repository-holdout", rh),
    )
    runtime: BaselineRuntimeMetadata = generator.runtime_metadata(suites)
    if runtime.total_requests != 675:
        raise PythonRankSweepEvaluationError("runtime evidence does not cover all 675 requests")
    write_runtime_metadata(runtime, output_dir)
    write_json(
        output_dir / STAGE_MANIFEST,
        _stage_payload(
            output_dir=output_dir,
            source_git_sha=source_git_sha,
            candidate=candidate,
            evaluation=evaluation,
            settings=settings,
            system_prompt=system_prompt,
            generation_contract=generation_contract,
            adapter=adapter,
        ),
    )
    return output_dir / STAGE_MANIFEST


def score_rank_stage(
    *,
    rank: int,
    baseline_dir: Path,
    repo_root: Path = Path("."),
) -> Path:
    registry = load_rank_candidate_registry()
    candidate = registry.candidate(rank)
    evaluation, settings, base_model, system_version, system_prompt = _evaluation_context(candidate)
    source_git_sha, _ = _preflight_source_tree(repo_root)
    output_dir = Path(evaluation.output_dir)
    stage = read_json(output_dir / STAGE_MANIFEST, context="P9-001 generation stage")
    adapter = _adapter_from_stage(stage, candidate)
    generation_contract = _adapter_contract(
        base_model=base_model,
        settings=settings,
        system_prompt_version=system_version,
        system_prompt=system_prompt,
        adapter=adapter,
    )
    _validate_stage(
        output_dir=output_dir,
        source_git_sha=source_git_sha,
        candidate=candidate,
        evaluation=evaluation,
        settings=settings,
        system_prompt=system_prompt,
        generation_contract=generation_contract,
        adapter=adapter,
    )
    baseline_manifest, baseline = load_baseline(baseline_dir, base_model)

    runtime = discover_oci_runtime()
    harness = ConstrainedExecutionHarness(runtime=runtime)
    humaneval, he_problems, mbpp, mbpp_problems, holdout = _load_inputs(
        evaluation=evaluation,
        settings=settings,
        base_model=base_model,
        candidate=candidate,
        harness=harness,
    )
    _preflight_execution_images(
        runtime,
        (
            humaneval.runner.execution_image,
            mbpp.runner.execution_image,
            holdout.suite.execution_image,
        ),
    )
    resolved_he, he, resolved_mbpp, mb, resolved_holdout, rh = _generate_suites(
        evaluation=evaluation,
        settings=settings,
        base_model=base_model,
        candidate=candidate,
        generator=_CheckpointOnlyGenerator(),
        system_prompt=system_prompt,
        generation_contract=generation_contract,
        output_dir=output_dir,
        harness=harness,
    )
    if (
        resolved_he != he_problems
        or resolved_mbpp != mbpp_problems
        or resolved_holdout.suite != holdout.suite
    ):
        raise PythonRankSweepEvaluationError("protected benchmark inputs changed during scoring")

    he_result = humaneval.evaluate_suite(
        resolved_he,
        tuple(
            HumanEvalCompletion(
                task_id=problem.task_id,
                generated_text=response.generated_text,
                generation=response.generation,
            )
            for problem, response in zip(resolved_he, he, strict=True)
        ),
    )
    humaneval.write_artifacts(he_result, output_dir / "humaneval")
    mbpp_result = mbpp.evaluate_suite(
        resolved_mbpp,
        tuple(
            MBPPCompletion(
                task_id=problem.task_id,
                generated_text=response.generated_text,
                generation=response.generation,
            )
            for problem, response in zip(resolved_mbpp, mb, strict=True)
        ),
    )
    mbpp.write_artifacts(mbpp_result, output_dir / "mbpp")
    holdout_result = holdout.evaluate_suite(
        tuple(
            RepositoryHoldoutCompletion(
                problem_id=task.problem_id,
                generated_text=response.generated_text,
                generation=response.generation,
            )
            for task, response in zip(resolved_holdout.suite.tasks, rh, strict=True)
        )
    )
    holdout.write_artifacts(holdout_result, output_dir / "repository-holdout")

    adapted = {
        "humaneval": aggregate(
            output_dir / "humaneval/humaneval-aggregate.json",
            suite_id="humaneval",
            base_model=base_model,
            adapter_id=candidate.adapter_id,
        ),
        "mbpp": aggregate(
            output_dir / "mbpp/mbpp-aggregate.json",
            suite_id="mbpp",
            base_model=base_model,
            adapter_id=candidate.adapter_id,
        ),
        "repository-holdout": aggregate(
            output_dir / "repository-holdout/repository-holdout-aggregate.json",
            suite_id="repository-holdout",
            base_model=base_model,
            adapter_id=candidate.adapter_id,
        ),
    }
    write_json(
        output_dir / COMPARISON,
        _comparison(
            source_git_sha=source_git_sha,
            candidate=candidate,
            base_model=base_model,
            adapter=adapter,
            settings=settings,
            generation_contract=generation_contract,
            baseline_manifest=baseline_manifest,
            baseline=baseline,
            adapted=adapted,
        ),
    )
    return _write_evaluation_manifest(output_dir, source_git_sha, rank)


def verify_rank_evaluation(
    *,
    rank: int,
    baseline_dir: Path,
    repo_root: Path = Path("."),
) -> dict[str, object]:
    registry = load_rank_candidate_registry()
    candidate = registry.candidate(rank)
    evaluation, settings, base_model, system_version, system_prompt = _evaluation_context(candidate)
    source_git_sha, _ = _preflight_source_tree(repo_root)
    output_dir = Path(evaluation.output_dir)
    stage = read_json(output_dir / STAGE_MANIFEST, context="P9-001 generation stage")
    adapter = _adapter_from_stage(stage, candidate)
    generation_contract = _adapter_contract(
        base_model=base_model,
        settings=settings,
        system_prompt_version=system_version,
        system_prompt=system_prompt,
        adapter=adapter,
    )
    _validate_stage(
        output_dir=output_dir,
        source_git_sha=source_git_sha,
        candidate=candidate,
        evaluation=evaluation,
        settings=settings,
        system_prompt=system_prompt,
        generation_contract=generation_contract,
        adapter=adapter,
    )
    baseline_manifest, baseline = load_baseline(baseline_dir, base_model)
    adapted = {
        suite: aggregate(
            output_dir
            / (
                "repository-holdout/repository-holdout-aggregate.json"
                if suite == "repository-holdout"
                else f"{suite}/{suite}-aggregate.json"
            ),
            suite_id=suite,
            base_model=base_model,
            adapter_id=candidate.adapter_id,
        )
        for suite in EXPECTED_SUITES
    }
    expected_comparison = _comparison(
        source_git_sha=source_git_sha,
        candidate=candidate,
        base_model=base_model,
        adapter=adapter,
        settings=settings,
        generation_contract=generation_contract,
        baseline_manifest=baseline_manifest,
        baseline=baseline,
        adapted=adapted,
    )
    actual_comparison = read_json(output_dir / COMPARISON, context="P9-001 comparison")
    if actual_comparison != expected_comparison:
        raise PythonRankSweepEvaluationError(
            "persisted comparison does not match recomputed metrics"
        )
    manifest = read_json(output_dir / EVALUATION_MANIFEST, context="P9-001 evaluation manifest")
    expected_manifest = _evaluation_manifest_payload(output_dir, source_git_sha, rank)
    if manifest != expected_manifest:
        raise PythonRankSweepEvaluationError("final P9-001 artifact-set manifest drifted")
    return actual_comparison


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run/verify P9-001 protected rank evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("generate", "score", "verify"):
        item = sub.add_parser(name)
        item.add_argument("--rank", type=int, choices=_EVALUATED_RANKS, required=True)
        item.add_argument("--repo-root", type=Path, default=Path("."))
        if name == "generate":
            item.add_argument("--training-output", type=Path, required=True)
            item.add_argument("--device-index", type=int, default=0)
        else:
            item.add_argument("--baseline-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    command = cast(str, args.command)
    rank = cast(int, args.rank)
    repo_root = cast(Path, args.repo_root)
    if command == "generate":
        print(
            generate_rank_stage(
                rank=rank,
                training_output=cast(Path, args.training_output),
                device_index=cast(int, args.device_index),
                repo_root=repo_root,
            )
        )
    elif command == "score":
        print(
            score_rank_stage(
                rank=rank,
                baseline_dir=cast(Path, args.baseline_dir),
                repo_root=repo_root,
            )
        )
    else:
        print(
            json.dumps(
                verify_rank_evaluation(
                    rank=rank,
                    baseline_dir=cast(Path, args.baseline_dir),
                    repo_root=repo_root,
                ),
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
