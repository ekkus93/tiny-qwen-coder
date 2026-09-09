"""FTR-102 no-op adapter equivalence control.

This module proves that exercising the PEFT adapter-loading path does not by
itself change the canonical Qwen3.5-4B evaluation behavior.  It creates an
exactly-zero LoRA adapter, loads it with ``PeftModel.from_pretrained``, runs the
same frozen prompts and greedy decoding contract used by the unchanged-base
baseline, scores the generated responses under the same OCI harness, and
compares task-level behavior against the exact FTR-101 reproduction artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Protocol, cast

import torch
from torch import nn

from tiny_qwen_coder.config import EvaluationConfig, load_evaluation_config
from tiny_qwen_coder.evaluation._baseline_artifacts import (
    file_sha256,
    write_regression_baseline_artifacts,
    write_runtime_metadata,
)
from tiny_qwen_coder.evaluation._baseline_generation import (
    BaselineGenerator,
    HuggingFaceBaselineGenerator,
    generation_contract_sha256,
)
from tiny_qwen_coder.evaluation._baseline_provenance import load_baseline_base_model_identity
from tiny_qwen_coder.evaluation._baseline_runner import (
    _CANONICAL_BASELINE_CONFIG_PATH,
    _generate_items,
    _preflight_execution_images,
    _preflight_source_tree,
    _regression_aggregate,
    _regression_results,
    _validate_baseline_contract,
)
from tiny_qwen_coder.evaluation._baseline_stages import (
    _generate_all_suites,
    _suite_performance_from_responses,
)
from tiny_qwen_coder.evaluation._baseline_types import BaselineGeneratedResponse
from tiny_qwen_coder.evaluation.execution import (
    ConstrainedExecutionHarness,
    OciRuntime,
    OciRuntimeSpec,
)
from tiny_qwen_coder.evaluation.humaneval import HumanEvalCompletion, HumanEvalEvaluator
from tiny_qwen_coder.evaluation.mbpp import MBPPCompletion, MBPPEvaluator
from tiny_qwen_coder.evaluation.python_ftr_base_reproduction import tokenizer_identity_payload
from tiny_qwen_coder.evaluation.regression import load_frozen_general_tool_regression_suite
from tiny_qwen_coder.evaluation.repository_holdout import (
    RepositoryHoldoutCompletion,
    RepositoryHoldoutEvaluator,
)
from tiny_qwen_coder.evaluation.settings import (
    FrozenEvaluationSettings,
    evaluation_settings_sha256,
    load_frozen_evaluation_settings,
)
from tiny_qwen_coder.identities import AdapterIdentity, BaseModelIdentity
from tiny_qwen_coder.reproducibility import seed_everything

_PROTOCOL_CONFIG = Path("configs/eval/python/ftr_102_noop_adapter_equivalence_v1.json")
_DEFAULT_OUTPUT_DIR = Path("artifacts/eval/python/ftr-102-noop-adapter-v1")
_CONTROL_IDENTITY_FILE = "ftr-102-control-identity.json"
_GENERATION_MANIFEST_FILE = "ftr-102-generation-manifest.json"
_CONTROL_ADAPTER = AdapterIdentity(
    family="control",
    adapter_id="control/python/ftr-102-noop",
)
_CHECKPOINT_PATHS = (
    ".baseline-work/humaneval-generation.jsonl",
    ".baseline-work/mbpp-generation.jsonl",
    ".baseline-work/repository-holdout-generation.jsonl",
    ".baseline-work/general-tool-regression-generation.jsonl",
)
_BENCHMARK_SUITES: tuple[tuple[str, str, str], ...] = (
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
_AGGREGATE_ALLOWED_DIFFERENCES = frozenset({"adapter", "results_sha256"})


class FTRNoopAdapterError(RuntimeError):
    """Raised when FTR-102 cannot prove adapter-path equivalence."""


class _PeftStatusLike(Protocol):
    enabled: object
    active_adapters: object
    available_adapters: object
    merged_adapters: object
    trainable_params: object


def _strict_mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise FTRNoopAdapterError(f"{context} must be a JSON object")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise FTRNoopAdapterError(f"{context} keys must be strings")
        result[key] = item
    return result


def _read_json(path: Path, *, context: str) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FTRNoopAdapterError(f"could not read {context}: {path}") from exc
    return _strict_mapping(value, context=context)


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _expect_str(mapping: Mapping[str, object], key: str, *, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise FTRNoopAdapterError(f"{context}.{key} must be a non-empty string")
    return value


def _expect_int(mapping: Mapping[str, object], key: str, *, context: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise FTRNoopAdapterError(f"{context}.{key} must be an integer")
    return value


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _protocol_config(path: Path = _PROTOCOL_CONFIG) -> dict[str, object]:
    config = _read_json(path, context="FTR-102 protocol config")
    if _expect_int(config, "schema_version", context="FTR-102 protocol config") != 1:
        raise FTRNoopAdapterError("unsupported FTR-102 protocol config schema")
    if _expect_str(config, "id", context="FTR-102 protocol config") != (
        "python-ftr-102-noop-adapter-equivalence-v1"
    ):
        raise FTRNoopAdapterError("unexpected FTR-102 protocol config id")
    if _expect_str(config, "task_id", context="FTR-102 protocol config") != "FTR-102":
        raise FTRNoopAdapterError("unexpected FTR-102 task id")
    return config


def _protocol_adapter(config: Mapping[str, object]) -> dict[str, object]:
    return _strict_mapping(config.get("noop_adapter"), context="FTR-102 noop_adapter")


def _protocol_base(config: Mapping[str, object]) -> BaseModelIdentity:
    raw = _strict_mapping(config.get("base_model"), context="FTR-102 base_model")
    return BaseModelIdentity(
        repository=_expect_str(raw, "repository", context="FTR-102 base_model"),
        revision=_expect_str(raw, "revision", context="FTR-102 base_model"),
        tokenizer_repository=_expect_str(
            raw,
            "tokenizer_repository",
            context="FTR-102 base_model",
        ),
        tokenizer_revision=_expect_str(
            raw,
            "tokenizer_revision",
            context="FTR-102 base_model",
        ),
    )


def _target_modules(adapter_config: Mapping[str, object]) -> list[str]:
    raw = adapter_config.get("target_modules")
    if not isinstance(raw, list) or not raw or any(not isinstance(item, str) for item in raw):
        raise FTRNoopAdapterError("FTR-102 target_modules must be a non-empty string list")
    values = cast(list[str], raw)
    if len(values) != len(set(values)):
        raise FTRNoopAdapterError("FTR-102 target_modules must be unique")
    return values


def _qualified_class_name(value: object) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _resolved_revision(model: object) -> str:
    config: object = getattr(model, "config", None)
    value: object = getattr(config, "_commit_hash", None)
    if not isinstance(value, str) or not value:
        raise FTRNoopAdapterError("loaded model does not expose resolved revision")
    return value


def _floating_dtypes(model: nn.Module) -> tuple[str, ...]:
    values = {
        str(parameter.dtype) for parameter in model.parameters() if parameter.is_floating_point()
    }
    if not values:
        raise FTRNoopAdapterError("model exposes no floating parameters")
    return tuple(sorted(values))


def _status_strings(value: object, *, field: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise FTRNoopAdapterError(f"PEFT status {field} is not a sequence")
    items = tuple(value)
    if any(not isinstance(item, str) for item in items):
        raise FTRNoopAdapterError(f"PEFT status {field} contains a non-string")
    return cast(tuple[str, ...], items)


def _peft_status(model: object) -> dict[str, object]:
    getter = getattr(model, "get_model_status", None)
    if not callable(getter):
        raise FTRNoopAdapterError("loaded PEFT model does not expose get_model_status()")
    status = cast(_PeftStatusLike, getter())
    active = _status_strings(status.active_adapters, field="active_adapters")
    available = _status_strings(status.available_adapters, field="available_adapters")
    merged = _status_strings(status.merged_adapters, field="merged_adapters")
    trainable = status.trainable_params
    if status.enabled is not True:
        raise FTRNoopAdapterError("FTR-102 no-op adapter is not enabled")
    if active != ("default",):
        raise FTRNoopAdapterError(f"unexpected active adapters: {active!r}")
    if "default" not in available:
        raise FTRNoopAdapterError("default FTR-102 adapter is not available")
    if merged:
        raise FTRNoopAdapterError("FTR-102 requires an unmerged no-op adapter")
    if isinstance(trainable, bool) or not isinstance(trainable, int) or trainable != 0:
        raise FTRNoopAdapterError("FTR-102 evaluation adapter has trainable parameters")
    return {
        "enabled": True,
        "active_adapters": list(active),
        "available_adapters": list(available),
        "merged_adapters": list(merged),
        "trainable_params": trainable,
    }


def inspect_zero_adapter(adapter_dir: Path) -> dict[str, object]:
    """Fail closed unless every persisted adapter tensor is exactly zero."""

    from safetensors import safe_open

    model_path = adapter_dir / "adapter_model.safetensors"
    config_path = adapter_dir / "adapter_config.json"
    if not model_path.is_file() or not config_path.is_file():
        raise FTRNoopAdapterError("no-op adapter artifact set is incomplete")
    tensor_count = 0
    total_elements = 0
    nonzero_tensors: list[str] = []
    tensor_dtypes: set[str] = set()
    try:
        with safe_open(str(model_path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                tensor_count += 1
                total_elements += tensor.numel()
                tensor_dtypes.add(str(tensor.dtype))
                if int(torch.count_nonzero(tensor).item()) != 0:
                    nonzero_tensors.append(key)
    except (OSError, RuntimeError) as exc:
        raise FTRNoopAdapterError("could not inspect no-op adapter safetensors") from exc
    if tensor_count <= 0 or total_elements <= 0:
        raise FTRNoopAdapterError("no-op adapter contains no tensors")
    if nonzero_tensors:
        raise FTRNoopAdapterError(
            f"no-op adapter contains non-zero tensors: {nonzero_tensors[:5]!r}"
        )
    return {
        "adapter_config_sha256": file_sha256(config_path),
        "adapter_model_sha256": file_sha256(model_path),
        "adapter_model_size_bytes": model_path.stat().st_size,
        "tensor_count": tensor_count,
        "total_elements": total_elements,
        "tensor_dtypes": sorted(tensor_dtypes),
        "nonzero_tensor_count": 0,
        "all_saved_adapter_tensors_zero": True,
    }


def prepare_noop_adapter(
    *,
    adapter_dir: Path,
    manifest_path: Path,
    device_index: int = 0,
    repo_root: Path = Path("."),
    protocol_path: Path = _PROTOCOL_CONFIG,
) -> dict[str, object]:
    """Create a persisted zero-LoRA adapter using the canonical training target structure."""

    if not torch.cuda.is_available():
        raise FTRNoopAdapterError("FTR-102 adapter preparation requires CUDA")
    if not 0 <= device_index < torch.cuda.device_count():
        raise FTRNoopAdapterError(f"invalid CUDA device index: {device_index}")
    source_git_sha, _ = _preflight_source_tree(repo_root)
    protocol = _protocol_config(protocol_path)
    expected_base = _protocol_base(protocol)
    base_model = load_baseline_base_model_identity(Path("configs/base/qwen35-4b.yaml"))
    if base_model != expected_base:
        raise FTRNoopAdapterError("FTR-102 protocol base-model identity drift detected")
    adapter = _protocol_adapter(protocol)
    targets = _target_modules(adapter)
    rank = _expect_int(adapter, "rank", context="FTR-102 noop_adapter")
    alpha = _expect_int(adapter, "alpha", context="FTR-102 noop_adapter")
    dropout = adapter.get("dropout")
    if isinstance(dropout, bool) or not isinstance(dropout, int | float):
        raise FTRNoopAdapterError("FTR-102 noop_adapter.dropout must be numeric")
    bias = _expect_str(adapter, "bias", context="FTR-102 noop_adapter")
    task_type = _expect_str(adapter, "task_type", context="FTR-102 noop_adapter")

    if adapter_dir.exists():
        if any(adapter_dir.iterdir()):
            raise FTRNoopAdapterError(f"refusing non-empty adapter directory: {adapter_dir}")
    else:
        adapter_dir.mkdir(parents=True)

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForMultimodalLM

    torch.cuda.set_device(device_index)
    seed_everything(1729)
    loaded: object = AutoModelForMultimodalLM.from_pretrained(
        base_model.repository,
        revision=base_model.revision,
        dtype=torch.bfloat16,
        device_map={"": device_index},
        low_cpu_mem_usage=True,
    )
    if not isinstance(loaded, nn.Module):
        raise FTRNoopAdapterError("Transformers returned an unexpected model object")
    if _resolved_revision(loaded) != base_model.revision:
        raise FTRNoopAdapterError("FTR-102 preparation loaded the wrong base revision")
    peft_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=float(dropout),
        bias=bias,
        target_modules=targets,
        task_type=task_type,
    )
    adapted = cast(nn.Module, get_peft_model(loaded, peft_config))
    adapter_parameters = 0
    with torch.no_grad():
        for name, parameter in adapted.named_parameters():
            if "lora_" in name:
                parameter.zero_()
                adapter_parameters += parameter.numel()
    if adapter_parameters <= 0:
        raise FTRNoopAdapterError("PEFT created no LoRA parameters for FTR-102")
    adapted.save_pretrained(str(adapter_dir), safe_serialization=True)
    artifact_identity = inspect_zero_adapter(adapter_dir)
    payload: dict[str, object] = {
        "schema_version": 1,
        "task_id": "FTR-102",
        "source_git_sha": source_git_sha,
        "protocol_config_sha256": file_sha256(protocol_path),
        "base_model": asdict(base_model),
        "loader_under_test": _expect_str(adapter, "loader", context="FTR-102 noop_adapter"),
        "lora": {
            "rank": rank,
            "alpha": alpha,
            "dropout": float(dropout),
            "bias": bias,
            "task_type": task_type,
            "target_modules": targets,
            "zeroed_parameter_elements_before_save": adapter_parameters,
        },
        "artifacts": artifact_identity,
        "peft_version": importlib.metadata.version("peft"),
        "torch_version": torch.__version__,
        "transformers_version": importlib.metadata.version("transformers"),
    }
    _write_json(manifest_path, payload)
    return payload


class NoopAdapterGenerator(HuggingFaceBaselineGenerator):
    """Canonical base generator with a persisted exactly-zero PEFT adapter attached."""

    def __init__(
        self,
        *,
        adapter_dir: Path,
        adapter_manifest: Mapping[str, object],
        base_model: BaseModelIdentity,
        settings: FrozenEvaluationSettings,
        device_index: int = 0,
    ) -> None:
        super().__init__(base_model=base_model, settings=settings, device_index=device_index)
        from peft import PeftModel

        before_model = self._model
        before_class = _qualified_class_name(before_model)
        before_dtypes = _floating_dtypes(before_model)
        before_revision = _resolved_revision(before_model)
        tokenizer_payload = tokenizer_identity_payload(
            cast(Any, self._tokenizer),
            settings=settings,
            repository=base_model.tokenizer_repository,
            revision=base_model.tokenizer_revision,
        )
        persisted_identity = inspect_zero_adapter(adapter_dir)
        manifest_artifacts = _strict_mapping(
            adapter_manifest.get("artifacts"), context="FTR-102 adapter manifest artifacts"
        )
        for key in (
            "adapter_config_sha256",
            "adapter_model_sha256",
            "adapter_model_size_bytes",
            "tensor_count",
            "total_elements",
            "nonzero_tensor_count",
            "all_saved_adapter_tensors_zero",
        ):
            if manifest_artifacts.get(key) != persisted_identity.get(key):
                raise FTRNoopAdapterError(f"persisted no-op adapter identity drift: {key}")

        started = time.perf_counter()
        adapted_obj: object = cast(Any, PeftModel).from_pretrained(
            before_model,
            str(adapter_dir),
            adapter_name="default",
            is_trainable=False,
        )
        if not isinstance(adapted_obj, nn.Module):
            raise FTRNoopAdapterError("PEFT returned an unexpected model object")
        self._model = adapted_obj
        self._model.eval()
        self._model.requires_grad_(False)
        status = _peft_status(self._model)
        loaded_adapter_elements = 0
        nonzero_loaded: list[str] = []
        with torch.no_grad():
            for name, parameter in self._model.named_parameters():
                if "lora_" not in name:
                    continue
                loaded_adapter_elements += parameter.numel()
                if int(torch.count_nonzero(parameter).item()) != 0:
                    nonzero_loaded.append(name)
        if loaded_adapter_elements <= 0:
            raise FTRNoopAdapterError("loaded PEFT model exposes no LoRA parameters")
        if nonzero_loaded:
            raise FTRNoopAdapterError(
                f"loaded no-op adapter contains non-zero parameters: {nonzero_loaded[:5]!r}"
            )
        getter = getattr(self._model, "get_base_model", None)
        if not callable(getter):
            raise FTRNoopAdapterError("PEFT model does not expose get_base_model()")
        base_after = getter()
        after_revision = _resolved_revision(base_after)
        if after_revision != before_revision or after_revision != base_model.revision:
            raise FTRNoopAdapterError("base-model revision changed across adapter activation")
        self._load_seconds += time.perf_counter() - started
        self._parameter_dtypes = _floating_dtypes(self._model)
        self._resolved_model_revision = after_revision
        self.control_identity: dict[str, object] = {
            "schema_version": 1,
            "task_id": "FTR-102",
            "base_model": asdict(base_model),
            "tokenizer_identity": tokenizer_payload,
            "model_before_adapter": {
                "class": before_class,
                "parameter_dtypes": list(before_dtypes),
                "resolved_revision": before_revision,
            },
            "model_after_adapter": {
                "class": _qualified_class_name(self._model),
                "parameter_dtypes": list(self._parameter_dtypes),
                "base_resolved_revision": after_revision,
            },
            "adapter_load": {
                "loader": "peft.PeftModel.from_pretrained",
                "status": status,
                "loaded_adapter_parameter_elements": loaded_adapter_elements,
                "nonzero_loaded_adapter_parameters": 0,
                "all_loaded_adapter_parameters_zero": True,
                "persisted_artifacts": persisted_identity,
            },
        }


def _evaluation_context() -> tuple[
    EvaluationConfig,
    FrozenEvaluationSettings,
    BaseModelIdentity,
    str,
    str,
    str,
]:
    evaluation = load_evaluation_config(_CANONICAL_BASELINE_CONFIG_PATH)
    settings = load_frozen_evaluation_settings()
    base_model = load_baseline_base_model_identity(Path(evaluation.base_config))
    system_prompt_version, system_prompt = _validate_baseline_contract(
        evaluation,
        settings,
        base_model,
    )
    generation_contract = generation_contract_sha256(
        base_model=base_model,
        settings=settings,
        system_prompt_version=system_prompt_version,
        system_prompt=system_prompt,
    )
    return (
        evaluation,
        settings,
        base_model,
        system_prompt_version,
        system_prompt,
        generation_contract,
    )


def _write_generation_manifest(
    *,
    output_dir: Path,
    source_git_sha: str,
    protocol_path: Path,
    generation_contract: str,
) -> dict[str, object]:
    paths = (*_CHECKPOINT_PATHS, "runtime-metadata.json", _CONTROL_IDENTITY_FILE)
    artifacts: list[dict[str, object]] = []
    for relative in paths:
        path = output_dir / relative
        if not path.is_file():
            raise FTRNoopAdapterError(f"FTR-102 generation artifact is missing: {relative}")
        artifacts.append({"path": relative, "sha256": file_sha256(path)})
    manifest: dict[str, object] = {
        "schema_version": 1,
        "task_id": "FTR-102",
        "source_git_sha": source_git_sha,
        "protocol_config_sha256": file_sha256(protocol_path),
        "generation_contract_sha256": generation_contract,
        "evaluation_settings_sha256": evaluation_settings_sha256(load_frozen_evaluation_settings()),
        "artifacts": artifacts,
    }
    _write_json(output_dir / _GENERATION_MANIFEST_FILE, manifest)
    return manifest


def _validate_generation_manifest(
    *,
    output_dir: Path,
    source_git_sha: str,
    protocol_path: Path,
    generation_contract: str,
) -> dict[str, object]:
    manifest = _read_json(
        output_dir / _GENERATION_MANIFEST_FILE,
        context="FTR-102 generation manifest",
    )
    expected = {
        "schema_version": 1,
        "task_id": "FTR-102",
        "source_git_sha": source_git_sha,
        "protocol_config_sha256": file_sha256(protocol_path),
        "generation_contract_sha256": generation_contract,
        "evaluation_settings_sha256": evaluation_settings_sha256(load_frozen_evaluation_settings()),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise FTRNoopAdapterError(f"FTR-102 generation manifest drift: {key}")
    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise FTRNoopAdapterError("FTR-102 generation manifest artifacts must be a list")
    seen: list[str] = []
    for index, item in enumerate(raw_artifacts):
        mapping = _strict_mapping(item, context=f"FTR-102 artifacts[{index}]")
        relative = _expect_str(mapping, "path", context=f"FTR-102 artifacts[{index}]")
        digest = _expect_str(mapping, "sha256", context=f"FTR-102 artifacts[{index}]")
        if file_sha256(output_dir / relative) != digest:
            raise FTRNoopAdapterError(f"FTR-102 transported artifact digest drift: {relative}")
        seen.append(relative)
    expected_paths = [*_CHECKPOINT_PATHS, "runtime-metadata.json", _CONTROL_IDENTITY_FILE]
    if seen != expected_paths:
        raise FTRNoopAdapterError("FTR-102 generation artifact inventory drift detected")
    return manifest


def generate_noop_control(
    *,
    adapter_dir: Path,
    adapter_manifest_path: Path,
    output_dir: Path = _DEFAULT_OUTPUT_DIR,
    device_index: int = 0,
    repo_root: Path = Path("."),
    protocol_path: Path = _PROTOCOL_CONFIG,
) -> dict[str, object]:
    """Generate all canonical suites with the persisted zero adapter attached."""

    source_git_sha, _ = _preflight_source_tree(repo_root)
    protocol = _protocol_config(protocol_path)
    adapter_manifest = _read_json(adapter_manifest_path, context="FTR-102 adapter manifest")
    if adapter_manifest.get("source_git_sha") != source_git_sha:
        raise FTRNoopAdapterError("FTR-102 adapter was prepared from a different source SHA")
    if adapter_manifest.get("protocol_config_sha256") != file_sha256(protocol_path):
        raise FTRNoopAdapterError("FTR-102 adapter protocol identity drift detected")
    expected_base = _protocol_base(protocol)
    (
        evaluation,
        settings,
        base_model,
        _system_prompt_version,
        system_prompt,
        generation_contract,
    ) = _evaluation_context()
    if base_model != expected_base:
        raise FTRNoopAdapterError("FTR-102 evaluation base-model identity drift detected")
    generator = NoopAdapterGenerator(
        adapter_dir=adapter_dir,
        adapter_manifest=adapter_manifest,
        base_model=base_model,
        settings=settings,
        device_index=device_index,
    )
    (
        _humaneval_problems,
        humaneval_responses,
        _mbpp_problems,
        mbpp_responses,
        _holdout,
        holdout_responses,
        regression_responses,
    ) = _generate_all_suites(
        evaluation=evaluation,
        generator=generator,
        base_model=base_model,
        settings=settings,
        system_prompt=system_prompt,
        generation_contract=generation_contract,
        output_dir=output_dir,
    )
    suites = _suite_performance_from_responses(
        humaneval=humaneval_responses,
        mbpp=mbpp_responses,
        holdout=holdout_responses,
        regression=regression_responses,
    )
    runtime_metadata = generator.runtime_metadata(suites)
    write_runtime_metadata(runtime_metadata, output_dir)
    identity = dict(generator.control_identity)
    identity.update(
        {
            "source_git_sha": source_git_sha,
            "protocol_config_sha256": file_sha256(protocol_path),
            "adapter_manifest_sha256": file_sha256(adapter_manifest_path),
            "generation_contract_sha256": generation_contract,
            "evaluation_config": str(_CANONICAL_BASELINE_CONFIG_PATH),
            "evaluation_settings_sha256": evaluation_settings_sha256(settings),
        }
    )
    _write_json(output_dir / _CONTROL_IDENTITY_FILE, identity)
    manifest = _write_generation_manifest(
        output_dir=output_dir,
        source_git_sha=source_git_sha,
        protocol_path=protocol_path,
        generation_contract=generation_contract,
    )
    return manifest


def _control_evaluators(
    *,
    evaluation: EvaluationConfig,
    settings: FrozenEvaluationSettings,
    base_model: BaseModelIdentity,
    harness: ConstrainedExecutionHarness,
) -> tuple[HumanEvalEvaluator, MBPPEvaluator, RepositoryHoldoutEvaluator]:
    return (
        HumanEvalEvaluator(
            evaluation,
            base_model=base_model,
            adapter=_CONTROL_ADAPTER,
            settings=settings,
            harness=harness,
        ),
        MBPPEvaluator(
            evaluation,
            base_model=base_model,
            adapter=_CONTROL_ADAPTER,
            settings=settings,
            harness=harness,
        ),
        RepositoryHoldoutEvaluator(
            evaluation,
            base_model=base_model,
            adapter=_CONTROL_ADAPTER,
            settings=settings,
            harness=harness,
        ),
    )


class _CheckpointOnly(BaselineGenerator):
    def generate(self, *, system_prompt: str, user_prompt: str) -> BaselineGeneratedResponse:
        del system_prompt, user_prompt
        raise FTRNoopAdapterError("FTR-102 scoring attempted unexpected regeneration")


def _score_control(
    *,
    output_dir: Path,
    runtime: OciRuntimeSpec,
) -> None:
    (
        evaluation,
        settings,
        base_model,
        system_prompt_version,
        system_prompt,
        generation_contract,
    ) = _evaluation_context()
    harness = ConstrainedExecutionHarness(runtime=runtime)
    humaneval, mbpp, holdout = _control_evaluators(
        evaluation=evaluation,
        settings=settings,
        base_model=base_model,
        harness=harness,
    )
    humaneval_problems = humaneval.load_problems()
    mbpp_problems = mbpp.load_problems()
    _preflight_execution_images(
        runtime,
        (
            humaneval.runner.execution_image,
            mbpp.runner.execution_image,
            holdout.suite.execution_image,
        ),
    )
    checkpoint = _CheckpointOnly()
    humaneval_responses = _generate_items(
        suite_id="humaneval",
        prompts=tuple(
            (problem.task_id, humaneval.prompt_for(problem).user_content)
            for problem in humaneval_problems
        ),
        generator=checkpoint,
        system_prompt=system_prompt,
        generation_contract=generation_contract,
        output_dir=output_dir,
    )
    mbpp_responses = _generate_items(
        suite_id="mbpp",
        prompts=tuple(
            (problem.task_id, mbpp.prompt_for(problem).user_content) for problem in mbpp_problems
        ),
        generator=checkpoint,
        system_prompt=system_prompt,
        generation_contract=generation_contract,
        output_dir=output_dir,
    )
    holdout_responses = _generate_items(
        suite_id="repository-holdout",
        prompts=tuple(
            (task.problem_id, holdout.prompt_for(task).user_content) for task in holdout.suite.tasks
        ),
        generator=checkpoint,
        system_prompt=system_prompt,
        generation_contract=generation_contract,
        output_dir=output_dir,
    )
    regression_suite = load_frozen_general_tool_regression_suite()
    regression_responses = _generate_items(
        suite_id="general-tool-regression",
        prompts=tuple((case.id, case.prompt) for case in regression_suite.cases),
        generator=checkpoint,
        system_prompt=system_prompt,
        generation_contract=generation_contract,
        output_dir=output_dir,
    )

    humaneval_result = humaneval.evaluate_suite(
        humaneval_problems,
        tuple(
            HumanEvalCompletion(
                task_id=problem.task_id,
                generated_text=response.generated_text,
                generation=response.generation,
            )
            for problem, response in zip(humaneval_problems, humaneval_responses, strict=True)
        ),
    )
    humaneval.write_artifacts(humaneval_result, output_dir / "humaneval")

    mbpp_result = mbpp.evaluate_suite(
        mbpp_problems,
        tuple(
            MBPPCompletion(
                task_id=problem.task_id,
                generated_text=response.generated_text,
                generation=response.generation,
            )
            for problem, response in zip(mbpp_problems, mbpp_responses, strict=True)
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
            for task, response in zip(holdout.suite.tasks, holdout_responses, strict=True)
        )
    )
    holdout.write_artifacts(holdout_result, output_dir / "repository-holdout")

    regression_results = _regression_results(regression_suite, regression_responses)
    regression_aggregate = replace(
        _regression_aggregate(
            suite=regression_suite,
            results=regression_results,
            settings=settings,
            system_prompt_version=system_prompt_version,
            system_prompt=system_prompt,
            base_model=base_model,
        ),
        adapter=_CONTROL_ADAPTER,
    )
    write_regression_baseline_artifacts(
        results=regression_results,
        aggregate=regression_aggregate,
        output_dir=output_dir / "general-tool-regression",
    )


def _load_jsonl(path: Path, *, context: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise FTRNoopAdapterError(f"could not read {context}: {path}") from exc
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value: object = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FTRNoopAdapterError(f"invalid JSON in {context} line {index}") from exc
        rows.append(_strict_mapping(value, context=f"{context}[{index}]"))
    if not rows:
        raise FTRNoopAdapterError(f"{context} is empty")
    return rows


def _canonical_benchmark_row(row: Mapping[str, object], *, context: str) -> dict[str, object]:
    generation = _strict_mapping(row.get("generation"), context=f"{context}.generation")
    tests = _strict_mapping(row.get("tests"), context=f"{context}.tests")
    generated_text = row.get("generated_text")
    if not isinstance(generated_text, str):
        raise FTRNoopAdapterError(f"{context}.generated_text must be a string")
    return {
        "problem_id": _expect_str(row, "problem_id", context=context),
        "generated_text": generated_text,
        "generated_code": row.get("generated_code"),
        "prompt_tokens": _expect_int(generation, "prompt_tokens", context=f"{context}.generation"),
        "generated_tokens": _expect_int(
            generation,
            "generated_tokens",
            context=f"{context}.generation",
        ),
        "parse_status": row.get("parse_status"),
        "compile_status": row.get("compile_status"),
        "error_category": row.get("error_category"),
        "tests_passed": _expect_int(tests, "passed", context=f"{context}.tests"),
        "tests_total": _expect_int(tests, "total", context=f"{context}.tests"),
    }


def _canonical_regression_row(row: Mapping[str, object], *, context: str) -> dict[str, object]:
    generation = _strict_mapping(row.get("generation"), context=f"{context}.generation")
    generated_text = row.get("generated_text")
    if not isinstance(generated_text, str):
        raise FTRNoopAdapterError(f"{context}.generated_text must be a string")
    passed = row.get("passed")
    if not isinstance(passed, bool):
        raise FTRNoopAdapterError(f"{context}.passed must be boolean")
    return {
        "case_id": _expect_str(row, "case_id", context=context),
        "category": row.get("category"),
        "generated_text": generated_text,
        "passed": passed,
        "detail": row.get("detail"),
        "prompt_tokens": _expect_int(generation, "prompt_tokens", context=f"{context}.generation"),
        "generated_tokens": _expect_int(
            generation,
            "generated_tokens",
            context=f"{context}.generation",
        ),
    }


def _index_rows(
    path: Path,
    *,
    context: str,
    regression: bool = False,
) -> dict[str, dict[str, object]]:
    indexed: dict[str, dict[str, object]] = {}
    for index, row in enumerate(_load_jsonl(path, context=context)):
        canonical = (
            _canonical_regression_row(row, context=f"{context}[{index}]")
            if regression
            else _canonical_benchmark_row(row, context=f"{context}[{index}]")
        )
        key_name = "case_id" if regression else "problem_id"
        key = cast(str, canonical[key_name])
        if key in indexed:
            raise FTRNoopAdapterError(f"duplicate result id in {context}: {key}")
        indexed[key] = canonical
    return indexed


def _aggregate_comparison(frozen: Path, current: Path, *, context: str) -> dict[str, object]:
    frozen_payload = _read_json(frozen, context=f"frozen {context} aggregate")
    current_payload = _read_json(current, context=f"current {context} aggregate")
    frozen_stable = {
        key: value
        for key, value in frozen_payload.items()
        if key not in _AGGREGATE_ALLOWED_DIFFERENCES
    }
    current_stable = {
        key: value
        for key, value in current_payload.items()
        if key not in _AGGREGATE_ALLOWED_DIFFERENCES
    }
    changed = sorted(
        key
        for key in set(frozen_stable) | set(current_stable)
        if frozen_stable.get(key) != current_stable.get(key)
    )
    return {"exact": not changed, "changed_fields": changed}


def _result_comparison(
    *,
    frozen_path: Path,
    current_path: Path,
    suite_id: str,
    regression: bool = False,
) -> dict[str, object]:
    frozen = _index_rows(frozen_path, context=f"frozen {suite_id}", regression=regression)
    current = _index_rows(current_path, context=f"current {suite_id}", regression=regression)
    frozen_ids = set(frozen)
    current_ids = set(current)
    missing = sorted(frozen_ids - current_ids)
    unexpected = sorted(current_ids - frozen_ids)
    mismatches: list[dict[str, object]] = []
    for item_id in sorted(frozen_ids & current_ids):
        before = frozen[item_id]
        after = current[item_id]
        if before == after:
            continue
        changed = sorted(key for key in before if before.get(key) != after.get(key))
        mismatches.append(
            {
                "item_id": item_id,
                "changed_fields": changed,
                "frozen_generated_text_sha256": _sha256_text(cast(str, before["generated_text"])),
                "current_generated_text_sha256": _sha256_text(cast(str, after["generated_text"])),
            }
        )
    return {
        "suite_id": suite_id,
        "frozen_items": len(frozen),
        "current_items": len(current),
        "missing_ids": missing,
        "unexpected_ids": unexpected,
        "mismatches": mismatches,
        "exact_task_records": not missing and not unexpected and not mismatches,
    }


def _identity_comparison(
    *,
    frozen_dir: Path,
    frozen_identity_path: Path,
    current_identity_path: Path,
) -> dict[str, object]:
    frozen_identity = _read_json(frozen_identity_path, context="FTR-101 generation identity")
    current_identity = _read_json(current_identity_path, context="FTR-102 control identity")
    current_tokenizer = _strict_mapping(
        current_identity.get("tokenizer_identity"), context="FTR-102 tokenizer identity"
    )
    frozen_tokenizer = {key: frozen_identity.get(key) for key in current_tokenizer}
    tokenizer_exact = frozen_tokenizer == current_tokenizer
    base_exact = frozen_identity.get("base_model") == current_identity.get("base_model")
    frozen_runtime = _read_json(
        frozen_dir / "runtime-metadata.json", context="FTR-101 runtime metadata"
    )
    before = _strict_mapping(
        current_identity.get("model_before_adapter"), context="FTR-102 model_before_adapter"
    )
    pre_adapter_exact = (
        before.get("class") == frozen_runtime.get("model_class")
        and before.get("parameter_dtypes") == frozen_runtime.get("parameter_dtypes")
        and before.get("resolved_revision") == frozen_runtime.get("resolved_model_revision")
    )
    frozen_manifest = _read_json(
        frozen_dir / "baseline-manifest.json", context="FTR-101 baseline manifest"
    )
    generation_contract_exact = current_identity.get(
        "generation_contract_sha256"
    ) == frozen_manifest.get("generation_contract_sha256")
    after = _strict_mapping(
        current_identity.get("model_after_adapter"), context="FTR-102 model_after_adapter"
    )
    base_revision_preserved = (
        after.get("base_resolved_revision")
        == cast(dict[str, object], current_identity.get("base_model"))["revision"]
    )
    adapter_load = _strict_mapping(
        current_identity.get("adapter_load"), context="FTR-102 adapter_load"
    )
    status = _strict_mapping(adapter_load.get("status"), context="FTR-102 adapter status")
    adapter_proof = (
        adapter_load.get("loader") == "peft.PeftModel.from_pretrained"
        and adapter_load.get("all_loaded_adapter_parameters_zero") is True
        and adapter_load.get("nonzero_loaded_adapter_parameters") == 0
        and status.get("enabled") is True
        and status.get("active_adapters") == ["default"]
        and status.get("merged_adapters") == []
        and status.get("trainable_params") == 0
    )
    return {
        "tokenizer_identity_exact": tokenizer_exact,
        "base_model_identity_exact": base_exact,
        "pre_adapter_model_identity_exact": pre_adapter_exact,
        "generation_contract_exact": generation_contract_exact,
        "base_revision_preserved_after_adapter_load": base_revision_preserved,
        "adapter_loading_path_proven": adapter_proof,
        "exact": (
            tokenizer_exact
            and base_exact
            and pre_adapter_exact
            and generation_contract_exact
            and base_revision_preserved
            and adapter_proof
        ),
    }


def compare_control_to_frozen(
    *,
    frozen_dir: Path,
    frozen_identity_path: Path,
    current_dir: Path,
    current_identity_path: Path,
) -> dict[str, object]:
    """Compare FTR-102 output and harness behavior to the exact FTR-101 control."""

    suites: list[dict[str, object]] = []
    for suite_id, results_rel, aggregate_rel in _BENCHMARK_SUITES:
        result = _result_comparison(
            frozen_path=frozen_dir / results_rel,
            current_path=current_dir / results_rel,
            suite_id=suite_id,
        )
        aggregate = _aggregate_comparison(
            frozen_dir / aggregate_rel,
            current_dir / aggregate_rel,
            context=suite_id,
        )
        result["aggregate_exact_except_adapter_and_result_hash"] = aggregate["exact"]
        result["aggregate_changed_fields"] = aggregate["changed_fields"]
        result["exact_suite_equivalence"] = (
            result["exact_task_records"] is True and aggregate["exact"] is True
        )
        suites.append(result)

    regression = _result_comparison(
        frozen_path=frozen_dir / "general-tool-regression/general-tool-regression-results.jsonl",
        current_path=current_dir / "general-tool-regression/general-tool-regression-results.jsonl",
        suite_id="general-tool-regression",
        regression=True,
    )
    regression_aggregate = _aggregate_comparison(
        frozen_dir / "general-tool-regression/general-tool-regression-aggregate.json",
        current_dir / "general-tool-regression/general-tool-regression-aggregate.json",
        context="general-tool-regression",
    )
    regression["aggregate_exact_except_adapter_and_result_hash"] = regression_aggregate["exact"]
    regression["aggregate_changed_fields"] = regression_aggregate["changed_fields"]
    regression["exact_suite_equivalence"] = (
        regression["exact_task_records"] is True and regression_aggregate["exact"] is True
    )
    suites.append(regression)

    identity = _identity_comparison(
        frozen_dir=frozen_dir,
        frozen_identity_path=frozen_identity_path,
        current_identity_path=current_identity_path,
    )
    exact = identity["exact"] is True and all(
        item.get("exact_suite_equivalence") is True for item in suites
    )
    return {
        "schema_version": 1,
        "task_id": "FTR-102",
        "control": "exactly-zero LoRA loaded through peft.PeftModel.from_pretrained",
        "identity": identity,
        "suites": suites,
        "exact_noop_adapter_equivalence": exact,
    }


def score_and_compare(
    *,
    frozen_dir: Path,
    frozen_identity_path: Path,
    adapter_manifest_path: Path,
    output_dir: Path = _DEFAULT_OUTPUT_DIR,
    oci_runtime: Path,
    report_path: Path,
    repo_root: Path = Path("."),
    protocol_path: Path = _PROTOCOL_CONFIG,
) -> dict[str, object]:
    """Score FTR-102 and fail closed unless it is exactly equivalent to FTR-101."""

    source_git_sha, _ = _preflight_source_tree(repo_root)
    protocol = _protocol_config(protocol_path)
    adapter_manifest = _read_json(adapter_manifest_path, context="FTR-102 adapter manifest")
    if adapter_manifest.get("source_git_sha") != source_git_sha:
        raise FTRNoopAdapterError("FTR-102 transported adapter source SHA drift detected")
    if adapter_manifest.get("protocol_config_sha256") != file_sha256(protocol_path):
        raise FTRNoopAdapterError("FTR-102 transported adapter protocol drift detected")
    (
        _evaluation,
        _settings,
        base_model,
        _system_prompt_version,
        _system_prompt,
        generation_contract,
    ) = _evaluation_context()
    if base_model != _protocol_base(protocol):
        raise FTRNoopAdapterError("FTR-102 score-stage base identity drift detected")
    _validate_generation_manifest(
        output_dir=output_dir,
        source_git_sha=source_git_sha,
        protocol_path=protocol_path,
        generation_contract=generation_contract,
    )
    if not oci_runtime.is_absolute() or not oci_runtime.is_file():
        raise FTRNoopAdapterError("FTR-102 OCI runtime must be an existing absolute path")
    runtime = OciRuntimeSpec(kind=OciRuntime.DOCKER, executable=oci_runtime)
    _score_control(output_dir=output_dir, runtime=runtime)
    report = compare_control_to_frozen(
        frozen_dir=frozen_dir,
        frozen_identity_path=frozen_identity_path,
        current_dir=output_dir,
        current_identity_path=output_dir / _CONTROL_IDENTITY_FILE,
    )
    report.update(
        {
            "source_git_sha": source_git_sha,
            "protocol_config_sha256": file_sha256(protocol_path),
            "adapter_manifest_sha256": file_sha256(adapter_manifest_path),
            "frozen_reference": _strict_mapping(
                protocol.get("ftr_101_reference"), context="FTR-102 ftr_101_reference"
            ),
        }
    )
    _write_json(report_path, report)
    if report["exact_noop_adapter_equivalence"] is not True:
        raise FTRNoopAdapterError(
            "FTR-102 no-op adapter changed canonical evaluation behavior; see report"
        )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run FTR-102 no-op adapter equivalence control")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-adapter")
    prepare.add_argument("--adapter-dir", type=Path, required=True)
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--device-index", type=int, default=0)
    prepare.add_argument("--repo-root", type=Path, default=Path("."))

    generate = subparsers.add_parser("generate")
    generate.add_argument("--adapter-dir", type=Path, required=True)
    generate.add_argument("--adapter-manifest", type=Path, required=True)
    generate.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
    generate.add_argument("--device-index", type=int, default=0)
    generate.add_argument("--repo-root", type=Path, default=Path("."))

    score = subparsers.add_parser("score-and-compare")
    score.add_argument("--frozen-dir", type=Path, required=True)
    score.add_argument("--frozen-identity", type=Path, required=True)
    score.add_argument("--adapter-manifest", type=Path, required=True)
    score.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
    score.add_argument("--oci-runtime", type=Path, required=True)
    score.add_argument("--report", type=Path, required=True)
    score.add_argument("--repo-root", type=Path, default=Path("."))
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    command = cast(str, args.command)
    if command == "prepare-adapter":
        payload = prepare_noop_adapter(
            adapter_dir=cast(Path, args.adapter_dir),
            manifest_path=cast(Path, args.manifest),
            device_index=cast(int, args.device_index),
            repo_root=cast(Path, args.repo_root),
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if command == "generate":
        payload = generate_noop_control(
            adapter_dir=cast(Path, args.adapter_dir),
            adapter_manifest_path=cast(Path, args.adapter_manifest),
            output_dir=cast(Path, args.output_dir),
            device_index=cast(int, args.device_index),
            repo_root=cast(Path, args.repo_root),
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    payload = score_and_compare(
        frozen_dir=cast(Path, args.frozen_dir),
        frozen_identity_path=cast(Path, args.frozen_identity),
        adapter_manifest_path=cast(Path, args.adapter_manifest),
        output_dir=cast(Path, args.output_dir),
        oci_runtime=cast(Path, args.oci_runtime),
        report_path=cast(Path, args.report),
        repo_root=cast(Path, args.repo_root),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
