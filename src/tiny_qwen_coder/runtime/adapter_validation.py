"""Fail-closed P7-007 adapter load/inference validation on the canonical CUDA base."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import torch
from torch import nn

from tiny_qwen_coder.adapters import AdapterManifest, load_adapter_manifest
from tiny_qwen_coder.identities import BaseModelIdentity
from tiny_qwen_coder.languages.python import load_python_plugin
from tiny_qwen_coder.reporting import load_base_model_identity
from tiny_qwen_coder.reproducibility import seed_everything

_SCHEMA_VERSION = 1
_TASK_ID = "P7-007"
_DEFAULT_SEED = 1729
_DEFAULT_MAX_NEW_TOKENS = 128
_DEFAULT_BASE_CONFIG = Path("configs/base/qwen35-4b.yaml")
_DEFAULT_OUTPUT = Path("artifacts/eval/python/p7-007/adapter-inference-validation.json")
_ENABLE_THINKING = False
_EXPECTED_ADAPTER_ID = "language/python/p0"
_EXPECTED_FAMILY = "language"
_EXPECTED_LANGUAGE = "python"
_REQUIRED_OUTPUT_FILES = (
    "adapter-manifest.json",
    "training-config.json",
    "training-report.json",
    "run-manifest.json",
    "adapter/adapter_config.json",
    "adapter/adapter_model.safetensors",
    "adapter/chat_template.jinja",
)
_FORBIDDEN_MODEL_PATTERNS = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
    "model-*.safetensors",
    "pytorch_model-*.bin",
)
_SMOKE_PROMPTS = (
    (
        "clamp-function",
        "Implement a Python function clamp(value, low, high) that returns low when value is "
        "below low, high when value is above high, and value otherwise. Raise ValueError when "
        "low is greater than high. Return only Python code.",
    ),
    (
        "stable-unique",
        "Implement a Python function unique_preserve_order(items) that returns a list containing "
        "the first occurrence of each hashable item while preserving input order. Return only "
        "Python code.",
    ),
)


class AdapterInferenceValidationError(RuntimeError):
    """Raised when P7-007 cannot prove the adapter inference contract."""


@dataclass(frozen=True, slots=True)
class VerifiedAdapterArtifacts:
    """Validated filesystem and provenance identity for one trained adapter output."""

    output_dir: Path
    adapter_dir: Path
    manifest: AdapterManifest
    adapter_model_sha256: str
    adapter_model_size_bytes: int
    training_run_id: str
    training_git_sha: str
    inference_chat_template_sha256: str


@dataclass(frozen=True, slots=True)
class GenerationObservation:
    """One deterministic generated suffix and its bounded runtime measurements."""

    text: str
    token_ids: tuple[int, ...]
    prompt_tokens: int
    generated_tokens: int
    latency_seconds: float

    def __post_init__(self) -> None:
        if not self.token_ids:
            raise AdapterInferenceValidationError("generation produced no new tokens")
        if self.generated_tokens != len(self.token_ids):
            raise AdapterInferenceValidationError(
                "generated_tokens must equal the number of recorded token IDs"
            )
        if self.prompt_tokens <= 0:
            raise AdapterInferenceValidationError("prompt_tokens must be greater than zero")
        if not math.isfinite(self.latency_seconds) or self.latency_seconds <= 0:
            raise AdapterInferenceValidationError(
                "generation latency must be finite and greater than zero"
            )


@dataclass(frozen=True, slots=True)
class SmokePromptValidation:
    """Enable/disable/re-enable observations for one fixed smoke prompt."""

    prompt_id: str
    prompt: str
    base: GenerationObservation
    adapter_enabled: GenerationObservation
    adapter_disabled: GenerationObservation
    adapter_reenabled: GenerationObservation
    base_recovered_exactly: bool
    adapter_reenabled_exactly: bool
    adapter_changed_output: bool


@dataclass(frozen=True, slots=True)
class AdapterStatusSnapshot:
    """Small stable subset of PEFT model status relevant to P7-007."""

    enabled: bool
    active_adapters: tuple[str, ...]
    available_adapters: tuple[str, ...]
    merged_adapters: tuple[str, ...]
    num_adapter_layers: int
    trainable_params: int


@dataclass(frozen=True, slots=True)
class AdapterInferenceValidationReport:
    """Machine-readable acceptance evidence for P7-007."""

    schema_version: int
    task_id: str
    accepted: bool
    base_model: BaseModelIdentity
    resolved_model_revision: str
    adapter_id: str
    adapter_model_sha256: str
    adapter_model_size_bytes: int
    source_training_run_id: str
    source_training_git_sha: str
    base_load_count: int
    same_base_object_after_attach: bool
    gpu_name: str
    cuda_total_bytes: int
    peak_allocated_vram_bytes: int
    peak_reserved_vram_bytes: int
    base_load_seconds: float
    adapter_load_seconds: float
    adapter_status_enabled: AdapterStatusSnapshot
    adapter_status_disabled: AdapterStatusSnapshot
    adapter_status_reenabled: AdapterStatusSnapshot
    prompts: tuple[SmokePromptValidation, ...]

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise AdapterInferenceValidationError(
                f"unsupported report schema version: {self.schema_version}"
            )
        if self.task_id != _TASK_ID:
            raise AdapterInferenceValidationError(f"unexpected task_id: {self.task_id!r}")
        if not self.accepted:
            raise AdapterInferenceValidationError("P7-007 report cannot be emitted as unaccepted")
        if self.base_load_count != 1:
            raise AdapterInferenceValidationError("canonical base must be loaded exactly once")
        if not self.same_base_object_after_attach:
            raise AdapterInferenceValidationError(
                "PEFT attachment rebuilt or replaced the base model"
            )
        if not self.prompts:
            raise AdapterInferenceValidationError("P7-007 requires fixed smoke prompts")
        if any(not item.base_recovered_exactly for item in self.prompts):
            raise AdapterInferenceValidationError(
                "disabled adapter did not recover exact base behavior"
            )
        if any(not item.adapter_reenabled_exactly for item in self.prompts):
            raise AdapterInferenceValidationError(
                "re-enabled adapter did not recover adapted behavior"
            )


class _Tokenizer(Protocol):
    chat_template: str | None

    def apply_chat_template(self, *args: object, **kwargs: object) -> object: ...

    def decode(self, token_ids: list[int], **kwargs: object) -> object: ...


class _GenerateCapable(Protocol):
    def generate(self, **kwargs: object) -> object: ...


def _load_json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdapterInferenceValidationError(f"could not read {label}: {path}") from exc
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise AdapterInferenceValidationError(f"{label} must be a JSON object")
    return cast(dict[str, object], value)


def _require_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise AdapterInferenceValidationError(f"{field} must be an object")
    return cast(Mapping[str, object], value)


def _require_str(mapping: Mapping[str, object], key: str, *, field: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise AdapterInferenceValidationError(f"{field}.{key} must be a non-empty string")
    return value


def _require_int(mapping: Mapping[str, object], key: str, *, field: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise AdapterInferenceValidationError(f"{field}.{key} must be an integer")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise AdapterInferenceValidationError(f"could not hash required artifact: {path}") from exc
    return digest.hexdigest()


def _relative_leaves(names: Sequence[str]) -> tuple[str, ...]:
    leaves = tuple(sorted({name.rsplit(".", maxsplit=1)[-1] for name in names}))
    if not leaves:
        raise AdapterInferenceValidationError("adapter manifest contains no LoRA target modules")
    return leaves


def _require_equal(actual: object, expected: object, *, field: str) -> None:
    if actual != expected:
        raise AdapterInferenceValidationError(
            f"{field} mismatch: observed {actual!r}; expected {expected!r}"
        )


def _persisted_artifact(
    report: Mapping[str, object],
    relative_path: str,
) -> Mapping[str, object]:
    value = report.get("persisted_artifacts")
    if not isinstance(value, list):
        raise AdapterInferenceValidationError("training report persisted_artifacts must be a list")
    matches: list[Mapping[str, object]] = []
    for item in value:
        if isinstance(item, Mapping) and item.get("path") == relative_path:
            matches.append(cast(Mapping[str, object], item))
    if len(matches) != 1:
        raise AdapterInferenceValidationError(
            f"training report must contain exactly one persisted artifact for {relative_path!r}"
        )
    return matches[0]


def validate_adapter_artifacts(
    output_dir: Path,
    base_model: BaseModelIdentity,
) -> VerifiedAdapterArtifacts:
    """Verify identity and hashes before loading the expensive base model."""

    root = output_dir.resolve()
    if not root.is_dir():
        raise AdapterInferenceValidationError(f"training output directory does not exist: {root}")
    for relative in _REQUIRED_OUTPUT_FILES:
        path = root / relative
        if not path.is_file():
            raise AdapterInferenceValidationError(
                f"required P7-006 artifact is missing: {relative}"
            )
    for pattern in _FORBIDDEN_MODEL_PATTERNS:
        matches = tuple((root / "adapter").glob(pattern))
        if matches:
            raise AdapterInferenceValidationError(
                "adapter directory contains forbidden merged/full-model weights: "
                + ", ".join(sorted(path.name for path in matches))
            )

    manifest = load_adapter_manifest(root / "adapter-manifest.json")
    _require_equal(manifest.adapter_id, _EXPECTED_ADAPTER_ID, field="adapter_manifest.adapter_id")
    _require_equal(manifest.family, _EXPECTED_FAMILY, field="adapter_manifest.family")
    _require_equal(manifest.language, _EXPECTED_LANGUAGE, field="adapter_manifest.language")
    _require_equal(
        manifest.base_model.repository,
        base_model.repository,
        field="adapter_manifest.base_model.repository",
    )
    _require_equal(
        manifest.base_model.revision,
        base_model.revision,
        field="adapter_manifest.base_model.revision",
    )
    _require_equal(
        manifest.tokenizer.repository,
        base_model.tokenizer_repository,
        field="adapter_manifest.tokenizer.repository",
    )
    _require_equal(
        manifest.tokenizer.revision,
        base_model.tokenizer_revision,
        field="adapter_manifest.tokenizer.revision",
    )

    training_report = _load_json_object(root / "training-report.json", label="training report")
    training_config = _load_json_object(root / "training-config.json", label="training config")
    run_manifest = _load_json_object(root / "run-manifest.json", label="run manifest")
    adapter_config = _load_json_object(
        root / "adapter" / "adapter_config.json", label="PEFT config"
    )

    _require_equal(
        training_report.get("adapter_id"), manifest.adapter_id, field="training_report.adapter_id"
    )
    _require_equal(
        training_report.get("language"), manifest.language, field="training_report.language"
    )
    _require_equal(training_report.get("global_steps"), 4750, field="training_report.global_steps")
    _require_equal(
        training_report.get("source_training_config"),
        "configs/train/python/p0.yaml",
        field="training_report.source_training_config",
    )
    _require_equal(
        training_report.get("source_training_config_sha256"),
        manifest.training.config_sha256,
        field="training_report.source_training_config_sha256",
    )
    _require_equal(
        training_config.get("config_sha256"),
        manifest.training.config_sha256,
        field="training_config.config_sha256",
    )

    resolved_base = _require_mapping(training_config.get("base"), field="training_config.base")
    _require_equal(
        resolved_base.get("model_repository"),
        base_model.repository,
        field="training_config.base.model_repository",
    )
    _require_equal(
        resolved_base.get("model_revision"),
        base_model.revision,
        field="training_config.base.model_revision",
    )
    _require_equal(
        resolved_base.get("tokenizer_repository"),
        base_model.tokenizer_repository,
        field="training_config.base.tokenizer_repository",
    )
    _require_equal(
        resolved_base.get("tokenizer_revision"),
        base_model.tokenizer_revision,
        field="training_config.base.tokenizer_revision",
    )

    dataset = _require_mapping(training_config.get("dataset"), field="training_config.dataset")
    inference_template_sha = _require_str(
        dataset,
        "chat_template_sha256",
        field="training_config.dataset",
    )
    saved_template = root / "adapter" / "chat_template.jinja"
    _require_equal(
        _sha256(saved_template),
        inference_template_sha,
        field="adapter.chat_template.jinja.sha256",
    )

    run_base = _require_mapping(run_manifest.get("base_model"), field="run_manifest.base_model")
    _require_equal(
        run_base.get("repository"),
        base_model.repository,
        field="run_manifest.base_model.repository",
    )
    _require_equal(
        run_base.get("revision"), base_model.revision, field="run_manifest.base_model.revision"
    )
    _require_equal(
        run_base.get("tokenizer_repository"),
        base_model.tokenizer_repository,
        field="run_manifest.base_model.tokenizer_repository",
    )
    _require_equal(
        run_base.get("tokenizer_revision"),
        base_model.tokenizer_revision,
        field="run_manifest.base_model.tokenizer_revision",
    )
    run_adapter = _require_mapping(run_manifest.get("adapter"), field="run_manifest.adapter")
    _require_equal(
        run_adapter.get("adapter_id"), manifest.adapter_id, field="run_manifest.adapter.adapter_id"
    )
    _require_equal(run_adapter.get("family"), manifest.family, field="run_manifest.adapter.family")
    _require_equal(run_manifest.get("language"), manifest.language, field="run_manifest.language")
    run_git = _require_mapping(run_manifest.get("git"), field="run_manifest.git")
    training_git_sha = _require_str(run_git, "sha", field="run_manifest.git")
    _require_equal(training_git_sha, manifest.training.git_sha, field="run_manifest.git.sha")
    training_run_id = _require_str(run_manifest, "run_id", field="run_manifest")
    _require_equal(training_run_id, manifest.training.run_id, field="run_manifest.run_id")

    _require_equal(
        str(adapter_config.get("peft_type", "")).upper(), "LORA", field="adapter_config.peft_type"
    )
    _require_equal(adapter_config.get("task_type"), "CAUSAL_LM", field="adapter_config.task_type")
    _require_equal(
        adapter_config.get("inference_mode"), True, field="adapter_config.inference_mode"
    )
    _require_equal(
        adapter_config.get("base_model_name_or_path"),
        base_model.repository,
        field="adapter_config.base_model_name_or_path",
    )
    _require_equal(adapter_config.get("r"), manifest.lora.rank, field="adapter_config.r")
    _require_equal(
        adapter_config.get("lora_alpha"), manifest.lora.alpha, field="adapter_config.lora_alpha"
    )
    _require_equal(
        adapter_config.get("lora_dropout"),
        manifest.lora.dropout,
        field="adapter_config.lora_dropout",
    )
    _require_equal(adapter_config.get("bias"), manifest.lora.bias, field="adapter_config.bias")
    _require_equal(
        adapter_config.get("peft_version"),
        manifest.training.peft_version,
        field="adapter_config.peft_version",
    )
    config_targets = adapter_config.get("target_modules")
    if not isinstance(config_targets, list) or any(
        not isinstance(item, str) for item in config_targets
    ):
        raise AdapterInferenceValidationError("adapter_config.target_modules must be a string list")
    _require_equal(
        tuple(sorted(cast(list[str], config_targets))),
        _relative_leaves(manifest.lora.target_modules),
        field="adapter_config.target_modules",
    )

    adapter_model = root / "adapter" / "adapter_model.safetensors"
    adapter_size = adapter_model.stat().st_size
    if adapter_size <= 0:
        raise AdapterInferenceValidationError("adapter_model.safetensors is empty")
    adapter_sha = _sha256(adapter_model)
    persisted_weights = _persisted_artifact(training_report, "adapter/adapter_model.safetensors")
    _require_equal(
        persisted_weights.get("size_bytes"),
        adapter_size,
        field="persisted adapter weight size",
    )
    _require_equal(
        persisted_weights.get("sha256"),
        adapter_sha,
        field="persisted adapter weight sha256",
    )
    persisted_config = _persisted_artifact(training_report, "adapter/adapter_config.json")
    config_path = root / "adapter" / "adapter_config.json"
    _require_equal(
        persisted_config.get("size_bytes"),
        config_path.stat().st_size,
        field="persisted adapter config size",
    )
    _require_equal(
        persisted_config.get("sha256"),
        _sha256(config_path),
        field="persisted adapter config sha256",
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


def _resolved_revision(model: nn.Module) -> str:
    config: object = getattr(model, "config", None)
    value: object = getattr(config, "_commit_hash", None)
    if not isinstance(value, str) or not value:
        raise AdapterInferenceValidationError("loaded base model does not expose resolved revision")
    return value


def _floating_parameter_dtypes(model: nn.Module) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                str(parameter.dtype)
                for parameter in model.parameters()
                if parameter.is_floating_point()
            }
        )
    )


def _prepare_inputs(
    tokenizer: _Tokenizer,
    device: torch.device,
    *,
    system_prompt: str,
    user_prompt: str,
) -> dict[str, torch.Tensor]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=_ENABLE_THINKING,
    )
    if not isinstance(encoded, Mapping):
        raise AdapterInferenceValidationError("chat template did not return a mapping")
    inputs: dict[str, torch.Tensor] = {}
    for key, value in encoded.items():
        if isinstance(value, torch.Tensor):
            inputs[str(key)] = value.to(device)
    ids = inputs.get("input_ids")
    if ids is None or ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] <= 0:
        raise AdapterInferenceValidationError(
            "smoke prompt did not tokenize to one non-empty batch"
        )
    return inputs


def _generate(
    model: nn.Module,
    tokenizer: _Tokenizer,
    device: torch.device,
    *,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int,
) -> GenerationObservation:
    inputs = _prepare_inputs(
        tokenizer,
        device,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    prompt_tokens = int(inputs["input_ids"].shape[1])
    kwargs: dict[str, object] = dict(inputs)
    kwargs.update(
        {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
            "use_cache": True,
        }
    )
    generate_model = cast(_GenerateCapable, model)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = generate_model.generate(**kwargs)
    torch.cuda.synchronize(device)
    latency = time.perf_counter() - started
    if not isinstance(output, torch.Tensor) or output.ndim != 2 or output.shape[0] != 1:
        raise AdapterInferenceValidationError("model.generate returned an unexpected value")
    if output.shape[1] <= prompt_tokens:
        raise AdapterInferenceValidationError("model.generate produced no new tokens")
    suffix = output[0, prompt_tokens:].detach().cpu()
    token_ids = tuple(int(item) for item in suffix.tolist())
    decoded = tokenizer.decode(
        list(token_ids),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    if not isinstance(decoded, str):
        raise AdapterInferenceValidationError("tokenizer.decode returned a non-string response")
    return GenerationObservation(
        text=decoded,
        token_ids=token_ids,
        prompt_tokens=prompt_tokens,
        generated_tokens=len(token_ids),
        latency_seconds=latency,
    )


def _freeze_inference_parameters(model: nn.Module) -> None:
    """Force and verify a fully frozen inference-only parameter state."""

    model.requires_grad_(False)
    trainable_names = tuple(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    if trainable_names:
        preview = ", ".join(trainable_names[:5])
        raise AdapterInferenceValidationError(
            "could not freeze all inference parameters; still trainable: " + preview
        )


def _status_snapshot(model: object) -> AdapterStatusSnapshot:
    getter = getattr(model, "get_model_status", None)
    if not callable(getter):
        raise AdapterInferenceValidationError(
            "loaded PEFT model does not expose get_model_status()"
        )
    status: object = getter()

    def tuple_of_strings(name: str) -> tuple[str, ...]:
        value: object = getattr(status, name, None)
        if isinstance(value, str) or not isinstance(value, Sequence):
            raise AdapterInferenceValidationError(f"PEFT status {name} is not a sequence")
        items = tuple(value)
        if any(not isinstance(item, str) for item in items):
            raise AdapterInferenceValidationError(f"PEFT status {name} contains a non-string")
        return cast(tuple[str, ...], items)

    enabled: object = getattr(status, "enabled", None)
    layer_count: object = getattr(status, "num_adapter_layers", None)
    trainable: object = getattr(status, "trainable_params", None)
    if not isinstance(enabled, bool):
        raise AdapterInferenceValidationError("PEFT status enabled is not boolean")
    if isinstance(layer_count, bool) or not isinstance(layer_count, int) or layer_count <= 0:
        raise AdapterInferenceValidationError("PEFT status num_adapter_layers is not positive")
    if isinstance(trainable, bool) or not isinstance(trainable, int) or trainable < 0:
        raise AdapterInferenceValidationError("PEFT status trainable_params is invalid")
    return AdapterStatusSnapshot(
        enabled=enabled,
        active_adapters=tuple_of_strings("active_adapters"),
        available_adapters=tuple_of_strings("available_adapters"),
        merged_adapters=tuple_of_strings("merged_adapters"),
        num_adapter_layers=layer_count,
        trainable_params=trainable,
    )


def _require_enabled_status(status: AdapterStatusSnapshot) -> None:
    if not status.enabled:
        raise AdapterInferenceValidationError("PEFT adapter layers unexpectedly report disabled")
    if status.active_adapters != ("default",):
        raise AdapterInferenceValidationError(
            f"unexpected active PEFT adapters: {status.active_adapters!r}"
        )
    if "default" not in status.available_adapters:
        raise AdapterInferenceValidationError("default PEFT adapter is not available")
    if status.merged_adapters:
        raise AdapterInferenceValidationError("P7-007 requires an unmerged LoRA adapter")
    if status.trainable_params != 0:
        raise AdapterInferenceValidationError(
            "inference adapter unexpectedly has trainable parameters"
        )


def _require_disabled_status(status: AdapterStatusSnapshot) -> None:
    if status.enabled:
        raise AdapterInferenceValidationError("disable_adapter() did not disable adapter layers")
    if status.merged_adapters:
        raise AdapterInferenceValidationError("disabled adapter unexpectedly reports merged layers")
    if status.trainable_params != 0:
        raise AdapterInferenceValidationError("disabled inference adapter has trainable parameters")


def restore_inference_only_adapter(model: object) -> None:
    """Restore PEFT adapter state after disable_adapter() without rebuilding the base."""

    setter = getattr(model, "set_adapter", None)
    if not callable(setter):
        raise AdapterInferenceValidationError("PEFT model does not expose set_adapter()")
    try:
        setter("default", inference_mode=True)
    except (TypeError, ValueError) as exc:
        raise AdapterInferenceValidationError(
            "could not restore the default adapter in inference-only mode after disable_adapter()"
        ) from exc


def validate_generation_recovery(
    *,
    prompt_id: str,
    prompt: str,
    base: GenerationObservation,
    adapter_enabled: GenerationObservation,
    adapter_disabled: GenerationObservation,
    adapter_reenabled: GenerationObservation,
) -> SmokePromptValidation:
    """Require exact deterministic recovery for disable and re-enable transitions."""

    base_recovered = (
        base.token_ids == adapter_disabled.token_ids and base.text == adapter_disabled.text
    )
    if not base_recovered:
        raise AdapterInferenceValidationError(
            f"prompt {prompt_id!r} did not recover exact base output while adapter was disabled"
        )
    reenabled = (
        adapter_enabled.token_ids == adapter_reenabled.token_ids
        and adapter_enabled.text == adapter_reenabled.text
    )
    if not reenabled:
        raise AdapterInferenceValidationError(
            f"prompt {prompt_id!r} did not recover exact adapted output after re-enable"
        )
    return SmokePromptValidation(
        prompt_id=prompt_id,
        prompt=prompt,
        base=base,
        adapter_enabled=adapter_enabled,
        adapter_disabled=adapter_disabled,
        adapter_reenabled=adapter_reenabled,
        base_recovered_exactly=True,
        adapter_reenabled_exactly=True,
        adapter_changed_output=(
            base.token_ids != adapter_enabled.token_ids or base.text != adapter_enabled.text
        ),
    )


def run_adapter_inference_validation(
    output_dir: Path,
    *,
    base_config: Path = _DEFAULT_BASE_CONFIG,
    device_index: int = 0,
    max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
) -> AdapterInferenceValidationReport:
    """Run the canonical P7-007 validation against a completed P7-006 output directory."""

    if max_new_tokens <= 0:
        raise AdapterInferenceValidationError("max_new_tokens must be greater than zero")
    if not torch.cuda.is_available():
        raise AdapterInferenceValidationError("P7-007 requires CUDA")
    if not torch.cuda.is_bf16_supported():
        raise AdapterInferenceValidationError("P7-007 requires a BF16-capable CUDA device")
    if not 0 <= device_index < torch.cuda.device_count():
        raise AdapterInferenceValidationError(f"invalid CUDA device index: {device_index}")

    base_model = load_base_model_identity(base_config)
    artifacts = validate_adapter_artifacts(output_dir, base_model)

    from peft import PeftModel
    from transformers import AutoModelForMultimodalLM, AutoTokenizer, PreTrainedTokenizerBase

    device = torch.device("cuda", device_index)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    seed_everything(_DEFAULT_SEED)

    tokenizer_obj: object = AutoTokenizer.from_pretrained(
        base_model.tokenizer_repository,
        revision=base_model.tokenizer_revision,
    )
    if not isinstance(tokenizer_obj, PreTrainedTokenizerBase):
        raise AdapterInferenceValidationError(
            "Transformers returned an unexpected tokenizer object"
        )
    tokenizer = cast(_Tokenizer, tokenizer_obj)
    if not isinstance(tokenizer.chat_template, str) or not tokenizer.chat_template:
        raise AdapterInferenceValidationError("canonical tokenizer does not expose a chat template")
    template_sha = hashlib.sha256(tokenizer.chat_template.encode("utf-8")).hexdigest()
    _require_equal(
        template_sha,
        artifacts.inference_chat_template_sha256,
        field="canonical inference chat template sha256",
    )

    model_factory = cast(Any, AutoModelForMultimodalLM)
    base_load_started = time.perf_counter()
    loaded: object = model_factory.from_pretrained(
        base_model.repository,
        revision=base_model.revision,
        dtype=torch.bfloat16,
        device_map={"": device_index},
        low_cpu_mem_usage=True,
    )
    base_load_seconds = time.perf_counter() - base_load_started
    if not isinstance(loaded, nn.Module):
        raise AdapterInferenceValidationError("Transformers returned an unexpected model object")
    base = loaded
    base.eval()
    resolved_revision = _resolved_revision(base)
    _require_equal(resolved_revision, base_model.revision, field="loaded base revision")
    dtypes = _floating_parameter_dtypes(base)
    if dtypes != ("torch.bfloat16",):
        raise AdapterInferenceValidationError(
            "P7-007 requires canonical BF16 floating parameters before adapter attachment; "
            f"observed {dtypes!r}"
        )

    system_prompt = load_python_plugin().spec.config.system_prompt.text
    base_observations = tuple(
        _generate(
            base,
            tokenizer,
            device,
            system_prompt=system_prompt,
            user_prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        for _prompt_id, prompt in _SMOKE_PROMPTS
    )

    base_object_id = id(base)
    peft_factory = cast(Any, PeftModel)
    adapter_load_started = time.perf_counter()
    adapted_obj: object = peft_factory.from_pretrained(
        base,
        str(artifacts.adapter_dir),
        adapter_name="default",
        is_trainable=False,
    )
    adapter_load_seconds = time.perf_counter() - adapter_load_started
    if not isinstance(adapted_obj, nn.Module):
        raise AdapterInferenceValidationError("PEFT returned an unexpected adapted model object")
    adapted = adapted_obj
    adapted.eval()
    _freeze_inference_parameters(adapted)
    get_base = getattr(adapted, "get_base_model", None)
    if not callable(get_base):
        raise AdapterInferenceValidationError("PEFT model does not expose get_base_model()")
    same_base_object = id(get_base()) == base_object_id
    if not same_base_object:
        raise AdapterInferenceValidationError(
            "PEFT attachment did not preserve the loaded base object"
        )

    enabled_status = _status_snapshot(adapted)
    _require_enabled_status(enabled_status)
    adapted_observations = tuple(
        _generate(
            adapted,
            tokenizer,
            device,
            system_prompt=system_prompt,
            user_prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        for _prompt_id, prompt in _SMOKE_PROMPTS
    )

    disable = getattr(adapted, "disable_adapter", None)
    if not callable(disable):
        raise AdapterInferenceValidationError("PEFT model does not expose disable_adapter()")
    with disable():
        disabled_status = _status_snapshot(adapted)
        _require_disabled_status(disabled_status)
        disabled_observations = tuple(
            _generate(
                adapted,
                tokenizer,
                device,
                system_prompt=system_prompt,
                user_prompt=prompt,
                max_new_tokens=max_new_tokens,
            )
            for _prompt_id, prompt in _SMOKE_PROMPTS
        )

    # PEFT 0.20 disable_adapter() re-enables through inference_mode=False.
    # Restore the public PEFT inference contract, then independently verify all params are frozen.
    restore_inference_only_adapter(adapted)
    _freeze_inference_parameters(adapted)
    reenabled_status = _status_snapshot(adapted)
    _require_enabled_status(reenabled_status)
    reenabled_observations = tuple(
        _generate(
            adapted,
            tokenizer,
            device,
            system_prompt=system_prompt,
            user_prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        for _prompt_id, prompt in _SMOKE_PROMPTS
    )

    prompt_reports = tuple(
        validate_generation_recovery(
            prompt_id=prompt_id,
            prompt=prompt,
            base=base_observations[index],
            adapter_enabled=adapted_observations[index],
            adapter_disabled=disabled_observations[index],
            adapter_reenabled=reenabled_observations[index],
        )
        for index, (prompt_id, prompt) in enumerate(_SMOKE_PROMPTS)
    )
    torch.cuda.synchronize(device)
    total_memory = torch.cuda.get_device_properties(device).total_memory
    return AdapterInferenceValidationReport(
        schema_version=_SCHEMA_VERSION,
        task_id=_TASK_ID,
        accepted=True,
        base_model=base_model,
        resolved_model_revision=resolved_revision,
        adapter_id=artifacts.manifest.adapter_id,
        adapter_model_sha256=artifacts.adapter_model_sha256,
        adapter_model_size_bytes=artifacts.adapter_model_size_bytes,
        source_training_run_id=artifacts.training_run_id,
        source_training_git_sha=artifacts.training_git_sha,
        base_load_count=1,
        same_base_object_after_attach=True,
        gpu_name=torch.cuda.get_device_name(device),
        cuda_total_bytes=total_memory,
        peak_allocated_vram_bytes=torch.cuda.max_memory_allocated(device),
        peak_reserved_vram_bytes=torch.cuda.max_memory_reserved(device),
        base_load_seconds=base_load_seconds,
        adapter_load_seconds=adapter_load_seconds,
        adapter_status_enabled=enabled_status,
        adapter_status_disabled=disabled_status,
        adapter_status_reenabled=reenabled_status,
        prompts=prompt_reports,
    )


def adapter_inference_validation_json(report: AdapterInferenceValidationReport) -> str:
    """Serialize P7-007 acceptance evidence deterministically."""

    return json.dumps(asdict(report), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_adapter_inference_validation_report(
    report: AdapterInferenceValidationReport,
    output: Path,
) -> Path:
    """Atomically persist the accepted P7-007 report."""

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(adapter_inference_validation_json(report), encoding="utf-8")
    temporary.replace(output)
    return output


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate P7-007 LoRA load, deterministic inference, disable, and re-enable."
    )
    parser.add_argument(
        "--training-output",
        type=Path,
        required=True,
        help="Extracted canonical P7-006 artifacts/train/python/p0 directory.",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=_DEFAULT_BASE_CONFIG,
        help="Canonical exact-revision base config.",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="CUDA device index.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=_DEFAULT_MAX_NEW_TOKENS,
        help="Bounded generated-token budget per fixed smoke prompt.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="Machine-readable acceptance report path.",
    )
    return parser


def adapter_inference_validation_main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point for P7-007."""

    args = _argument_parser().parse_args(argv)
    report = run_adapter_inference_validation(
        cast(Path, args.training_output),
        base_config=cast(Path, args.base_config),
        device_index=cast(int, args.device_index),
        max_new_tokens=cast(int, args.max_new_tokens),
    )
    output = write_adapter_inference_validation_report(report, cast(Path, args.output))
    print(adapter_inference_validation_json(report), end="")
    print(f"P7-007 report: {output}")


if __name__ == "__main__":
    adapter_inference_validation_main()
