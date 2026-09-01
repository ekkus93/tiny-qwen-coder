"""Canonical CUDA generation with the accepted Python P0 LoRA adapter."""

from __future__ import annotations

import hashlib
import importlib.metadata
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import torch
from torch import nn

from tiny_qwen_coder.evaluation._baseline_types import (
    BaselineGeneratedResponse,
    BaselineRuntimeMetadata,
    BaselineSuitePerformance,
)
from tiny_qwen_coder.evaluation.results import GenerationStats
from tiny_qwen_coder.evaluation.settings import FrozenEvaluationSettings
from tiny_qwen_coder.identities import BaseModelIdentity
from tiny_qwen_coder.reproducibility import seed_everything

EXPECTED_ADAPTER_ID = "language/python/p0"
EXPECTED_ADAPTER_FAMILY = "language"
EXPECTED_ADAPTER_SHA256 = "c94606250e112f72362eb883a55f7b2c8af854d445f6bb6194352c2806a8f276"
EXPECTED_ADAPTER_SIZE_BYTES = 65_004_840
EXPECTED_TRAINING_GIT_SHA = "02df92a9c2d347b9fb013dc25714fe066c6bcafe"
EXPECTED_TRAINING_RUN_ID = "training-python-20260831T180916446466Z-02df92a9-eafc119d"
_ENABLE_THINKING = False


class PythonP0GenerationError(RuntimeError):
    """Raised when canonical Python P0 generation cannot be proven safe/correct."""


@dataclass(frozen=True, slots=True)
class VerifiedPythonP0Adapter:
    """Exact immutable identity of the adapter attached for evaluation."""

    adapter_dir: Path
    adapter_id: str
    family: str
    adapter_model_sha256: str
    adapter_model_size_bytes: int
    training_run_id: str
    training_git_sha: str
    inference_chat_template_sha256: str


class _GenerateCapable(Protocol):
    def generate(self, **kwargs: object) -> torch.Tensor: ...


class _Tokenizer(Protocol):
    chat_template: object

    def apply_chat_template(self, *args: object, **kwargs: object) -> object: ...

    def decode(self, token_ids: list[int], **kwargs: object) -> object: ...


def _resolved_revision(model: nn.Module) -> str:
    config: object = getattr(model, "config", None)
    value: object = getattr(config, "_commit_hash", None)
    if not isinstance(value, str) or not value:
        raise PythonP0GenerationError("loaded base model does not expose resolved revision")
    return value


def _parameter_dtypes(model: nn.Module) -> tuple[str, ...]:
    values = {
        str(parameter.dtype) for parameter in model.parameters() if parameter.is_floating_point()
    }
    if not values:
        raise PythonP0GenerationError("adapted model exposes no floating parameters")
    return tuple(sorted(values))


def _qualified_class_name(value: object) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _status_value(status: object, name: str) -> object:
    return getattr(status, name, None)


def _status_strings(status: object, name: str) -> tuple[str, ...]:
    value = _status_value(status, name)
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise PythonP0GenerationError(f"PEFT status {name} is not a sequence")
    items = tuple(value)
    if any(not isinstance(item, str) for item in items):
        raise PythonP0GenerationError(f"PEFT status {name} contains a non-string")
    return cast(tuple[str, ...], items)


def _validate_peft_status(model: object) -> None:
    getter = getattr(model, "get_model_status", None)
    if not callable(getter):
        raise PythonP0GenerationError("loaded PEFT model does not expose get_model_status()")
    status = getter()
    if _status_value(status, "enabled") is not True:
        raise PythonP0GenerationError("Python P0 adapter is not enabled")
    active = _status_strings(status, "active_adapters")
    available = _status_strings(status, "available_adapters")
    merged = _status_strings(status, "merged_adapters")
    trainable = _status_value(status, "trainable_params")
    if active != ("default",):
        raise PythonP0GenerationError(f"unexpected active adapters: {active!r}")
    if "default" not in available:
        raise PythonP0GenerationError("default Python P0 adapter is not available")
    if merged:
        raise PythonP0GenerationError("P8-001 requires an unmerged LoRA adapter")
    if isinstance(trainable, bool) or not isinstance(trainable, int) or trainable != 0:
        raise PythonP0GenerationError("evaluation adapter unexpectedly has trainable parameters")


def validate_python_p0_adapter(
    training_output: Path,
    base_model: BaseModelIdentity,
) -> VerifiedPythonP0Adapter:
    """Reuse P7-007's fail-closed artifact validation and pin the accepted P7-006 weights."""

    try:
        from tiny_qwen_coder.runtime.adapter_validation import (
            AdapterInferenceValidationError,
            validate_adapter_artifacts,
        )
    except ImportError as exc:
        raise PythonP0GenerationError("P7-007 adapter validator is unavailable") from exc
    try:
        artifacts = validate_adapter_artifacts(training_output, base_model)
    except AdapterInferenceValidationError as exc:
        raise PythonP0GenerationError(f"P7-006 adapter validation failed: {exc}") from exc

    manifest = artifacts.manifest
    if manifest.adapter_id != EXPECTED_ADAPTER_ID or manifest.family != EXPECTED_ADAPTER_FAMILY:
        raise PythonP0GenerationError("training output is not the canonical Python P0 adapter")
    if artifacts.adapter_model_sha256 != EXPECTED_ADAPTER_SHA256:
        raise PythonP0GenerationError("Python P0 adapter SHA-256 does not match accepted P7-006")
    if artifacts.adapter_model_size_bytes != EXPECTED_ADAPTER_SIZE_BYTES:
        raise PythonP0GenerationError("Python P0 adapter byte size does not match accepted P7-006")
    if artifacts.training_git_sha != EXPECTED_TRAINING_GIT_SHA:
        raise PythonP0GenerationError("Python P0 adapter training Git SHA is not canonical")
    if artifacts.training_run_id != EXPECTED_TRAINING_RUN_ID:
        raise PythonP0GenerationError("Python P0 adapter training run ID is not canonical")
    return VerifiedPythonP0Adapter(
        adapter_dir=artifacts.adapter_dir,
        adapter_id=manifest.adapter_id,
        family=manifest.family,
        adapter_model_sha256=artifacts.adapter_model_sha256,
        adapter_model_size_bytes=artifacts.adapter_model_size_bytes,
        training_run_id=artifacts.training_run_id,
        training_git_sha=artifacts.training_git_sha,
        inference_chat_template_sha256=artifacts.inference_chat_template_sha256,
    )


class HuggingFacePythonP0Generator:
    """Load the canonical base once, attach exact P0 weights, and generate greedily on CUDA."""

    def __init__(
        self,
        *,
        training_output: Path,
        base_model: BaseModelIdentity,
        settings: FrozenEvaluationSettings,
        device_index: int = 0,
    ) -> None:
        if not torch.cuda.is_available():
            raise PythonP0GenerationError("P8-001 generation requires CUDA")
        if not torch.cuda.is_bf16_supported():
            raise PythonP0GenerationError("P8-001 requires a BF16-capable CUDA device")
        if not 0 <= device_index < torch.cuda.device_count():
            raise PythonP0GenerationError(f"invalid CUDA device index: {device_index}")
        if settings.generation.decoding_strategy != "greedy":
            raise PythonP0GenerationError("P8-001 requires frozen greedy generation")

        self.adapter = validate_python_p0_adapter(training_output, base_model)
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
            raise PythonP0GenerationError("Transformers returned an unexpected tokenizer object")
        tokenizer = cast(_Tokenizer, tokenizer_obj)
        if not isinstance(tokenizer.chat_template, str) or not tokenizer.chat_template:
            raise PythonP0GenerationError("canonical tokenizer does not expose a chat template")
        template_sha = hashlib.sha256(tokenizer.chat_template.encode("utf-8")).hexdigest()
        if template_sha != self.adapter.inference_chat_template_sha256:
            raise PythonP0GenerationError("inference chat template does not match P7-006 training")
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
            raise PythonP0GenerationError("Transformers returned an unexpected model object")
        base = loaded
        base.eval()
        if _resolved_revision(base) != base_model.revision:
            raise PythonP0GenerationError("loaded base revision is not canonical")
        base_dtypes = _parameter_dtypes(base)
        if base_dtypes != ("torch.bfloat16",):
            raise PythonP0GenerationError(
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
            raise PythonP0GenerationError("PEFT returned an unexpected model object")
        self._model = adapted_obj
        self._model.eval()
        self._model.requires_grad_(False)
        _validate_peft_status(self._model)
        self._adapter_load_seconds = time.perf_counter() - adapter_started
        torch.cuda.synchronize(self._device)

        free_after, total_after = torch.cuda.mem_get_info(self._device)
        if total_after != total_bytes:
            raise PythonP0GenerationError("CUDA total memory changed unexpectedly during load")
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

    def _prepare_inputs(self, system_prompt: str, user_prompt: str) -> dict[str, torch.Tensor]:
        encoded = self._tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=_ENABLE_THINKING,
        )
        if not isinstance(encoded, Mapping):
            raise PythonP0GenerationError("chat template did not return a mapping")
        inputs = {
            str(key): value.to(self._device)
            for key, value in encoded.items()
            if isinstance(value, torch.Tensor)
        }
        ids = inputs.get("input_ids")
        if ids is None or ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] <= 0:
            raise PythonP0GenerationError(
                "evaluation prompt did not tokenize to one non-empty batch"
            )
        return inputs

    def generate(self, *, system_prompt: str, user_prompt: str) -> BaselineGeneratedResponse:
        """Generate one deterministic adapted response under the frozen P6 decoding contract."""

        inputs = self._prepare_inputs(system_prompt, user_prompt)
        prompt_tokens = int(inputs["input_ids"].shape[1])
        kwargs: dict[str, object] = dict(inputs)
        kwargs.update(
            {
                "max_new_tokens": self._settings.generation.max_new_tokens,
                "do_sample": False,
                "num_beams": 1,
                "use_cache": True,
            }
        )
        torch.cuda.reset_peak_memory_stats(self._device)
        torch.cuda.synchronize(self._device)
        started = time.perf_counter()
        with torch.inference_mode():
            output = cast(_GenerateCapable, self._model).generate(**kwargs)
        torch.cuda.synchronize(self._device)
        latency = time.perf_counter() - started
        self._generation_peak_allocated_bytes = max(
            self._generation_peak_allocated_bytes, torch.cuda.max_memory_allocated(self._device)
        )
        self._generation_peak_reserved_bytes = max(
            self._generation_peak_reserved_bytes, torch.cuda.max_memory_reserved(self._device)
        )
        if not isinstance(output, torch.Tensor) or output.ndim != 2 or output.shape[0] != 1:
            raise PythonP0GenerationError("model.generate returned an unexpected tensor")
        if output.shape[1] <= prompt_tokens:
            raise PythonP0GenerationError("model.generate produced no new tokens")
        suffix = output[0, prompt_tokens:].detach().cpu()
        token_ids = [int(item) for item in suffix.tolist()]
        decoded = self._tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not isinstance(decoded, str):
            raise PythonP0GenerationError("tokenizer.decode returned a non-string response")
        generated_tokens = len(token_ids)
        if generated_tokens <= 0 or latency <= 0:
            raise PythonP0GenerationError("generation produced invalid token/timing measurements")
        return BaselineGeneratedResponse(
            generated_text=decoded,
            generation=GenerationStats(
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                latency_seconds=latency,
                tokens_per_second=generated_tokens / latency,
            ),
        )

    def runtime_metadata(
        self,
        suites: Sequence[BaselineSuitePerformance],
    ) -> BaselineRuntimeMetadata:
        """Return bounded CUDA/runtime evidence for the adapted generation pass."""

        suite_items = tuple(suites)
        total_requests = sum(item.requests for item in suite_items)
        total_prompt_tokens = sum(item.prompt_tokens for item in suite_items)
        total_generated_tokens = sum(item.generated_tokens for item in suite_items)
        total_latency = sum(item.generation_latency_seconds for item in suite_items)
        if total_requests != 675 or total_generated_tokens <= 0 or total_latency <= 0:
            raise PythonP0GenerationError("P8-001 runtime totals are incomplete")
        capability = torch.cuda.get_device_capability(self._device)
        return BaselineRuntimeMetadata(
            schema_version=1,
            device=str(self._device),
            gpu_name=torch.cuda.get_device_name(self._device),
            gpu_compute_capability=f"{capability[0]}.{capability[1]}",
            torch_version=torch.__version__,
            transformers_version=importlib.metadata.version("transformers"),
            model_class=_qualified_class_name(self._model),
            parameter_dtypes=self._parameter_dtypes,
            resolved_model_revision=self._resolved_model_revision,
            cuda_total_bytes=self._cuda_total_bytes,
            cuda_free_before_load_bytes=self._cuda_free_before_load_bytes,
            cuda_free_after_load_bytes=self._cuda_free_after_load_bytes,
            torch_allocated_after_load_bytes=self._torch_allocated_after_load_bytes,
            torch_reserved_after_load_bytes=self._torch_reserved_after_load_bytes,
            load_peak_allocated_bytes=self._load_peak_allocated_bytes,
            load_peak_reserved_bytes=self._load_peak_reserved_bytes,
            generation_peak_allocated_bytes=self._generation_peak_allocated_bytes,
            generation_peak_reserved_bytes=self._generation_peak_reserved_bytes,
            model_load_seconds=self._base_load_seconds + self._adapter_load_seconds,
            total_wall_seconds=time.perf_counter() - self._started_wall,
            total_requests=total_requests,
            total_prompt_tokens=total_prompt_tokens,
            total_generated_tokens=total_generated_tokens,
            total_generation_latency_seconds=total_latency,
            overall_tokens_per_second=total_generated_tokens / total_latency,
            suites=suite_items,
        )
