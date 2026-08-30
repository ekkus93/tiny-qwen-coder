"""Canonical CUDA-backed generation for the unchanged-base Python baseline."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Protocol, cast

import torch
from torch import nn

from tiny_qwen_coder.evaluation._baseline_types import (
    BaselineGeneratedResponse,
    BaselineRuntimeMetadata,
    BaselineSuitePerformance,
    PythonBaselineError,
)
from tiny_qwen_coder.evaluation.results import GenerationStats
from tiny_qwen_coder.evaluation.settings import FrozenEvaluationSettings
from tiny_qwen_coder.identities import BaseModelIdentity
from tiny_qwen_coder.reproducibility import seed_everything

_BASELINE_ENABLE_THINKING = False


class BaselineGenerator(Protocol):
    """Minimal generation backend required by the baseline orchestration layer."""

    def generate(self, *, system_prompt: str, user_prompt: str) -> BaselineGeneratedResponse: ...


class _GenerateCapable(Protocol):
    def generate(self, **kwargs: object) -> torch.Tensor: ...


class _Tokenizer(Protocol):
    def apply_chat_template(self, *args: object, **kwargs: object) -> object: ...

    def decode(self, token_ids: list[int], **kwargs: object) -> object: ...


def _qualified_class_name(value: object) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _resolved_revision(model: nn.Module) -> str:
    config: object = getattr(model, "config", None)
    revision: object = getattr(config, "_commit_hash", None)
    if not isinstance(revision, str) or not revision:
        raise PythonBaselineError("loaded base model does not expose its resolved revision")
    return revision


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


def generation_contract_sha256(
    *,
    base_model: BaseModelIdentity,
    settings: FrozenEvaluationSettings,
    system_prompt_version: str,
    system_prompt: str,
) -> str:
    """Hash the exact model, decoding, template, and system-prompt generation contract."""

    payload = json.dumps(
        {
            "base_model": asdict(base_model),
            "settings": asdict(settings),
            "system_prompt_version": system_prompt_version,
            "system_prompt": system_prompt,
            "chat_template_kwargs": {"enable_thinking": _BASELINE_ENABLE_THINKING},
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prompt_sha256(*, item_id: str, user_prompt: str) -> str:
    """Bind one stable item identity to the exact user prompt presented to the model."""

    payload = json.dumps(
        {"item_id": item_id, "user_prompt": user_prompt},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class HuggingFaceBaselineGenerator:
    """Load the exact pinned Qwen3.5-4B base once and generate greedily on CUDA."""

    def __init__(
        self,
        *,
        base_model: BaseModelIdentity,
        settings: FrozenEvaluationSettings,
        device_index: int = 0,
    ) -> None:
        if not torch.cuda.is_available():
            raise PythonBaselineError("P6-005 unchanged-base baseline requires CUDA")
        if not 0 <= device_index < torch.cuda.device_count():
            raise PythonBaselineError(f"invalid CUDA device index: {device_index}")
        if settings.generation.decoding_strategy != "greedy":
            raise PythonBaselineError("P6-005 requires frozen greedy generation")

        from transformers import AutoModelForMultimodalLM, AutoTokenizer, PreTrainedTokenizerBase

        self._base_model = base_model
        self._settings = settings
        self._device = torch.device("cuda", device_index)
        seed_everything(settings.seed)
        torch.cuda.set_device(self._device)
        torch.cuda.empty_cache()
        torch.cuda.synchronize(self._device)
        free_before_load, total_bytes = torch.cuda.mem_get_info(self._device)
        torch.cuda.reset_peak_memory_stats(self._device)

        started = time.perf_counter()
        tokenizer_obj: object = AutoTokenizer.from_pretrained(
            base_model.tokenizer_repository,
            revision=base_model.tokenizer_revision,
        )
        if not isinstance(tokenizer_obj, PreTrainedTokenizerBase):
            raise PythonBaselineError("Transformers returned an unexpected tokenizer object")
        loaded: object = AutoModelForMultimodalLM.from_pretrained(
            base_model.repository,
            revision=base_model.revision,
            dtype=torch.bfloat16,
            device_map={"": device_index},
            low_cpu_mem_usage=True,
        )
        if not isinstance(loaded, nn.Module):
            raise PythonBaselineError("Transformers returned an unexpected model object")
        self._model = loaded
        self._tokenizer = cast(_Tokenizer, tokenizer_obj)
        self._model.eval()
        torch.cuda.synchronize(self._device)
        self._load_seconds = time.perf_counter() - started

        resolved_revision = _resolved_revision(self._model)
        if resolved_revision != base_model.revision:
            raise PythonBaselineError(
                "loaded base revision does not match the canonical revision: "
                f"{resolved_revision} != {base_model.revision}"
            )
        parameter_dtypes = _floating_parameter_dtypes(self._model)
        if parameter_dtypes != ("torch.bfloat16",):
            raise PythonBaselineError(
                "canonical unchanged-base evaluation requires all floating parameters in BF16; "
                f"observed {parameter_dtypes!r}"
            )

        free_after_load, total_after_load = torch.cuda.mem_get_info(self._device)
        if total_after_load != total_bytes:
            raise PythonBaselineError("CUDA total memory changed unexpectedly during model load")
        self._cuda_total_bytes = total_bytes
        self._cuda_free_before_load_bytes = free_before_load
        self._cuda_free_after_load_bytes = free_after_load
        self._torch_allocated_after_load_bytes = torch.cuda.memory_allocated(self._device)
        self._torch_reserved_after_load_bytes = torch.cuda.memory_reserved(self._device)
        self._load_peak_allocated_bytes = torch.cuda.max_memory_allocated(self._device)
        self._load_peak_reserved_bytes = torch.cuda.max_memory_reserved(self._device)
        self._generation_peak_allocated_bytes = self._torch_allocated_after_load_bytes
        self._generation_peak_reserved_bytes = self._torch_reserved_after_load_bytes
        self._resolved_model_revision = resolved_revision
        self._parameter_dtypes = parameter_dtypes
        self._started_wall = time.perf_counter()

    @property
    def device(self) -> torch.device:
        return self._device

    def _prepare_inputs(self, system_prompt: str, user_prompt: str) -> dict[str, torch.Tensor]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        encoded: object = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=_BASELINE_ENABLE_THINKING,
        )
        if not isinstance(encoded, Mapping):
            raise PythonBaselineError("chat template did not return a mapping")
        inputs: dict[str, torch.Tensor] = {}
        for key, value in encoded.items():
            if isinstance(value, torch.Tensor):
                inputs[str(key)] = value.to(self._device)
        input_ids = inputs.get("input_ids")
        if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise PythonBaselineError("baseline prompt did not tokenize to one batch")
        if input_ids.shape[1] <= 0:
            raise PythonBaselineError("baseline prompt tokenized to zero tokens")
        return inputs

    def generate(self, *, system_prompt: str, user_prompt: str) -> BaselineGeneratedResponse:
        """Generate exactly one greedy response and measure its latency/throughput."""

        inputs = self._prepare_inputs(system_prompt, user_prompt)
        input_ids = inputs["input_ids"]
        prompt_tokens = int(input_ids.shape[1])
        torch.cuda.reset_peak_memory_stats(self._device)
        generate_model = cast(_GenerateCapable, self._model)
        kwargs: dict[str, object] = dict(inputs)
        kwargs.update(
            {
                "max_new_tokens": self._settings.generation.max_new_tokens,
                "do_sample": False,
                "num_beams": 1,
                "use_cache": True,
            }
        )
        torch.cuda.synchronize(self._device)
        started = time.perf_counter()
        with torch.inference_mode():
            output: object = generate_model.generate(**kwargs)
        torch.cuda.synchronize(self._device)
        latency = time.perf_counter() - started
        self._generation_peak_allocated_bytes = max(
            self._generation_peak_allocated_bytes,
            torch.cuda.max_memory_allocated(self._device),
        )
        self._generation_peak_reserved_bytes = max(
            self._generation_peak_reserved_bytes,
            torch.cuda.max_memory_reserved(self._device),
        )
        if not isinstance(output, torch.Tensor) or output.ndim != 2 or output.shape[0] != 1:
            raise PythonBaselineError("model.generate returned an unexpected tensor")
        if output.shape[1] <= prompt_tokens:
            raise PythonBaselineError("model.generate produced no new tokens")
        suffix = output[0, prompt_tokens:].detach().cpu()
        token_ids = tuple(int(token_id) for token_id in suffix.tolist())
        decoded: object = self._tokenizer.decode(
            list(token_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not isinstance(decoded, str):
            raise PythonBaselineError("tokenizer.decode returned a non-string response")
        generated_tokens = len(token_ids)
        tokens_per_second = generated_tokens / latency if latency > 0 else 0.0
        return BaselineGeneratedResponse(
            generated_text=decoded,
            generation=GenerationStats(
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                latency_seconds=latency,
                tokens_per_second=tokens_per_second,
            ),
        )

    def runtime_metadata(
        self,
        suite_performance: Sequence[BaselineSuitePerformance],
    ) -> BaselineRuntimeMetadata:
        """Finalize measured model-load, CUDA-memory, and generation-throughput metadata."""

        suites = tuple(suite_performance)
        total_requests = sum(item.requests for item in suites)
        total_prompt_tokens = sum(item.prompt_tokens for item in suites)
        total_generated_tokens = sum(item.generated_tokens for item in suites)
        total_latency = sum(item.generation_latency_seconds for item in suites)
        overall_tps = total_generated_tokens / total_latency if total_latency > 0 else 0.0
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
            model_load_seconds=self._load_seconds,
            total_wall_seconds=time.perf_counter() - self._started_wall,
            total_requests=total_requests,
            total_prompt_tokens=total_prompt_tokens,
            total_generated_tokens=total_generated_tokens,
            total_generation_latency_seconds=total_latency,
            overall_tokens_per_second=overall_tps,
            suites=suites,
        )
