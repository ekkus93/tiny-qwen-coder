"""vLLM-backed Qwen teacher generation loaded lazily for Colab/GPU use."""

from __future__ import annotations

import importlib.metadata
from typing import Any

from tiny_qwen_coder.data.records import TrainingMessage
from tiny_qwen_coder.distillation.config import TeacherDistillationConfig
from tiny_qwen_coder.distillation.generation import (
    TeacherCompletion,
    TeacherGenerationError,
)


class VllmTeacherBackend:
    """Batch Qwen3.8 inference using vLLM with BF16 or optional 4-bit weights."""

    @staticmethod
    def _verify_runtime(config: TeacherDistillationConfig) -> None:
        required = {"vllm": config.runtime.vllm_version}
        if config.teacher.quantization == "bitsandbytes":
            required.update(
                {
                    "vllm-bnb-plugin": config.runtime.vllm_bnb_plugin_version,
                    "bitsandbytes": config.runtime.bitsandbytes_version,
                }
            )
        for distribution, expected in required.items():
            try:
                actual = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError as exc:
                raise TeacherGenerationError(
                    f"required teacher runtime package is not installed: {distribution}=={expected}"
                ) from exc
            if actual != expected:
                raise TeacherGenerationError(
                    f"teacher runtime requires {distribution}=={expected}; found {actual}"
                )

    def __init__(self, config: TeacherDistillationConfig) -> None:
        self._verify_runtime(config)
        try:
            from vllm import LLM, SamplingParams  # type: ignore[import-not-found]
        except ImportError as exc:
            raise TeacherGenerationError(
                "vLLM is unavailable; Colab setup must install the pinned teacher runtime "
                "before generation"
            ) from exc
        self._sampling_params_factory: Any = SamplingParams
        try:
            self._llm: Any = LLM(
                model=config.teacher.repository,
                revision=config.teacher.revision,
                dtype=config.teacher.dtype,
                quantization=(
                    None if config.teacher.quantization == "none" else config.teacher.quantization
                ),
                max_model_len=config.teacher.max_model_len,
                gpu_memory_utilization=config.teacher.gpu_memory_utilization,
                language_model_only=True,
                enable_prefix_caching=True,
                max_num_seqs=config.checkpoint.shard_size,
                trust_remote_code=False,
            )
        except Exception as exc:
            raise TeacherGenerationError(f"could not initialize vLLM teacher: {exc}") from exc
        self._config = config

    def generate(
        self,
        conversations: tuple[tuple[TrainingMessage, ...], ...],
        *,
        seeds: tuple[int, ...],
    ) -> tuple[TeacherCompletion, ...]:
        if len(conversations) != len(seeds):
            raise TeacherGenerationError("conversation and seed counts must match")
        messages: list[list[dict[str, str]]] = [
            [{"role": message.role, "content": message.content} for message in conversation]
            for conversation in conversations
        ]
        generation = self._config.generation
        params = [
            self._sampling_params_factory(
                temperature=generation.temperature,
                top_p=generation.top_p,
                top_k=generation.top_k,
                min_p=generation.min_p,
                presence_penalty=generation.presence_penalty,
                repetition_penalty=generation.repetition_penalty,
                max_tokens=generation.max_tokens,
                seed=seed,
            )
            for seed in seeds
        ]
        try:
            outputs: list[Any] = self._llm.chat(
                messages,
                sampling_params=params,
                use_tqdm=True,
                chat_template_kwargs={
                    "enable_thinking": generation.thinking,
                    "preserve_thinking": generation.preserve_thinking,
                    "reasoning_effort": generation.reasoning_effort,
                },
            )
        except Exception as exc:
            raise TeacherGenerationError(f"vLLM teacher generation failed: {exc}") from exc
        if len(outputs) != len(conversations):
            raise TeacherGenerationError("vLLM returned an unexpected number of outputs")

        completions: list[TeacherCompletion] = []
        for output in outputs:
            choices = getattr(output, "outputs", None)
            if not isinstance(choices, list) or len(choices) != 1:
                raise TeacherGenerationError("vLLM output must contain exactly one completion")
            choice = choices[0]
            text = getattr(choice, "text", None)
            token_ids = getattr(choice, "token_ids", None)
            prompt_token_ids = getattr(output, "prompt_token_ids", None)
            finish_reason = getattr(choice, "finish_reason", None)
            if not isinstance(text, str):
                raise TeacherGenerationError("vLLM completion text is not a string")
            completions.append(
                TeacherCompletion(
                    text=text,
                    finish_reason=str(finish_reason or "unknown"),
                    prompt_tokens=len(prompt_token_ids) if prompt_token_ids is not None else 0,
                    completion_tokens=len(token_ids) if token_ids is not None else 0,
                )
            )
        return tuple(completions)
