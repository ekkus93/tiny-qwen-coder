"""Strict configuration for resumable teacher-data distillation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

import yaml

TeacherBackendName = Literal["vllm"]
TeacherDtype = Literal["bfloat16"]
TeacherQuantization = Literal["bitsandbytes", "none"]
TeacherReasoningEffort = Literal["low", "medium", "xhigh"]


class TeacherDistillationConfigError(ValueError):
    """Raised when a teacher-distillation config is invalid."""


@dataclass(frozen=True, slots=True)
class TeacherRuntimeConfig:
    """Exact inference-package versions required for reproducible resumed shards."""

    vllm_version: str
    vllm_bnb_plugin_version: str
    bitsandbytes_version: str

    def __post_init__(self) -> None:
        for field_name, value in (
            ("vllm_version", self.vllm_version),
            ("vllm_bnb_plugin_version", self.vllm_bnb_plugin_version),
            ("bitsandbytes_version", self.bitsandbytes_version),
        ):
            if not value.strip():
                raise TeacherDistillationConfigError(f"runtime.{field_name} must not be empty")


@dataclass(frozen=True, slots=True)
class TeacherModelConfig:
    """Pinned teacher-model and inference-runtime identity."""

    repository: str
    revision: str
    backend: TeacherBackendName
    dtype: TeacherDtype
    quantization: TeacherQuantization
    max_model_len: int
    gpu_memory_utilization: float

    def __post_init__(self) -> None:
        if not self.repository.strip():
            raise TeacherDistillationConfigError("teacher.repository must not be empty")
        if len(self.revision) != 40 or any(c not in "0123456789abcdef" for c in self.revision):
            raise TeacherDistillationConfigError(
                "teacher.revision must be a pinned lowercase 40-character Git SHA"
            )
        if self.max_model_len <= 0:
            raise TeacherDistillationConfigError("teacher.max_model_len must be greater than zero")
        if not 0.0 < self.gpu_memory_utilization < 1.0:
            raise TeacherDistillationConfigError(
                "teacher.gpu_memory_utilization must be greater than zero and less than one"
            )


@dataclass(frozen=True, slots=True)
class TeacherGenerationConfig:
    """Frozen sampling contract for one candidate answer per source prompt."""

    thinking: bool
    preserve_thinking: bool
    reasoning_effort: TeacherReasoningEffort
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    presence_penalty: float
    repetition_penalty: float
    max_tokens: int
    seed: int

    def __post_init__(self) -> None:
        if self.reasoning_effort not in {"low", "medium", "xhigh"}:
            raise TeacherDistillationConfigError(
                "generation.reasoning_effort must be one of: low, medium, xhigh"
            )
        if self.temperature < 0:
            raise TeacherDistillationConfigError("generation.temperature must be non-negative")
        if not 0.0 < self.top_p <= 1.0:
            raise TeacherDistillationConfigError("generation.top_p must be in (0, 1]")
        if self.top_k < 0:
            raise TeacherDistillationConfigError("generation.top_k must be non-negative")
        if not 0.0 <= self.min_p <= 1.0:
            raise TeacherDistillationConfigError("generation.min_p must be in [0, 1]")
        if self.presence_penalty < -2.0 or self.presence_penalty > 2.0:
            raise TeacherDistillationConfigError(
                "generation.presence_penalty must be between -2 and 2"
            )
        if self.repetition_penalty <= 0:
            raise TeacherDistillationConfigError(
                "generation.repetition_penalty must be greater than zero"
            )
        if self.max_tokens <= 0:
            raise TeacherDistillationConfigError("generation.max_tokens must be greater than zero")
        if self.seed < 0:
            raise TeacherDistillationConfigError("generation.seed must be non-negative")
        if not self.thinking and self.preserve_thinking:
            raise TeacherDistillationConfigError(
                "generation.preserve_thinking cannot be true when thinking is disabled"
            )


@dataclass(frozen=True, slots=True)
class TeacherCheckpointConfig:
    """Granularity for interruption-safe durable shard checkpoints."""

    shard_size: int

    def __post_init__(self) -> None:
        if self.shard_size <= 0:
            raise TeacherDistillationConfigError("checkpoint.shard_size must be greater than zero")


@dataclass(frozen=True, slots=True)
class TeacherDistillationConfig:
    """Complete deterministic teacher-data generation contract."""

    schema_version: int
    id: str
    language: str
    input_records: str
    runtime: TeacherRuntimeConfig
    teacher: TeacherModelConfig
    generation: TeacherGenerationConfig
    checkpoint: TeacherCheckpointConfig

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise TeacherDistillationConfigError("unsupported distillation schema_version")
        for field_name, value in (
            ("id", self.id),
            ("language", self.language),
            ("input_records", self.input_records),
        ):
            if not value.strip():
                raise TeacherDistillationConfigError(f"{field_name} must not be empty")
        if self.teacher.max_model_len < self.generation.max_tokens + 2048:
            raise TeacherDistillationConfigError(
                "teacher.max_model_len must leave at least 2048 tokens for the input prompt"
            )


def _mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TeacherDistillationConfigError(f"{context} must be a mapping")
    output: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TeacherDistillationConfigError(f"{context} keys must be strings")
        output[key] = item
    return output


def _require_keys(mapping: dict[str, object], *, required: frozenset[str], context: str) -> None:
    missing = sorted(required - set(mapping))
    unknown = sorted(set(mapping) - required)
    if missing:
        raise TeacherDistillationConfigError(
            f"{context} is missing required field(s): {', '.join(missing)}"
        )
    if unknown:
        raise TeacherDistillationConfigError(
            f"{context} contains unknown field(s): {', '.join(unknown)}"
        )


def _string(mapping: dict[str, object], key: str, *, context: str) -> str:
    value = mapping[key]
    if not isinstance(value, str):
        raise TeacherDistillationConfigError(f"{context}.{key} must be a string")
    return value


def _integer(mapping: dict[str, object], key: str, *, context: str) -> int:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TeacherDistillationConfigError(f"{context}.{key} must be an integer")
    return value


def _number(mapping: dict[str, object], key: str, *, context: str) -> float:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TeacherDistillationConfigError(f"{context}.{key} must be numeric")
    return float(value)


def _boolean(mapping: dict[str, object], key: str, *, context: str) -> bool:
    value = mapping[key]
    if not isinstance(value, bool):
        raise TeacherDistillationConfigError(f"{context}.{key} must be boolean")
    return value


def parse_teacher_distillation_config(value: object) -> TeacherDistillationConfig:
    """Parse one strict teacher-distillation configuration mapping."""

    root = _mapping(value, context="distillation")
    _require_keys(
        root,
        required=frozenset(
            {
                "schema_version",
                "id",
                "language",
                "input_records",
                "runtime",
                "teacher",
                "generation",
                "checkpoint",
            }
        ),
        context="distillation",
    )
    runtime = _mapping(root["runtime"], context="distillation.runtime")
    _require_keys(
        runtime,
        required=frozenset({"vllm_version", "vllm_bnb_plugin_version", "bitsandbytes_version"}),
        context="distillation.runtime",
    )
    teacher = _mapping(root["teacher"], context="distillation.teacher")
    _require_keys(
        teacher,
        required=frozenset(
            {
                "repository",
                "revision",
                "backend",
                "dtype",
                "quantization",
                "max_model_len",
                "gpu_memory_utilization",
            }
        ),
        context="distillation.teacher",
    )
    generation = _mapping(root["generation"], context="distillation.generation")
    _require_keys(
        generation,
        required=frozenset(
            {
                "thinking",
                "preserve_thinking",
                "reasoning_effort",
                "temperature",
                "top_p",
                "top_k",
                "min_p",
                "presence_penalty",
                "repetition_penalty",
                "max_tokens",
                "seed",
            }
        ),
        context="distillation.generation",
    )
    checkpoint = _mapping(root["checkpoint"], context="distillation.checkpoint")
    _require_keys(
        checkpoint,
        required=frozenset({"shard_size"}),
        context="distillation.checkpoint",
    )

    backend = _string(teacher, "backend", context="distillation.teacher")
    if backend != "vllm":
        raise TeacherDistillationConfigError("teacher.backend must be 'vllm'")
    dtype = _string(teacher, "dtype", context="distillation.teacher")
    if dtype != "bfloat16":
        raise TeacherDistillationConfigError("teacher.dtype must be 'bfloat16'")
    quantization = _string(teacher, "quantization", context="distillation.teacher")
    if quantization not in {"bitsandbytes", "none"}:
        raise TeacherDistillationConfigError(
            "teacher.quantization must be one of: bitsandbytes, none"
        )
    reasoning_effort = _string(generation, "reasoning_effort", context="distillation.generation")
    if reasoning_effort not in {"low", "medium", "xhigh"}:
        raise TeacherDistillationConfigError(
            "generation.reasoning_effort must be one of: low, medium, xhigh"
        )

    return TeacherDistillationConfig(
        schema_version=_integer(root, "schema_version", context="distillation"),
        id=_string(root, "id", context="distillation"),
        language=_string(root, "language", context="distillation"),
        input_records=_string(root, "input_records", context="distillation"),
        runtime=TeacherRuntimeConfig(
            vllm_version=_string(runtime, "vllm_version", context="distillation.runtime"),
            vllm_bnb_plugin_version=_string(
                runtime, "vllm_bnb_plugin_version", context="distillation.runtime"
            ),
            bitsandbytes_version=_string(
                runtime, "bitsandbytes_version", context="distillation.runtime"
            ),
        ),
        teacher=TeacherModelConfig(
            repository=_string(teacher, "repository", context="distillation.teacher"),
            revision=_string(teacher, "revision", context="distillation.teacher"),
            backend=cast(TeacherBackendName, backend),
            dtype=cast(TeacherDtype, dtype),
            quantization=cast(TeacherQuantization, quantization),
            max_model_len=_integer(teacher, "max_model_len", context="distillation.teacher"),
            gpu_memory_utilization=_number(
                teacher, "gpu_memory_utilization", context="distillation.teacher"
            ),
        ),
        generation=TeacherGenerationConfig(
            thinking=_boolean(generation, "thinking", context="distillation.generation"),
            preserve_thinking=_boolean(
                generation, "preserve_thinking", context="distillation.generation"
            ),
            reasoning_effort=cast(TeacherReasoningEffort, reasoning_effort),
            temperature=_number(generation, "temperature", context="distillation.generation"),
            top_p=_number(generation, "top_p", context="distillation.generation"),
            top_k=_integer(generation, "top_k", context="distillation.generation"),
            min_p=_number(generation, "min_p", context="distillation.generation"),
            presence_penalty=_number(
                generation, "presence_penalty", context="distillation.generation"
            ),
            repetition_penalty=_number(
                generation, "repetition_penalty", context="distillation.generation"
            ),
            max_tokens=_integer(generation, "max_tokens", context="distillation.generation"),
            seed=_integer(generation, "seed", context="distillation.generation"),
        ),
        checkpoint=TeacherCheckpointConfig(
            shard_size=_integer(checkpoint, "shard_size", context="distillation.checkpoint")
        ),
    )


def load_teacher_distillation_config(path: Path) -> TeacherDistillationConfig:
    """Load a strict teacher-distillation YAML configuration."""

    try:
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TeacherDistillationConfigError(
            f"could not read distillation config {path}: {exc}"
        ) from exc
    except yaml.YAMLError as exc:
        raise TeacherDistillationConfigError(
            f"invalid YAML in distillation config {path}: {exc}"
        ) from exc
    return parse_teacher_distillation_config(raw)


def teacher_distillation_config_sha256(config: TeacherDistillationConfig) -> str:
    """Hash the exact semantic config payload independently of YAML formatting."""

    payload = json.dumps(
        asdict(config), ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()
