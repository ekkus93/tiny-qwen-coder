"""Canonical Qwen3.5-4B BF16 load/generation smoke test and memory report."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, cast

import torch
from torch import nn
from transformers import PreTrainedTokenizerBase

from tiny_qwen_coder.model.inspection import InspectionTarget, load_inspection_target
from tiny_qwen_coder.reproducibility import seed_everything

_SCHEMA_VERSION = 1
_DEFAULT_SEED = 0
_DEFAULT_SYSTEM_PROMPT = "You are a concise coding assistant."
_DEFAULT_USER_PROMPT = "Reply with exactly the word OK."
_DEFAULT_MAX_NEW_TOKENS = 16


class ModelSmokeTestError(RuntimeError):
    """Raised when the canonical model cannot satisfy the smoke-test contract."""


class _GenerateCapable(Protocol):
    def generate(self, **kwargs: object) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True)
class SmokeMemoryReport:
    """CUDA memory observations around canonical BF16 load and generation."""

    cuda_total_bytes: int
    cuda_free_before_load_bytes: int
    cuda_free_after_load_bytes: int
    torch_allocated_after_load_bytes: int
    torch_reserved_after_load_bytes: int
    load_peak_allocated_bytes: int
    load_peak_reserved_bytes: int
    torch_allocated_before_generation_bytes: int
    torch_reserved_before_generation_bytes: int
    generation_peak_allocated_bytes: int
    generation_peak_reserved_bytes: int

    def __post_init__(self) -> None:
        non_negative_fields = (
            self.cuda_total_bytes,
            self.cuda_free_before_load_bytes,
            self.cuda_free_after_load_bytes,
            self.torch_allocated_after_load_bytes,
            self.torch_reserved_after_load_bytes,
            self.load_peak_allocated_bytes,
            self.load_peak_reserved_bytes,
            self.torch_allocated_before_generation_bytes,
            self.torch_reserved_before_generation_bytes,
            self.generation_peak_allocated_bytes,
            self.generation_peak_reserved_bytes,
        )
        if self.cuda_total_bytes <= 0:
            raise ModelSmokeTestError("cuda_total_bytes must be greater than zero")
        if any(value < 0 for value in non_negative_fields):
            raise ModelSmokeTestError("memory observations must not be negative")
        if self.cuda_free_before_load_bytes > self.cuda_total_bytes:
            raise ModelSmokeTestError("free CUDA memory before load exceeds total memory")
        if self.cuda_free_after_load_bytes > self.cuda_total_bytes:
            raise ModelSmokeTestError("free CUDA memory after load exceeds total memory")
        if self.torch_reserved_after_load_bytes < self.torch_allocated_after_load_bytes:
            raise ModelSmokeTestError("reserved memory after load is smaller than allocated memory")
        if self.load_peak_allocated_bytes < self.torch_allocated_after_load_bytes:
            raise ModelSmokeTestError(
                "load peak allocated memory is smaller than post-load allocation"
            )
        if self.load_peak_reserved_bytes < self.torch_reserved_after_load_bytes:
            raise ModelSmokeTestError(
                "load peak reserved memory is smaller than post-load reservation"
            )
        if self.generation_peak_allocated_bytes < self.torch_allocated_before_generation_bytes:
            raise ModelSmokeTestError(
                "generation peak allocated memory is smaller than its starting allocation"
            )
        if self.generation_peak_reserved_bytes < self.torch_reserved_before_generation_bytes:
            raise ModelSmokeTestError(
                "generation peak reserved memory is smaller than its starting reservation"
            )


@dataclass(frozen=True, slots=True)
class ModelSmokeTestReport:
    """Machine-readable proof that the pinned canonical base loads and generates."""

    schema_version: int
    target: InspectionTarget
    torch_version: str
    transformers_version: str
    model_class: str
    resolved_model_revision: str
    parameter_dtypes: tuple[str, ...]
    device: str
    gpu_name: str
    gpu_compute_capability: str
    seed: int
    system_prompt: str
    user_prompt: str
    max_new_tokens: int
    input_token_count: int
    generated_token_count: int
    generated_token_ids: tuple[int, ...]
    generated_text: str
    deterministic_repeat: bool
    memory: SmokeMemoryReport

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise ModelSmokeTestError(
                f"unsupported smoke-test schema version {self.schema_version}; expected {_SCHEMA_VERSION}"
            )
        if self.target.model_load_dtype != "bfloat16":
            raise ModelSmokeTestError("canonical P2-007 smoke test requires bfloat16 model loading")
        if self.resolved_model_revision != self.target.model_revision:
            raise ModelSmokeTestError(
                "loaded model revision does not match the canonical pinned revision"
            )
        if self.parameter_dtypes != ("torch.bfloat16",):
            raise ModelSmokeTestError(
                "canonical BF16 smoke test requires every floating model parameter to be bfloat16"
            )
        if not self.device.startswith("cuda:"):
            raise ModelSmokeTestError("canonical P2-007 smoke test requires a CUDA device")
        for field_name, value in (
            ("model_class", self.model_class),
            ("gpu_name", self.gpu_name),
            ("gpu_compute_capability", self.gpu_compute_capability),
            ("system_prompt", self.system_prompt),
            ("user_prompt", self.user_prompt),
            ("generated_text", self.generated_text),
        ):
            if not value.strip():
                raise ModelSmokeTestError(f"{field_name} must not be empty")
        if self.max_new_tokens <= 0:
            raise ModelSmokeTestError("max_new_tokens must be greater than zero")
        if self.input_token_count <= 0:
            raise ModelSmokeTestError("input_token_count must be greater than zero")
        if self.generated_token_count <= 0:
            raise ModelSmokeTestError("generated_token_count must be greater than zero")
        if self.generated_token_count > self.max_new_tokens:
            raise ModelSmokeTestError("generated token count exceeds max_new_tokens")
        if self.generated_token_count != len(self.generated_token_ids):
            raise ModelSmokeTestError("generated token count does not match generated_token_ids")
        if not self.deterministic_repeat:
            raise ModelSmokeTestError("repeated greedy generation was not deterministic")


@dataclass(frozen=True, slots=True)
class _GenerationResult:
    token_ids: tuple[int, ...]
    text: str
    peak_allocated_bytes: int
    peak_reserved_bytes: int


def _qualified_class_name(value: object) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _resolved_revision(model: nn.Module) -> str:
    config: object = getattr(model, "config", None)
    revision: object = getattr(config, "_commit_hash", None)
    if not isinstance(revision, str) or not revision:
        raise ModelSmokeTestError("loaded model does not expose a resolved upstream revision")
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


def _prepare_prompt_inputs(
    tokenizer: PreTrainedTokenizerBase,
    *,
    device: torch.device,
    system_prompt: str,
    user_prompt: str,
) -> dict[str, torch.Tensor]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    encoded: object = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    if not isinstance(encoded, Mapping):
        raise ModelSmokeTestError("chat template did not return a mapping of model inputs")

    inputs: dict[str, torch.Tensor] = {}
    for key in ("input_ids", "attention_mask"):
        value: object = encoded.get(key)
        if value is None and key == "attention_mask":
            continue
        if not isinstance(value, torch.Tensor):
            raise ModelSmokeTestError(f"chat template output {key!r} is not a torch Tensor")
        inputs[key] = value.to(device)

    input_ids = inputs.get("input_ids")
    if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ModelSmokeTestError("smoke-test prompt must tokenize to one non-empty batch")
    if input_ids.shape[1] <= 0:
        raise ModelSmokeTestError("smoke-test prompt tokenized to zero tokens")
    return inputs


def _decode_generated_tokens(
    tokenizer: PreTrainedTokenizerBase,
    token_ids: tuple[int, ...],
) -> str:
    decoded: object = tokenizer.decode(
        list(token_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if not isinstance(decoded, str) or not decoded.strip():
        raise ModelSmokeTestError("generated token suffix did not decode to non-empty text")
    return decoded


def _generate_once(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    inputs: Mapping[str, torch.Tensor],
    *,
    input_token_count: int,
    max_new_tokens: int,
    device: torch.device,
) -> _GenerationResult:
    torch.cuda.reset_peak_memory_stats(device)
    generate_model = cast(_GenerateCapable, model)
    generate_kwargs: dict[str, object] = {key: value for key, value in inputs.items()}
    generate_kwargs.update(
        {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
            "use_cache": True,
        }
    )
    with torch.inference_mode():
        output: object = generate_model.generate(**generate_kwargs)
    torch.cuda.synchronize(device)
    if not isinstance(output, torch.Tensor):
        raise ModelSmokeTestError("model.generate did not return a token tensor")
    if output.ndim != 2 or output.shape[0] != 1:
        raise ModelSmokeTestError("model.generate returned an unexpected batch shape")
    if output.shape[1] <= input_token_count:
        raise ModelSmokeTestError("model.generate did not produce any new tokens")

    suffix = output[0, input_token_count:].detach().cpu()
    token_ids = tuple(int(token_id) for token_id in suffix.tolist())
    if not token_ids:
        raise ModelSmokeTestError("model.generate produced an empty token suffix")
    return _GenerationResult(
        token_ids=token_ids,
        text=_decode_generated_tokens(tokenizer, token_ids),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
    )


def run_canonical_model_smoke_test(
    target: InspectionTarget,
    *,
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
    user_prompt: str = _DEFAULT_USER_PROMPT,
    max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
    seed: int = _DEFAULT_SEED,
    device_index: int = 0,
) -> ModelSmokeTestReport:
    """Load the exact pinned base in BF16, generate twice, and record CUDA memory."""

    if target.model_load_dtype != "bfloat16":
        raise ModelSmokeTestError("P2-007 requires the canonical model_load_dtype to be bfloat16")
    if max_new_tokens <= 0:
        raise ModelSmokeTestError("max_new_tokens must be greater than zero")
    if not torch.cuda.is_available():
        raise ModelSmokeTestError("P2-007 requires CUDA")
    if not 0 <= device_index < torch.cuda.device_count():
        raise ModelSmokeTestError(f"invalid CUDA device index: {device_index}")

    from transformers import AutoModelForMultimodalLM, AutoTokenizer

    device = torch.device("cuda", device_index)
    seed_everything(seed)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    free_before_load, total_bytes = torch.cuda.mem_get_info(device)
    torch.cuda.reset_peak_memory_stats(device)

    tokenizer_obj: object = AutoTokenizer.from_pretrained(
        target.tokenizer_repository,
        revision=target.tokenizer_revision,
    )
    if not isinstance(tokenizer_obj, PreTrainedTokenizerBase):
        raise ModelSmokeTestError("Transformers returned an unexpected tokenizer object")
    tokenizer = tokenizer_obj

    loaded: object = AutoModelForMultimodalLM.from_pretrained(
        target.model_repository,
        revision=target.model_revision,
        dtype=torch.bfloat16,
        device_map={"": device_index},
        low_cpu_mem_usage=True,
    )
    if not isinstance(loaded, nn.Module):
        raise ModelSmokeTestError("Transformers returned an object that is not a torch.nn.Module")
    model = loaded
    model.eval()
    torch.cuda.synchronize(device)

    resolved_revision = _resolved_revision(model)
    if resolved_revision != target.model_revision:
        raise ModelSmokeTestError(
            "loaded model resolved to an unexpected revision: "
            f"{resolved_revision} != {target.model_revision}"
        )
    parameter_dtypes = _floating_parameter_dtypes(model)
    if parameter_dtypes != ("torch.bfloat16",):
        raise ModelSmokeTestError(
            "canonical model was not loaded entirely in BF16; "
            f"observed floating parameter dtypes: {parameter_dtypes}"
        )

    free_after_load, total_after_load = torch.cuda.mem_get_info(device)
    if total_after_load != total_bytes:
        raise ModelSmokeTestError("CUDA total memory changed unexpectedly during model load")
    allocated_after_load = torch.cuda.memory_allocated(device)
    reserved_after_load = torch.cuda.memory_reserved(device)
    load_peak_allocated = torch.cuda.max_memory_allocated(device)
    load_peak_reserved = torch.cuda.max_memory_reserved(device)

    inputs = _prepare_prompt_inputs(
        tokenizer,
        device=device,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    input_token_count = int(inputs["input_ids"].shape[1])
    allocated_before_generation = torch.cuda.memory_allocated(device)
    reserved_before_generation = torch.cuda.memory_reserved(device)

    first = _generate_once(
        model,
        tokenizer,
        inputs,
        input_token_count=input_token_count,
        max_new_tokens=max_new_tokens,
        device=device,
    )
    second = _generate_once(
        model,
        tokenizer,
        inputs,
        input_token_count=input_token_count,
        max_new_tokens=max_new_tokens,
        device=device,
    )
    deterministic_repeat = first.token_ids == second.token_ids
    if not deterministic_repeat:
        raise ModelSmokeTestError("repeated greedy generation produced different token IDs")

    capability = torch.cuda.get_device_capability(device)
    memory = SmokeMemoryReport(
        cuda_total_bytes=total_bytes,
        cuda_free_before_load_bytes=free_before_load,
        cuda_free_after_load_bytes=free_after_load,
        torch_allocated_after_load_bytes=allocated_after_load,
        torch_reserved_after_load_bytes=reserved_after_load,
        load_peak_allocated_bytes=load_peak_allocated,
        load_peak_reserved_bytes=load_peak_reserved,
        torch_allocated_before_generation_bytes=allocated_before_generation,
        torch_reserved_before_generation_bytes=reserved_before_generation,
        generation_peak_allocated_bytes=max(
            first.peak_allocated_bytes,
            second.peak_allocated_bytes,
        ),
        generation_peak_reserved_bytes=max(
            first.peak_reserved_bytes,
            second.peak_reserved_bytes,
        ),
    )
    return ModelSmokeTestReport(
        schema_version=_SCHEMA_VERSION,
        target=target,
        torch_version=torch.__version__,
        transformers_version=importlib.metadata.version("transformers"),
        model_class=_qualified_class_name(model),
        resolved_model_revision=resolved_revision,
        parameter_dtypes=parameter_dtypes,
        device=str(device),
        gpu_name=torch.cuda.get_device_name(device),
        gpu_compute_capability=f"{capability[0]}.{capability[1]}",
        seed=seed,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_new_tokens=max_new_tokens,
        input_token_count=input_token_count,
        generated_token_count=len(first.token_ids),
        generated_token_ids=first.token_ids,
        generated_text=first.text,
        deterministic_repeat=deterministic_repeat,
        memory=memory,
    )


def model_smoke_report_json(report: ModelSmokeTestReport) -> str:
    """Serialize the canonical model smoke-test report deterministically."""

    return json.dumps(asdict(report), indent=2, sort_keys=True) + "\n"


def _gib(value: int) -> float:
    return value / (1024**3)


def model_smoke_report_text(report: ModelSmokeTestReport) -> str:
    """Render a concise human-readable canonical model smoke-test report."""

    memory = report.memory
    lines = [
        "Tiny Qwen Coder canonical model smoke test",
        "===========================================",
        f"Base: {report.target.model_repository}@{report.target.model_revision}",
        f"Tokenizer: {report.target.tokenizer_repository}@{report.target.tokenizer_revision}",
        f"Model class: {report.model_class}",
        f"Parameter dtype(s): {', '.join(report.parameter_dtypes)}",
        f"Device: {report.device} ({report.gpu_name}, compute {report.gpu_compute_capability})",
        f"PyTorch / Transformers: {report.torch_version} / {report.transformers_version}",
        f"Input tokens: {report.input_token_count}",
        f"Generated tokens: {report.generated_token_count}",
        f"Deterministic repeat: {report.deterministic_repeat}",
        f"Generated text: {report.generated_text!r}",
        "",
        "CUDA memory",
        "-----------",
        f"Total VRAM: {_gib(memory.cuda_total_bytes):.3f} GiB",
        f"Free before load: {_gib(memory.cuda_free_before_load_bytes):.3f} GiB",
        f"Free after load: {_gib(memory.cuda_free_after_load_bytes):.3f} GiB",
        f"PyTorch allocated after load: {_gib(memory.torch_allocated_after_load_bytes):.3f} GiB",
        f"PyTorch reserved after load: {_gib(memory.torch_reserved_after_load_bytes):.3f} GiB",
        f"Load peak allocated: {_gib(memory.load_peak_allocated_bytes):.3f} GiB",
        f"Load peak reserved: {_gib(memory.load_peak_reserved_bytes):.3f} GiB",
        f"Generation peak allocated: {_gib(memory.generation_peak_allocated_bytes):.3f} GiB",
        f"Generation peak reserved: {_gib(memory.generation_peak_reserved_bytes):.3f} GiB",
    ]
    return "\n".join(lines) + "\n"


def _write_output(text: str, output: Path | None) -> None:
    if output is None:
        print(text, end="")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(output)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load and generate with the exact pinned Qwen3.5-4B base in BF16."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/base/qwen35-4b.yaml"),
        help="Canonical base-model config path.",
    )
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--output", type=Path, help="Optional report output path.")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    parser.add_argument("--system-prompt", default=_DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--user-prompt", default=_DEFAULT_USER_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=_DEFAULT_MAX_NEW_TOKENS)
    return parser


def smoke_test_model(argv: Sequence[str] | None = None) -> None:
    """CLI entry point for the canonical BF16 model load/generation smoke test."""

    args = _argument_parser().parse_args(argv)
    target = load_inspection_target(cast(Path, args.config))
    report = run_canonical_model_smoke_test(
        target,
        system_prompt=cast(str, args.system_prompt),
        user_prompt=cast(str, args.user_prompt),
        max_new_tokens=cast(int, args.max_new_tokens),
        seed=cast(int, args.seed),
        device_index=cast(int, args.device_index),
    )
    rendered = (
        model_smoke_report_json(report)
        if cast(str, args.format) == "json"
        else model_smoke_report_text(report)
    )
    _write_output(rendered, cast(Path | None, args.output))


if __name__ == "__main__":
    smoke_test_model()
