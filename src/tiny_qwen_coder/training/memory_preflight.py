"""Measured QLoRA training-memory preflight for the canonical 16 GiB GPU target."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

import torch

from tiny_qwen_coder.model import InspectionTarget, load_inspection_target
from tiny_qwen_coder.reproducibility import seed_everything

ReportFormat: TypeAlias = Literal["text", "json"]

_SCHEMA_VERSION = 1
_GIB = 1024**3
_DEFAULT_SEQUENCE_LENGTH = 2048
_DEFAULT_MICRO_BATCH_SIZE = 1
_DEFAULT_RANK = 16
_DEFAULT_ALPHA = 32
_DEFAULT_DROPOUT = 0.05
_DEFAULT_SEED = 0
_DEFAULT_LEARNING_RATE = 2e-4
_MINIMUM_HEADROOM_BYTES = int(1.5 * _GIB)
_MINIMUM_HEADROOM_FRACTION = 0.10
_CANONICAL_TARGET_MODULES = (
    "down_proj",
    "gate_proj",
    "in_proj_a",
    "in_proj_b",
    "in_proj_qkv",
    "in_proj_z",
    "k_proj",
    "o_proj",
    "out_proj",
    "q_proj",
    "up_proj",
    "v_proj",
)


class TrainingMemoryPreflightError(RuntimeError):
    """Raised when the canonical training-memory preflight cannot complete safely."""


@dataclass(frozen=True, slots=True)
class QuantizationSpec:
    """Frozen P0 QLoRA quantization settings."""

    bits: int = 4
    quant_type: str = "nf4"
    double_quant: bool = True
    compute_dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        if self.bits != 4:
            raise TrainingMemoryPreflightError("canonical QLoRA quantization must use 4 bits")
        if self.quant_type != "nf4":
            raise TrainingMemoryPreflightError("canonical QLoRA quantization must use NF4")
        if not self.double_quant:
            raise TrainingMemoryPreflightError("canonical QLoRA must enable double quantization")
        if self.compute_dtype != "bfloat16":
            raise TrainingMemoryPreflightError("canonical QLoRA compute dtype must be bfloat16")


@dataclass(frozen=True, slots=True)
class CudaMemorySnapshot:
    """One synchronized CUDA allocator/device-memory observation."""

    allocated_bytes: int
    reserved_bytes: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    free_bytes: int
    total_bytes: int

    def __post_init__(self) -> None:
        values = (
            self.allocated_bytes,
            self.reserved_bytes,
            self.peak_allocated_bytes,
            self.peak_reserved_bytes,
            self.free_bytes,
            self.total_bytes,
        )
        if any(value < 0 for value in values):
            raise TrainingMemoryPreflightError("CUDA memory observations must be non-negative")
        if self.total_bytes <= 0:
            raise TrainingMemoryPreflightError("total CUDA memory must be greater than zero")
        if self.free_bytes > self.total_bytes:
            raise TrainingMemoryPreflightError("free CUDA memory exceeds total CUDA memory")
        if self.reserved_bytes < self.allocated_bytes:
            raise TrainingMemoryPreflightError("reserved CUDA memory is smaller than allocated")
        if self.peak_allocated_bytes < self.allocated_bytes:
            raise TrainingMemoryPreflightError("peak allocated CUDA memory is inconsistent")
        if self.peak_reserved_bytes < self.reserved_bytes:
            raise TrainingMemoryPreflightError("peak reserved CUDA memory is inconsistent")


@dataclass(frozen=True, slots=True)
class TrainingMemoryPreflightReport:
    """Machine-readable proof of the selected canonical P0 training-memory strategy."""

    schema_version: int
    target: InspectionTarget
    mode: str
    quantization: QuantizationSpec
    sequence_length: int
    micro_batch_size: int
    gradient_checkpointing: bool
    rank: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]
    trainable_parameters: int
    four_bit_module_count: int
    loss: float
    forward_seconds: float
    backward_seconds: float
    optimizer_seconds: float
    attached: CudaMemorySnapshot
    after_forward: CudaMemorySnapshot
    after_backward: CudaMemorySnapshot
    after_optimizer: CudaMemorySnapshot
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    safety_headroom_bytes: int
    required_headroom_bytes: int
    comfortable: bool
    bitsandbytes_version: str
    torch_version: str
    gpu_name: str
    compute_capability: str

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise TrainingMemoryPreflightError("unsupported training-memory report schema version")
        if self.mode != "qlora_4bit":
            raise TrainingMemoryPreflightError("canonical P0 training mode must be qlora_4bit")
        if self.sequence_length != _DEFAULT_SEQUENCE_LENGTH:
            raise TrainingMemoryPreflightError("canonical preflight sequence length must be 2048")
        if self.micro_batch_size != _DEFAULT_MICRO_BATCH_SIZE:
            raise TrainingMemoryPreflightError("canonical preflight micro-batch size must be 1")
        if not self.gradient_checkpointing:
            raise TrainingMemoryPreflightError(
                "canonical QLoRA preflight requires gradient checkpointing"
            )
        if self.rank != _DEFAULT_RANK or self.alpha != _DEFAULT_ALPHA:
            raise TrainingMemoryPreflightError("canonical LoRA rank/alpha must be 16/32")
        if self.dropout != _DEFAULT_DROPOUT:
            raise TrainingMemoryPreflightError("canonical LoRA dropout must be 0.05")
        if self.target_modules != _CANONICAL_TARGET_MODULES:
            raise TrainingMemoryPreflightError("canonical selective target modules changed")
        if self.trainable_parameters <= 0 or self.four_bit_module_count <= 0:
            raise TrainingMemoryPreflightError("preflight must observe trainable and 4-bit modules")
        if not math.isfinite(self.loss):
            raise TrainingMemoryPreflightError("preflight loss must be finite")
        if min(self.forward_seconds, self.backward_seconds, self.optimizer_seconds) < 0:
            raise TrainingMemoryPreflightError("preflight timings must be non-negative")
        if self.peak_allocated_bytes <= 0 or self.peak_reserved_bytes <= 0:
            raise TrainingMemoryPreflightError("preflight peak memory must be measured")
        if self.peak_reserved_bytes < self.peak_allocated_bytes:
            raise TrainingMemoryPreflightError(
                "peak reserved memory is smaller than peak allocated"
            )
        if (
            self.safety_headroom_bytes
            != self.after_optimizer.total_bytes - self.peak_reserved_bytes
        ):
            raise TrainingMemoryPreflightError("safety headroom is inconsistent with measured peak")
        expected_required = required_safety_headroom_bytes(self.after_optimizer.total_bytes)
        if self.required_headroom_bytes != expected_required:
            raise TrainingMemoryPreflightError("required safety headroom is inconsistent")
        if self.comfortable != (self.safety_headroom_bytes >= self.required_headroom_bytes):
            raise TrainingMemoryPreflightError("comfortable flag is inconsistent with headroom")
        if not self.comfortable:
            raise TrainingMemoryPreflightError("canonical QLoRA preflight lacks safe VRAM headroom")
        for field_name, value in (
            ("bitsandbytes_version", self.bitsandbytes_version),
            ("torch_version", self.torch_version),
            ("gpu_name", self.gpu_name),
            ("compute_capability", self.compute_capability),
        ):
            if not value.strip():
                raise TrainingMemoryPreflightError(f"{field_name} must not be empty")


def required_safety_headroom_bytes(total_bytes: int) -> int:
    """Return the P2-008 safety threshold: max(1.5 GiB, 10% physical VRAM)."""

    if total_bytes <= 0:
        raise TrainingMemoryPreflightError("total_bytes must be greater than zero")
    return max(_MINIMUM_HEADROOM_BYTES, math.ceil(total_bytes * _MINIMUM_HEADROOM_FRACTION))


def _memory_snapshot(device: torch.device) -> CudaMemorySnapshot:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return CudaMemorySnapshot(
        allocated_bytes=torch.cuda.memory_allocated(device),
        reserved_bytes=torch.cuda.memory_reserved(device),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        free_bytes=free_bytes,
        total_bytes=total_bytes,
    )


def _timed_cuda_operation(operation: Any, device: torch.device) -> float:
    start = time.perf_counter()
    operation()
    torch.cuda.synchronize(device)
    return time.perf_counter() - start


def run_canonical_qlora_memory_preflight(
    *,
    config_path: Path = Path("configs/base/qwen35-4b.yaml"),
) -> TrainingMemoryPreflightReport:
    """Run one 2,048-token QLoRA forward/backward/optimizer step on CUDA device zero."""

    if not torch.cuda.is_available():
        raise TrainingMemoryPreflightError("canonical training-memory preflight requires CUDA")

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForMultimodalLM, BitsAndBytesConfig

    seed_everything(_DEFAULT_SEED)
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    target = load_inspection_target(config_path)
    quantization = QuantizationSpec()
    bitsandbytes_config_factory = cast(Any, BitsAndBytesConfig)
    bnb_config = bitsandbytes_config_factory(
        load_in_4bit=True,
        bnb_4bit_quant_type=quantization.quant_type,
        bnb_4bit_use_double_quant=quantization.double_quant,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model: Any = AutoModelForMultimodalLM.from_pretrained(
        target.model_repository,
        revision=target.model_revision,
        dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map={"": 0},
    )
    model.config.use_cache = False
    prepare_for_kbit_training = cast(Any, prepare_model_for_kbit_training)
    model = prepare_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=_DEFAULT_RANK,
            lora_alpha=_DEFAULT_ALPHA,
            lora_dropout=_DEFAULT_DROPOUT,
            bias="none",
            target_modules=list(_CANONICAL_TARGET_MODULES),
            task_type="CAUSAL_LM",
        ),
    )
    model.train()

    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    four_bit_module_count = sum(type(module).__name__ == "Linear4bit" for module in model.modules())

    text_config: Any = model.config.text_config
    generator = torch.Generator(device=device).manual_seed(_DEFAULT_SEED)
    input_ids = torch.randint(
        0,
        int(text_config.vocab_size),
        (_DEFAULT_MICRO_BATCH_SIZE, _DEFAULT_SEQUENCE_LENGTH),
        generator=generator,
        device=device,
    )
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    torch.cuda.reset_peak_memory_stats(device)
    attached = _memory_snapshot(device)
    outputs: Any = None

    def forward() -> None:
        nonlocal outputs
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    forward_seconds = _timed_cuda_operation(forward, device)
    loss: torch.Tensor | None = getattr(outputs, "loss", None)
    if loss is None or not bool(torch.isfinite(loss).item()):
        raise TrainingMemoryPreflightError("QLoRA preflight produced a missing or non-finite loss")
    after_forward = _memory_snapshot(device)
    backward_seconds = _timed_cuda_operation(loss.backward, device)
    after_backward = _memory_snapshot(device)

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=_DEFAULT_LEARNING_RATE,
    )

    def optimizer_step() -> None:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    optimizer_seconds = _timed_cuda_operation(optimizer_step, device)
    after_optimizer = _memory_snapshot(device)

    peak_allocated = max(
        after_forward.peak_allocated_bytes,
        after_backward.peak_allocated_bytes,
        after_optimizer.peak_allocated_bytes,
    )
    peak_reserved = max(
        after_forward.peak_reserved_bytes,
        after_backward.peak_reserved_bytes,
        after_optimizer.peak_reserved_bytes,
    )
    required_headroom = required_safety_headroom_bytes(after_optimizer.total_bytes)
    headroom = after_optimizer.total_bytes - peak_reserved

    return TrainingMemoryPreflightReport(
        schema_version=_SCHEMA_VERSION,
        target=target,
        mode="qlora_4bit",
        quantization=quantization,
        sequence_length=_DEFAULT_SEQUENCE_LENGTH,
        micro_batch_size=_DEFAULT_MICRO_BATCH_SIZE,
        gradient_checkpointing=True,
        rank=_DEFAULT_RANK,
        alpha=_DEFAULT_ALPHA,
        dropout=_DEFAULT_DROPOUT,
        target_modules=_CANONICAL_TARGET_MODULES,
        trainable_parameters=trainable_parameters,
        four_bit_module_count=four_bit_module_count,
        loss=float(loss.detach().cpu()),
        forward_seconds=forward_seconds,
        backward_seconds=backward_seconds,
        optimizer_seconds=optimizer_seconds,
        attached=attached,
        after_forward=after_forward,
        after_backward=after_backward,
        after_optimizer=after_optimizer,
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
        safety_headroom_bytes=headroom,
        required_headroom_bytes=required_headroom,
        comfortable=headroom >= required_headroom,
        bitsandbytes_version=importlib.metadata.version("bitsandbytes"),
        torch_version=torch.__version__,
        gpu_name=torch.cuda.get_device_name(device),
        compute_capability=".".join(map(str, torch.cuda.get_device_capability(device))),
    )


def training_memory_report_json(report: TrainingMemoryPreflightReport) -> str:
    """Serialize a training-memory preflight report deterministically."""

    return json.dumps(asdict(report), indent=2, sort_keys=True) + "\n"


def training_memory_report_text(report: TrainingMemoryPreflightReport) -> str:
    """Render a concise human-readable preflight summary."""

    gib = _GIB
    return "\n".join(
        (
            "Tiny Qwen Coder canonical training-memory preflight",
            "===================================================",
            f"Base: {report.target.model_repository}@{report.target.model_revision}",
            f"Mode: {report.mode}",
            "Quantization: 4-bit NF4; double-quant=true; compute=bfloat16",
            f"Sequence/micro-batch: {report.sequence_length}/{report.micro_batch_size}",
            f"LoRA: r={report.rank} alpha={report.alpha} dropout={report.dropout}",
            f"Trainable parameters: {report.trainable_parameters:,}",
            f"4-bit modules: {report.four_bit_module_count}",
            f"Peak allocated: {report.peak_allocated_bytes / gib:.3f} GiB",
            f"Peak reserved: {report.peak_reserved_bytes / gib:.3f} GiB",
            f"Safety headroom: {report.safety_headroom_bytes / gib:.3f} GiB",
            f"Required headroom: {report.required_headroom_bytes / gib:.3f} GiB",
            f"Comfortable: {report.comfortable}",
            f"GPU: {report.gpu_name} (cc {report.compute_capability})",
            f"PyTorch/bitsandbytes: {report.torch_version}/{report.bitsandbytes_version}",
            "",
        )
    )


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
        description="Run the frozen Qwen3.5-4B 4-bit QLoRA training-memory preflight."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/base/qwen35-4b.yaml"),
        help="Canonical base-model config path.",
    )
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--output", type=Path)
    return parser


def training_memory_preflight_main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point for the frozen P0 QLoRA memory validation."""

    args = _argument_parser().parse_args(argv)
    report = run_canonical_qlora_memory_preflight(config_path=cast(Path, args.config))
    report_format = cast(ReportFormat, args.format)
    rendered = (
        training_memory_report_json(report)
        if report_format == "json"
        else training_memory_report_text(report)
    )
    _write_output(rendered, cast(Path | None, args.output))


__all__ = [
    "CudaMemorySnapshot",
    "QuantizationSpec",
    "TrainingMemoryPreflightError",
    "TrainingMemoryPreflightReport",
    "required_safety_headroom_bytes",
    "run_canonical_qlora_memory_preflight",
    "training_memory_preflight_main",
    "training_memory_report_json",
    "training_memory_report_text",
]
