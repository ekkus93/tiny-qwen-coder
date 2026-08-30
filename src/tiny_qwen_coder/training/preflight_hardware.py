"""GPU/training-mode compatibility checks for adapter-training preflight."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Protocol

import torch

from tiny_qwen_coder.training.plan import AdapterTrainingError, AdapterTrainingPlan


@dataclass(frozen=True, slots=True)
class HardwareSnapshot:
    """Minimal hardware facts needed to approve a training mode."""

    cuda_available: bool
    device_count: int
    device_name: str | None
    total_vram_bytes: int | None
    bf16_supported: bool
    bitsandbytes_available: bool


@dataclass(frozen=True, slots=True)
class HardwarePreflightEvidence:
    """Approved hardware/training-mode combination."""

    training_mode: str
    compute_dtype: str
    cuda_available: bool
    device_count: int
    device_name: str
    total_vram_bytes: int
    bf16_supported: bool
    bitsandbytes_available: bool


class HardwareProbe(Protocol):
    def snapshot(self) -> HardwareSnapshot: ...


class TorchHardwareProbe:
    """Production hardware probe backed by the pinned PyTorch runtime."""

    def snapshot(self) -> HardwareSnapshot:
        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if cuda_available else 0
        device_name: str | None = None
        total_vram_bytes: int | None = None
        if device_count:
            properties = torch.cuda.get_device_properties(0)
            device_name = str(properties.name)
            total_vram_bytes = int(properties.total_memory)
        return HardwareSnapshot(
            cuda_available=cuda_available,
            device_count=device_count,
            device_name=device_name,
            total_vram_bytes=total_vram_bytes,
            bf16_supported=bool(cuda_available and torch.cuda.is_bf16_supported()),
            bitsandbytes_available=importlib.util.find_spec("bitsandbytes") is not None,
        )


def verify_training_hardware(
    plan: AdapterTrainingPlan,
    *,
    probe: HardwareProbe | None = None,
) -> HardwarePreflightEvidence:
    """Fail closed when the visible GPU cannot execute the configured training mode."""

    snapshot = (probe or TorchHardwareProbe()).snapshot()
    if not snapshot.cuda_available or snapshot.device_count <= 0:
        raise AdapterTrainingError("training preflight requires a CUDA-visible GPU")
    if snapshot.device_name is None or snapshot.total_vram_bytes is None:
        raise AdapterTrainingError("CUDA device 0 did not report complete hardware properties")
    if plan.config.compute_dtype == "bfloat16" and not snapshot.bf16_supported:
        raise AdapterTrainingError("configured BF16 compute is not supported by CUDA device 0")
    if plan.config.training_mode == "qlora_4bit" and not snapshot.bitsandbytes_available:
        raise AdapterTrainingError("QLoRA training requires the bitsandbytes runtime")
    return HardwarePreflightEvidence(
        training_mode=plan.config.training_mode,
        compute_dtype=plan.config.compute_dtype,
        cuda_available=snapshot.cuda_available,
        device_count=snapshot.device_count,
        device_name=snapshot.device_name,
        total_vram_bytes=snapshot.total_vram_bytes,
        bf16_supported=snapshot.bf16_supported,
        bitsandbytes_available=snapshot.bitsandbytes_available,
    )
