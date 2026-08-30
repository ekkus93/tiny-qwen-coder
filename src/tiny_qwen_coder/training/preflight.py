"""Fail-fast validation for adapter training before expensive model loading."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NoReturn

from transformers import PreTrainedTokenizerBase

from tiny_qwen_coder.training.plan import AdapterTrainingPlan, resolve_adapter_training_plan
from tiny_qwen_coder.training.preflight_dataset import (
    DatasetPreflightEvidence,
    verify_frozen_training_dataset,
)
from tiny_qwen_coder.training.preflight_hardware import (
    HardwarePreflightEvidence,
    HardwareProbe,
    verify_training_hardware,
)
from tiny_qwen_coder.training.preflight_loss import LossPreflightEvidence, verify_training_loss_mask
from tiny_qwen_coder.training.preflight_output import (
    OutputPreflightEvidence,
    verify_training_output_path,
)
from tiny_qwen_coder.training.preflight_targets import (
    TargetPreflightEvidence,
    verify_frozen_lora_targets,
)

_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class TrainingPreflightReport:
    """Machine-readable proof that one training run is safe to start."""

    schema_version: int
    config_sha256: str
    language: str
    adapter_id: str
    dataset: DatasetPreflightEvidence
    targets: TargetPreflightEvidence
    loss: LossPreflightEvidence
    output: OutputPreflightEvidence
    hardware: HardwarePreflightEvidence


def run_training_preflight(
    plan: AdapterTrainingPlan,
    *,
    repo_root: Path = Path("."),
    hardware_probe: HardwareProbe | None = None,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> TrainingPreflightReport:
    """Run every P7-004 gate without creating output artifacts or loading model weights."""

    targets = verify_frozen_lora_targets(plan)
    output = verify_training_output_path(plan, repo_root=repo_root)
    hardware = verify_training_hardware(plan, probe=hardware_probe)
    dataset = verify_frozen_training_dataset(plan)
    if tokenizer is None:
        loss = verify_training_loss_mask(plan)
    else:
        loss = verify_training_loss_mask(plan, tokenizer=tokenizer)
    return TrainingPreflightReport(
        schema_version=_SCHEMA_VERSION,
        config_sha256=plan.config_sha256,
        language=plan.language,
        adapter_id=plan.config.adapter_id,
        dataset=dataset,
        targets=targets,
        loss=loss,
        output=output,
        hardware=hardware,
    )


def training_preflight_json(report: TrainingPreflightReport) -> str:
    """Serialize P7-004 evidence deterministically."""

    return json.dumps(asdict(report), indent=2, sort_keys=True) + "\n"


def training_preflight_main(argv: list[str] | None = None) -> NoReturn:
    """CLI entry point for fail-fast training preflight."""

    parser = argparse.ArgumentParser(description="Validate a Tiny Qwen Coder training run")
    parser.add_argument("--config", type=Path, required=True, help="Adapter training YAML config")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root used to enforce safe output placement",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report destination")
    args = parser.parse_args(argv)
    plan = resolve_adapter_training_plan(args.config)
    report = run_training_preflight(plan, repo_root=args.repo_root)
    payload = training_preflight_json(report)
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    raise SystemExit(0)
