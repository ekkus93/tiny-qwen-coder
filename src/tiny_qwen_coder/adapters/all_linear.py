"""Validate literal PEFT all-linear targeting against the inspected canonical model."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Literal, TypeAlias, cast

from torch import nn
from transformers import PreTrainedModel

from tiny_qwen_coder.adapters.targets import (
    PeftTargetDiscoveryReport,
    TargetCategory,
    discover_peft_targets,
)
from tiny_qwen_coder.model import (
    InspectionTarget,
    build_inspection_report,
    load_inspection_target,
    load_model_for_inspection,
)

ReportFormat: TypeAlias = Literal["text", "json"]

_SCHEMA_VERSION = 1
_DEFAULT_RANK = 16
_DEFAULT_ALPHA = 32
_DEFAULT_DROPOUT = 0.05
_DEFAULT_BIAS: Literal["none"] = "none"
_CATEGORY_ORDER: tuple[TargetCategory, ...] = (
    "full_attention",
    "mlp",
    "gated_deltanet",
    "language_output",
    "vision_encoder",
    "multimodal_projector",
    "unclassified_text",
    "other",
)


class AllLinearValidationError(ValueError):
    """Raised when literal PEFT all-linear validation is incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class AllLinearCategorySummary:
    """Matched-module count for one P2 target-discovery category."""

    category: TargetCategory
    matched_module_count: int


@dataclass(frozen=True, slots=True)
class AllLinearValidationReport:
    """Observed result of attaching literal PEFT ``target_modules='all-linear'``."""

    schema_version: int
    model_repository: str
    model_revision: str
    peft_version: str
    target_modules: str
    rank: int
    alpha: int
    dropout: float
    bias: str
    observed_linear_module_count: int
    selective_candidate_module_count: int
    matched_module_count: int
    trainable_parameter_count: int
    selective_expected_trainable_parameter_count: int
    selective_overlap_count: int
    extra_vs_selective_count: int
    missing_vs_selective_count: int
    literal_all_linear_is_language_only: bool
    language_scoped_all_linear_is_selective_equivalent: bool
    categories: tuple[AllLinearCategorySummary, ...]
    matched_modules: tuple[str, ...]
    extra_vs_selective_modules: tuple[str, ...]
    missing_vs_selective_modules: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise AllLinearValidationError(
                f"unsupported all-linear schema_version {self.schema_version}; "
                f"expected {_SCHEMA_VERSION}"
            )
        if self.target_modules != "all-linear":
            raise AllLinearValidationError("target_modules must be literal 'all-linear'")
        if self.rank <= 0 or self.alpha <= 0:
            raise AllLinearValidationError("rank and alpha must be greater than zero")
        if not 0.0 <= self.dropout < 1.0:
            raise AllLinearValidationError("dropout must be in [0, 1)")
        if self.trainable_parameter_count <= 0:
            raise AllLinearValidationError("trainable_parameter_count must be greater than zero")
        if tuple(summary.category for summary in self.categories) != _CATEGORY_ORDER:
            raise AllLinearValidationError("category summaries must use canonical order")
        if self.matched_module_count != len(self.matched_modules):
            raise AllLinearValidationError("matched_module_count is inconsistent")
        if self.extra_vs_selective_count != len(self.extra_vs_selective_modules):
            raise AllLinearValidationError("extra_vs_selective_count is inconsistent")
        if self.missing_vs_selective_count != len(self.missing_vs_selective_modules):
            raise AllLinearValidationError("missing_vs_selective_count is inconsistent")
        if (
            self.selective_overlap_count + self.extra_vs_selective_count
            != self.matched_module_count
        ):
            raise AllLinearValidationError("matched-module overlap accounting is inconsistent")


def _expected_lora_parameter_count(
    discovery: PeftTargetDiscoveryReport,
    *,
    rank: int,
) -> int:
    return sum(
        rank * (module.in_features + module.out_features)
        for module in discovery.modules
        if module.selected_by_default
    )


def build_all_linear_validation_report(
    discovery: PeftTargetDiscoveryReport,
    target: InspectionTarget,
    *,
    matched_module_names: Sequence[str],
    trainable_parameter_count: int,
    peft_version: str,
    rank: int = _DEFAULT_RANK,
    alpha: int = _DEFAULT_ALPHA,
    dropout: float = _DEFAULT_DROPOUT,
    bias: str = _DEFAULT_BIAS,
) -> AllLinearValidationReport:
    """Compare observed literal all-linear PEFT matches with the P2 selective candidate."""

    if discovery.model_repository != target.model_repository:
        raise AllLinearValidationError(
            "discovery model repository does not match inspection target"
        )
    if discovery.model_revision != target.model_revision:
        raise AllLinearValidationError("discovery model revision does not match inspection target")
    if rank <= 0:
        raise AllLinearValidationError("rank must be greater than zero")
    if trainable_parameter_count <= 0:
        raise AllLinearValidationError("trainable_parameter_count must be greater than zero")

    module_by_name = {module.name: module for module in discovery.modules}
    matched = tuple(sorted(set(matched_module_names)))
    unknown = tuple(name for name in matched if name not in module_by_name)
    if unknown:
        raise AllLinearValidationError(
            "PEFT matched module(s) absent from the inspected Linear inventory: "
            + ", ".join(unknown)
        )

    selective = frozenset(module.name for module in discovery.modules if module.selected_by_default)
    matched_set = frozenset(matched)
    missing = tuple(sorted(selective - matched_set))
    if missing:
        raise AllLinearValidationError(
            "literal all-linear unexpectedly missed selective module(s): " + ", ".join(missing)
        )
    extra = tuple(sorted(matched_set - selective))

    categories = tuple(
        AllLinearCategorySummary(
            category=category,
            matched_module_count=sum(module_by_name[name].category == category for name in matched),
        )
        for category in _CATEGORY_ORDER
    )
    unsafe_categories = {"vision_encoder", "multimodal_projector", "other"}
    literal_is_language_only = not any(
        module_by_name[name].category in unsafe_categories for name in matched
    )

    language_body = frozenset(
        module.name
        for module in discovery.modules
        if module.component == "text_backbone" and module.category != "language_output"
    )
    scoped_equivalent = language_body == selective

    return AllLinearValidationReport(
        schema_version=_SCHEMA_VERSION,
        model_repository=target.model_repository,
        model_revision=target.model_revision,
        peft_version=peft_version,
        target_modules="all-linear",
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        bias=bias,
        observed_linear_module_count=discovery.linear_module_count,
        selective_candidate_module_count=len(selective),
        matched_module_count=len(matched),
        trainable_parameter_count=trainable_parameter_count,
        selective_expected_trainable_parameter_count=_expected_lora_parameter_count(
            discovery,
            rank=rank,
        ),
        selective_overlap_count=len(matched_set & selective),
        extra_vs_selective_count=len(extra),
        missing_vs_selective_count=len(missing),
        literal_all_linear_is_language_only=literal_is_language_only,
        language_scoped_all_linear_is_selective_equivalent=scoped_equivalent,
        categories=categories,
        matched_modules=matched,
        extra_vs_selective_modules=extra,
        missing_vs_selective_modules=missing,
    )


def _normalize_peft_module_name(name: str) -> str:
    for prefix in ("base_model.model.", "base_model."):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _matched_lora_module_names(model: nn.Module) -> tuple[str, ...]:
    matched: list[str] = []
    for name, module in model.named_modules():
        lora_a: object = getattr(module, "lora_A", None)
        lora_b: object = getattr(module, "lora_B", None)
        if isinstance(lora_a, nn.ModuleDict) and isinstance(lora_b, nn.ModuleDict):
            matched.append(_normalize_peft_module_name(name))
    return tuple(sorted(set(matched)))


def validate_literal_all_linear(
    model: nn.Module,
    discovery: PeftTargetDiscoveryReport,
    target: InspectionTarget,
    *,
    rank: int = _DEFAULT_RANK,
    alpha: int = _DEFAULT_ALPHA,
    dropout: float = _DEFAULT_DROPOUT,
) -> AllLinearValidationReport:
    """Attach literal PEFT all-linear LoRA and report exactly what PEFT wrapped."""

    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias=_DEFAULT_BIAS,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    attached = cast(nn.Module, get_peft_model(cast(PreTrainedModel, model), config))
    matched = _matched_lora_module_names(attached)
    trainable = sum(
        parameter.numel() for parameter in attached.parameters() if parameter.requires_grad
    )
    return build_all_linear_validation_report(
        discovery,
        target,
        matched_module_names=matched,
        trainable_parameter_count=trainable,
        peft_version=version("peft"),
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    )


def all_linear_report_json(report: AllLinearValidationReport) -> str:
    """Serialize the validation report deterministically as JSON."""

    return json.dumps(asdict(report), indent=2, sort_keys=True) + "\n"


def all_linear_report_text(report: AllLinearValidationReport) -> str:
    """Render a concise human-readable validation report."""

    lines = [
        "Tiny Qwen Coder PEFT all-linear validation",
        "==========================================",
        f"Base: {report.model_repository}@{report.model_revision}",
        f"PEFT: {report.peft_version}",
        f"LoRA: r={report.rank} alpha={report.alpha} dropout={report.dropout} bias={report.bias}",
        f"Observed Linear modules: {report.observed_linear_module_count}",
        f"Literal all-linear matches: {report.matched_module_count}",
        f"Trainable parameters: {report.trainable_parameter_count:,}",
        f"Selective candidate matches: {report.selective_candidate_module_count}",
        f"Selective overlap: {report.selective_overlap_count}",
        f"Extra vs selective: {report.extra_vs_selective_count}",
        f"Missing vs selective: {report.missing_vs_selective_count}",
        f"Literal all-linear language-only safe: {report.literal_all_linear_is_language_only}",
        "Language-scoped all-linear equals selective candidate: "
        f"{report.language_scoped_all_linear_is_selective_equivalent}",
        "",
        "Matched categories",
        "------------------",
    ]
    for summary in report.categories:
        lines.append(f"{summary.category}: {summary.matched_module_count}")
    lines.extend(["", "Extra modules vs selective", "--------------------------"])
    lines.extend(report.extra_vs_selective_modules or ("none",))
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
        description="Validate literal PEFT all-linear targeting on the pinned canonical model."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/base/qwen35-4b.yaml"),
        help="Canonical base-model config path.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Report format written to stdout or --output.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device placement used while attaching the PEFT probe adapter.",
    )
    parser.add_argument("--output", type=Path, help="Optional report output path.")
    return parser


def validate_all_linear_main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point for revision-bound literal PEFT all-linear validation."""

    args = _argument_parser().parse_args(argv)
    target = load_inspection_target(cast(Path, args.config))
    model, tokenizer = load_model_for_inspection(target, device=cast(str, args.device))
    inspection = build_inspection_report(model, tokenizer, target)
    discovery = discover_peft_targets(inspection.linear_modules, target)
    report = validate_literal_all_linear(model, discovery, target)
    report_format = cast(ReportFormat, args.format)
    rendered = (
        all_linear_report_json(report)
        if report_format == "json"
        else all_linear_report_text(report)
    )
    _write_output(rendered, cast(Path | None, args.output))
