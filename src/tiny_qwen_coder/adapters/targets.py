"""Discover PEFT target modules from an inspected Qwen3.5 module hierarchy."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, TypeAlias, cast

from tiny_qwen_coder.model import (
    InspectionTarget,
    LinearModuleRecord,
    build_inspection_report,
    load_inspection_target,
    load_model_for_inspection,
)

TargetCategory: TypeAlias = Literal[
    "full_attention",
    "mlp",
    "gated_deltanet",
    "language_output",
    "vision_encoder",
    "multimodal_projector",
    "unclassified_text",
    "other",
]
ReportFormat: TypeAlias = Literal["text", "json"]

_SCHEMA_VERSION = 1
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
_SELECTED_CATEGORIES = frozenset({"full_attention", "mlp", "gated_deltanet"})
_FULL_ATTENTION_LEAVES = frozenset({"q_proj", "k_proj", "v_proj", "o_proj"})
_MLP_LEAVES = frozenset({"gate_proj", "up_proj", "down_proj"})
_GATED_DELTANET_LEAVES = frozenset(
    {"in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"}
)


class TargetDiscoveryError(ValueError):
    """Raised when language-LoRA target discovery cannot complete safely."""


@dataclass(frozen=True, slots=True)
class TargetModuleRecord:
    """One observed Linear module with its PEFT-target classification."""

    name: str
    leaf_name: str
    category: TargetCategory
    component: str
    in_features: int
    out_features: int
    has_bias: bool
    selected_by_default: bool


@dataclass(frozen=True, slots=True)
class TargetCategorySummary:
    """Deterministic summary for one PEFT target-discovery category."""

    category: TargetCategory
    module_count: int
    leaf_names: tuple[str, ...]
    selected_by_default: bool


@dataclass(frozen=True, slots=True)
class PeftTargetDiscoveryReport:
    """Observed PEFT target inventory and initial selective language-LoRA candidate."""

    schema_version: int
    model_repository: str
    model_revision: str
    linear_module_count: int
    selective_matched_module_count: int
    excluded_module_count: int
    unclassified_text_module_count: int
    selective_target_modules: tuple[str, ...]
    categories: tuple[TargetCategorySummary, ...]
    modules: tuple[TargetModuleRecord, ...]

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise TargetDiscoveryError(
                f"unsupported target-discovery schema_version {self.schema_version}; "
                f"expected {_SCHEMA_VERSION}"
            )
        if tuple(summary.category for summary in self.categories) != _CATEGORY_ORDER:
            raise TargetDiscoveryError("category summaries must use canonical deterministic order")
        if self.linear_module_count != len(self.modules):
            raise TargetDiscoveryError(
                "linear_module_count must equal the number of module records"
            )
        selected_count = sum(module.selected_by_default for module in self.modules)
        if self.selective_matched_module_count != selected_count:
            raise TargetDiscoveryError(
                "selective_matched_module_count must equal selected module records"
            )
        if self.excluded_module_count != self.linear_module_count - selected_count:
            raise TargetDiscoveryError("excluded_module_count is inconsistent with module records")
        if self.unclassified_text_module_count != sum(
            module.category == "unclassified_text" for module in self.modules
        ):
            raise TargetDiscoveryError(
                "unclassified_text_module_count is inconsistent with module records"
            )
        if tuple(sorted(set(self.selective_target_modules))) != self.selective_target_modules:
            raise TargetDiscoveryError("selective_target_modules must be sorted and unique")
        observed_selected_leaves = tuple(
            sorted({module.leaf_name for module in self.modules if module.selected_by_default})
        )
        if self.selective_target_modules != observed_selected_leaves:
            raise TargetDiscoveryError(
                "selective_target_modules must be derived from observed selected modules"
            )


def _leaf_name(name: str) -> str:
    return name.rsplit(".", maxsplit=1)[-1]


def _classify_module(module: LinearModuleRecord) -> TargetCategory:
    if module.component == "vision_encoder":
        return "vision_encoder"
    if module.component == "multimodal_projector":
        return "multimodal_projector"
    if module.component == "other":
        return "other"
    if module.component != "text_backbone":
        return "other"

    if module.name == "lm_head" or module.name.startswith("lm_head."):
        return "language_output"

    leaf_name = _leaf_name(module.name)
    if ".self_attn." in module.name and leaf_name in _FULL_ATTENTION_LEAVES:
        return "full_attention"
    if ".mlp." in module.name and leaf_name in _MLP_LEAVES:
        return "mlp"
    if ".linear_attn." in module.name and leaf_name in _GATED_DELTANET_LEAVES:
        return "gated_deltanet"
    return "unclassified_text"


def discover_peft_targets(
    linear_modules: Sequence[LinearModuleRecord],
    target: InspectionTarget,
    *,
    require_complete: bool = True,
) -> PeftTargetDiscoveryReport:
    """Classify observed linears and derive a selective language-LoRA target candidate."""

    classified: list[TargetModuleRecord] = []
    for module in sorted(linear_modules, key=lambda item: item.name):
        category = _classify_module(module)
        classified.append(
            TargetModuleRecord(
                name=module.name,
                leaf_name=_leaf_name(module.name),
                category=category,
                component=module.component,
                in_features=module.in_features,
                out_features=module.out_features,
                has_bias=module.has_bias,
                selected_by_default=category in _SELECTED_CATEGORIES,
            )
        )

    unclassified = tuple(
        module.name for module in classified if module.category == "unclassified_text"
    )
    if require_complete and unclassified:
        raise TargetDiscoveryError(
            "unclassified text-backbone Linear module(s) prevent freezing a selective candidate: "
            + ", ".join(unclassified)
        )

    modules = tuple(classified)
    categories = tuple(
        TargetCategorySummary(
            category=category,
            module_count=sum(module.category == category for module in modules),
            leaf_names=tuple(
                sorted({module.leaf_name for module in modules if module.category == category})
            ),
            selected_by_default=category in _SELECTED_CATEGORIES,
        )
        for category in _CATEGORY_ORDER
    )
    selective_target_modules = tuple(
        sorted({module.leaf_name for module in modules if module.selected_by_default})
    )
    selected_count = sum(module.selected_by_default for module in modules)
    return PeftTargetDiscoveryReport(
        schema_version=_SCHEMA_VERSION,
        model_repository=target.model_repository,
        model_revision=target.model_revision,
        linear_module_count=len(modules),
        selective_matched_module_count=selected_count,
        excluded_module_count=len(modules) - selected_count,
        unclassified_text_module_count=len(unclassified),
        selective_target_modules=selective_target_modules,
        categories=categories,
        modules=modules,
    )


def peft_target_report_json(report: PeftTargetDiscoveryReport) -> str:
    """Serialize a PEFT target-discovery report deterministically as JSON."""

    return json.dumps(asdict(report), indent=2, sort_keys=True) + "\n"


def peft_target_report_text(report: PeftTargetDiscoveryReport) -> str:
    """Render PEFT target discovery in a deterministic human-readable form."""

    lines = [
        "Tiny Qwen Coder PEFT target discovery",
        "=====================================",
        f"Base: {report.model_repository}@{report.model_revision}",
        f"Observed Linear/projection modules: {report.linear_module_count}",
        f"Selective language-LoRA modules: {report.selective_matched_module_count}",
        f"Excluded by default: {report.excluded_module_count}",
        f"Unclassified text linears: {report.unclassified_text_module_count}",
        "",
        "Selective PEFT target module names",
        "----------------------------------",
    ]
    lines.extend(f"- {name}" for name in report.selective_target_modules)
    lines.extend(["", "Categories", "----------"])
    for summary in report.categories:
        leaf_names = ", ".join(summary.leaf_names) if summary.leaf_names else "none"
        lines.append(
            f"{summary.category}: {summary.module_count} modules; "
            f"selected={summary.selected_by_default}; leaves={leaf_names}"
        )

    lines.extend(["", "Observed module hierarchy", "-------------------------"])
    for module in report.modules:
        lines.append(
            f"{module.name} [{module.category}] {module.in_features}->{module.out_features} "
            f"bias={module.has_bias} selected={module.selected_by_default}"
        )
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
        description="Discover selective PEFT targets from the pinned Qwen module hierarchy."
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
        help="Device placement policy used while loading the checkpoint.",
    )
    parser.add_argument("--output", type=Path, help="Optional report output path.")
    return parser


def discover_peft_targets_main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point for observed, revision-bound PEFT target discovery."""

    args = _argument_parser().parse_args(argv)
    target = load_inspection_target(cast(Path, args.config))
    model, tokenizer = load_model_for_inspection(target, device=cast(str, args.device))
    inspection = build_inspection_report(model, tokenizer, target)
    report = discover_peft_targets(inspection.linear_modules, target)
    report_format = cast(ReportFormat, args.format)
    rendered = (
        peft_target_report_json(report)
        if report_format == "json"
        else peft_target_report_text(report)
    )
    _write_output(rendered, cast(Path | None, args.output))
