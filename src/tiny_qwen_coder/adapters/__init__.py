"""LoRA adapter metadata, compatibility, and lifecycle services."""

from tiny_qwen_coder.adapters.targets import (
    PeftTargetDiscoveryReport,
    TargetCategorySummary,
    TargetDiscoveryError,
    TargetModuleRecord,
    discover_peft_targets,
    discover_peft_targets_main,
    peft_target_report_json,
    peft_target_report_text,
)

__all__ = [
    "PeftTargetDiscoveryReport",
    "TargetCategorySummary",
    "TargetDiscoveryError",
    "TargetModuleRecord",
    "discover_peft_targets",
    "discover_peft_targets_main",
    "peft_target_report_json",
    "peft_target_report_text",
]
