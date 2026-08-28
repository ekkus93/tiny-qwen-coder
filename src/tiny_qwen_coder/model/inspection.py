"""Canonical base-model inspection with deterministic text and JSON reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence, Sized
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, TypeAlias, cast

import torch
import yaml
from torch import nn

ComponentName: TypeAlias = Literal[
    "text_backbone",
    "vision_encoder",
    "multimodal_projector",
    "other",
]
ReportFormat: TypeAlias = Literal["text", "json"]

_SCHEMA_VERSION = 1
_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_COMPONENT_ORDER: tuple[ComponentName, ...] = (
    "text_backbone",
    "vision_encoder",
    "multimodal_projector",
    "other",
)
_REQUIRED_CANONICAL_MODULES = (
    "model.language_model",
    "model.visual",
    "model.visual.merger",
    "lm_head",
)
_DTYPE_BY_NAME: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class InspectionError(ValueError):
    """Raised when canonical model inspection cannot proceed safely."""


@dataclass(frozen=True, slots=True)
class InspectionTarget:
    """Exact model/tokenizer target loaded from the canonical base config."""

    config_id: str
    model_repository: str
    model_revision: str
    tokenizer_repository: str
    tokenizer_revision: str
    model_load_dtype: str

    def __post_init__(self) -> None:
        for field_name, value in (
            ("config_id", self.config_id),
            ("model_repository", self.model_repository),
            ("tokenizer_repository", self.tokenizer_repository),
        ):
            if not value.strip():
                raise InspectionError(f"{field_name} must not be empty")
        for field_name, value in (
            ("model_revision", self.model_revision),
            ("tokenizer_revision", self.tokenizer_revision),
        ):
            if not _SHA_PATTERN.fullmatch(value):
                raise InspectionError(f"{field_name} must be an immutable 40-character Git SHA")
        if self.model_load_dtype not in _DTYPE_BY_NAME:
            raise InspectionError(
                f"unsupported model_load_dtype {self.model_load_dtype!r}; "
                f"expected one of {sorted(_DTYPE_BY_NAME)}"
            )


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Transformers model/config identity observed from the loaded checkpoint."""

    model_class: str
    config_class: str
    model_type: str | None
    architectures: tuple[str, ...]
    resolved_revision: str | None
    text_model_type: str | None
    vision_model_type: str | None
    text_layer_types: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ComponentSummary:
    """Unique parameter and linear-module ownership for one canonical component."""

    component: ComponentName
    parameter_count: int
    linear_module_count: int


@dataclass(frozen=True, slots=True)
class LinearModuleRecord:
    """One observed torch Linear module relevant to later LoRA target discovery."""

    name: str
    component: ComponentName
    class_name: str
    in_features: int
    out_features: int
    has_bias: bool


@dataclass(frozen=True, slots=True)
class TokenizerMetadata:
    """Tokenizer and chat-template identity from the pinned tokenizer revision."""

    tokenizer_class: str
    vocab_size: int | None
    tokenizer_length: int | None
    model_max_length: int | None
    bos_token_id: int | None
    eos_token_id: int | None
    pad_token_id: int | None
    padding_side: str | None
    truncation_side: str | None
    chat_template_present: bool
    chat_template_sha256: str | None
    chat_template_length: int | None


@dataclass(frozen=True, slots=True)
class ModelInspectionReport:
    """Stable machine-readable description of the canonical loaded base model."""

    schema_version: int
    target: InspectionTarget
    model: ModelMetadata
    total_parameters: int
    trainable_parameters: int
    components: tuple[ComponentSummary, ...]
    linear_modules: tuple[LinearModuleRecord, ...]
    tokenizer: TokenizerMetadata

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise InspectionError(
                f"unsupported inspection schema_version {self.schema_version}; "
                f"expected {_SCHEMA_VERSION}"
            )
        if self.total_parameters <= 0:
            raise InspectionError("total_parameters must be greater than zero")
        if not 0 <= self.trainable_parameters <= self.total_parameters:
            raise InspectionError("trainable_parameters must be between zero and total_parameters")
        if tuple(summary.component for summary in self.components) != _COMPONENT_ORDER:
            raise InspectionError("component summaries must use the canonical deterministic order")
        if sum(summary.parameter_count for summary in self.components) != self.total_parameters:
            raise InspectionError("component parameter counts must sum to total_parameters")


def _require_mapping(value: object, *, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise InspectionError(f"{field_name} must be a mapping")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise InspectionError(f"{field_name} keys must be strings")
        result[key] = item
    return result


def _require_string(mapping: Mapping[str, object], key: str, *, field_name: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InspectionError(f"{field_name}.{key} must be a non-empty string")
    return value


def load_inspection_target(path: Path) -> InspectionTarget:
    """Load the exact inspection target from a canonical base-model YAML config."""

    try:
        payload: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise InspectionError(f"unable to load base config {path}: {exc}") from exc

    root = _require_mapping(payload, field_name="base config")
    model = _require_mapping(root.get("model"), field_name="model")
    tokenizer = _require_mapping(root.get("tokenizer"), field_name="tokenizer")
    precision = _require_mapping(root.get("precision"), field_name="precision")
    return InspectionTarget(
        config_id=_require_string(root, "id", field_name="base config"),
        model_repository=_require_string(model, "repository", field_name="model"),
        model_revision=_require_string(model, "revision", field_name="model"),
        tokenizer_repository=_require_string(tokenizer, "repository", field_name="tokenizer"),
        tokenizer_revision=_require_string(tokenizer, "revision", field_name="tokenizer"),
        model_load_dtype=_require_string(precision, "model_load_dtype", field_name="precision"),
    )


def _qualified_class_name(value: object) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _optional_string_attr(value: object, name: str) -> str | None:
    attribute: object = getattr(value, name, None)
    return attribute if isinstance(attribute, str) else None


def _optional_int_attr(value: object, name: str) -> int | None:
    attribute: object = getattr(value, name, None)
    if isinstance(attribute, bool) or not isinstance(attribute, int):
        return None
    return attribute


def _string_tuple_attr(value: object, name: str) -> tuple[str, ...]:
    attribute: object = getattr(value, name, None)
    if not isinstance(attribute, Sequence) or isinstance(attribute, str | bytes):
        return ()
    result: list[str] = []
    for item in attribute:
        if isinstance(item, str):
            result.append(item)
    return tuple(result)


def _nested_object(value: object, name: str) -> object | None:
    attribute: object = getattr(value, name, None)
    return attribute


def _component_for_path(name: str) -> ComponentName:
    if name == "model.visual.merger" or name.startswith("model.visual.merger."):
        return "multimodal_projector"
    if name == "model.visual" or name.startswith("model.visual."):
        return "vision_encoder"
    if (
        name == "model.language_model"
        or name.startswith("model.language_model.")
        or name == "lm_head"
        or name.startswith("lm_head.")
    ):
        return "text_backbone"
    return "other"


def _validate_canonical_module_roots(model: nn.Module) -> None:
    names = {name for name, _module in model.named_modules()}
    missing = [name for name in _REQUIRED_CANONICAL_MODULES if name not in names]
    if missing:
        raise InspectionError(
            "canonical Qwen3.5 component roots changed or are missing: " + ", ".join(missing)
        )


def _component_parameter_counts(model: nn.Module) -> dict[ComponentName, int]:
    counts: dict[ComponentName, int] = {component: 0 for component in _COMPONENT_ORDER}
    for name, parameter in model.named_parameters(remove_duplicate=True):
        counts[_component_for_path(name)] += parameter.numel()
    return counts


def enumerate_linear_modules(model: nn.Module) -> tuple[LinearModuleRecord, ...]:
    """Enumerate observed Linear modules without selecting canonical LoRA targets."""

    records: list[LinearModuleRecord] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        records.append(
            LinearModuleRecord(
                name=name,
                component=_component_for_path(name),
                class_name=_qualified_class_name(module),
                in_features=module.in_features,
                out_features=module.out_features,
                has_bias=module.bias is not None,
            )
        )
    return tuple(sorted(records, key=lambda record: record.name))


def inspect_tokenizer(tokenizer: object) -> TokenizerMetadata:
    """Collect tokenizer/chat-template metadata without requiring the multimodal processor."""

    chat_template_value: object = getattr(tokenizer, "chat_template", None)
    chat_template = chat_template_value if isinstance(chat_template_value, str) else None
    tokenizer_length: int | None = None
    if isinstance(tokenizer, Sized):
        tokenizer_length = len(tokenizer)

    return TokenizerMetadata(
        tokenizer_class=_qualified_class_name(tokenizer),
        vocab_size=_optional_int_attr(tokenizer, "vocab_size"),
        tokenizer_length=tokenizer_length,
        model_max_length=_optional_int_attr(tokenizer, "model_max_length"),
        bos_token_id=_optional_int_attr(tokenizer, "bos_token_id"),
        eos_token_id=_optional_int_attr(tokenizer, "eos_token_id"),
        pad_token_id=_optional_int_attr(tokenizer, "pad_token_id"),
        padding_side=_optional_string_attr(tokenizer, "padding_side"),
        truncation_side=_optional_string_attr(tokenizer, "truncation_side"),
        chat_template_present=chat_template is not None,
        chat_template_sha256=(
            hashlib.sha256(chat_template.encode("utf-8")).hexdigest()
            if chat_template is not None
            else None
        ),
        chat_template_length=len(chat_template) if chat_template is not None else None,
    )


def _model_metadata(model: nn.Module) -> ModelMetadata:
    config: object = getattr(model, "config", None)
    if config is None:
        raise InspectionError("loaded model does not expose a Transformers config")

    text_config = _nested_object(config, "text_config")
    vision_config = _nested_object(config, "vision_config")
    return ModelMetadata(
        model_class=_qualified_class_name(model),
        config_class=_qualified_class_name(config),
        model_type=_optional_string_attr(config, "model_type"),
        architectures=_string_tuple_attr(config, "architectures"),
        resolved_revision=_optional_string_attr(config, "_commit_hash"),
        text_model_type=(
            _optional_string_attr(text_config, "model_type") if text_config is not None else None
        ),
        vision_model_type=(
            _optional_string_attr(vision_config, "model_type")
            if vision_config is not None
            else None
        ),
        text_layer_types=(
            _string_tuple_attr(text_config, "layer_types") if text_config is not None else ()
        ),
    )


def build_inspection_report(
    model: nn.Module,
    tokenizer: object,
    target: InspectionTarget,
) -> ModelInspectionReport:
    """Build a deterministic report from an already-loaded canonical checkpoint."""

    _validate_canonical_module_roots(model)
    metadata = _model_metadata(model)
    if (
        metadata.resolved_revision is not None
        and metadata.resolved_revision != target.model_revision
    ):
        raise InspectionError(
            "loaded model resolved to an unexpected upstream revision: "
            f"{metadata.resolved_revision} != {target.model_revision}"
        )

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    linear_modules = enumerate_linear_modules(model)
    parameter_counts = _component_parameter_counts(model)
    linear_counts: dict[ComponentName, int] = {component: 0 for component in _COMPONENT_ORDER}
    for module in linear_modules:
        linear_counts[module.component] += 1

    components = tuple(
        ComponentSummary(
            component=component,
            parameter_count=parameter_counts[component],
            linear_module_count=linear_counts[component],
        )
        for component in _COMPONENT_ORDER
    )
    return ModelInspectionReport(
        schema_version=_SCHEMA_VERSION,
        target=target,
        model=metadata,
        total_parameters=total_parameters,
        trainable_parameters=trainable_parameters,
        components=components,
        linear_modules=linear_modules,
        tokenizer=inspect_tokenizer(tokenizer),
    )


def inspection_report_json(report: ModelInspectionReport) -> str:
    """Serialize an inspection report deterministically as JSON."""

    return json.dumps(asdict(report), indent=2, sort_keys=True) + "\n"


def inspection_report_text(report: ModelInspectionReport) -> str:
    """Render the same inspection data as a deterministic human-readable report."""

    lines = [
        "Tiny Qwen Coder model inspection",
        "================================",
        f"Base: {report.target.model_repository}@{report.target.model_revision}",
        f"Tokenizer: {report.target.tokenizer_repository}@{report.target.tokenizer_revision}",
        f"Load dtype: {report.target.model_load_dtype}",
        f"Model class: {report.model.model_class}",
        f"Config class: {report.model.config_class}",
        f"Model type: {report.model.model_type or 'unknown'}",
        f"Total parameters: {report.total_parameters:,}",
        f"Trainable parameters: {report.trainable_parameters:,}",
        "",
        "Components",
        "----------",
    ]
    for summary in report.components:
        lines.append(
            f"{summary.component}: {summary.parameter_count:,} parameters; "
            f"{summary.linear_module_count} Linear modules"
        )

    tokenizer = report.tokenizer
    lines.extend(
        [
            "",
            "Tokenizer / chat template",
            "-------------------------",
            f"Class: {tokenizer.tokenizer_class}",
            f"Vocab size: {tokenizer.vocab_size}",
            f"Tokenizer length: {tokenizer.tokenizer_length}",
            f"Model max length: {tokenizer.model_max_length}",
            f"BOS/EOS/PAD IDs: {tokenizer.bos_token_id}/{tokenizer.eos_token_id}/{tokenizer.pad_token_id}",
            f"Chat template present: {tokenizer.chat_template_present}",
            f"Chat template SHA-256: {tokenizer.chat_template_sha256 or 'none'}",
            "",
            "LoRA-relevant Linear module hierarchy",
            "-------------------------------------",
        ]
    )
    for module in report.linear_modules:
        lines.append(
            f"{module.name} [{module.component}] "
            f"{module.in_features}->{module.out_features} bias={module.has_bias}"
        )
    return "\n".join(lines) + "\n"


def _device_map_for_cli(device: str) -> str | dict[str, str | int]:
    if device == "auto":
        return "auto"
    if device == "cpu":
        return {"": "cpu"}
    if device == "cuda":
        if not torch.cuda.is_available():
            raise InspectionError("--device cuda requested but CUDA is unavailable")
        return {"": 0}
    raise InspectionError("--device must be one of: auto, cpu, cuda")


def load_model_for_inspection(
    target: InspectionTarget,
    *,
    device: str,
) -> tuple[nn.Module, object]:
    """Load the exact pinned model and tokenizer needed for structural inspection."""

    from transformers import AutoModelForMultimodalLM, AutoTokenizer

    tokenizer: object = AutoTokenizer.from_pretrained(
        target.tokenizer_repository,
        revision=target.tokenizer_revision,
    )
    loaded: object = AutoModelForMultimodalLM.from_pretrained(
        target.model_repository,
        revision=target.model_revision,
        dtype=_DTYPE_BY_NAME[target.model_load_dtype],
        device_map=_device_map_for_cli(device),
        low_cpu_mem_usage=True,
    )
    if not isinstance(loaded, nn.Module):
        raise InspectionError("Transformers returned an object that is not a torch.nn.Module")
    return loaded, tokenizer


def _write_output(text: str, output: Path | None) -> None:
    if output is None:
        print(text, end="")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(output)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect the pinned canonical Qwen base model.")
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


def inspect_model(argv: Sequence[str] | None = None) -> None:
    """CLI entry point for human- or machine-readable canonical model inspection."""

    args = _argument_parser().parse_args(argv)
    target = load_inspection_target(cast(Path, args.config))
    model, tokenizer = load_model_for_inspection(target, device=cast(str, args.device))
    report = build_inspection_report(model, tokenizer, target)
    report_format = cast(ReportFormat, args.format)
    rendered = (
        inspection_report_json(report)
        if report_format == "json"
        else inspection_report_text(report)
    )
    _write_output(rendered, cast(Path | None, args.output))
