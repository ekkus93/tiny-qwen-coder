"""Fail-closed compatibility validation for portable adapter manifests."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Literal, TypeAlias

from tiny_qwen_coder.adapters.manifest import AdapterManifest
from tiny_qwen_coder.adapters.targets import PeftTargetDiscoveryReport
from tiny_qwen_coder.model import ModelInspectionReport

CompatibilityIssueCode: TypeAlias = Literal[
    "base_repository_mismatch",
    "base_revision_mismatch",
    "tokenizer_repository_mismatch",
    "tokenizer_revision_mismatch",
    "chat_template_identifier_mismatch",
    "chat_template_hash_mismatch",
    "chat_template_hash_unverifiable",
    "lora_target_module_missing",
    "lora_target_scope_mismatch",
    "lora_trainable_parameter_count_mismatch",
]

_GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class AdapterCompatibilityError(ValueError):
    """Raised when an adapter cannot be proven compatible with a target base."""


def _require_non_empty(value: str, *, field_name: str) -> None:
    if not value.strip():
        raise AdapterCompatibilityError(f"{field_name} must not be empty")


def _require_git_sha(value: str, *, field_name: str) -> None:
    if not _GIT_SHA_PATTERN.fullmatch(value):
        raise AdapterCompatibilityError(f"{field_name} must be a lowercase 40-character Git SHA")


def _require_sha256(value: str, *, field_name: str) -> None:
    if not _SHA256_PATTERN.fullmatch(value):
        raise AdapterCompatibilityError(f"{field_name} must be a lowercase 64-character SHA-256")


@dataclass(frozen=True, slots=True)
class CompatibleLinearModule:
    """Observed Linear geometry and language-LoRA scope evidence."""

    name: str
    in_features: int
    out_features: int
    has_bias: bool
    language_lora_allowed: bool

    def __post_init__(self) -> None:
        _require_non_empty(self.name, field_name="linear_modules[].name")
        if self.in_features <= 0:
            raise AdapterCompatibilityError(
                "linear_modules[].in_features must be greater than zero"
            )
        if self.out_features <= 0:
            raise AdapterCompatibilityError(
                "linear_modules[].out_features must be greater than zero"
            )


@dataclass(frozen=True, slots=True)
class AdapterCompatibilityTarget:
    """Exact loaded-base identity and architecture evidence used for validation."""

    base_repository: str
    base_revision: str
    tokenizer_repository: str
    tokenizer_revision: str
    chat_template_identifier: str
    chat_template_sha256: str | None
    linear_modules: tuple[CompatibleLinearModule, ...]

    def __post_init__(self) -> None:
        _require_non_empty(self.base_repository, field_name="base_repository")
        _require_git_sha(self.base_revision, field_name="base_revision")
        _require_non_empty(self.tokenizer_repository, field_name="tokenizer_repository")
        _require_git_sha(self.tokenizer_revision, field_name="tokenizer_revision")
        _require_non_empty(
            self.chat_template_identifier,
            field_name="chat_template_identifier",
        )
        if self.chat_template_sha256 is not None:
            _require_sha256(
                self.chat_template_sha256,
                field_name="chat_template_sha256",
            )
        if not self.linear_modules:
            raise AdapterCompatibilityError("linear_modules must contain observed Linear modules")
        names = tuple(module.name for module in self.linear_modules)
        if names != tuple(sorted(names)):
            raise AdapterCompatibilityError("linear_modules must be sorted by module name")
        if len(names) != len(set(names)):
            raise AdapterCompatibilityError("linear_modules must not contain duplicate names")


@dataclass(frozen=True, slots=True)
class CompatibilityIssue:
    """One deterministic reason an adapter is incompatible with a target."""

    code: CompatibilityIssueCode
    field: str
    adapter_value: str
    target_value: str


@dataclass(frozen=True, slots=True)
class AdapterCompatibilityReport:
    """Deterministic compatibility result for one adapter and one target base."""

    adapter_id: str
    compatible: bool
    issues: tuple[CompatibilityIssue, ...]
    expected_lora_trainable_parameters: int | None

    def __post_init__(self) -> None:
        if self.compatible != (not self.issues):
            raise AdapterCompatibilityError("compatible must be true exactly when issues is empty")
        if (
            self.expected_lora_trainable_parameters is not None
            and self.expected_lora_trainable_parameters <= 0
        ):
            raise AdapterCompatibilityError(
                "expected_lora_trainable_parameters must be greater than zero"
            )


def build_compatibility_target(
    inspection: ModelInspectionReport,
    discovery: PeftTargetDiscoveryReport,
    *,
    chat_template_identifier: str,
) -> AdapterCompatibilityTarget:
    """Build compatibility evidence from P2 inspection and target-discovery reports."""

    target = inspection.target
    if discovery.model_repository != target.model_repository:
        raise AdapterCompatibilityError(
            "target discovery repository does not match model inspection target"
        )
    if discovery.model_revision != target.model_revision:
        raise AdapterCompatibilityError(
            "target discovery revision does not match model inspection target"
        )
    if not inspection.tokenizer.chat_template_present:
        raise AdapterCompatibilityError(
            "model inspection target does not expose the required chat template"
        )

    modules = tuple(
        CompatibleLinearModule(
            name=module.name,
            in_features=module.in_features,
            out_features=module.out_features,
            has_bias=module.has_bias,
            language_lora_allowed=module.selected_by_default,
        )
        for module in sorted(discovery.modules, key=lambda item: item.name)
    )
    return AdapterCompatibilityTarget(
        base_repository=target.model_repository,
        base_revision=target.model_revision,
        tokenizer_repository=target.tokenizer_repository,
        tokenizer_revision=target.tokenizer_revision,
        chat_template_identifier=chat_template_identifier,
        chat_template_sha256=inspection.tokenizer.chat_template_sha256,
        linear_modules=modules,
    )


def _issue(
    code: CompatibilityIssueCode,
    *,
    field: str,
    adapter_value: object,
    target_value: object,
) -> CompatibilityIssue:
    return CompatibilityIssue(
        code=code,
        field=field,
        adapter_value=str(adapter_value),
        target_value=str(target_value),
    )


def _expected_lora_parameter_count(
    manifest: AdapterManifest,
    module_by_name: dict[str, CompatibleLinearModule],
) -> int | None:
    """Return exact LoRA A/B parameter count when the manifest bias mode permits it."""

    if manifest.lora.bias != "none":
        return None
    if any(name not in module_by_name for name in manifest.lora.target_modules):
        return None
    return sum(
        manifest.lora.rank * (module_by_name[name].in_features + module_by_name[name].out_features)
        for name in manifest.lora.target_modules
    )


def validate_adapter_compatibility(
    manifest: AdapterManifest,
    target: AdapterCompatibilityTarget,
) -> AdapterCompatibilityReport:
    """Compare an adapter manifest with exact loaded-base compatibility evidence."""

    issues: list[CompatibilityIssue] = []

    if manifest.base_model.repository != target.base_repository:
        issues.append(
            _issue(
                "base_repository_mismatch",
                field="base_model.repository",
                adapter_value=manifest.base_model.repository,
                target_value=target.base_repository,
            )
        )
    if manifest.base_model.revision != target.base_revision:
        issues.append(
            _issue(
                "base_revision_mismatch",
                field="base_model.revision",
                adapter_value=manifest.base_model.revision,
                target_value=target.base_revision,
            )
        )
    if manifest.tokenizer.repository != target.tokenizer_repository:
        issues.append(
            _issue(
                "tokenizer_repository_mismatch",
                field="tokenizer.repository",
                adapter_value=manifest.tokenizer.repository,
                target_value=target.tokenizer_repository,
            )
        )
    if manifest.tokenizer.revision != target.tokenizer_revision:
        issues.append(
            _issue(
                "tokenizer_revision_mismatch",
                field="tokenizer.revision",
                adapter_value=manifest.tokenizer.revision,
                target_value=target.tokenizer_revision,
            )
        )
    if manifest.tokenizer.chat_template.identifier != target.chat_template_identifier:
        issues.append(
            _issue(
                "chat_template_identifier_mismatch",
                field="tokenizer.chat_template.identifier",
                adapter_value=manifest.tokenizer.chat_template.identifier,
                target_value=target.chat_template_identifier,
            )
        )

    adapter_template_hash = manifest.tokenizer.chat_template.sha256
    if adapter_template_hash is not None:
        if target.chat_template_sha256 is None:
            issues.append(
                _issue(
                    "chat_template_hash_unverifiable",
                    field="tokenizer.chat_template.sha256",
                    adapter_value=adapter_template_hash,
                    target_value="<unavailable>",
                )
            )
        elif adapter_template_hash != target.chat_template_sha256:
            issues.append(
                _issue(
                    "chat_template_hash_mismatch",
                    field="tokenizer.chat_template.sha256",
                    adapter_value=adapter_template_hash,
                    target_value=target.chat_template_sha256,
                )
            )

    module_by_name = {module.name: module for module in target.linear_modules}
    missing_modules = tuple(
        sorted(name for name in manifest.lora.target_modules if name not in module_by_name)
    )
    if missing_modules:
        issues.append(
            _issue(
                "lora_target_module_missing",
                field="lora.target_modules",
                adapter_value=", ".join(missing_modules),
                target_value="<missing from observed Linear inventory>",
            )
        )

    if manifest.family == "language":
        out_of_scope_modules = tuple(
            sorted(
                name
                for name in manifest.lora.target_modules
                if name in module_by_name and not module_by_name[name].language_lora_allowed
            )
        )
        if out_of_scope_modules:
            issues.append(
                _issue(
                    "lora_target_scope_mismatch",
                    field="lora.target_modules",
                    adapter_value=", ".join(out_of_scope_modules),
                    target_value="<language-LoRA-allowed modules only>",
                )
            )

    expected_trainable = _expected_lora_parameter_count(manifest, module_by_name)
    if expected_trainable is not None and manifest.lora.trainable_parameters != expected_trainable:
        issues.append(
            _issue(
                "lora_trainable_parameter_count_mismatch",
                field="lora.trainable_parameters",
                adapter_value=manifest.lora.trainable_parameters,
                target_value=expected_trainable,
            )
        )

    return AdapterCompatibilityReport(
        adapter_id=manifest.adapter_id,
        compatible=not issues,
        issues=tuple(issues),
        expected_lora_trainable_parameters=expected_trainable,
    )


def require_adapter_compatible(
    manifest: AdapterManifest,
    target: AdapterCompatibilityTarget,
) -> AdapterCompatibilityReport:
    """Return a compatible report or raise with deterministic incompatibility reasons."""

    report = validate_adapter_compatibility(manifest, target)
    if report.compatible:
        return report

    details = "; ".join(
        f"{issue.code} ({issue.field}: adapter={issue.adapter_value!r}, "
        f"target={issue.target_value!r})"
        for issue in report.issues
    )
    raise AdapterCompatibilityError(f"adapter {manifest.adapter_id!r} is incompatible: {details}")


def compatibility_report_json(report: AdapterCompatibilityReport) -> str:
    """Serialize a compatibility result deterministically as JSON."""

    return json.dumps(asdict(report), indent=2, sort_keys=True) + "\n"
