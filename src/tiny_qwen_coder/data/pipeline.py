"""Generic CPU-testable dataset preparation orchestration.

P3-009 connects the generic Phase 3 stages without loading source datasets or
model weights. Concrete language/source plugins remain responsible for creating
``NormalizedTrainingRecord`` inputs; this module owns the common validation,
filtering, deduplication, splitting, and manifest sequence after normalization.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol, cast

from tiny_qwen_coder.config import DataPreparationConfig
from tiny_qwen_coder.data.deduplication import ExactDeduplicationReport, deduplicate_exact_records
from tiny_qwen_coder.data.filtering import RequiredContentFilterReport, filter_required_content
from tiny_qwen_coder.data.length_filtering import (
    LengthFilterConfig,
    TokenLengthFilterReport,
    TruncationPolicy,
    filter_by_token_length,
)
from tiny_qwen_coder.data.records import (
    NormalizedTrainingRecord,
    ValidationMetadata,
    ValidationResult,
)
from tiny_qwen_coder.data.splitting import DeduplicatedDatasetSplit, split_deduplicated_records
from tiny_qwen_coder.evaluation.contamination import (
    HighOverlapConfig,
    ProtectedBenchmarkExample,
    check_training_contamination,
)
from tiny_qwen_coder.evaluation.protected_benchmarks import ProtectedBenchmarkRegistry
from tiny_qwen_coder.languages.spec import LanguageComponentRef, LanguagePlugin
from tiny_qwen_coder.model.inspection import InspectionTarget
from tiny_qwen_coder.reporting.dataset_manifest import (
    ContaminationSummary,
    DatasetManifest,
    create_dataset_manifest,
)
from tiny_qwen_coder.reporting.manifest import GitMetadata


class DatasetPipelineError(ValueError):
    """Raised when the generic dataset pipeline cannot proceed safely."""


class LanguageRecordValidator(Protocol):
    """Callable contract for one language-plugin record validator."""

    def __call__(self, record: NormalizedTrainingRecord, /) -> ValidationResult:
        """Validate one normalized record and return stable validation evidence."""

        ...


ValidatorResolver = Callable[[LanguageComponentRef], LanguageRecordValidator]


@dataclass(frozen=True, slots=True)
class DatasetPipelineResult:
    """All auditable generic stage outputs from one dataset preparation pass."""

    content_filter: RequiredContentFilterReport
    language_validated_records: tuple[NormalizedTrainingRecord, ...]
    length_filter: TokenLengthFilterReport
    deduplication: ExactDeduplicationReport
    split: DeduplicatedDatasetSplit
    manifest: DatasetManifest

    def __post_init__(self) -> None:
        if len(self.language_validated_records) != self.content_filter.accepted_count:
            raise DatasetPipelineError(
                "language-validated record count must match content-filter accepted count"
            )
        if self.length_filter.total_records != len(self.language_validated_records):
            raise DatasetPipelineError(
                "length-filter input count must match language-validated record count"
            )
        if self.deduplication.total_records != self.length_filter.accepted_count:
            raise DatasetPipelineError(
                "deduplication input count must match length-filter accepted count"
            )
        if self.split.total_records != self.deduplication.unique_count:
            raise DatasetPipelineError("split input count must match deduplicated unique count")
        if self.manifest.counts.deduplicated_unique != self.split.total_records:
            raise DatasetPipelineError("dataset manifest must describe the pipeline split corpus")


def resolve_language_validator(component: LanguageComponentRef) -> LanguageRecordValidator:
    """Resolve one validated ``package.module:attribute`` component reference."""

    module_name, attribute_path = component.import_ref.split(":", maxsplit=1)
    try:
        resolved: object = importlib.import_module(module_name)
        for attribute in attribute_path.split("."):
            resolved = getattr(resolved, attribute)
    except (ImportError, AttributeError) as exc:
        raise DatasetPipelineError(
            f"could not resolve validator {component.id!r} from {component.import_ref!r}"
        ) from exc
    if not callable(resolved):
        raise DatasetPipelineError(f"validator {component.id!r} resolved to a non-callable object")
    return cast(LanguageRecordValidator, resolved)


def apply_language_validators(
    records: Sequence[NormalizedTrainingRecord],
    plugin: LanguagePlugin,
    *,
    resolver: ValidatorResolver = resolve_language_validator,
) -> tuple[NormalizedTrainingRecord, ...]:
    """Attach plugin-validator evidence without mutating or filtering record content.

    Validators run in the declaration order from ``LanguageSpec.validators``.
    Each validator must return ``ValidationResult`` with an ID equal to its
    component ID. Existing validation metadata is preserved, but ID collisions
    fail closed so no source/plugin evidence is silently overwritten.
    """

    resolved_validators = tuple(
        (component, resolver(component)) for component in plugin.spec.validators
    )
    output: list[NormalizedTrainingRecord] = []

    for record in records:
        existing_results = record.validation.results if record.validation is not None else ()
        existing_ids = {result.validator_id for result in existing_results}
        plugin_ids = {component.id for component, _ in resolved_validators}
        collisions = sorted(existing_ids & plugin_ids)
        if collisions:
            raise DatasetPipelineError(
                "plugin validator IDs collide with existing validation metadata: "
                + ", ".join(collisions)
            )

        new_results: list[ValidationResult] = list(existing_results)
        for component, validator in resolved_validators:
            try:
                result = validator(record)
            except Exception as exc:
                record_id = record.provenance.record_id or "<unknown>"
                raise DatasetPipelineError(
                    f"validator {component.id!r} failed for source record {record_id!r}"
                ) from exc
            if not isinstance(result, ValidationResult):
                raise DatasetPipelineError(
                    f"validator {component.id!r} must return ValidationResult"
                )
            if result.validator_id != component.id:
                raise DatasetPipelineError(
                    f"validator {component.id!r} returned mismatched validator_id "
                    f"{result.validator_id!r}"
                )
            new_results.append(result)

        if new_results:
            output.append(
                replace(record, validation=ValidationMetadata(results=tuple(new_results)))
            )
        else:
            output.append(record)

    return tuple(output)


def _validate_pipeline_config(config: DataPreparationConfig, plugin: LanguagePlugin) -> None:
    if config.language != plugin.spec.id:
        raise DatasetPipelineError(
            f"data-preparation language {config.language!r} does not match plugin {plugin.spec.id!r}"
        )
    if config.min_tokens < 1:
        raise DatasetPipelineError("generic Phase 3 pipeline requires min_tokens >= 1")
    if config.truncation_policy != "reject":
        raise DatasetPipelineError(
            "generic Phase 3 pipeline supports only explicit overlength rejection"
        )
    if not config.deduplicate:
        raise DatasetPipelineError(
            "generic Phase 3 pipeline requires deduplicate=true before leakage-safe splitting"
        )


def run_dataset_pipeline(
    input_records: Sequence[NormalizedTrainingRecord],
    *,
    config: DataPreparationConfig,
    plugin: LanguagePlugin,
    tokenizer: object,
    target: InspectionTarget,
    validator_resolver: ValidatorResolver = resolve_language_validator,
    contamination: ContaminationSummary | None = None,
    protected_benchmarks: ProtectedBenchmarkRegistry | None = None,
    protected_examples: Sequence[ProtectedBenchmarkExample] | None = None,
    contamination_overlap: HighOverlapConfig | None = None,
    repo_root: Path = Path("."),
    git: GitMetadata | None = None,
) -> DatasetPipelineResult:
    """Run the common Phase 3 pipeline over already-normalized upstream records.

    Protected-benchmark registration and SFT source access are checked before
    any record processing. Stage order is then fixed and auditable:

    1. generic required-content normalization/filtering (P3-004),
    2. language-plugin validation evidence,
    3. full tokenizer-aware length filtering with no truncation (P3-005),
    4. exact deduplication (P3-006),
    5. protected-benchmark contamination checks when examples are supplied (P4-002),
    6. linkage-safe deterministic splitting (P3-007), and
    7. deterministic dataset-manifest generation (P3-008).

    Language-specific source loading and normalization into
    ``NormalizedTrainingRecord`` remain outside this function.
    """

    _validate_pipeline_config(config, plugin)
    benchmark_registry = (
        protected_benchmarks
        if protected_benchmarks is not None
        else ProtectedBenchmarkRegistry()
    )
    benchmark_registry.assert_plugin_registration_matches(plugin)
    benchmark_registry.assert_sft_config_allowed(config)
    if contamination is not None and protected_examples is not None:
        raise DatasetPipelineError(
            "provide either external contamination evidence or protected_examples, not both"
        )
    if contamination_overlap is not None and protected_examples is None:
        raise DatasetPipelineError("contamination_overlap requires protected_examples")
    records = tuple(input_records)
    content_filter = filter_required_content(records)
    validated_records = apply_language_validators(
        content_filter.accepted_records,
        plugin,
        resolver=validator_resolver,
    )
    length_filter = filter_by_token_length(
        validated_records,
        tokenizer,
        target,
        config=LengthFilterConfig(
            min_tokens=config.min_tokens,
            max_tokens=config.max_tokens,
            truncation_policy=TruncationPolicy.REJECT,
        ),
    )
    deduplication = deduplicate_exact_records(length_filter.accepted_records)
    selected_contamination = (
        check_training_contamination(
            deduplication.unique_records,
            protected_examples,
            language=config.language,
            registry=benchmark_registry,
            overlap=contamination_overlap,
        )
        if protected_examples is not None
        else contamination or ContaminationSummary.not_run()
    )
    split = split_deduplicated_records(
        deduplication,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
    )
    manifest = create_dataset_manifest(
        config=config,
        input_records=records,
        content_filter=content_filter,
        length_filter=length_filter,
        deduplication=deduplication,
        split=split,
        contamination=selected_contamination,
        repo_root=repo_root,
        git=git,
    )
    return DatasetPipelineResult(
        content_filter=content_filter,
        language_validated_records=validated_records,
        length_filter=length_filter,
        deduplication=deduplication,
        split=split,
        manifest=manifest,
    )
