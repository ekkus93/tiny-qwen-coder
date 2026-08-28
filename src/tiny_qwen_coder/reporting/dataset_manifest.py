"""Deterministic, auditable manifests for prepared training corpora."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from tiny_qwen_coder.config import DataPreparationConfig
from tiny_qwen_coder.data.deduplication import DuplicateReasonCount, ExactDeduplicationReport
from tiny_qwen_coder.data.filtering import RejectionReasonCount, RequiredContentFilterReport
from tiny_qwen_coder.data.length_filtering import (
    LengthRejectionCount,
    TokenLengthDistribution,
    TokenLengthFilterReport,
)
from tiny_qwen_coder.data.records import LicenseMetadata, NormalizedTrainingRecord
from tiny_qwen_coder.data.splitting import DatasetSplitMembership, DeduplicatedDatasetSplit
from tiny_qwen_coder.reporting.manifest import GitMetadata, collect_git_metadata

_SCHEMA_VERSION = 1
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_COMPONENT_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]*$")


class DatasetManifestError(ValueError):
    """Raised when prepared-dataset provenance is incomplete or inconsistent."""


class ContaminationStatus(StrEnum):
    """Whether protected-data contamination checks ran and what they found."""

    NOT_RUN = "not_run"
    CLEAN = "clean"
    FINDINGS = "findings"


@dataclass(frozen=True, slots=True)
class DatasetSourceSummary:
    """Auditable upstream source/revision/license identity and record counts."""

    source_id: str
    revision: str
    license: LicenseMetadata
    input_records: int
    prepared_records: int

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise DatasetManifestError("source_id must not be empty")
        if not self.revision.strip():
            raise DatasetManifestError("source revision must not be empty")
        if self.input_records <= 0:
            raise DatasetManifestError("source input_records must be greater than zero")
        if self.prepared_records < 0:
            raise DatasetManifestError("source prepared_records must not be negative")
        if self.prepared_records > self.input_records:
            raise DatasetManifestError("source prepared_records must not exceed input_records")


@dataclass(frozen=True, slots=True)
class DatasetPreparationIdentity:
    """Exact code and data-preparation configuration identity."""

    git: GitMetadata
    config_sha256: str
    config: DataPreparationConfig

    def __post_init__(self) -> None:
        _require_sha256(self.config_sha256, field_name="config_sha256")


@dataclass(frozen=True, slots=True)
class DatasetCountSummary:
    """Record counts at each generic preparation boundary."""

    input_records: int
    content_accepted: int
    content_rejected: int
    length_accepted: int
    length_rejected: int
    deduplicated_unique: int
    duplicates_removed: int
    train_records: int
    validation_records: int

    def __post_init__(self) -> None:
        values = asdict(self)
        if any(value < 0 for value in values.values()):
            raise DatasetManifestError("dataset counts must not be negative")
        if self.input_records != self.content_accepted + self.content_rejected:
            raise DatasetManifestError("input count must equal content accepted plus rejected")
        if self.content_accepted != self.length_accepted + self.length_rejected:
            raise DatasetManifestError("content accepted must equal length accepted plus rejected")
        if self.length_accepted != self.deduplicated_unique + self.duplicates_removed:
            raise DatasetManifestError("length accepted must equal unique plus duplicates removed")
        if self.deduplicated_unique != self.train_records + self.validation_records:
            raise DatasetManifestError("unique count must equal train plus validation")


@dataclass(frozen=True, slots=True)
class DatasetTokenizerSummary:
    """Tokenizer/template identity and observed token-length statistics."""

    repository: str
    revision: str
    tokenizer_class: str
    chat_template_sha256: str
    min_tokens: int
    max_tokens: int
    truncation_policy: str
    measured_distribution: TokenLengthDistribution
    accepted_distribution: TokenLengthDistribution

    def __post_init__(self) -> None:
        if not self.repository.strip():
            raise DatasetManifestError("tokenizer repository must not be empty")
        if not self.revision.strip():
            raise DatasetManifestError("tokenizer revision must not be empty")
        if not self.tokenizer_class.strip():
            raise DatasetManifestError("tokenizer class must not be empty")
        _require_sha256(self.chat_template_sha256, field_name="chat_template_sha256")
        if self.min_tokens < 1:
            raise DatasetManifestError("min_tokens must be at least one")
        if self.max_tokens < self.min_tokens:
            raise DatasetManifestError("max_tokens must be greater than or equal to min_tokens")
        if self.truncation_policy != "reject":
            raise DatasetManifestError(
                "P3 dataset manifests require explicit reject truncation policy"
            )


@dataclass(frozen=True, slots=True)
class DatasetSplitSummary:
    """Deterministic linkage-aware train/validation split evidence."""

    requested_validation_fraction: float
    target_validation_records: int
    actual_validation_fraction: float
    linked_prompt_group_count: int

    def __post_init__(self) -> None:
        for field_name, value in (
            ("requested_validation_fraction", self.requested_validation_fraction),
            ("actual_validation_fraction", self.actual_validation_fraction),
        ):
            if not 0.0 < value < 1.0:
                raise DatasetManifestError(f"{field_name} must be greater than 0 and less than 1")
        if self.target_validation_records <= 0:
            raise DatasetManifestError("target_validation_records must be greater than zero")
        if self.linked_prompt_group_count < 2:
            raise DatasetManifestError("linked_prompt_group_count must be at least two")


@dataclass(frozen=True, slots=True)
class DatasetChecksums:
    """Stable checksums for input, corpus, partitions, and membership evidence."""

    input_records_sha256: str
    unique_corpus_sha256: str
    train_records_sha256: str
    validation_records_sha256: str
    split_membership_sha256: str

    def __post_init__(self) -> None:
        for field_name, value in asdict(self).items():
            _require_sha256(value, field_name=field_name)


@dataclass(frozen=True, slots=True)
class ContaminationFinding:
    """One finding emitted by a present or future protected-data checker."""

    checker_id: str
    protected_dataset_id: str
    finding_type: str
    training_record_sha256: str
    protected_record_id: str | None = None
    score: float | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        for field_name, value in (
            ("checker_id", self.checker_id),
            ("protected_dataset_id", self.protected_dataset_id),
            ("finding_type", self.finding_type),
        ):
            if not _COMPONENT_ID_PATTERN.fullmatch(value):
                raise DatasetManifestError(
                    f"{field_name} must be a stable lowercase component identifier"
                )
        _require_sha256(self.training_record_sha256, field_name="training_record_sha256")
        if self.protected_record_id is not None and not self.protected_record_id.strip():
            raise DatasetManifestError("protected_record_id must not be empty when provided")
        if self.score is not None and not 0.0 <= self.score <= 1.0:
            raise DatasetManifestError("contamination score must be between 0 and 1")
        if self.detail is not None and not self.detail.strip():
            raise DatasetManifestError("contamination detail must not be empty when provided")


@dataclass(frozen=True, slots=True)
class ContaminationSummary:
    """Explicit contamination-check state so 'not run' is never confused with clean."""

    status: ContaminationStatus
    check_ids: tuple[str, ...]
    findings: tuple[ContaminationFinding, ...]

    def __post_init__(self) -> None:
        if tuple(sorted(self.check_ids)) != self.check_ids:
            raise DatasetManifestError("contamination check_ids must be sorted")
        if len(self.check_ids) != len(set(self.check_ids)):
            raise DatasetManifestError("contamination check_ids must be unique")
        for check_id in self.check_ids:
            if not _COMPONENT_ID_PATTERN.fullmatch(check_id):
                raise DatasetManifestError(
                    "contamination check_ids must be stable lowercase component identifiers"
                )
        finding_check_ids = {finding.checker_id for finding in self.findings}
        if not finding_check_ids.issubset(self.check_ids):
            raise DatasetManifestError(
                "every contamination finding must reference a declared check"
            )
        if self.status is ContaminationStatus.NOT_RUN:
            if self.check_ids or self.findings:
                raise DatasetManifestError(
                    "not_run contamination state must contain no checks/findings"
                )
        elif self.status is ContaminationStatus.CLEAN:
            if not self.check_ids or self.findings:
                raise DatasetManifestError(
                    "clean contamination state requires checks and no findings"
                )
        elif self.status is ContaminationStatus.FINDINGS and (
            not self.check_ids or not self.findings
        ):
            raise DatasetManifestError("findings contamination state requires checks and findings")

    @classmethod
    def not_run(cls) -> ContaminationSummary:
        """Return the explicit state used before P4-002 contamination checks exist."""

        return cls(status=ContaminationStatus.NOT_RUN, check_ids=(), findings=())


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Complete audit envelope for one prepared corpus without embedding example text."""

    schema_version: int
    language: str
    seed: int
    identity: DatasetPreparationIdentity
    sources: tuple[DatasetSourceSummary, ...]
    counts: DatasetCountSummary
    content_rejection_counts: tuple[RejectionReasonCount, ...]
    length_rejection_counts: tuple[LengthRejectionCount, ...]
    duplicate_reason_counts: tuple[DuplicateReasonCount, ...]
    tokenizer: DatasetTokenizerSummary
    split: DatasetSplitSummary
    memberships: tuple[DatasetSplitMembership, ...]
    checksums: DatasetChecksums
    contamination: ContaminationSummary

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise DatasetManifestError(
                f"unsupported dataset manifest schema_version {self.schema_version}; "
                f"expected {_SCHEMA_VERSION}"
            )
        if self.language != self.identity.config.language:
            raise DatasetManifestError("manifest language must match preparation config language")
        if self.seed != self.identity.config.seed:
            raise DatasetManifestError("manifest seed must match preparation config seed")
        if not self.sources:
            raise DatasetManifestError("dataset manifest must contain at least one source")
        if tuple(_source_sort_key(source) for source in self.sources) != tuple(
            sorted(_source_sort_key(source) for source in self.sources)
        ):
            raise DatasetManifestError("dataset sources must use deterministic identity order")
        if sum(source.input_records for source in self.sources) != self.counts.input_records:
            raise DatasetManifestError("source input counts must equal manifest input count")
        if (
            sum(source.prepared_records for source in self.sources)
            != self.counts.deduplicated_unique
        ):
            raise DatasetManifestError("source prepared counts must equal manifest unique count")
        if len(self.memberships) != self.counts.deduplicated_unique:
            raise DatasetManifestError(
                "split membership count must equal deduplicated unique count"
            )
        prepared_hashes = {membership.record_sha256 for membership in self.memberships}
        for finding in self.contamination.findings:
            if finding.training_record_sha256 not in prepared_hashes:
                raise DatasetManifestError(
                    "contamination finding references a record outside the prepared corpus"
                )


def _require_sha256(value: str, *, field_name: str) -> None:
    if not _SHA256_PATTERN.fullmatch(value):
        raise DatasetManifestError(f"{field_name} must be a lowercase SHA-256 digest")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _source_key(record: NormalizedTrainingRecord) -> tuple[str, str]:
    return (record.provenance.source_id, record.provenance.revision)


def _license_key(license_metadata: LicenseMetadata) -> tuple[str, str, str]:
    return (
        license_metadata.name,
        license_metadata.url or "",
        license_metadata.attribution or "",
    )


def _source_sort_key(source: DatasetSourceSummary) -> tuple[str, str, str, str, str]:
    return (
        source.source_id,
        source.revision,
        source.license.name,
        source.license.url or "",
        source.license.attribution or "",
    )


def _config_sha256(config: DataPreparationConfig) -> str:
    return _sha256(asdict(config))


def _record_audit_payload(record: NormalizedTrainingRecord) -> object:
    """Return complete record/provenance data for an input checksum, not manifest storage."""

    return asdict(record)


def _ordered_record_hashes(split_records: Sequence[object]) -> str:
    return _sha256(tuple(split_records))


def _validate_pipeline_consistency(
    *,
    config: DataPreparationConfig,
    input_records: Sequence[NormalizedTrainingRecord],
    content_filter: RequiredContentFilterReport,
    length_filter: TokenLengthFilterReport,
    deduplication: ExactDeduplicationReport,
    split: DeduplicatedDatasetSplit,
) -> None:
    if not input_records:
        raise DatasetManifestError("dataset manifest requires at least one input record")
    if any(record.language != config.language for record in input_records):
        raise DatasetManifestError("every input record must match the configured language")
    if content_filter.total_records != len(input_records):
        raise DatasetManifestError("content-filter report does not cover every input record")
    if length_filter.total_records != content_filter.accepted_count:
        raise DatasetManifestError(
            "length-filter input count must equal content-filter accepted count"
        )
    if deduplication.total_records != length_filter.accepted_count:
        raise DatasetManifestError(
            "deduplication input count must equal length-filter accepted count"
        )
    if split.total_records != deduplication.unique_count:
        raise DatasetManifestError("split input count must equal deduplicated unique count")
    if split.seed != config.seed:
        raise DatasetManifestError("split seed must match data-preparation config seed")
    if split.requested_validation_fraction != config.validation_fraction:
        raise DatasetManifestError(
            "split validation fraction must match data-preparation config validation_fraction"
        )
    if length_filter.config.min_tokens != config.min_tokens:
        raise DatasetManifestError("tokenizer min_tokens must match data-preparation config")
    if length_filter.config.max_tokens != config.max_tokens:
        raise DatasetManifestError("tokenizer max_tokens must match data-preparation config")
    if str(length_filter.config.truncation_policy) != config.truncation_policy:
        raise DatasetManifestError("tokenizer truncation policy must match data-preparation config")
    if not config.deduplicate:
        raise DatasetManifestError(
            "P3 linkage-safe dataset manifests require data-preparation deduplicate=true"
        )


def _build_source_summaries(
    input_records: Sequence[NormalizedTrainingRecord],
    prepared_records: Sequence[NormalizedTrainingRecord],
) -> tuple[DatasetSourceSummary, ...]:
    licenses: dict[tuple[str, str], LicenseMetadata] = {}
    input_counts: Counter[tuple[str, str]] = Counter()
    prepared_counts: Counter[tuple[str, str]] = Counter()

    for record in input_records:
        key = _source_key(record)
        license_metadata = record.provenance.license
        previous = licenses.get(key)
        if previous is not None and _license_key(previous) != _license_key(license_metadata):
            raise DatasetManifestError(
                "one source ID/revision appears with inconsistent license metadata: "
                f"{key[0]}@{key[1]}"
            )
        licenses[key] = license_metadata
        input_counts[key] += 1

    for record in prepared_records:
        key = _source_key(record)
        if key not in licenses:
            raise DatasetManifestError(
                "prepared record references a source absent from input records"
            )
        if _license_key(licenses[key]) != _license_key(record.provenance.license):
            raise DatasetManifestError(
                "prepared record license does not match input source license"
            )
        prepared_counts[key] += 1

    summaries = tuple(
        DatasetSourceSummary(
            source_id=source_id,
            revision=revision,
            license=licenses[(source_id, revision)],
            input_records=input_counts[(source_id, revision)],
            prepared_records=prepared_counts[(source_id, revision)],
        )
        for source_id, revision in sorted(licenses)
    )
    return tuple(sorted(summaries, key=_source_sort_key))


def _build_checksums(
    *,
    input_records: Sequence[NormalizedTrainingRecord],
    deduplication: ExactDeduplicationReport,
    split: DeduplicatedDatasetSplit,
) -> DatasetChecksums:
    return DatasetChecksums(
        input_records_sha256=_sha256(
            tuple(_record_audit_payload(record) for record in input_records)
        ),
        unique_corpus_sha256=_ordered_record_hashes(
            tuple(item.record_sha256 for item in deduplication.unique_fingerprints)
        ),
        train_records_sha256=_ordered_record_hashes(
            tuple(item.record_sha256 for item in split.train_fingerprints)
        ),
        validation_records_sha256=_ordered_record_hashes(
            tuple(item.record_sha256 for item in split.validation_fingerprints)
        ),
        split_membership_sha256=_sha256(tuple(asdict(item) for item in split.memberships)),
    )


def create_dataset_manifest(
    *,
    config: DataPreparationConfig,
    input_records: Sequence[NormalizedTrainingRecord],
    content_filter: RequiredContentFilterReport,
    length_filter: TokenLengthFilterReport,
    deduplication: ExactDeduplicationReport,
    split: DeduplicatedDatasetSplit,
    contamination: ContaminationSummary | None = None,
    repo_root: Path = Path("."),
    git: GitMetadata | None = None,
) -> DatasetManifest:
    """Create and cross-check one deterministic prepared-corpus audit manifest."""

    _validate_pipeline_consistency(
        config=config,
        input_records=input_records,
        content_filter=content_filter,
        length_filter=length_filter,
        deduplication=deduplication,
        split=split,
    )
    prepared_records = deduplication.unique_records
    counts = DatasetCountSummary(
        input_records=len(input_records),
        content_accepted=content_filter.accepted_count,
        content_rejected=content_filter.rejected_count,
        length_accepted=length_filter.accepted_count,
        length_rejected=length_filter.rejected_count,
        deduplicated_unique=deduplication.unique_count,
        duplicates_removed=deduplication.duplicate_count,
        train_records=len(split.train_records),
        validation_records=len(split.validation_records),
    )
    tokenizer = DatasetTokenizerSummary(
        repository=length_filter.target.tokenizer_repository,
        revision=length_filter.target.tokenizer_revision,
        tokenizer_class=length_filter.tokenizer_class,
        chat_template_sha256=length_filter.chat_template_sha256,
        min_tokens=length_filter.config.min_tokens,
        max_tokens=length_filter.config.max_tokens,
        truncation_policy=str(length_filter.config.truncation_policy),
        measured_distribution=length_filter.input_distribution,
        accepted_distribution=length_filter.accepted_distribution,
    )
    split_summary = DatasetSplitSummary(
        requested_validation_fraction=split.requested_validation_fraction,
        target_validation_records=split.target_validation_records,
        actual_validation_fraction=split.actual_validation_fraction,
        linked_prompt_group_count=split.linked_prompt_group_count,
    )
    selected_contamination = contamination or ContaminationSummary.not_run()
    return DatasetManifest(
        schema_version=_SCHEMA_VERSION,
        language=config.language,
        seed=config.seed,
        identity=DatasetPreparationIdentity(
            git=git or collect_git_metadata(repo_root),
            config_sha256=_config_sha256(config),
            config=config,
        ),
        sources=_build_source_summaries(input_records, prepared_records),
        counts=counts,
        content_rejection_counts=content_filter.reason_counts,
        length_rejection_counts=length_filter.rejection_counts,
        duplicate_reason_counts=deduplication.reason_counts,
        tokenizer=tokenizer,
        split=split_summary,
        memberships=split.memberships,
        checksums=_build_checksums(
            input_records=input_records,
            deduplication=deduplication,
            split=split,
        ),
        contamination=selected_contamination,
    )


def dataset_manifest_json(manifest: DatasetManifest) -> str:
    """Serialize a dataset manifest deterministically without example content."""

    return json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n"


def dataset_manifest_sha256(manifest: DatasetManifest) -> str:
    """Return the SHA-256 of the exact deterministic manifest JSON bytes."""

    return hashlib.sha256(dataset_manifest_json(manifest).encode("utf-8")).hexdigest()


def write_dataset_manifest(manifest: DatasetManifest, output_dir: Path) -> Path:
    """Atomically write ``dataset-manifest.json`` into a prepared-corpus directory."""

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "dataset-manifest.json"
    temporary = output_dir / ".dataset-manifest.json.tmp"
    temporary.write_text(dataset_manifest_json(manifest), encoding="utf-8")
    temporary.replace(destination)
    return destination
