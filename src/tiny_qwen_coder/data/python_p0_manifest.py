"""Freeze deterministic audit evidence for the canonical Python P0 dataset."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from tiny_qwen_coder.data.deduplication import (
    deduplicate_exact_records,
    normalized_record_fingerprint,
)
from tiny_qwen_coder.data.python_corpus import (
    PythonP0CorpusResult,
    PythonP0RejectionCount,
    PythonP0SourceStats,
)
from tiny_qwen_coder.data.python_corpus_config import PythonP0CorpusConfig
from tiny_qwen_coder.data.records import NormalizedTrainingRecord
from tiny_qwen_coder.data.source_config import DatasetSourceConfig
from tiny_qwen_coder.data.splitting import (
    DatasetSplitMembership,
    DeduplicatedDatasetSplit,
    split_deduplicated_records,
)
from tiny_qwen_coder.reporting.dataset_manifest import (
    ContaminationSummary,
    DatasetSplitSummary,
    DatasetTokenizerSummary,
)
from tiny_qwen_coder.reporting.manifest import GitMetadata, collect_git_metadata

_SCHEMA_VERSION = 1
_MANIFEST_ID = "dataset/python/p0"
_SHA256_LENGTH = 64


class PythonP0ManifestError(ValueError):
    """Raised when a Python P0 corpus cannot be frozen reproducibly."""


@dataclass(frozen=True, slots=True)
class PythonP0ManifestSource:
    """Pinned source config plus measured P5-005 composition for one source."""

    source_config_sha256: str
    config: DatasetSourceConfig
    stats: PythonP0SourceStats
    fill_accepted: int

    def __post_init__(self) -> None:
        _require_sha256(self.source_config_sha256, field_name="source_config_sha256")
        if self.config.id != self.stats.source_id:
            raise PythonP0ManifestError("source config id must match measured source stats")
        if self.fill_accepted != self.stats.fill_accepted:
            raise PythonP0ManifestError("source fill_accepted must match measured source stats")


@dataclass(frozen=True, slots=True)
class PythonP0ManifestCounts:
    """Measured record counts from source scan through final train/validation split."""

    scanned_records: int
    content_rejected: int
    validation_rejected: int
    length_rejected: int
    duplicate_rejected: int
    accepted_records: int
    train_records: int
    validation_records: int

    def __post_init__(self) -> None:
        if any(value < 0 for value in asdict(self).values()):
            raise PythonP0ManifestError("manifest counts must not be negative")
        classified = (
            self.content_rejected
            + self.validation_rejected
            + self.length_rejected
            + self.duplicate_rejected
            + self.accepted_records
        )
        if classified != self.scanned_records:
            raise PythonP0ManifestError("scanned count must equal rejected plus accepted records")
        if self.accepted_records != self.train_records + self.validation_records:
            raise PythonP0ManifestError("accepted count must equal train plus validation records")


@dataclass(frozen=True, slots=True)
class PythonP0ManifestChecksums:
    """Content/provenance fingerprints that make the frozen P0 corpus auditable."""

    accepted_audit_sha256: str
    accepted_content_sha256: str
    train_content_sha256: str
    validation_content_sha256: str
    split_membership_sha256: str
    composition_sha256: str

    def __post_init__(self) -> None:
        for field_name, value in asdict(self).items():
            _require_sha256(value, field_name=field_name)


@dataclass(frozen=True, slots=True)
class PythonP0DatasetManifest:
    """Frozen manifest for the selected, split canonical Python P0 corpus."""

    schema_version: int
    manifest_id: str
    corpus_id: str
    language: str
    seed: int
    config_sha256: str
    config: PythonP0CorpusConfig
    git: GitMetadata
    sources: tuple[PythonP0ManifestSource, ...]
    counts: PythonP0ManifestCounts
    rejection_counts: tuple[PythonP0RejectionCount, ...]
    tokenizer: DatasetTokenizerSummary
    split: DatasetSplitSummary
    memberships: tuple[DatasetSplitMembership, ...]
    checksums: PythonP0ManifestChecksums
    contamination: ContaminationSummary

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise PythonP0ManifestError("unsupported Python P0 manifest schema version")
        if self.manifest_id != _MANIFEST_ID:
            raise PythonP0ManifestError(f"manifest_id must be {_MANIFEST_ID!r}")
        if self.corpus_id != self.config.id:
            raise PythonP0ManifestError("manifest corpus_id must match P0 config id")
        if self.language != self.config.language:
            raise PythonP0ManifestError("manifest language must match P0 config language")
        if self.seed != self.config.seed:
            raise PythonP0ManifestError("manifest seed must match P0 config seed")
        _require_sha256(self.config_sha256, field_name="config_sha256")
        if self.git.dirty:
            raise PythonP0ManifestError("frozen Python P0 manifest requires a clean Git tree")
        if tuple(source.config.id for source in self.sources) != tuple(
            budget.id for budget in self.config.sources
        ):
            raise PythonP0ManifestError("manifest sources must follow configured P0 source order")
        if len(self.memberships) != self.counts.accepted_records:
            raise PythonP0ManifestError("split memberships must cover every accepted P0 record")
        if self.tokenizer.accepted_distribution.count != self.counts.accepted_records:
            raise PythonP0ManifestError("tokenizer accepted distribution must cover final corpus")
        prepared_hashes = {membership.record_sha256 for membership in self.memberships}
        for finding in self.contamination.findings:
            if finding.training_record_sha256 not in prepared_hashes:
                raise PythonP0ManifestError(
                    "contamination finding references a record outside the frozen P0 corpus"
                )


@dataclass(frozen=True, slots=True)
class FrozenPythonP0ManifestFiles:
    """Paths emitted when one Python P0 manifest is frozen to disk."""

    manifest: Path
    checksum: Path


def _require_sha256(value: str, *, field_name: str) -> None:
    if len(value) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise PythonP0ManifestError(f"{field_name} must be a lowercase SHA-256 digest")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _ordered_content_checksum(records: Sequence[NormalizedTrainingRecord]) -> str:
    return _sha256(tuple(normalized_record_fingerprint(record).record_sha256 for record in records))


def _build_counts(
    result: PythonP0CorpusResult,
    split: DeduplicatedDatasetSplit,
) -> PythonP0ManifestCounts:
    return PythonP0ManifestCounts(
        scanned_records=sum(item.scanned for item in result.source_stats),
        content_rejected=sum(item.content_rejected for item in result.source_stats),
        validation_rejected=sum(item.validation_rejected for item in result.source_stats),
        length_rejected=sum(item.length_rejected for item in result.source_stats),
        duplicate_rejected=sum(item.duplicate_rejected for item in result.source_stats),
        accepted_records=result.accepted_total,
        train_records=len(split.train_records),
        validation_records=len(split.validation_records),
    )


def _source_config_sha256(source: DatasetSourceConfig) -> str:
    return _sha256(asdict(source))


def _build_sources(
    result: PythonP0CorpusResult,
    source_configs: Mapping[str, DatasetSourceConfig],
) -> tuple[PythonP0ManifestSource, ...]:
    dataset_identities = tuple(
        (source.dataset.repository, source.dataset.revision, source.dataset.split)
        for source in source_configs.values()
    )
    if len(dataset_identities) != len(set(dataset_identities)):
        raise PythonP0ManifestError(
            "configured P0 sources must use distinct repository/revision/split identities"
        )
    actual_by_repository: Counter[tuple[str, str]] = Counter(
        (record.provenance.source_id, record.provenance.revision)
        for record in result.accepted_records
    )
    summaries: list[PythonP0ManifestSource] = []
    for budget, stats in zip(result.config.sources, result.source_stats, strict=True):
        source = source_configs[budget.id]
        if source.id != budget.id:
            raise PythonP0ManifestError(
                f"source config key {budget.id!r} resolved to source id {source.id!r}"
            )
        if source.language != result.config.language:
            raise PythonP0ManifestError(
                f"source {source.id!r} language does not match Python P0 config"
            )
        actual = actual_by_repository[(source.dataset.repository, source.dataset.revision)]
        if actual != stats.accepted:
            raise PythonP0ManifestError(
                f"source {source.id!r} accepted count does not match retained provenance"
            )
        summaries.append(
            PythonP0ManifestSource(
                source_config_sha256=_source_config_sha256(source),
                config=source,
                stats=stats,
                fill_accepted=stats.fill_accepted,
            )
        )
    return tuple(summaries)


def _build_tokenizer(result: PythonP0CorpusResult) -> DatasetTokenizerSummary:
    token_stats = result.token_stats
    return DatasetTokenizerSummary(
        repository=token_stats.repository,
        revision=token_stats.revision,
        tokenizer_class=token_stats.tokenizer_class,
        chat_template_sha256=token_stats.chat_template_sha256,
        min_tokens=token_stats.min_tokens,
        max_tokens=token_stats.max_tokens,
        truncation_policy=token_stats.truncation_policy,
        measured_distribution=token_stats.measured_distribution,
        accepted_distribution=token_stats.accepted_distribution,
    )


def split_python_p0_corpus(result: PythonP0CorpusResult) -> DeduplicatedDatasetSplit:
    """Verify P5-005 uniqueness and create the frozen linkage-safe 95/5 split."""

    if result.shortfall:
        raise PythonP0ManifestError(
            f"cannot freeze Python P0 with {result.shortfall} accepted-record shortfall"
        )
    verification = deduplicate_exact_records(result.accepted_records)
    if verification.duplicate_count:
        raise PythonP0ManifestError(
            "P5-005 accepted corpus is not exactly deduplicated; refusing to freeze"
        )
    if verification.unique_records != result.accepted_records:
        raise PythonP0ManifestError("deduplication verification changed accepted record order")
    return split_deduplicated_records(
        verification,
        validation_fraction=result.config.validation_fraction,
        seed=result.config.seed,
    )


def create_python_p0_dataset_manifest(
    result: PythonP0CorpusResult,
    *,
    source_configs: Mapping[str, DatasetSourceConfig],
    split: DeduplicatedDatasetSplit,
    contamination: ContaminationSummary | None = None,
    repo_root: Path = Path("."),
    git: GitMetadata | None = None,
) -> PythonP0DatasetManifest:
    """Create one content-free frozen manifest from measured P5-005 evidence."""

    expected_source_ids = {source.id for source in result.config.sources}
    if set(source_configs) != expected_source_ids:
        raise PythonP0ManifestError("source_configs must exactly match configured P0 sources")
    if result.shortfall:
        raise PythonP0ManifestError("cannot freeze a shortfall Python P0 corpus")
    if split.seed != result.config.seed:
        raise PythonP0ManifestError("split seed must match Python P0 config")
    if split.requested_validation_fraction != result.config.validation_fraction:
        raise PythonP0ManifestError("split validation fraction must match Python P0 config")
    if split.total_records != result.accepted_total:
        raise PythonP0ManifestError("split must cover every accepted Python P0 record")

    accepted_hashes = tuple(
        normalized_record_fingerprint(record).record_sha256 for record in result.accepted_records
    )
    membership_hashes = tuple(item.record_sha256 for item in split.memberships)
    if set(membership_hashes) != set(accepted_hashes):
        raise PythonP0ManifestError("split membership fingerprints do not match accepted corpus")

    selected_git = git or collect_git_metadata(repo_root)
    counts = _build_counts(result, split)
    tokenizer = _build_tokenizer(result)
    split_summary = DatasetSplitSummary(
        requested_validation_fraction=split.requested_validation_fraction,
        target_validation_records=split.target_validation_records,
        actual_validation_fraction=split.actual_validation_fraction,
        linked_prompt_group_count=split.linked_prompt_group_count,
    )
    sources = _build_sources(result, source_configs)
    selected_contamination = contamination or ContaminationSummary.not_run()
    checksums = PythonP0ManifestChecksums(
        accepted_audit_sha256=_sha256(tuple(asdict(record) for record in result.accepted_records)),
        accepted_content_sha256=_sha256(accepted_hashes),
        train_content_sha256=_ordered_content_checksum(split.train_records),
        validation_content_sha256=_ordered_content_checksum(split.validation_records),
        split_membership_sha256=_sha256(tuple(asdict(item) for item in split.memberships)),
        composition_sha256=_sha256(
            {
                "source_stats": tuple(asdict(item) for item in result.source_stats),
                "rejection_counts": tuple(asdict(item) for item in result.rejection_counts),
                "token_stats": asdict(result.token_stats),
            }
        ),
    )
    return PythonP0DatasetManifest(
        schema_version=_SCHEMA_VERSION,
        manifest_id=_MANIFEST_ID,
        corpus_id=result.config.id,
        language=result.config.language,
        seed=result.config.seed,
        config_sha256=_sha256(asdict(result.config)),
        config=result.config,
        git=selected_git,
        sources=sources,
        counts=counts,
        rejection_counts=result.rejection_counts,
        tokenizer=tokenizer,
        split=split_summary,
        memberships=split.memberships,
        checksums=checksums,
        contamination=selected_contamination,
    )


def python_p0_dataset_manifest_json(manifest: PythonP0DatasetManifest) -> str:
    """Serialize the frozen P0 manifest deterministically without training text."""

    return json.dumps(asdict(manifest), ensure_ascii=True, indent=2, sort_keys=True) + "\n"


def python_p0_dataset_manifest_sha256(manifest: PythonP0DatasetManifest) -> str:
    """Hash the exact deterministic manifest JSON bytes."""

    return hashlib.sha256(python_p0_dataset_manifest_json(manifest).encode("utf-8")).hexdigest()


def write_python_p0_dataset_manifest(
    manifest: PythonP0DatasetManifest,
    output_dir: Path,
) -> FrozenPythonP0ManifestFiles:
    """Atomically write the frozen manifest and its external SHA-256 sidecar."""

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "dataset-manifest.json"
    checksum_path = output_dir / "dataset-manifest.sha256"
    manifest_tmp = output_dir / ".dataset-manifest.json.tmp"
    checksum_tmp = output_dir / ".dataset-manifest.sha256.tmp"

    digest = python_p0_dataset_manifest_sha256(manifest)
    manifest_tmp.write_text(python_p0_dataset_manifest_json(manifest), encoding="utf-8")
    checksum_tmp.write_text(f"{digest}  dataset-manifest.json\n", encoding="ascii")
    manifest_tmp.replace(manifest_path)
    checksum_tmp.replace(checksum_path)
    return FrozenPythonP0ManifestFiles(manifest=manifest_path, checksum=checksum_path)


__all__ = [
    "FrozenPythonP0ManifestFiles",
    "PythonP0DatasetManifest",
    "PythonP0ManifestChecksums",
    "PythonP0ManifestCounts",
    "PythonP0ManifestError",
    "PythonP0ManifestSource",
    "create_python_p0_dataset_manifest",
    "python_p0_dataset_manifest_json",
    "python_p0_dataset_manifest_sha256",
    "split_python_p0_corpus",
    "write_python_p0_dataset_manifest",
]
