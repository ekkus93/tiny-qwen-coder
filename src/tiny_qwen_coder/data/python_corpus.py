"""Deterministic construction of the canonical pre-split Python P0 corpus."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum

from tiny_qwen_coder.data.deduplication import (
    RecordContentFingerprint,
    SourceRecordIdentity,
    normalized_record_fingerprint,
    source_record_identity,
)
from tiny_qwen_coder.data.filtering import filter_required_content
from tiny_qwen_coder.data.length_filtering import (
    LengthFilterConfig,
    TruncationPolicy,
    filter_by_token_length,
)
from tiny_qwen_coder.data.magicoder_python import iter_magicoder_python
from tiny_qwen_coder.data.olmo_python import iter_olmo_python_instruct
from tiny_qwen_coder.data.pipeline import apply_language_validators
from tiny_qwen_coder.data.python_corpus_config import (
    PythonP0CorpusConfig,
    PythonP0CorpusError,
    PythonP0SourceBudget,
    load_python_p0_config,
    parse_python_p0_config,
)
from tiny_qwen_coder.data.records import NormalizedTrainingRecord, TrainingMessage
from tiny_qwen_coder.data.source_config import DatasetSourceConfig
from tiny_qwen_coder.languages.schema import LanguageConfig
from tiny_qwen_coder.languages.spec import LanguagePlugin
from tiny_qwen_coder.model.inspection import InspectionTarget

_SCHEMA_VERSION = 1


class PythonP0RejectionStage(StrEnum):
    """Stable P5-005 rejection stages applied before corpus counting."""

    CONTENT = "content"
    VALIDATION = "validation"
    LENGTH = "length"
    DUPLICATE = "duplicate"


@dataclass(frozen=True, slots=True)
class PythonP0SourceStats:
    """Measured P5-005 composition statistics for one source."""

    source_id: str
    target_accepted: int
    requested_accepted: int
    scanned: int
    content_rejected: int
    validation_rejected: int
    length_rejected: int
    duplicate_rejected: int
    accepted: int

    def __post_init__(self) -> None:
        if (
            min(
                self.target_accepted,
                self.requested_accepted,
                self.scanned,
                self.content_rejected,
                self.validation_rejected,
                self.length_rejected,
                self.duplicate_rejected,
                self.accepted,
            )
            < 0
        ):
            raise PythonP0CorpusError("source statistics must not contain negative counts")
        classified = (
            self.content_rejected
            + self.validation_rejected
            + self.length_rejected
            + self.duplicate_rejected
            + self.accepted
        )
        if classified != self.scanned:
            raise PythonP0CorpusError("source scanned count must equal classified outcomes")
        if self.accepted > self.requested_accepted:
            raise PythonP0CorpusError("source accepted count exceeds requested accepted count")

    @property
    def fill_accepted(self) -> int:
        """Return accepted records above this source's base composition target."""

        return max(0, self.accepted - self.target_accepted)


@dataclass(frozen=True, slots=True)
class PythonP0RejectionCount:
    """Measured count for one stable rejection stage/reason pair."""

    stage: PythonP0RejectionStage
    reason: str
    count: int

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise PythonP0CorpusError("rejection reason must not be empty")
        if self.count <= 0:
            raise PythonP0CorpusError("rejection count must be greater than zero")


@dataclass(frozen=True, slots=True)
class PythonP0CorpusResult:
    """Accepted unique records plus measured composition and rejection evidence."""

    config: PythonP0CorpusConfig
    source_stats: tuple[PythonP0SourceStats, ...]
    rejection_counts: tuple[PythonP0RejectionCount, ...]
    accepted_records: tuple[NormalizedTrainingRecord, ...]

    def __post_init__(self) -> None:
        if tuple(item.source_id for item in self.source_stats) != tuple(
            source.id for source in self.config.sources
        ):
            raise PythonP0CorpusError("source statistics must follow configured source order")
        if sum(item.accepted for item in self.source_stats) != len(self.accepted_records):
            raise PythonP0CorpusError("source accepted counts must equal accepted record count")
        rejection_order = tuple((item.stage.value, item.reason) for item in self.rejection_counts)
        if rejection_order != tuple(sorted(rejection_order)):
            raise PythonP0CorpusError("rejection counts must use stable stage/reason order")

    @property
    def accepted_total(self) -> int:
        return len(self.accepted_records)

    @property
    def shortfall(self) -> int:
        return max(0, self.config.target_total - self.accepted_total)

    @property
    def fill_accepted(self) -> int:
        return sum(item.fill_accepted for item in self.source_stats)


RecordStreamFactory = Callable[
    [DatasetSourceConfig, LanguageConfig], Iterable[NormalizedTrainingRecord]
]

_DEFAULT_STREAM_FACTORIES: Mapping[str, RecordStreamFactory] = {
    "magicoder-oss-instruct-75k": iter_magicoder_python,
    "olmo-starcoder-python-instruct": iter_olmo_python_instruct,
}


def _validation_failure_reasons(record: NormalizedTrainingRecord) -> tuple[str, ...]:
    metadata = record.validation
    if metadata is None:
        return ("missing_validation_metadata",)

    failures: list[str] = []
    for result in metadata.results:
        if result.passed:
            continue
        detail = result.detail.split(";", maxsplit=1)[0] if result.detail else "failed"
        failures.append(f"{result.validator_id}:{detail}")
    return tuple(failures)


def _validate_record_source(
    record: NormalizedTrainingRecord,
    *,
    source: DatasetSourceConfig,
    language: str,
) -> None:
    if record.language != language:
        raise PythonP0CorpusError(
            f"source {source.id!r} yielded language {record.language!r}; expected {language!r}"
        )

    provenance = record.provenance
    identity_fields = (
        ("source_id", provenance.source_id, source.dataset.repository),
        ("revision", provenance.revision, source.dataset.revision),
        ("split", provenance.split, source.dataset.split),
    )
    for field_name, actual, expected in identity_fields:
        if actual != expected:
            raise PythonP0CorpusError(
                f"source {source.id!r} yielded mismatched provenance {field_name}: "
                f"{actual!r} != {expected!r}"
            )
    if provenance.license != source.license:
        raise PythonP0CorpusError(
            f"source {source.id!r} yielded license metadata that does not match its pinned config"
        )
    if provenance.record_id is None:
        raise PythonP0CorpusError(f"source {source.id!r} yielded a record without record_id")


def _process_candidate(
    record: NormalizedTrainingRecord,
    *,
    plugin: LanguagePlugin,
    tokenizer: object,
    target: InspectionTarget,
    length_config: LengthFilterConfig,
) -> tuple[NormalizedTrainingRecord | None, PythonP0RejectionStage | None, tuple[str, ...]]:
    content = filter_required_content((record,))
    if content.rejected_records:
        reasons = tuple(reason.value for reason in content.rejected_records[0].reasons)
        return None, PythonP0RejectionStage.CONTENT, reasons

    validated = apply_language_validators(content.accepted_records, plugin)
    candidate = validated[0]
    failures = _validation_failure_reasons(candidate)
    if failures:
        return None, PythonP0RejectionStage.VALIDATION, failures

    length = filter_by_token_length((candidate,), tokenizer, target, config=length_config)
    if length.rejected_records:
        return (
            None,
            PythonP0RejectionStage.LENGTH,
            (length.rejected_records[0].reason.value,),
        )
    return length.accepted_records[0], None, ()


def _is_duplicate(
    record: NormalizedTrainingRecord,
    *,
    content_seen: dict[str, tuple[RecordContentFingerprint, tuple[TrainingMessage, ...]]],
    source_seen: dict[SourceRecordIdentity, RecordContentFingerprint],
) -> tuple[bool, tuple[str, ...]]:
    fingerprint = normalized_record_fingerprint(record)
    content_match = content_seen.get(fingerprint.record_sha256)
    if content_match is not None and content_match[1] != record.messages:
        raise PythonP0CorpusError("SHA-256 content collision while composing the Python P0 corpus")
    source_identity = source_record_identity(record)
    source_match = source_seen.get(source_identity) if source_identity is not None else None
    if source_match is not None and source_match != fingerprint:
        raise PythonP0CorpusError(
            "source-record identity conflict while composing Python P0: "
            f"{source_identity!r} appears with different normalized content"
        )

    reasons: list[str] = []
    if content_match is not None:
        reasons.append("exact_content")
    if source_match is not None:
        reasons.append("source_identity")
    if source_identity is not None and source_match is None:
        source_seen[source_identity] = fingerprint
    if reasons:
        return True, tuple(reasons)

    content_seen[fingerprint.record_sha256] = (fingerprint, record.messages)
    return False, ()


def _processing_order(config: PythonP0CorpusConfig) -> tuple[PythonP0SourceBudget, ...]:
    fill_source = config.fill_shortfall_from
    if fill_source is None:
        return config.sources
    non_fill = tuple(source for source in config.sources if source.id != fill_source)
    fill = tuple(source for source in config.sources if source.id == fill_source)
    return non_fill + fill


def build_python_p0_corpus(
    config: PythonP0CorpusConfig,
    *,
    plugin: LanguagePlugin,
    tokenizer: object,
    target: InspectionTarget,
    source_configs: Mapping[str, DatasetSourceConfig],
    stream_factories: Mapping[str, RecordStreamFactory] = _DEFAULT_STREAM_FACTORIES,
) -> PythonP0CorpusResult:
    """Build measured P0 composition from streamed, post-filter unique records."""

    if plugin.spec.id != config.language:
        raise PythonP0CorpusError(
            f"plugin language {plugin.spec.id!r} does not match corpus {config.language!r}"
        )
    expected_ids = {source.id for source in config.sources}
    if set(source_configs) != expected_ids:
        raise PythonP0CorpusError("source_configs must exactly match configured source IDs")
    adapter_refs = {component.id: component.import_ref for component in plugin.spec.data_adapters}
    for source_id, source in source_configs.items():
        if adapter_refs.get(source_id) != source.adapter:
            raise PythonP0CorpusError(
                f"plugin adapter registration does not match source {source_id!r}"
            )

    length_config = LengthFilterConfig(
        min_tokens=config.min_tokens,
        max_tokens=config.max_tokens,
        truncation_policy=TruncationPolicy.REJECT,
    )
    accepted_by_source: dict[str, list[NormalizedTrainingRecord]] = {
        source.id: [] for source in config.sources
    }
    stats_by_source: dict[str, PythonP0SourceStats] = {}
    rejection_counter: Counter[tuple[PythonP0RejectionStage, str]] = Counter()
    content_seen: dict[str, tuple[RecordContentFingerprint, tuple[TrainingMessage, ...]]] = {}
    source_seen: dict[SourceRecordIdentity, RecordContentFingerprint] = {}
    accumulated_shortfall = 0

    for budget in _processing_order(config):
        source = source_configs[budget.id]
        if source.id != budget.id:
            raise PythonP0CorpusError(
                f"source config key {budget.id!r} resolved to source id {source.id!r}"
            )
        if source.language != config.language:
            raise PythonP0CorpusError(
                f"source {source.id!r} language {source.language!r} does not match "
                f"corpus language {config.language!r}"
            )
        stream_factory = stream_factories.get(budget.id)
        if stream_factory is None:
            raise PythonP0CorpusError(f"no record stream factory registered for {budget.id!r}")
        requested = budget.target_accepted
        if budget.id == config.fill_shortfall_from:
            requested += accumulated_shortfall

        scanned = 0
        content_rejected = 0
        validation_rejected = 0
        length_rejected = 0
        duplicate_rejected = 0
        accepted = accepted_by_source[budget.id]

        for record in stream_factory(source, plugin.spec.config):
            _validate_record_source(record, source=source, language=config.language)
            scanned += 1
            candidate, stage, reasons = _process_candidate(
                record,
                plugin=plugin,
                tokenizer=tokenizer,
                target=target,
                length_config=length_config,
            )
            if stage is not None:
                for reason in reasons:
                    rejection_counter[(stage, reason)] += 1
                if stage is PythonP0RejectionStage.CONTENT:
                    content_rejected += 1
                elif stage is PythonP0RejectionStage.VALIDATION:
                    validation_rejected += 1
                elif stage is PythonP0RejectionStage.LENGTH:
                    length_rejected += 1
                continue
            assert candidate is not None

            duplicate, duplicate_reasons = _is_duplicate(
                candidate,
                content_seen=content_seen,
                source_seen=source_seen,
            )
            if duplicate:
                duplicate_rejected += 1
                for reason in duplicate_reasons:
                    rejection_counter[(PythonP0RejectionStage.DUPLICATE, reason)] += 1
                continue
            accepted.append(candidate)
            if len(accepted) >= requested:
                break

        source_shortfall = max(0, requested - len(accepted))
        if budget.id != config.fill_shortfall_from:
            accumulated_shortfall += source_shortfall
        else:
            accumulated_shortfall = source_shortfall
        stats_by_source[budget.id] = PythonP0SourceStats(
            source_id=budget.id,
            target_accepted=budget.target_accepted,
            requested_accepted=requested,
            scanned=scanned,
            content_rejected=content_rejected,
            validation_rejected=validation_rejected,
            length_rejected=length_rejected,
            duplicate_rejected=duplicate_rejected,
            accepted=len(accepted),
        )

    ordered_stats = tuple(stats_by_source[source.id] for source in config.sources)
    accepted_records = tuple(
        record for source in config.sources for record in accepted_by_source[source.id]
    )
    rejection_counts = tuple(
        PythonP0RejectionCount(stage=stage, reason=reason, count=count)
        for (stage, reason), count in sorted(
            rejection_counter.items(), key=lambda item: (item[0][0].value, item[0][1])
        )
    )
    return PythonP0CorpusResult(
        config=config,
        source_stats=ordered_stats,
        rejection_counts=rejection_counts,
        accepted_records=accepted_records,
    )


__all__ = [
    "PythonP0CorpusConfig",
    "PythonP0CorpusError",
    "PythonP0CorpusResult",
    "PythonP0RejectionCount",
    "PythonP0RejectionStage",
    "PythonP0SourceBudget",
    "PythonP0SourceStats",
    "RecordStreamFactory",
    "build_python_p0_corpus",
    "load_python_p0_config",
    "parse_python_p0_config",
]
