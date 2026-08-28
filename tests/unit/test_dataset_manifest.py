from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tiny_qwen_coder.config import DataPreparationConfig
from tiny_qwen_coder.data import (
    DeduplicatedDatasetSplit,
    ExactDeduplicationReport,
    LengthFilterConfig,
    LicenseMetadata,
    NormalizedTrainingRecord,
    RequiredContentFilterReport,
    SourceProvenance,
    TokenLengthFilterReport,
    deduplicate_exact_records,
    filter_by_token_length,
    filter_required_content,
    single_turn_messages,
    split_deduplicated_records,
)
from tiny_qwen_coder.model import InspectionTarget
from tiny_qwen_coder.reporting import (
    ContaminationFinding,
    ContaminationStatus,
    ContaminationSummary,
    DatasetManifest,
    DatasetManifestError,
    GitMetadata,
    create_dataset_manifest,
    dataset_manifest_json,
    dataset_manifest_sha256,
    write_dataset_manifest,
)

_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
_GIT_SHA = "a" * 40


class FakeTokenizer:
    chat_template = "fixture canonical template"

    def __init__(self) -> None:
        self.init_kwargs: dict[str, object] = {"_commit_hash": _REVISION}

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        truncation: bool,
        return_dict: bool,
        chat_template: str,
    ) -> list[int]:
        assert tokenize is True
        assert add_generation_prompt is False
        assert truncation is False
        assert return_dict is False
        assert chat_template == self.chat_template
        token_count = 2 + sum(len(message["content"].split()) for message in conversation)
        return list(range(token_count))


def _target() -> InspectionTarget:
    return InspectionTarget(
        config_id="qwen35-4b",
        model_repository="Qwen/Qwen3.5-4B",
        model_revision=_REVISION,
        tokenizer_repository="Qwen/Qwen3.5-4B",
        tokenizer_revision=_REVISION,
        model_load_dtype="bfloat16",
    )


def _record(
    *,
    user: str,
    assistant: str,
    source_id: str,
    record_id: str,
    license_name: str,
) -> NormalizedTrainingRecord:
    return NormalizedTrainingRecord(
        schema_version=1,
        language="python",
        messages=single_turn_messages(system=None, user=user, assistant=assistant),
        provenance=SourceProvenance(
            source_id=source_id,
            revision=_REVISION,
            license=LicenseMetadata(name=license_name),
            split="train",
            record_id=record_id,
        ),
        validation=None,
    )


def _inputs() -> tuple[NormalizedTrainingRecord, ...]:
    return (
        _record(
            user="alpha",
            assistant="one",
            source_id="fixture/source-a",
            record_id="a-1",
            license_name="Apache-2.0",
        ),
        _record(
            user="alpha",
            assistant="one",
            source_id="fixture/source-a",
            record_id="a-2",
            license_name="Apache-2.0",
        ),
        _record(
            user="alpha",
            assistant="alternate",
            source_id="fixture/source-a",
            record_id="a-3",
            license_name="Apache-2.0",
        ),
        _record(
            user="beta",
            assistant="two",
            source_id="fixture/source-b",
            record_id="b-1",
            license_name="MIT",
        ),
        _record(
            user="   ",
            assistant="rejected",
            source_id="fixture/source-b",
            record_id="b-2",
            license_name="MIT",
        ),
        _record(
            user="gamma",
            assistant="one two three four five six seven eight",
            source_id="fixture/source-b",
            record_id="b-3",
            license_name="MIT",
        ),
        _record(
            user="delta",
            assistant="four",
            source_id="fixture/source-b",
            record_id="b-4",
            license_name="MIT",
        ),
    )


def _config(
    *,
    output_dir: str = "data/python/p0",
    seed: int = 17,
    max_tokens: int = 8,
    deduplicate: bool = True,
) -> DataPreparationConfig:
    return DataPreparationConfig(
        schema_version=1,
        language="python",
        source_configs=("configs/data/python/a.yaml", "configs/data/python/b.yaml"),
        output_dir=output_dir,
        seed=seed,
        validation_fraction=0.25,
        min_tokens=1,
        max_tokens=max_tokens,
        truncation_policy="reject",
        deduplicate=deduplicate,
    )


def _pipeline() -> tuple[
    tuple[NormalizedTrainingRecord, ...],
    RequiredContentFilterReport,
    TokenLengthFilterReport,
    ExactDeduplicationReport,
    DeduplicatedDatasetSplit,
]:
    records = _inputs()
    content = filter_required_content(records)
    lengths = filter_by_token_length(
        content.accepted_records,
        FakeTokenizer(),
        _target(),
        config=LengthFilterConfig(min_tokens=1, max_tokens=8),
    )
    deduplication = deduplicate_exact_records(lengths.accepted_records)
    split = split_deduplicated_records(
        deduplication,
        validation_fraction=0.25,
        seed=17,
    )
    return records, content, lengths, deduplication, split


def _manifest(
    *,
    contamination: ContaminationSummary | None = None,
    config: DataPreparationConfig | None = None,
) -> DatasetManifest:
    records, content, lengths, deduplication, split = _pipeline()
    return create_dataset_manifest(
        config=config or _config(),
        input_records=records,
        content_filter=content,
        length_filter=lengths,
        deduplication=deduplication,
        split=split,
        contamination=contamination,
        git=GitMetadata(sha=_GIT_SHA, dirty=False),
    )


def test_manifest_captures_full_audit_surface_without_example_text() -> None:
    records, content, lengths, deduplication, _ = _pipeline()
    finding = ContaminationFinding(
        checker_id="p4.exact_prompt",
        protected_dataset_id="humaneval",
        finding_type="exact_prompt",
        training_record_sha256=deduplication.unique_fingerprints[0].record_sha256,
        protected_record_id="HumanEval/0",
    )
    contamination = ContaminationSummary(
        status=ContaminationStatus.FINDINGS,
        check_ids=("p4.exact_prompt",),
        findings=(finding,),
    )

    manifest = _manifest(contamination=contamination)

    assert manifest.language == "python"
    assert manifest.seed == 17
    assert manifest.identity.git.sha == _GIT_SHA
    assert len(manifest.identity.config_sha256) == 64
    assert tuple(
        (
            source.source_id,
            source.revision,
            source.license.name,
            source.input_records,
            source.prepared_records,
        )
        for source in manifest.sources
    ) == (
        ("fixture/source-a", _REVISION, "Apache-2.0", 3, 2),
        ("fixture/source-b", _REVISION, "MIT", 4, 2),
    )
    assert manifest.counts.input_records == len(records) == 7
    assert manifest.counts.content_rejected == content.rejected_count == 1
    assert manifest.counts.length_rejected == lengths.rejected_count == 1
    assert manifest.counts.duplicates_removed == deduplication.duplicate_count == 1
    assert manifest.counts.deduplicated_unique == 4
    assert manifest.counts.train_records + manifest.counts.validation_records == 4
    assert manifest.tokenizer.repository == "Qwen/Qwen3.5-4B"
    assert manifest.tokenizer.revision == _REVISION
    assert manifest.tokenizer.measured_distribution.count == 6
    assert manifest.tokenizer.accepted_distribution.count == 5
    assert manifest.split.requested_validation_fraction == 0.25
    assert len(manifest.memberships) == 4
    assert manifest.contamination.status is ContaminationStatus.FINDINGS
    assert manifest.contamination.findings == (finding,)

    rendered = dataset_manifest_json(manifest)
    assert "alpha" not in rendered
    assert "alternate" not in rendered
    assert "one two three" not in rendered
    assert "fixture/source-a" in rendered
    assert "Apache-2.0" in rendered


def test_manifest_records_all_rejection_and_duplicate_reason_counters() -> None:
    manifest = _manifest()

    content_counts = {item.reason.value: item.count for item in manifest.content_rejection_counts}
    length_counts = {item.reason.value: item.count for item in manifest.length_rejection_counts}
    duplicate_counts = {item.reason.value: item.count for item in manifest.duplicate_reason_counts}

    assert content_counts["empty_prompt"] == 1
    assert length_counts["too_long"] == 1
    assert duplicate_counts["exact_content"] == 1
    assert duplicate_counts["source_identity"] == 0


def test_manifest_json_and_checksum_are_deterministic() -> None:
    first = _manifest()
    second = _manifest()

    assert first == second
    assert dataset_manifest_json(first) == dataset_manifest_json(second)
    assert dataset_manifest_sha256(first) == dataset_manifest_sha256(second)
    assert len(dataset_manifest_sha256(first)) == 64


def test_config_identity_changes_without_changing_corpus_checksums() -> None:
    first = _manifest()
    second = _manifest(config=_config(output_dir="data/python/alternate"))

    assert first.identity.config_sha256 != second.identity.config_sha256
    assert first.checksums == second.checksums


def test_default_contamination_state_is_explicitly_not_run() -> None:
    manifest = _manifest()

    assert manifest.contamination == ContaminationSummary.not_run()
    assert manifest.contamination.status is ContaminationStatus.NOT_RUN


def test_clean_contamination_state_requires_declared_checks() -> None:
    clean = ContaminationSummary(
        status=ContaminationStatus.CLEAN,
        check_ids=("p4.exact_prompt",),
        findings=(),
    )
    assert _manifest(contamination=clean).contamination.status is ContaminationStatus.CLEAN

    with pytest.raises(DatasetManifestError, match="requires checks"):
        ContaminationSummary(status=ContaminationStatus.CLEAN, check_ids=(), findings=())


def test_contamination_finding_must_reference_prepared_record() -> None:
    contamination = ContaminationSummary(
        status=ContaminationStatus.FINDINGS,
        check_ids=("p4.exact_prompt",),
        findings=(
            ContaminationFinding(
                checker_id="p4.exact_prompt",
                protected_dataset_id="humaneval",
                finding_type="exact_prompt",
                training_record_sha256="f" * 64,
            ),
        ),
    )

    with pytest.raises(DatasetManifestError, match="outside the prepared corpus"):
        _manifest(contamination=contamination)


def test_manifest_rejects_seed_drift_between_config_and_split() -> None:
    with pytest.raises(DatasetManifestError, match="split seed"):
        _manifest(config=_config(seed=18))


def test_manifest_rejects_length_policy_drift() -> None:
    with pytest.raises(DatasetManifestError, match="max_tokens"):
        _manifest(config=_config(max_tokens=9))


def test_manifest_requires_deduplication_for_linkage_safe_split() -> None:
    with pytest.raises(DatasetManifestError, match="deduplicate=true"):
        _manifest(config=_config(deduplicate=False))


def test_manifest_rejects_inconsistent_license_for_same_source_revision() -> None:
    records, content, lengths, deduplication, split = _pipeline()
    first = records[0]
    conflicting = replace(
        first,
        provenance=replace(first.provenance, license=LicenseMetadata(name="MIT")),
    )
    changed_records = (conflicting, *records[1:])

    with pytest.raises(DatasetManifestError, match="inconsistent license metadata"):
        create_dataset_manifest(
            config=_config(),
            input_records=changed_records,
            content_filter=content,
            length_filter=lengths,
            deduplication=deduplication,
            split=split,
            git=GitMetadata(sha=_GIT_SHA, dirty=False),
        )


def test_manifest_rejects_cross_stage_count_drift() -> None:
    records, content, lengths, deduplication, split = _pipeline()

    with pytest.raises(DatasetManifestError, match="content-filter report"):
        create_dataset_manifest(
            config=_config(),
            input_records=records[:-1],
            content_filter=content,
            length_filter=lengths,
            deduplication=deduplication,
            split=split,
            git=GitMetadata(sha=_GIT_SHA, dirty=False),
        )


def test_write_dataset_manifest_is_stable_and_atomic(tmp_path: Path) -> None:
    manifest = _manifest()

    path = write_dataset_manifest(manifest, tmp_path)

    assert path == tmp_path / "dataset-manifest.json"
    assert path.read_text(encoding="utf-8") == dataset_manifest_json(manifest)
    assert not (tmp_path / ".dataset-manifest.json.tmp").exists()
