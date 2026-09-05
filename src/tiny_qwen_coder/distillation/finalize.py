"""Finalize durable teacher shards into a student-tokenizer-ready training corpus."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from tiny_qwen_coder.config import DataPreparationConfig, load_data_preparation_config
from tiny_qwen_coder.data.length_filtering import load_canonical_tokenizer
from tiny_qwen_coder.data.pipeline import DatasetPipelineResult, run_dataset_pipeline
from tiny_qwen_coder.data.records import NormalizedTrainingRecord
from tiny_qwen_coder.distillation.config import (
    TeacherDistillationConfig,
    load_teacher_distillation_config,
    teacher_distillation_config_sha256,
)
from tiny_qwen_coder.distillation.generation import load_completed_distilled_records
from tiny_qwen_coder.languages.python import (
    load_python_plugin,
    load_python_protected_benchmark_registry,
)
from tiny_qwen_coder.languages.python_quality import validate_python_quality
from tiny_qwen_coder.model.inspection import load_inspection_target
from tiny_qwen_coder.reporting.dataset_manifest import (
    dataset_manifest_json,
    dataset_manifest_sha256,
)


class TeacherFinalizationError(RuntimeError):
    """Raised when generated teacher data cannot be safely finalized."""


@dataclass(frozen=True, slots=True)
class TeacherFinalizationSummary:
    """Bounded rejection accounting before and during generic preparation."""

    schema_version: int
    distillation_config_sha256: str
    generated_candidates: int
    finish_reason_rejected: int
    python_quality_rejected: int
    pre_pipeline_accepted: int
    content_rejected: int
    length_rejected: int
    duplicates_removed: int
    prepared_unique: int
    train_records: int
    validation_records: int
    finish_reason_counts: tuple[tuple[str, int], ...]
    python_quality_rejection_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class FinalizedTeacherCorpus:
    """Paths and counts emitted for a fully prepared distilled corpus."""

    output_dir: Path
    accepted_path: Path
    train_path: Path
    validation_path: Path
    manifest_path: Path
    manifest_checksum_path: Path
    summary_path: Path
    summary: TeacherFinalizationSummary


def _atomic_write_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding=encoding)
    os.replace(temporary, path)


def _write_records(path: Path, records: tuple[NormalizedTrainingRecord, ...]) -> None:
    content = "".join(
        json.dumps(asdict(record), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    )
    _atomic_write_text(path, content)


def _finish_reason(record: NormalizedTrainingRecord) -> str:
    value = dict(record.provenance.source_metadata).get("distillation.finish_reason")
    return value or "missing"


def _quality_rejection_reason(record: NormalizedTrainingRecord) -> str | None:
    result = validate_python_quality(record)
    if result.passed:
        return None
    return result.detail or result.validator_id


def _prefilter_candidates(
    records: tuple[NormalizedTrainingRecord, ...],
) -> tuple[
    tuple[NormalizedTrainingRecord, ...],
    Counter[str],
    Counter[str],
]:
    finish_rejections: Counter[str] = Counter()
    quality_rejections: Counter[str] = Counter()
    accepted: list[NormalizedTrainingRecord] = []
    for record in records:
        finish_reason = _finish_reason(record)
        if finish_reason != "stop":
            finish_rejections[finish_reason] += 1
            continue
        quality_reason = _quality_rejection_reason(record)
        if quality_reason is not None:
            quality_rejections[quality_reason] += 1
            continue
        accepted.append(record)
    return tuple(accepted), finish_rejections, quality_rejections


def _summary(
    *,
    config: TeacherDistillationConfig,
    generated: int,
    prefiltered: int,
    finish_rejections: Counter[str],
    quality_rejections: Counter[str],
    pipeline: DatasetPipelineResult,
) -> TeacherFinalizationSummary:
    return TeacherFinalizationSummary(
        schema_version=1,
        distillation_config_sha256=teacher_distillation_config_sha256(config),
        generated_candidates=generated,
        finish_reason_rejected=sum(finish_rejections.values()),
        python_quality_rejected=sum(quality_rejections.values()),
        pre_pipeline_accepted=prefiltered,
        content_rejected=pipeline.content_filter.rejected_count,
        length_rejected=pipeline.length_filter.rejected_count,
        duplicates_removed=pipeline.deduplication.duplicate_count,
        prepared_unique=pipeline.deduplication.unique_count,
        train_records=len(pipeline.split.train_records),
        validation_records=len(pipeline.split.validation_records),
        finish_reason_counts=tuple(sorted(finish_rejections.items())),
        python_quality_rejection_counts=tuple(sorted(quality_rejections.items())),
    )


def finalize_teacher_corpus(
    *,
    distillation_config: TeacherDistillationConfig,
    data_config: DataPreparationConfig,
    checkpoint_dir: Path,
    input_path: Path | None = None,
    base_config: Path = Path("configs/base/qwen35-4b.yaml"),
    output_dir: Path | None = None,
    local_files_only: bool = False,
    limit: int | None = None,
) -> FinalizedTeacherCorpus:
    """Validate every durable shard and prepare train/validation files for the student."""

    if data_config.language != distillation_config.language:
        raise TeacherFinalizationError("data and distillation configs must use the same language")
    if data_config.max_tokens != 2048:
        raise TeacherFinalizationError(
            "Qwen3.5-4B distilled v1 preparation must retain the 2048-token student boundary"
        )
    generated = load_completed_distilled_records(
        distillation_config,
        checkpoint_dir=checkpoint_dir,
        input_path=input_path,
        limit=limit,
    )
    prefiltered, finish_rejections, quality_rejections = _prefilter_candidates(generated)
    if len(prefiltered) < 2:
        raise TeacherFinalizationError("fewer than two teacher candidates survived prefiltering")

    selected_output = output_dir or Path(data_config.output_dir)
    effective_data_config = replace(data_config, output_dir=str(selected_output))
    target = load_inspection_target(base_config)
    tokenizer = load_canonical_tokenizer(target, local_files_only=local_files_only)
    plugin = load_python_plugin()
    registry = load_python_protected_benchmark_registry()
    pipeline = run_dataset_pipeline(
        prefiltered,
        config=effective_data_config,
        plugin=plugin,
        tokenizer=tokenizer,
        target=target,
        protected_benchmarks=registry,
    )
    summary = _summary(
        config=distillation_config,
        generated=len(generated),
        prefiltered=len(prefiltered),
        finish_rejections=finish_rejections,
        quality_rejections=quality_rejections,
        pipeline=pipeline,
    )

    selected_output.mkdir(parents=True, exist_ok=True)
    accepted_path = selected_output / "accepted.jsonl"
    train_path = selected_output / "train.jsonl"
    validation_path = selected_output / "validation.jsonl"
    manifest_path = selected_output / "dataset-manifest.json"
    manifest_checksum_path = selected_output / "dataset-manifest.sha256"
    summary_path = selected_output / "teacher-finalization.json"

    _write_records(accepted_path, pipeline.deduplication.unique_records)
    _write_records(train_path, pipeline.split.train_records)
    _write_records(validation_path, pipeline.split.validation_records)
    manifest_text = dataset_manifest_json(pipeline.manifest)
    _atomic_write_text(manifest_path, manifest_text)
    digest = dataset_manifest_sha256(pipeline.manifest)
    _atomic_write_text(
        manifest_checksum_path,
        f"{digest}  dataset-manifest.json\n",
        encoding="ascii",
    )
    _atomic_write_text(
        summary_path,
        json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n",
    )

    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != digest:
        raise TeacherFinalizationError("written dataset manifest checksum verification failed")
    return FinalizedTeacherCorpus(
        output_dir=selected_output,
        accepted_path=accepted_path,
        train_path=train_path,
        validation_path=validation_path,
        manifest_path=manifest_path,
        manifest_checksum_path=manifest_checksum_path,
        summary_path=summary_path,
        summary=summary,
    )


def finalize_teacher_corpus_from_paths(
    *,
    distillation_config_path: Path,
    data_config_path: Path,
    checkpoint_dir: Path,
    input_path: Path | None = None,
    base_config: Path = Path("configs/base/qwen35-4b.yaml"),
    output_dir: Path | None = None,
    local_files_only: bool = False,
    limit: int | None = None,
) -> FinalizedTeacherCorpus:
    """Load configs then finalize one complete durable teacher run."""

    return finalize_teacher_corpus(
        distillation_config=load_teacher_distillation_config(distillation_config_path),
        data_config=load_data_preparation_config(data_config_path),
        checkpoint_dir=checkpoint_dir,
        input_path=input_path,
        base_config=base_config,
        output_dir=output_dir,
        local_files_only=local_files_only,
        limit=limit,
    )
