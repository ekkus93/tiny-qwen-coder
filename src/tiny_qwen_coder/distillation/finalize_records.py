"""Finalize an already student-shaped distilled corpus through the protected data pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

from tiny_qwen_coder.config import DataPreparationConfig
from tiny_qwen_coder.data.length_filtering import load_canonical_tokenizer
from tiny_qwen_coder.data.loading import load_normalized_training_records_jsonl
from tiny_qwen_coder.data.pipeline import run_dataset_pipeline
from tiny_qwen_coder.distillation.config import TeacherDistillationConfig
from tiny_qwen_coder.distillation.finalize import (
    FinalizedTeacherCorpus,
    TeacherFinalizationError,
    _atomic_write_text,
    _prefilter_candidates,
    _require_clean_contamination,
    _require_no_reasoning_markers,
    _summary,
    _write_records,
)
from tiny_qwen_coder.evaluation.python_protected_examples import load_python_protected_examples
from tiny_qwen_coder.languages.python import (
    load_python_plugin,
    load_python_protected_benchmark_registry,
)
from tiny_qwen_coder.model.inspection import load_inspection_target
from tiny_qwen_coder.reporting.dataset_manifest import (
    dataset_manifest_json,
    dataset_manifest_sha256,
)


def finalize_prepared_teacher_records(
    *,
    distillation_config: TeacherDistillationConfig,
    data_config: DataPreparationConfig,
    input_path: Path,
    base_config: Path = Path("configs/base/qwen35-4b.yaml"),
    output_dir: Path | None = None,
    local_files_only: bool = False,
) -> FinalizedTeacherCorpus:
    """Prepare one merged v4 corpus without reinterpreting teacher-only prompt policies."""

    if data_config.language != distillation_config.language:
        raise TeacherFinalizationError("data and distillation configs must use the same language")
    if data_config.max_tokens != 2048:
        raise TeacherFinalizationError(
            "Qwen3.5-4B distilled preparation must retain the 2048-token student boundary"
        )
    records = load_normalized_training_records_jsonl(
        input_path,
        expected_language=distillation_config.language,
    )
    if not records:
        raise TeacherFinalizationError("prepared teacher corpus is empty")
    for record in records:
        metadata = dict(record.provenance.source_metadata)
        if (
            "distillation.input_policy" in metadata
            or "distillation.input_policy_sha256" in metadata
        ):
            raise TeacherFinalizationError(
                "prepared teacher corpus still contains an active teacher-only input policy"
            )
    _require_no_reasoning_markers(records)

    prefiltered, finish_rejections, quality_rejections = _prefilter_candidates(records)
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
        protected_examples=load_python_protected_examples(registry),
    )
    _require_clean_contamination(pipeline)
    summary = _summary(
        config=distillation_config,
        generated=len(records),
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


__all__ = ["finalize_prepared_teacher_records"]
