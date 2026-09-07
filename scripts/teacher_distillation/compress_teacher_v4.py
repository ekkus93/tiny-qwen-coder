#!/usr/bin/env python3
"""Selectively compress v3 overlength answers into a durable v4 checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from tiny_qwen_coder.data.length_filtering import load_canonical_tokenizer
from tiny_qwen_coder.distillation.config import load_teacher_distillation_config
from tiny_qwen_coder.distillation.generation import load_completed_distilled_records
from tiny_qwen_coder.distillation.v4_compression import (
    build_v4_merged_records,
    inspect_v4_compression,
    run_v4_compression,
    write_v4_merged_records,
)
from tiny_qwen_coder.model.inspection import load_inspection_target


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-config",
        type=Path,
        default=Path("configs/distillation/python/qwen38_27b_v3.yaml"),
    )
    parser.add_argument("--source-input", type=Path, required=True)
    parser.add_argument("--source-checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--compression-config",
        type=Path,
        default=Path("configs/distillation/python/qwen38_27b_v4_compression.yaml"),
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/base/qwen35-4b.yaml"),
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source_config = load_teacher_distillation_config(args.source_config)
    compression_config = load_teacher_distillation_config(args.compression_config)
    if source_config.language != compression_config.language:
        raise RuntimeError("source and compression configs must use the same language")

    source_records = load_completed_distilled_records(
        source_config,
        checkpoint_dir=args.source_checkpoint_dir,
        input_path=args.source_input,
    )
    target = load_inspection_target(args.base_config)
    student_tokenizer = load_canonical_tokenizer(
        target,
        local_files_only=args.local_files_only,
    )

    if args.status_only:
        status = inspect_v4_compression(
            source_records=source_records,
            source_checkpoint_dir=args.source_checkpoint_dir,
            compression_config=compression_config,
            checkpoint_dir=args.checkpoint_dir,
            student_tokenizer=student_tokenizer,
        )
        print(f"source_records={status.source_records}")
        print(f"compression_targets={status.compression_targets}")
        print(f"completed_targets={status.completed_targets}/{status.compression_targets}")
        print(f"completed_shards={status.completed_shards}/{status.total_shards}")
        print(f"missing_shards={list(status.missing_shards)}")
        return

    from tiny_qwen_coder.distillation.vllm_backend import VllmTeacherBackend

    backend = VllmTeacherBackend(compression_config)
    status = run_v4_compression(
        source_records=source_records,
        source_checkpoint_dir=args.source_checkpoint_dir,
        compression_config=compression_config,
        backend=backend,
        checkpoint_dir=args.checkpoint_dir,
        work_dir=args.work_dir,
        student_tokenizer=student_tokenizer,
    )
    merged, summary = build_v4_merged_records(
        source_records=source_records,
        compression_config=compression_config,
        status=status,
        student_tokenizer=student_tokenizer,
    )
    write_v4_merged_records(records=merged, summary=summary, output_path=args.output)

    print(f"source_records={summary.source_records}")
    print(f"source_stop_records={summary.source_stop_records}")
    print(f"source_student_length_accepted={summary.source_student_length_accepted}")
    print(f"compression_targets={summary.compression_targets}")
    print(f"compression_stop_records={summary.compression_stop_records}")
    print(f"compressed_within_budget={summary.compressed_within_budget}")
    print(f"compressed_student_length_accepted={summary.compressed_student_length_accepted}")
    print(f"final_stop_records={summary.final_stop_records}")
    print(f"final_student_length_accepted={summary.final_student_length_accepted}")
    print(f"rescued_student_length_records={summary.rescued_student_length_records}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
