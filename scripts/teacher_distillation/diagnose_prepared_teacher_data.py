#!/usr/bin/env python3
"""Diagnose an already student-shaped distilled JSONL corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

from tiny_qwen_coder.data.length_filtering import load_canonical_tokenizer
from tiny_qwen_coder.data.loading import load_normalized_training_records_jsonl
from tiny_qwen_coder.distillation.config import load_teacher_distillation_config
from tiny_qwen_coder.distillation.diagnostics import diagnose_teacher_records, write_teacher_diagnostics
from tiny_qwen_coder.model.inspection import load_inspection_target


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--distillation-config",
        type=Path,
        default=Path("configs/distillation/python/qwen38_27b_v4_compression.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/base/qwen35-4b.yaml"),
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    from transformers import AutoTokenizer

    args = _parse_args()
    config = load_teacher_distillation_config(args.distillation_config)
    records = load_normalized_training_records_jsonl(
        args.input,
        expected_language=config.language,
    )
    for record in records:
        metadata = dict(record.provenance.source_metadata)
        if "distillation.input_policy" in metadata or "distillation.input_policy_sha256" in metadata:
            raise RuntimeError(
                "prepared teacher records still contain an active teacher-only input policy"
            )

    teacher_tokenizer = AutoTokenizer.from_pretrained(
        config.teacher.repository,
        revision=config.teacher.revision,
        trust_remote_code=False,
        local_files_only=args.local_files_only,
    )
    target = load_inspection_target(args.base_config)
    student_tokenizer = load_canonical_tokenizer(
        target,
        local_files_only=args.local_files_only,
    )
    diagnostics = diagnose_teacher_records(
        records,
        teacher_tokenizer=teacher_tokenizer,
        student_tokenizer=student_tokenizer,
        student_max_tokens=2048,
    )
    write_teacher_diagnostics(diagnostics, args.output_dir)
    print(f"total_records={diagnostics.summary.total_records}")
    print(f"stop_records={diagnostics.summary.stop_records}")
    print(f"stop_rate={diagnostics.summary.stop_rate:.6f}")
    print(
        "student_length_accept_rate_given_stop="
        f"{diagnostics.summary.student_length_accept_rate_given_stop:.6f}"
    )
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
