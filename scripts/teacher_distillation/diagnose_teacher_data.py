"""Measure teacher/runtime and student-tokenizer lengths for a durable checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from tiny_qwen_coder.distillation.diagnostics import (
    diagnose_teacher_checkpoint,
    write_teacher_diagnostics,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--distillation-config",
        type=Path,
        default=Path("configs/distillation/python/qwen38_27b_v1.yaml"),
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/base/qwen35-4b.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--student-max-tokens", type=int, default=2048)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    diagnostics = diagnose_teacher_checkpoint(
        distillation_config_path=args.distillation_config,
        checkpoint_dir=args.checkpoint_dir,
        input_path=args.input,
        base_config=args.base_config,
        local_files_only=args.local_files_only,
        limit=args.limit,
        student_max_tokens=args.student_max_tokens,
    )
    write_teacher_diagnostics(diagnostics, args.output_dir)
    summary = asdict(diagnostics.summary)
    print(f"total_records={summary['total_records']}")
    print(f"stop_records={summary['stop_records']}")
    print(f"stop_rate={summary['stop_rate']:.6f}")
    print(
        "student_length_accept_rate_given_stop="
        f"{summary['student_length_accept_rate_given_stop']:.6f}"
    )
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
