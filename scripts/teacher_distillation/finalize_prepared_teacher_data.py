#!/usr/bin/env python3
"""Finalize an already student-shaped distilled JSONL corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

from tiny_qwen_coder.config import load_data_preparation_config
from tiny_qwen_coder.distillation.config import load_teacher_distillation_config
from tiny_qwen_coder.distillation.finalize_records import finalize_prepared_teacher_records


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--distillation-config",
        type=Path,
        default=Path("configs/distillation/python/qwen38_27b_v4_compression.yaml"),
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/data/python/qwen38_27b_distilled_v4.yaml"),
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
    args = _parse_args()
    result = finalize_prepared_teacher_records(
        distillation_config=load_teacher_distillation_config(args.distillation_config),
        data_config=load_data_preparation_config(args.data_config),
        input_path=args.input,
        base_config=args.base_config,
        output_dir=args.output_dir,
        local_files_only=args.local_files_only,
    )
    summary = result.summary
    print(f"generated_candidates={summary.generated_candidates}")
    print(f"prepared_unique={summary.prepared_unique}")
    print(f"train_records={summary.train_records}")
    print(f"validation_records={summary.validation_records}")
    print(f"output_dir={result.output_dir}")


if __name__ == "__main__":
    main()
