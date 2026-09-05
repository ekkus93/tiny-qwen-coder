"""Finalize durable Qwen teacher shards into Qwen3.5-4B-ready JSONL splits and manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from tiny_qwen_coder.distillation.finalize import finalize_teacher_corpus_from_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--distillation-config",
        type=Path,
        default=Path("configs/distillation/python/qwen38_27b_v1.yaml"),
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/data/python/qwen38_27b_distilled_v1.yaml"),
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/base/qwen35-4b.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    result = finalize_teacher_corpus_from_paths(
        distillation_config_path=args.distillation_config,
        data_config_path=args.data_config,
        checkpoint_dir=args.checkpoint_dir,
        input_path=args.input,
        base_config=args.base_config,
        output_dir=args.output_dir,
        local_files_only=args.local_files_only,
        limit=args.limit,
    )
    print(f"generated_candidates={result.summary.generated_candidates}")
    print(f"prepared_unique={result.summary.prepared_unique}")
    print(f"train_records={result.summary.train_records}")
    print(f"validation_records={result.summary.validation_records}")
    print(f"output_dir={result.output_dir}")


if __name__ == "__main__":
    main()
