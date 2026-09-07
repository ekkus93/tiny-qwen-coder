#!/usr/bin/env python3
"""Sanitize an affected legacy Qwen generation checkpoint without regenerating it."""

from __future__ import annotations

import argparse
from pathlib import Path

from tiny_qwen_coder.distillation.config import load_teacher_distillation_config
from tiny_qwen_coder.distillation.legacy_salvage import write_salvaged_teacher_records


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/distillation/python/qwen38_27b_v3.yaml"),
    )
    parser.add_argument("--source-input", type=Path, required=True)
    parser.add_argument("--legacy-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--identity-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = write_salvaged_teacher_records(
        config=load_teacher_distillation_config(args.config),
        source_input_path=args.source_input,
        legacy_checkpoint_dir=args.legacy_checkpoint_dir,
        output_path=args.output,
        identity_dir=args.identity_dir,
    )
    print(f"source_records={summary.source_records}")
    print(f"stop_records={summary.stop_records}")
    print(f"length_records={summary.length_records}")
    print(f"closing_only_records={summary.closing_only_records}")
    print(f"wrapped_records={summary.wrapped_records}")
    print(f"truncated_reasoning_only_records={summary.truncated_reasoning_only_records}")
    print(f"source_run_identity_sha256={summary.source_run_identity_sha256}")
    print(f"source_implementation_sha256={summary.source_implementation_sha256}")
    print(f"salvage_implementation_sha256={summary.salvage_implementation_sha256}")
    print(f"output_sha256={summary.output_sha256}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
