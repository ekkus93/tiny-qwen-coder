"""Generate resumable Qwen teacher candidates, checkpointing complete shards to durable storage."""

from __future__ import annotations

import argparse
from pathlib import Path

from tiny_qwen_coder.distillation import (
    inspect_teacher_generation,
    load_teacher_distillation_config,
    run_teacher_generation,
)
from tiny_qwen_coder.distillation.vllm_backend import VllmTeacherBackend


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/distillation/python/qwen38_27b_v1.yaml"),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Override config input_records (useful for a Google Drive copy)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Durable checkpoint directory; on Colab point this at mounted Google Drive",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/content/tiny-qwen-coder-distillation"),
        help="Fast local scratch directory used before atomic durable copies",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Bounded smoke-run record count; use a separate checkpoint directory for smoke runs",
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Validate durable progress and exit without loading the teacher model",
    )
    args = parser.parse_args()

    config = load_teacher_distillation_config(args.config)
    status = inspect_teacher_generation(
        config,
        checkpoint_dir=args.checkpoint_dir,
        input_path=args.input,
        limit=args.limit,
    )
    print(f"verified_records={status.completed_records}/{status.total_records}")
    print(f"verified_shards={status.completed_shards}/{status.total_shards}")
    if status.complete:
        print("generation already complete; teacher model was not loaded")
        return
    print(f"missing_shards={len(status.missing_shards)}")
    if args.status_only:
        return
    backend = VllmTeacherBackend(config)
    result = run_teacher_generation(
        config,
        backend=backend,
        checkpoint_dir=args.checkpoint_dir,
        work_dir=args.work_dir,
        input_path=args.input,
        limit=args.limit,
    )
    print(f"completed_records={result.completed_records}/{result.total_records}")
    print(f"completed_shards={result.completed_shards}/{result.total_shards}")
    print(f"checkpoint_dir={result.checkpoint_dir}")


if __name__ == "__main__":
    main()
