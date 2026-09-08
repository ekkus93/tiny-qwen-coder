#!/usr/bin/env python3
"""Execute completed P9-009 contracts and freeze the split-preserving survivor census."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from tiny_qwen_coder.distillation.config import (
    load_teacher_distillation_config,
    teacher_distillation_config_sha256,
)
from tiny_qwen_coder.distillation.generation import load_completed_distilled_records
from tiny_qwen_coder.distillation.semantic_contracts import build_semantically_filtered_corpus


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--contract-input", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/distillation/python/qwen38_27b_semantic_contract_v1.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = load_teacher_distillation_config(args.config)
    contracts = load_completed_distilled_records(
        config,
        checkpoint_dir=args.checkpoint_dir,
        input_path=args.contract_input,
    )
    summary = build_semantically_filtered_corpus(
        source_dir=args.source_dir,
        contract_records=contracts,
        output_dir=args.output_dir,
        contract_input_sha256=_sha256(args.contract_input),
        contract_config_sha256=teacher_distillation_config_sha256(config),
    )
    print(f"input_records={summary.input_records}")
    print(f"semantic_verified={summary.semantic_verified}")
    print(f"semantic_rejected={summary.semantic_rejected}")
    print(f"train_records={summary.train_records}")
    print(f"validation_records={summary.validation_records}")
    print(f"result_sha256={summary.result_sha256}")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
