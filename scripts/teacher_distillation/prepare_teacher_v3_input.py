"""Apply and seal the per-record answer-budget policy for the bounded v3 teacher study."""

from __future__ import annotations

import argparse
from pathlib import Path

from tiny_qwen_coder.data.length_filtering import load_canonical_tokenizer
from tiny_qwen_coder.distillation.v3_input import write_v3_teacher_input
from tiny_qwen_coder.model.inspection import load_inspection_target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language", default="python")
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/base/qwen35-4b.yaml"),
    )
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    target = load_inspection_target(args.base_config)
    tokenizer = load_canonical_tokenizer(target, local_files_only=args.local_files_only)
    summary = write_v3_teacher_input(
        input_path=args.input,
        output_path=args.output,
        student_tokenizer=tokenizer,
        language=args.language,
    )
    print(f"records={summary.output_records}")
    print(f"input_sha256={summary.input_sha256}")
    print(f"output_sha256={summary.output_sha256}")
    print(f"policy_sha256={summary.policy_sha256}")
    print(f"minimum_answer_budget_tokens={summary.minimum_answer_budget_tokens}")
    print(f"maximum_answer_budget_tokens={summary.maximum_answer_budget_tokens}")


if __name__ == "__main__":
    main()
