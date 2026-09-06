"""Apply and seal the concise-final-answer policy for the bounded v2 teacher study."""

from __future__ import annotations

import argparse
from pathlib import Path

from tiny_qwen_coder.distillation.v2_input import write_v2_teacher_input


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language", default="python")
    args = parser.parse_args()

    summary = write_v2_teacher_input(
        input_path=args.input,
        output_path=args.output,
        language=args.language,
    )
    print(f"records={summary.output_records}")
    print(f"input_sha256={summary.input_sha256}")
    print(f"output_sha256={summary.output_sha256}")
    print(f"policy_sha256={summary.policy_sha256}")


if __name__ == "__main__":
    main()
