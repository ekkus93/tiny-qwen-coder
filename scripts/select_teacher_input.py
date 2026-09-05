"""Create a sealed source-stratified subset of a canonical teacher-input JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path

from tiny_qwen_coder.distillation.subset import write_teacher_input_subset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--language", default="python")
    args = parser.parse_args()

    summary = write_teacher_input_subset(
        input_path=args.input,
        output_path=args.output,
        count=args.count,
        seed=args.seed,
        language=args.language,
    )
    print(f"selected_records={summary.selected_records}/{summary.input_records}")
    print(f"input_sha256={summary.input_sha256}")
    print(f"output_sha256={summary.output_sha256}")
    for source in summary.sources:
        print(
            f"source={source.source_id}@{source.revision} "
            f"selected={source.selected_records}/{source.population_records}"
        )


if __name__ == "__main__":
    main()
