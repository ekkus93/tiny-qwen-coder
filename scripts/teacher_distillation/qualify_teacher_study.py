"""Qualify a bounded teacher study before scaling to a larger generation run."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from tiny_qwen_coder.distillation.qualification import qualify_teacher_study_from_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics-summary", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--minimum-stop-rate", type=float, default=0.90)
    parser.add_argument(
        "--minimum-student-length-accept-rate-given-stop",
        type=float,
        default=0.80,
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result = qualify_teacher_study_from_paths(
        diagnostics_summary_path=args.diagnostics_summary,
        dataset_manifest_path=args.dataset_manifest,
        minimum_stop_rate=args.minimum_stop_rate,
        minimum_student_length_accept_rate_given_stop=(
            args.minimum_student_length_accept_rate_given_stop
        ),
    )
    payload = json.dumps(asdict(result), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")

    if not result.qualified:
        raise SystemExit(
            "teacher study did not qualify for scaling: " + ", ".join(result.failed_gates)
        )


if __name__ == "__main__":
    main()
