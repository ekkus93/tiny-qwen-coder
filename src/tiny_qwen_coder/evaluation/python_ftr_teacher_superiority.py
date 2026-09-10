"""Public CLI/API for FTR-202/FTR-203 teacher-superiority evaluation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from tiny_qwen_coder.evaluation._ftr_teacher_common import FTRTeacherSuperiorityError
from tiny_qwen_coder.evaluation._ftr_teacher_compare import compare_teacher_to_base
from tiny_qwen_coder.evaluation._ftr_teacher_protocol import (
    GATE_PATH,
    audit_teacher_benchmark_protocol,
)

__all__ = [
    "FTRTeacherSuperiorityError",
    "audit_teacher_benchmark_protocol",
    "compare_teacher_to_base",
    "write_report",
]


def write_report(path: Path, report: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FTR-202/FTR-203 teacher superiority protocol")
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--repo-root", type=Path, default=Path("."))
    audit.add_argument("--source-git-sha", required=True)
    audit.add_argument("--output", type=Path, default=None)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--repo-root", type=Path, default=Path("."))
    compare.add_argument("--base-dir", type=Path, required=True)
    compare.add_argument("--teacher-dir", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "audit":
        report = audit_teacher_benchmark_protocol(
            repo_root=cast(Path, args.repo_root),
            source_git_sha=cast(str, args.source_git_sha),
        )
        output = cast(Path | None, args.output)
        if output is not None:
            write_report(output, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    repo_root = cast(Path, args.repo_root)
    report = compare_teacher_to_base(
        base_dir=cast(Path, args.base_dir),
        teacher_dir=cast(Path, args.teacher_dir),
        gate_path=repo_root / GATE_PATH,
    )
    write_report(cast(Path, args.output), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
