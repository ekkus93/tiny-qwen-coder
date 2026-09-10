#!/usr/bin/env python3
"""Score transported FTR-202 generation evidence under credential-free OCI isolation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

_OUTPUT_RELATIVE = Path("artifacts/eval/python/ftr-201-teacher-direct-v1")
_FORBIDDEN_ENV = (
    "GOOGLE_APPLICATION_CREDENTIALS",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
)


def _git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(repo_root), *args),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _preflight(repo_root: Path, generation_dir: Path) -> str:
    if any(os.environ.get(name) for name in _FORBIDDEN_ENV):
        raise SystemExit("FTR-202 scoring refuses cloud/model credentials in the environment")
    if Path("/content/drive").is_mount():
        raise SystemExit("FTR-202 scoring refuses to run while Google Drive is mounted")
    if _git(repo_root, "status", "--porcelain"):
        raise SystemExit("FTR-202 scoring requires a clean exact source checkout")
    sha = _git(repo_root, "rev-parse", "HEAD")
    stage = json.loads((generation_dir / "generation-stage.json").read_text(encoding="utf-8"))
    if not isinstance(stage, dict) or stage.get("source_git_sha") != sha:
        raise SystemExit("generation evidence was not produced by this exact source SHA")
    return sha


def main() -> None:
    parser = argparse.ArgumentParser(description="Score and compare FTR-202 teacher evidence")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/eval/python/ftr-202-teacher-superiority.json"),
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    generation_dir = args.generation_dir.resolve()
    source_sha = _preflight(repo_root, generation_dir)

    destination = repo_root / _OUTPUT_RELATIVE
    if destination.exists() or destination.is_symlink():
        raise SystemExit(f"refusing existing scoring destination: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(generation_dir, destination)

    from tiny_qwen_coder.evaluation.python_ftr_teacher_direct import score_teacher_stage
    from tiny_qwen_coder.evaluation.python_ftr_teacher_superiority import (
        audit_teacher_benchmark_protocol,
        compare_teacher_to_base,
        write_report,
    )

    audit_teacher_benchmark_protocol(repo_root=repo_root, source_git_sha=source_sha)
    score_teacher_stage(repo_root=repo_root)
    report = compare_teacher_to_base(
        base_dir=args.base_dir.resolve(),
        teacher_dir=destination,
        gate_path=repo_root / "configs/eval/python/ftr_203_teacher_superiority_gate_v1.json",
    )
    write_report(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["teacher_superiority_demonstrated"] is True else 3)


if __name__ == "__main__":
    main()
