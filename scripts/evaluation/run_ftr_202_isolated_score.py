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
_EXECUTION_IMAGE = (
    "python:3.11.14-slim@sha256:c8271b1f627d0068857dce5b53e14a9558603b527e46f1f901722f935b786a39"
)
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


def _docker_executable() -> Path:
    docker = shutil.which("docker")
    if docker is None:
        raise SystemExit("FTR-202 scoring requires Docker")
    completed = subprocess.run(
        (docker, "image", "inspect", _EXECUTION_IMAGE),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise SystemExit(
            "FTR-202 requires the exact frozen execution image to be preloaded; "
            f"run: docker pull {_EXECUTION_IMAGE}"
        )
    return Path(docker).resolve()


def _preflight(repo_root: Path, generation_dir: Path) -> tuple[str, Path]:
    if any(os.environ.get(name) for name in _FORBIDDEN_ENV):
        raise SystemExit("FTR-202 scoring refuses cloud/model credentials in the environment")
    if Path("/content/drive").is_mount():
        raise SystemExit("FTR-202 scoring refuses to run while Google Drive is mounted")
    if _git(repo_root, "status", "--porcelain"):
        raise SystemExit("FTR-202 scoring requires a clean exact source checkout")
    sha = _git(repo_root, "rev-parse", "HEAD")

    from tiny_qwen_coder.evaluation._ftr_teacher_transport import (
        FTRTeacherTransportError,
        verify_generation_handoff,
    )

    try:
        verify_generation_handoff(
            generation_dir=generation_dir,
            expected_source_sha=sha,
        )
    except FTRTeacherTransportError as exc:
        raise SystemExit(f"FTR-202 transport verification failed: {exc}") from exc
    return sha, _docker_executable()


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
    source_sha, docker = _preflight(repo_root, generation_dir)

    destination = repo_root / _OUTPUT_RELATIVE
    if destination.exists() or destination.is_symlink():
        raise SystemExit(f"refusing existing scoring destination: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(generation_dir, destination)

    import tiny_qwen_coder.evaluation._baseline_stages as baseline_stages
    from tiny_qwen_coder.evaluation.execution import OciRuntime, OciRuntimeSpec
    from tiny_qwen_coder.evaluation.python_ftr_teacher_direct import score_teacher_stage
    from tiny_qwen_coder.evaluation.python_ftr_teacher_superiority import (
        audit_teacher_benchmark_protocol,
        compare_teacher_to_base,
        write_report,
    )

    audit_teacher_benchmark_protocol(repo_root=repo_root, source_git_sha=source_sha)
    runtime = OciRuntimeSpec(kind=OciRuntime.DOCKER, executable=docker)
    baseline_stages.discover_oci_runtime = lambda: runtime
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
