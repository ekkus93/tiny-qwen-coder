#!/usr/bin/env python3
"""Run resumable FTR-202 teacher generation on a Google Colab A100 80 GB runtime.

Mount Google Drive before invoking this script. Generated-code execution is never
performed here: the canonical FTR-201 generation stage writes one checkpoint per
task through a symlink into Drive, making interruption recovery immediate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

_OUTPUT_RELATIVE = Path("artifacts/eval/python/ftr-201-teacher-direct-v1")
_MINIMUM_CUDA_BYTES = 75 * 1024**3


def _git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(repo_root), *args),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _source_sha(repo_root: Path) -> str:
    sha = _git(repo_root, "rev-parse", "HEAD")
    if len(sha) != 40:
        raise SystemExit("FTR-202 requires an exact Git commit checkout")
    if _git(repo_root, "status", "--porcelain"):
        raise SystemExit("FTR-202 requires a clean source tree")
    return sha


def _validate_a100() -> dict[str, object]:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("FTR-202 requires CUDA")
    properties = torch.cuda.get_device_properties(0)
    name = properties.name
    total = int(properties.total_memory)
    if "A100" not in name or total < _MINIMUM_CUDA_BYTES:
        raise SystemExit(
            f"FTR-202 requires an A100-class 80 GB GPU; observed {name!r} with {total} bytes"
        )
    return {
        "gpu_name": name,
        "cuda_total_bytes": total,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bind_persistent_output(*, repo_root: Path, persistent_root: Path, source_sha: str) -> Path:
    persistent_dir = persistent_root / source_sha / _OUTPUT_RELATIVE.name
    persistent_dir.mkdir(parents=True, exist_ok=True)
    local = repo_root / _OUTPUT_RELATIVE
    local.parent.mkdir(parents=True, exist_ok=True)
    if local.is_symlink():
        if local.resolve() != persistent_dir.resolve():
            raise SystemExit(f"existing output symlink points elsewhere: {local}")
    elif local.exists():
        raise SystemExit(
            f"refusing existing non-symlink output {local}; move it aside before FTR-202"
        )
    else:
        local.symlink_to(persistent_dir, target_is_directory=True)
    return persistent_dir


def _handoff_payload(
    *, persistent_dir: Path, source_sha: str, gpu: dict[str, object]
) -> dict[str, object]:
    stage_path = persistent_dir / "generation-stage.json"
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    if not isinstance(stage, dict) or stage.get("source_git_sha") != source_sha:
        raise SystemExit("generation-stage source SHA does not match this checkout")
    raw_artifacts = stage.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise SystemExit("generation-stage artifact inventory is invalid")
    artifacts: list[dict[str, str]] = []
    for item in raw_artifacts:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise SystemExit("generation-stage artifact entry is invalid")
        relative = Path(item["path"])
        path = persistent_dir / relative
        digest = _sha256(path)
        if digest != item.get("sha256"):
            raise SystemExit(f"generation artifact digest mismatch: {relative}")
        artifacts.append({"path": relative.as_posix(), "sha256": digest})
    return {
        "schema_version": 1,
        "task_id": "FTR-202",
        "source_git_sha": source_sha,
        "generation_stage_sha256": _sha256(stage_path),
        "teacher_generation_directory": persistent_dir.as_posix(),
        "gpu": gpu,
        "artifacts": artifacts,
        "scoring_performed": False,
        "candidate_execution_performed": False,
        "resume_policy": "checkpoint files are persisted directly to the mounted durable root",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FTR-202 teacher generation on Colab A100")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--persistent-root", type=Path, required=True)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    persistent_root = args.persistent_root.resolve()
    if not persistent_root.is_dir():
        raise SystemExit("--persistent-root must already exist (mount Drive first)")

    source_sha = _source_sha(repo_root)
    gpu = _validate_a100()
    persistent_dir = _bind_persistent_output(
        repo_root=repo_root,
        persistent_root=persistent_root,
        source_sha=source_sha,
    )

    from tiny_qwen_coder.evaluation.python_ftr_teacher_direct import (
        audit_teacher_direct_support,
        generate_teacher_stage,
    )
    from tiny_qwen_coder.evaluation.python_ftr_teacher_superiority import (
        audit_teacher_benchmark_protocol,
    )

    audit_teacher_direct_support(repo_root=repo_root, source_git_sha=source_sha)
    audit_teacher_benchmark_protocol(repo_root=repo_root, source_git_sha=source_sha)
    stage_path = generate_teacher_stage(repo_root=repo_root)
    if stage_path.resolve() != (persistent_dir / "generation-stage.json").resolve():
        raise SystemExit("teacher generation stage wrote to an unexpected destination")

    handoff = _handoff_payload(persistent_dir=persistent_dir, source_sha=source_sha, gpu=gpu)
    handoff_path = persistent_dir.parent / "FTR_202_GENERATION_HANDOFF.json"
    handoff_path.write_text(json.dumps(handoff, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(handoff_path)


if __name__ == "__main__":
    main()
