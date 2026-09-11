#!/usr/bin/env python3
"""Prepare and run isolated FTR-202 scoring from transported teacher evidence.

This convenience wrapper is not part of the experimental execution source. It
creates a clean checkout of the immutable FTR-202 execution SHA, restores the
frozen base artifact, preloads the pinned Docker image, strips credentials from
the scoring subprocess, and delegates the actual benchmark execution/gate to
the scorer frozen at that execution SHA.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import NoReturn

_EXECUTION_SHA = "90d42cca541cc7a96493d5d80d95718e5930d551"
_REPO_URL = "https://github.com/ekkus93/tiny-qwen-coder.git"
_REPO_SLUG = "ekkus93/tiny-qwen-coder"
_BASE_RUN_ID = "33301242379"
_BASE_ARTIFACT = "python-base-baseline-da537443ab80b1380bee0fc3c7d9d01ca0574f35"
_BASE_RELATIVE = Path("artifacts/eval/python/base-baseline-v1")
_HANDOFF_FILENAME = "FTR_202_GENERATION_HANDOFF.json"
_EXECUTION_IMAGE = (
    "python:3.11.14-slim@sha256:c8271b1f627d0068857dce5b53e14a9558603b527e46f1f901722f935b786a39"
)
_STRIP_ENV = {
    "GOOGLE_APPLICATION_CREDENTIALS",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "GH_TOKEN",
    "GITHUB_TOKEN",
}


def _die(message: str) -> NoReturn:
    raise SystemExit(message)


def _require_executable(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        _die(f"FTR-202 scoring bootstrap requires {name!r} on PATH")
    return executable


def _run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _output(command: list[str], *, cwd: Path | None = None) -> str:
    completed = subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def _validate_generation_dir(generation_dir: Path) -> None:
    if not generation_dir.is_dir():
        _die(f"generation directory does not exist: {generation_dir}")
    handoff_path = generation_dir / _HANDOFF_FILENAME
    try:
        raw: object = json.loads(handoff_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"could not read FTR-202 generation handoff: {handoff_path}") from exc
    if not isinstance(raw, dict):
        _die("FTR-202 generation handoff must be a JSON object")
    if raw.get("task_id") != "FTR-202":
        _die("generation handoff task_id is not FTR-202")
    if raw.get("source_git_sha") != _EXECUTION_SHA:
        _die(
            "generation evidence was not produced by the immutable FTR-202 execution SHA "
            f"{_EXECUTION_SHA}"
        )


def _find_base_dir(download_root: Path) -> Path:
    direct = download_root / _BASE_RELATIVE
    candidates = [direct] if direct.is_dir() else []
    candidates.extend(path for path in download_root.rglob(_BASE_RELATIVE.name) if path.is_dir())
    exact = [path for path in candidates if path.as_posix().endswith(_BASE_RELATIVE.as_posix())]
    unique = sorted({path.resolve() for path in exact})
    if len(unique) != 1:
        _die(
            "expected exactly one frozen base-baseline-v1 directory after artifact download; "
            f"found {len(unique)}"
        )
    return unique[0]


def _scrubbed_environment() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if key not in _STRIP_ENV}


def _prepare_checkout(workspace: Path, *, git: str) -> Path:
    checkout = workspace / "execution-source"
    _run([git, "clone", "--no-checkout", _REPO_URL, str(checkout)])
    _run([git, "checkout", "--detach", _EXECUTION_SHA], cwd=checkout)
    observed = _output([git, "rev-parse", "HEAD"], cwd=checkout)
    if observed != _EXECUTION_SHA:
        _die(f"execution checkout drifted: expected {_EXECUTION_SHA}, observed {observed}")
    if _output([git, "status", "--porcelain"], cwd=checkout):
        _die("execution checkout is not clean")
    return checkout


def _download_base(workspace: Path, *, gh: str) -> Path:
    download_root = workspace / "base-download"
    download_root.mkdir(parents=True, exist_ok=True)
    _run(
        [
            gh,
            "run",
            "download",
            _BASE_RUN_ID,
            "-R",
            _REPO_SLUG,
            "-n",
            _BASE_ARTIFACT,
            "-D",
            str(download_root),
        ]
    )
    return _find_base_dir(download_root)


def _prepare_runtime(*, docker: str, uv: str, checkout: Path) -> None:
    _run([uv, "sync", "--frozen"], cwd=checkout)
    _run([docker, "pull", _EXECUTION_IMAGE])
    _run([docker, "image", "inspect", _EXECUTION_IMAGE])


def _score(
    *,
    checkout: Path,
    generation_dir: Path,
    base_dir: Path,
    report: Path,
    uv: str,
) -> int:
    report.parent.mkdir(parents=True, exist_ok=True)
    command = [
        uv,
        "run",
        "--frozen",
        "python",
        "scripts/evaluation/run_ftr_202_isolated_score.py",
        "--repo-root",
        ".",
        "--generation-dir",
        str(generation_dir),
        "--base-dir",
        str(base_dir),
        "--report",
        str(report),
    ]
    completed = subprocess.run(command, cwd=checkout, env=_scrubbed_environment(), check=False)
    return completed.returncode


def _execute(
    *,
    workspace: Path,
    generation_dir: Path,
    report: Path,
    git: str,
    gh: str,
    docker: str,
    uv: str,
) -> int:
    checkout = _prepare_checkout(workspace, git=git)
    base_dir = _download_base(workspace, gh=gh)
    _prepare_runtime(docker=docker, uv=uv, checkout=checkout)
    print(f"execution_sha={_EXECUTION_SHA}")
    print(f"generation_dir={generation_dir}")
    print(f"base_dir={base_dir}")
    print(f"execution_image={_EXECUTION_IMAGE}")
    print(f"report={report}")
    return _score(
        checkout=checkout,
        generation_dir=generation_dir,
        base_dir=base_dir,
        report=report,
        uv=uv,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare immutable FTR-202 execution source and run isolated scoring"
    )
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("ftr-202-teacher-superiority.json"),
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        help="Optional persistent preparation directory; otherwise a temporary directory is used",
    )
    args = parser.parse_args()

    generation_dir = args.generation_dir.resolve()
    report = args.report.resolve()
    _validate_generation_dir(generation_dir)

    git = _require_executable("git")
    gh = _require_executable("gh")
    docker = _require_executable("docker")
    uv = _require_executable("uv")

    if args.workspace is None:
        with tempfile.TemporaryDirectory(prefix="ftr202-score-") as temporary:
            exit_code = _execute(
                workspace=Path(temporary),
                generation_dir=generation_dir,
                report=report,
                git=git,
                gh=gh,
                docker=docker,
                uv=uv,
            )
    else:
        workspace = args.workspace.resolve()
        if workspace.exists() and any(workspace.iterdir()):
            _die(f"--workspace must be empty or absent: {workspace}")
        workspace.mkdir(parents=True, exist_ok=True)
        exit_code = _execute(
            workspace=workspace,
            generation_dir=generation_dir,
            report=report,
            git=git,
            gh=gh,
            docker=docker,
            uv=uv,
        )

    if exit_code not in {0, 3}:
        _die(f"FTR-202 scoring failed with infrastructure/protocol exit code {exit_code}")
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
