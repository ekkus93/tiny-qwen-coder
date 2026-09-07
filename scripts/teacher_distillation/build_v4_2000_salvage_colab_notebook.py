#!/usr/bin/env python3
"""Build the executable v4-2000 reasoning-salvage Google Colab notebook."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(
    "scripts/teacher_distillation/qwen38_teacher_distillation_v4_2000_salvage_colab.ipynb"
)
RUFF_CONFIG = REPO_ROOT / "pyproject.toml"
CELL_IDS = (
    "7fb27b941602401d91542211134fc71a",
    "acae54e37e7d407bbb7b55eff062a284",
    "9a63283cbaf04dbcab1f6479b197f3a8",
    "8dd0d8092fe74a7c96281538738b07e2",
    "72eea5119410473aa328ad9291626812",
    "8edb47106e1a46a883d545849b8ab81b",
    "10185d26023b46108eb7d9f57d49d2b3",
    "8763a12b2bbd4a93a75aff182afb95dc",
    "7623eae2785240b9bd12b16a66d81610",
    "7cdc8c89c7104fffa095e18ddfef8986",
    "b118ea5561624da68c537baed56e602f",
    "938c804e27f84196a10c8828c723f798",
    "504fb2a444614c0babb325280ed9130a",
    "59bbdb311c014d738909a11f9e486628",
    "b43b363d81ae4b689946ece5c682cd59",
    "8a65eabff63a45729fe45fb5ade58bdc",
    "c3933fab20d04ec698c2621248eb3be0",
    "4dd4641cc4064e0191573fe9c69df29b",
    "8309879909854d7188b41380fd92a7c3",
    "3ed186c9a28b402fb0bc4494df01f08d",
    "cb1e1581032b452c9409d6c6813c49d1",
    "379cbbc1e968416e875cc15c1202d7eb",
    "277c27b1587741f2af2001be3712ef0d",
    "db7b79bc585a40fcaf58bf750017e135",
    "916684f9a58a4a2aa5f864670399430d",
    "1671c31a24314836a5b85d7ef7fbf015",
    "33b0902fd34d4ace834912fa1002cf8e",
    "f6fa52606d8c4a75a9b52967216f8f3f",
    "f5a1fa73e5044315a093ec459c9be902",
    "cdf66aed5cc84ca1b48e60bad68798a8",
    "28d3efd5258a48a79c179ea5c6759f01",
    "3f9bc0b9dd2c44919cc8dcca39b469f8",
    "0e382214b5f147d187d36a2058b9c724",
    "5b09d5ef5b5e4bb6ab9b829b10b6a29f",
    "a50416e276a0479cbe66534ed1713a40",
    "46a27a456b804aa2a380d5edf15a5daf",
    "1944c39560714e6e80c856f20744a8e5",
    "d6ca27006b894b04b6fc8b79396e2797",
)


def _lines(text: str) -> list[str]:
    return [line + "\n" for line in textwrap.dedent(text).strip().splitlines()]


def _markdown(text: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def _code(text: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(text),
    }


def _notebook() -> dict[str, object]:
    cells: list[dict[str, object]] = [
        _markdown(
            """
            # Qwen3.8-27B v4-2000 reasoning-leak salvage on Google Colab

            This notebook repairs the completed v4-2000 teacher run **without regenerating the
            2,000 high-reasoning first-pass answers**. The original v4-2000 `final/` corpus is
            frozen evidence and must not be used for student training because its assistant
            responses contain leaked Qwen reasoning before a closing `</think>` marker.

            The workflow validates every original generation shard, reconstructs reasoning-free
            first-pass answers, recomputes selective-compression targets, runs a **fresh**
            low-reasoning compression checkpoint only for remaining overlength budget violators,
            finalizes in a new namespace, and fails closed if any thinking marker survives.
            """
        ),
        _markdown(
            """
            ## 1. Mount Google Drive

            Use a fresh **A100 80 GB** runtime. Most repair steps are CPU-only; the A100 is used
            only if the recomputed selective-compression target set is non-empty.
            """
        ),
        _code(
            """
            from google.colab import drive

            drive.mount("/content/drive")
            """
        ),
        _markdown("## 2. Define frozen legacy evidence and fresh salvage paths"),
        _code(
            """
            import os
            from pathlib import Path

            os.environ["TQC_DRIVE"] = "/content/drive/MyDrive/tiny-qwen-coder"
            os.environ["CODE_DIR"] = f"{os.environ['TQC_DRIVE']}/code"
            os.environ["CODE_ARCHIVE"] = (
                f"{os.environ['CODE_DIR']}/tiny-qwen-coder-v4-2000-salvage-v1.zip"
            )

            os.environ["LEGACY_DIR"] = (
                f"{os.environ['TQC_DRIVE']}/distillation/qwen38-27b-v4-2000"
            )
            os.environ["LEGACY_SOURCE_INPUT"] = (
                f"{os.environ['LEGACY_DIR']}/input/p0-2000-v3.jsonl"
            )
            os.environ["LEGACY_GENERATION_CHECKPOINT"] = (
                f"{os.environ['LEGACY_DIR']}/generation-checkpoint"
            )

            os.environ["SALVAGE_DIR"] = (
                f"{os.environ['TQC_DRIVE']}/distillation/"
                "qwen38-27b-v4-2000-salvage-v1"
            )
            os.environ["SALVAGED_SOURCE_DIR"] = f"{os.environ['SALVAGE_DIR']}/source"
            os.environ["SALVAGED_SOURCE"] = (
                f"{os.environ['SALVAGED_SOURCE_DIR']}/p0-2000-v3-sanitized.jsonl"
            )
            os.environ["SALVAGE_IDENTITY_DIR"] = f"{os.environ['SALVAGE_DIR']}/source-identity"
            os.environ["SALVAGE_COMPRESSION_CHECKPOINT"] = (
                f"{os.environ['SALVAGE_DIR']}/compression-checkpoint"
            )
            os.environ["SALVAGE_MERGED_DIR"] = f"{os.environ['SALVAGE_DIR']}/merged"
            os.environ["SALVAGE_MERGED"] = (
                f"{os.environ['SALVAGE_MERGED_DIR']}/p0-2000-v4-sanitized.jsonl"
            )
            os.environ["SALVAGE_DIAG_DIR"] = f"{os.environ['SALVAGE_DIR']}/diagnostics"
            os.environ["SALVAGE_FINAL_DIR"] = f"{os.environ['SALVAGE_DIR']}/final"
            os.environ["SALVAGE_QUALIFICATION"] = (
                f"{os.environ['SALVAGE_DIR']}/qualification.json"
            )

            for key in (
                "CODE_DIR",
                "SALVAGE_DIR",
                "SALVAGED_SOURCE_DIR",
                "SALVAGE_IDENTITY_DIR",
                "SALVAGE_COMPRESSION_CHECKPOINT",
                "SALVAGE_MERGED_DIR",
                "SALVAGE_DIAG_DIR",
                "SALVAGE_FINAL_DIR",
            ):
                Path(os.environ[key]).mkdir(parents=True, exist_ok=True)

            print("Frozen affected run:", os.environ["LEGACY_DIR"])
            print("Fresh salvage namespace:", os.environ["SALVAGE_DIR"])
            print("Repository ZIP expected at:", os.environ["CODE_ARCHIVE"])
            """
        ),
        _markdown(
            """
            ## 3. Put the frozen repaired repository ZIP on Drive

            Download the exact repository ZIP for the commit that contains this notebook, rename
            it to `tiny-qwen-coder-v4-2000-salvage-v1.zip`, and place it under
            `MyDrive/tiny-qwen-coder/code/`. The cell can also upload it interactively.
            """
        ),
        _code(
            """
            archive = Path(os.environ["CODE_ARCHIVE"])
            if archive.exists():
                print("Repository ZIP already exists:", archive)
            else:
                from google.colab import files

                uploaded = files.upload()
                if len(uploaded) != 1:
                    raise RuntimeError("Upload exactly one repository ZIP.")
                uploaded_name, uploaded_bytes = next(iter(uploaded.items()))
                if not uploaded_name.lower().endswith(".zip"):
                    raise RuntimeError("The uploaded file must be a .zip archive.")
                archive.write_bytes(uploaded_bytes)
                print("Saved repository ZIP to:", archive)
            """
        ),
        _markdown("## 4. Verify/extract the frozen ZIP and create local manifest provenance"),
        _code(
            """
            import hashlib
            import shutil
            import subprocess
            from zipfile import ZipFile

            archive = Path(os.environ["CODE_ARCHIVE"])
            if not archive.is_file():
                raise FileNotFoundError(archive)

            def sha256_file(path: Path) -> str:
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                return digest.hexdigest()

            archive_sha256 = sha256_file(archive)
            checksum_path = archive.with_suffix(archive.suffix + ".sha256")
            if checksum_path.exists():
                expected = checksum_path.read_text(encoding="ascii").split()[0]
                if expected != archive_sha256:
                    raise RuntimeError(
                        "salvage repository ZIP checksum changed; use a new namespace."
                    )
            else:
                checksum_path.write_text(
                    f"{archive_sha256}  {archive.name}\\n",
                    encoding="ascii",
                )

            scratch_root = Path("/content/tiny-qwen-coder-salvage-code")
            if scratch_root.exists():
                shutil.rmtree(scratch_root)
            scratch_root.mkdir(parents=True)
            with ZipFile(archive) as zip_file:
                zip_file.extractall(scratch_root)

            repo_candidates = sorted(
                {
                    pyproject.parent
                    for pyproject in scratch_root.rglob("pyproject.toml")
                    if (pyproject.parent / "scripts/teacher_distillation/README.md").is_file()
                }
            )
            if len(repo_candidates) != 1:
                raise RuntimeError(
                    f"Expected one repository in the ZIP; found {len(repo_candidates)}."
                )

            repo = repo_candidates[0]
            os.environ["TQC_REPO"] = str(repo)
            os.environ["TQC_CODE_ARCHIVE_SHA256"] = archive_sha256
            git_env = os.environ.copy()
            git_env.update(
                {
                    "GIT_AUTHOR_NAME": "Tiny Qwen Coder Archive",
                    "GIT_AUTHOR_EMAIL": "archive@tiny-qwen-coder.invalid",
                    "GIT_COMMITTER_NAME": "Tiny Qwen Coder Archive",
                    "GIT_COMMITTER_EMAIL": "archive@tiny-qwen-coder.invalid",
                    "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+00:00",
                    "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+00:00",
                }
            )
            subprocess.run(["git", "init", "--quiet"], cwd=repo, check=True, env=git_env)
            subprocess.run(
                ["git", "-c", "core.autocrlf=false", "add", "--all"],
                cwd=repo,
                check=True,
                env=git_env,
            )
            subprocess.run(
                [
                    "git",
                    "commit",
                    "--quiet",
                    "--no-gpg-sign",
                    "-m",
                    f"Frozen repository archive sha256:{archive_sha256}",
                ],
                cwd=repo,
                check=True,
                env=git_env,
            )
            archive_git_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True, env=git_env
            ).strip()
            if subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=repo, text=True, env=git_env
            ).strip():
                raise RuntimeError("Extracted repository is unexpectedly dirty.")
            os.environ["TQC_ARCHIVE_GIT_SHA"] = archive_git_sha
            os.chdir(repo)

            print("repository:", repo)
            print("archive SHA-256:", archive_sha256)
            print("archive provenance Git SHA:", archive_git_sha)
            """
        ),
        _markdown("## 5. Create the isolated uv/vLLM CUDA 13.0 environment"),
        _code(
            """
            import sys

            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade", "uv"],
                check=True,
            )
            uv_executable = shutil.which("uv")
            if uv_executable is None:
                raise RuntimeError("uv was installed but is not on PATH.")

            venv = Path("/content/tqc-teacher-venv")
            if venv.exists():
                shutil.rmtree(venv)
            subprocess.run(
                [uv_executable, "venv", str(venv), "--python", sys.executable],
                check=True,
            )
            os.environ["TQC_UV"] = uv_executable
            os.environ["TQC_VENV"] = str(venv)
            os.environ["TQC_PYTHON"] = str(venv / "bin" / "python")
            os.environ["PATH"] = f"{venv / 'bin'}{os.pathsep}{os.environ['PATH']}"

            def run_uv(args: list[str]) -> None:
                command = [os.environ["TQC_UV"], *args]
                print("$", " ".join(command), flush=True)
                result = subprocess.run(
                    command,
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                print(result.stdout, end="", flush=True)
                result.check_returncode()

            run_uv(
                [
                    "pip",
                    "install",
                    "--python",
                    os.environ["TQC_PYTHON"],
                    "--torch-backend=cu130",
                    "-r",
                    "requirements/colab-teacher.txt",
                    "-e",
                    ".",
                ]
            )
            """
        ),
        _markdown("## 6. Verify CUDA, vLLM, Ninja, and the A100"),
        _code(
            """
            import textwrap

            !nvidia-smi

            verification_code = textwrap.dedent(
                '''
                import shutil
                import subprocess
                import sys
                import torch
                import vllm

                print("python:", sys.executable, flush=True)
                print("torch:", torch.__version__, flush=True)
                print("torch CUDA:", torch.version.cuda, flush=True)
                print("vllm:", vllm.__version__, flush=True)
                ninja = shutil.which("ninja")
                print("ninja:", ninja or "MISSING", flush=True)
                if ninja is None:
                    raise RuntimeError("Ninja is not on PATH.")
                print(
                    "ninja version:",
                    subprocess.check_output([ninja, "--version"], text=True).strip(),
                    flush=True,
                )
                print(
                    "gpu:",
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE",
                    flush=True,
                )
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA GPU is unavailable.")
                if torch.version.cuda != "13.0":
                    raise RuntimeError(
                        f"Expected PyTorch CUDA 13.0, found {torch.version.cuda!r}."
                    )
                if vllm.__version__ != "0.28.0":
                    raise RuntimeError(f"Unexpected vLLM version: {vllm.__version__}")
                '''
            )
            result = subprocess.run(
                [os.environ["TQC_PYTHON"], "-c", verification_code],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            print(result.stdout, end="", flush=True)
            result.check_returncode()
            """
        ),
        _markdown("## 7. Define a streaming subprocess helper"),
        _code(
            """
            import shlex

            def run_teacher(
                *args: str,
                check: bool = True,
            ) -> subprocess.CompletedProcess[str]:
                command = [os.environ["TQC_PYTHON"], "-u", *args]
                print("$", shlex.join(command), flush=True)
                with subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                ) as process:
                    assert process.stdout is not None
                    captured: list[str] = []
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        captured.append(line)
                    returncode = process.wait()
                result = subprocess.CompletedProcess(
                    command,
                    returncode,
                    "".join(captured),
                    None,
                )
                if check:
                    result.check_returncode()
                return result
            """
        ),
        _markdown("## 8. Verify the frozen affected generation evidence exists"),
        _code(
            """
            required = (
                Path(os.environ["LEGACY_SOURCE_INPUT"]),
                Path(os.environ["LEGACY_SOURCE_INPUT"] + ".sha256"),
                Path(os.environ["LEGACY_GENERATION_CHECKPOINT"]) / "run-identity.json",
            )
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    "Frozen v4-2000 generation evidence is incomplete: " + ", ".join(missing)
                )
            print("Frozen affected generation evidence found.")
            print("The old compression checkpoint and old final/ directory will not be reused.")
            """
        ),
        _markdown(
            """
            ## 9. Reconstruct a reasoning-free first-pass source from the sealed legacy shards

            This is CPU-only. It validates every old shard/checksum and writes a new deterministic
            source JSONL. It never edits the old generation checkpoint.
            """
        ),
        _code(
            """
            run_teacher(
                "scripts/teacher_distillation/salvage_teacher_reasoning.py",
                "--config",
                "configs/distillation/python/qwen38_27b_v3.yaml",
                "--source-input",
                os.environ["LEGACY_SOURCE_INPUT"],
                "--legacy-checkpoint-dir",
                os.environ["LEGACY_GENERATION_CHECKPOINT"],
                "--output",
                os.environ["SALVAGED_SOURCE"],
                "--identity-dir",
                os.environ["SALVAGE_IDENTITY_DIR"],
            )
            """
        ),
        _markdown("## 10. Inspect and pin the expected legacy-salvage accounting"),
        _code(
            """
            import json

            salvage_summary_path = Path(os.environ["SALVAGED_SOURCE"] + ".summary.json")
            salvage_summary = json.loads(salvage_summary_path.read_text(encoding="utf-8"))
            print(json.dumps(salvage_summary, indent=2, sort_keys=True))

            expected = {
                "source_records": 2000,
                "stop_records": 1983,
                "length_records": 17,
                "closing_only_records": 1987,
                "wrapped_records": 0,
                "truncated_reasoning_only_records": 13,
            }
            for key, value in expected.items():
                if salvage_summary.get(key) != value:
                    raise RuntimeError(
                        f"Unexpected legacy salvage accounting for {key}: "
                        f"expected {value}, found {salvage_summary.get(key)!r}"
                    )
            text = Path(os.environ["SALVAGED_SOURCE"]).read_text(encoding="utf-8")
            if "<think>" in text or "</think>" in text:
                raise RuntimeError("Sanitized source still contains thinking markup")
            print("Sanitized first-pass source matches the frozen affected-run accounting.")
            """
        ),
        _markdown(
            """
            ## 11. Recompute selective-compression targets without loading Qwen3.8

            This uses the sanitized answers and their original frozen v3 budgets. The target count
            is intentionally **not predeclared** because removing hidden reasoning changes the true
            student-token lengths. The computed value is authoritative.
            """
        ),
        _code(
            """
            run_teacher(
                "scripts/teacher_distillation/compress_teacher_v4.py",
                "--source-records",
                os.environ["SALVAGED_SOURCE"],
                "--source-checkpoint-dir",
                os.environ["SALVAGE_IDENTITY_DIR"],
                "--checkpoint-dir",
                os.environ["SALVAGE_COMPRESSION_CHECKPOINT"],
                "--work-dir",
                "/content/tqc-v4-2000-salvage-compression",
                "--output",
                os.environ["SALVAGE_MERGED"],
                "--status-only",
            )
            """
        ),
        _markdown(
            """
            ## 12. Run or resume the fresh selective-compression pass

            This is the only repair cell that loads Qwen3.8-27B. The old 475-target compression
            checkpoint is ignored. Only the newly recomputed sanitized-source targets are generated.
            """
        ),
        _code(
            """
            run_teacher(
                "scripts/teacher_distillation/compress_teacher_v4.py",
                "--source-records",
                os.environ["SALVAGED_SOURCE"],
                "--source-checkpoint-dir",
                os.environ["SALVAGE_IDENTITY_DIR"],
                "--checkpoint-dir",
                os.environ["SALVAGE_COMPRESSION_CHECKPOINT"],
                "--work-dir",
                "/content/tqc-v4-2000-salvage-compression",
                "--output",
                os.environ["SALVAGE_MERGED"],
            )
            """
        ),
        _markdown("## 13. Verify the fresh compression checkpoint without loading the teacher"),
        _code(
            """
            run_teacher(
                "scripts/teacher_distillation/compress_teacher_v4.py",
                "--source-records",
                os.environ["SALVAGED_SOURCE"],
                "--source-checkpoint-dir",
                os.environ["SALVAGE_IDENTITY_DIR"],
                "--checkpoint-dir",
                os.environ["SALVAGE_COMPRESSION_CHECKPOINT"],
                "--work-dir",
                "/content/tqc-v4-2000-salvage-compression",
                "--output",
                os.environ["SALVAGE_MERGED"],
                "--status-only",
            )
            """
        ),
        _markdown("## 14. Diagnose the repaired merged 2,000-record corpus"),
        _code(
            """
            run_teacher(
                "scripts/teacher_distillation/diagnose_prepared_teacher_data.py",
                "--input",
                os.environ["SALVAGE_MERGED"],
                "--output-dir",
                os.environ["SALVAGE_DIAG_DIR"],
            )
            """
        ),
        _markdown(
            "## 15. Finalize with Python quality, 2,048-token, dedup, and contamination gates"
        ),
        _code(
            """
            run_teacher(
                "scripts/teacher_distillation/finalize_prepared_teacher_data.py",
                "--input",
                os.environ["SALVAGE_MERGED"],
                "--output-dir",
                os.environ["SALVAGE_FINAL_DIR"],
            )
            """
        ),
        _markdown("## 16. Mechanically qualify the repaired 2,000-record corpus"),
        _code(
            """
            qualification = run_teacher(
                "scripts/teacher_distillation/qualify_teacher_study.py",
                "--diagnostics-summary",
                str(Path(os.environ["SALVAGE_DIAG_DIR"]) / "teacher-length-summary.json"),
                "--dataset-manifest",
                str(Path(os.environ["SALVAGE_FINAL_DIR"]) / "dataset-manifest.json"),
                "--minimum-student-length-accept-rate-given-stop",
                "0.85",
                "--output",
                os.environ["SALVAGE_QUALIFICATION"],
                check=False,
            )
            print("qualification exit code:", qualification.returncode)
            """
        ),
        _markdown(
            "## 17. Inspect final evidence and prove the train/validation files are reasoning-free"
        ),
        _code(
            """
            decision = json.loads(
                Path(os.environ["SALVAGE_QUALIFICATION"]).read_text(encoding="utf-8")
            )
            print("QUALIFICATION")
            print(json.dumps(decision, indent=2, sort_keys=True))

            final_dir = Path(os.environ["SALVAGE_FINAL_DIR"])
            summary = json.loads(
                (final_dir / "teacher-finalization.json").read_text(encoding="utf-8")
            )
            manifest = json.loads(
                (final_dir / "dataset-manifest.json").read_text(encoding="utf-8")
            )
            print()
            print("FINALIZATION")
            print(json.dumps(summary, indent=2, sort_keys=True))
            print()
            print("MANIFEST COUNTS")
            print(json.dumps(manifest["counts"], indent=2, sort_keys=True))
            print()
            print("CONTAMINATION")
            print(json.dumps(manifest["contamination"], indent=2, sort_keys=True))

            for filename in ("accepted.jsonl", "train.jsonl", "validation.jsonl"):
                path = final_dir / filename
                text = path.read_text(encoding="utf-8")
                if "<think>" in text or "</think>" in text:
                    raise RuntimeError(f"Thinking markup survived into {filename}")
                print(filename, "reasoning-marker audit: CLEAN")

            if decision.get("qualified") is True:
                print("REPAIRED V4-2000 READY. Preserve this evidence before training.")
            else:
                print("DO NOT TRAIN. The repaired corpus did not qualify.")
            """
        ),
        _markdown("## 18. Training-readiness guard"),
        _code(
            """
            if decision.get("qualified") is not True:
                raise RuntimeError(
                    "repaired v4-2000 did not qualify; student training is blocked"
                )
            print(
                "repaired v4-2000 qualified and passed the reasoning-marker audit; "
                "freeze the final artifacts before training."
            )
            """
        ),
        _markdown(
            """
            ## Recovery

            Google Drive is durable; `/content` is disposable. If Colab disconnects, start a new
            A100 runtime with the **same sealed repository ZIP** and rerun from the top. The salvage
            reconstruction is deterministic and the fresh compression checkpoint resumes sealed
            shards. Never edit the original v4-2000 evidence, the salvage run identity, or shard
            checksum sidecars.
            """
        ),
    ]
    if len(cells) != len(CELL_IDS):
        raise RuntimeError(
            f"Notebook cell count changed: expected {len(CELL_IDS)}, found {len(cells)}"
        )
    for cell, cell_id in zip(cells, CELL_IDS, strict=True):
        cell["id"] = cell_id

    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "name": "qwen38_teacher_distillation_v4_2000_salvage_colab.ipynb",
                "provenance": [],
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_notebook(output: Path = OUTPUT) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_notebook(), indent=1) + "\n",
        encoding="utf-8",
    )

    ruff = shutil.which("ruff")
    if ruff is None:
        raise RuntimeError(
            "Ruff is required to canonicalize the generated notebook; "
            "run this builder inside the repository's frozen development environment."
        )
    subprocess.run(
        [ruff, "format", "--config", str(RUFF_CONFIG), str(output)],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    build_notebook(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
