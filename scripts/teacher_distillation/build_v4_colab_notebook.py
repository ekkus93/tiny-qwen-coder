#!/usr/bin/env python3
"""Build the executable bounded-v4 Google Colab notebook from stdlib-only source."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

OUTPUT = Path("scripts/teacher_distillation/qwen38_teacher_distillation_v4_colab.ipynb")


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
            # Qwen3.8-27B bounded v4 selective-compression study on Google Colab

            v4 preserves the completed v3 high-reasoning answers and selectively rewrites only
            the normally stopped records that are both over the Qwen3.5-4B 2,048-token student
            envelope and over their v3 per-record answer budget.

            The compression pass uses the same pinned Qwen3.8-27B BF16 teacher with **low**
            reasoning. It never receives v3 hidden reasoning. It receives only the original
            conversation, the already-generated v3 final answer, and that record's frozen budget.

            Do not scale to 2,000 unless the merged 200-record v4 study has >=90% normal stops,
            >=85% student-length acceptance among normal stops, and contamination status `clean`.
            """
        ),
        _markdown(
            """
            ## 1. Mount Google Drive

            Select an **A100 80 GB** runtime first.
            """
        ),
        _code(
            """
            from google.colab import drive

            drive.mount("/content/drive")
            """
        ),
        _markdown("## 2. Define durable v3 evidence and fresh v4 paths"),
        _code(
            """
            import os
            from pathlib import Path

            os.environ["TQC_DRIVE"] = "/content/drive/MyDrive/tiny-qwen-coder"
            os.environ["CODE_DIR"] = f"{os.environ['TQC_DRIVE']}/code"
            os.environ["CODE_ARCHIVE"] = f"{os.environ['CODE_DIR']}/tiny-qwen-coder-v4.zip"

            os.environ["V3_DIR"] = f"{os.environ['TQC_DRIVE']}/distillation/qwen38-27b-v3-200"
            os.environ["V3_INPUT"] = f"{os.environ['V3_DIR']}/input/p0-200-v3.jsonl"
            os.environ["V3_CHECKPOINT_DIR"] = f"{os.environ['V3_DIR']}/checkpoint"

            os.environ["V4_DIR"] = (
                f"{os.environ['TQC_DRIVE']}/distillation/qwen38-27b-v4-compression-200"
            )
            os.environ["V4_CHECKPOINT_DIR"] = f"{os.environ['V4_DIR']}/compression-checkpoint"
            os.environ["V4_MERGED_DIR"] = f"{os.environ['V4_DIR']}/merged"
            os.environ["V4_MERGED"] = f"{os.environ['V4_MERGED_DIR']}/p0-200-v4.jsonl"
            os.environ["V4_DIAG_DIR"] = f"{os.environ['V4_DIR']}/diagnostics"
            os.environ["V4_FINAL_DIR"] = f"{os.environ['V4_DIR']}/final"
            os.environ["V4_QUALIFICATION"] = f"{os.environ['V4_DIR']}/qualification.json"

            for key in (
                "CODE_DIR",
                "V4_DIR",
                "V4_CHECKPOINT_DIR",
                "V4_MERGED_DIR",
                "V4_DIAG_DIR",
                "V4_FINAL_DIR",
            ):
                Path(os.environ[key]).mkdir(parents=True, exist_ok=True)

            print("Repository ZIP expected at:", os.environ["CODE_ARCHIVE"])
            print("Preserved v3 input:", os.environ["V3_INPUT"])
            print("Preserved v3 checkpoint:", os.environ["V3_CHECKPOINT_DIR"])
            print("Fresh v4 study:", os.environ["V4_DIR"])
            """
        ),
        _markdown(
            """
            ## 3. Put the frozen v4 repository ZIP on Drive

            Upload the exact v4 repository ZIP as:

            `MyDrive/tiny-qwen-coder/code/tiny-qwen-coder-v4.zip`

            Do not replace it after v4 compression has started.
            """
        ),
        _code(
            """
            from pathlib import Path

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
        _markdown(
            """
            ## 4. Verify/extract the frozen ZIP and create local provenance

            The `.sha256` sidecar seals the exact ZIP bytes. A deterministic local Git commit is
            created only so dataset manifests can record source-tree provenance; no remote or
            GitHub credentials are configured.
            """
        ),
        _code(
            """
            import hashlib
            import shutil
            import subprocess
            from pathlib import Path
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
                        "v4 repository ZIP checksum changed; use a new experiment namespace."
                    )
            else:
                checksum_path.write_text(
                    f"{archive_sha256}  {archive.name}\\n",
                    encoding="ascii",
                )

            scratch_root = Path("/content/tiny-qwen-coder-v4-code")
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

            print("repository:", repo)
            print("archive SHA-256:", archive_sha256)
            print("archive provenance Git SHA:", archive_git_sha)
            """
        ),
        _markdown("## 5. Create the isolated uv/vLLM CUDA 13.0 environment"),
        _code(
            """
            import os
            import shutil
            import subprocess
            import sys
            from pathlib import Path

            os.chdir(os.environ["TQC_REPO"])
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
            import subprocess
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
        _markdown(
            "## 7. Define a streaming subprocess helper and verify preserved v3 evidence"
        ),
        _code(
            """
            import shlex
            import subprocess
            from pathlib import Path

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

            required_v3 = (
                Path(os.environ["V3_INPUT"]),
                Path(os.environ["V3_CHECKPOINT_DIR"]) / "run-identity.json",
            )
            missing = [str(path) for path in required_v3 if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    "Preserved v3 evidence is incomplete: " + ", ".join(missing)
                )
            print("Preserved v3 evidence found.")
            """
        ),
        _markdown(
            """
            ## 8. Preflight selective v4 compression without loading Qwen3.8

            This validates the preserved v3 checkpoint under the unchanged generation
            implementation, recomputes the selective target set, and creates/validates the fresh
            v4 compression identity. For the observed v3-200 run, the target count should be 40.
            """
        ),
        _code(
            """
            run_teacher(
                "scripts/teacher_distillation/compress_teacher_v4.py",
                "--source-input",
                os.environ["V3_INPUT"],
                "--source-checkpoint-dir",
                os.environ["V3_CHECKPOINT_DIR"],
                "--checkpoint-dir",
                os.environ["V4_CHECKPOINT_DIR"],
                "--work-dir",
                "/content/tqc-distillation-v4-compression",
                "--output",
                os.environ["V4_MERGED"],
                "--status-only",
            )
            """
        ),
        _markdown(
            """
            ## 9. Run or resume the selective compression pass

            This is the only v4 cell that loads Qwen3.8-27B. Only missing compression shards are
            generated. Already student-length-accepted v3 answers are never regenerated.
            """
        ),
        _code(
            """
            run_teacher(
                "scripts/teacher_distillation/compress_teacher_v4.py",
                "--source-input",
                os.environ["V3_INPUT"],
                "--source-checkpoint-dir",
                os.environ["V3_CHECKPOINT_DIR"],
                "--checkpoint-dir",
                os.environ["V4_CHECKPOINT_DIR"],
                "--work-dir",
                "/content/tqc-distillation-v4-compression",
                "--output",
                os.environ["V4_MERGED"],
            )
            """
        ),
        _markdown("## 10. Verify the durable v4 checkpoint without loading the teacher"),
        _code(
            """
            run_teacher(
                "scripts/teacher_distillation/compress_teacher_v4.py",
                "--source-input",
                os.environ["V3_INPUT"],
                "--source-checkpoint-dir",
                os.environ["V3_CHECKPOINT_DIR"],
                "--checkpoint-dir",
                os.environ["V4_CHECKPOINT_DIR"],
                "--work-dir",
                "/content/tqc-distillation-v4-compression",
                "--output",
                os.environ["V4_MERGED"],
                "--status-only",
            )
            """
        ),
        _markdown("## 11. Diagnose the merged 200-record v4 corpus"),
        _code(
            """
            run_teacher(
                "scripts/teacher_distillation/diagnose_prepared_teacher_data.py",
                "--input",
                os.environ["V4_MERGED"],
                "--output-dir",
                os.environ["V4_DIAG_DIR"],
            )
            """
        ),
        _markdown(
            "## 12. Finalize v4 with fail-closed Python quality and protected-benchmark checks"
        ),
        _code(
            """
            run_teacher(
                "scripts/teacher_distillation/finalize_prepared_teacher_data.py",
                "--input",
                os.environ["V4_MERGED"],
                "--output-dir",
                os.environ["V4_FINAL_DIR"],
            )
            """
        ),
        _markdown(
            """
            ## 13. Mechanically qualify the bounded v4 study

            The scale gate remains >=90% normal stops, >=85% student-length acceptance among
            normal stops, and contamination status `clean`.
            """
        ),
        _code(
            """
            qualification = run_teacher(
                "scripts/teacher_distillation/qualify_teacher_study.py",
                "--diagnostics-summary",
                str(Path(os.environ["V4_DIAG_DIR"]) / "teacher-length-summary.json"),
                "--dataset-manifest",
                str(Path(os.environ["V4_FINAL_DIR"]) / "dataset-manifest.json"),
                "--minimum-student-length-accept-rate-given-stop",
                "0.85",
                "--output",
                os.environ["V4_QUALIFICATION"],
                check=False,
            )
            print("qualification exit code:", qualification.returncode)
            """
        ),
        _markdown("## 14. Inspect the scaling decision"),
        _code(
            """
            import json

            qualification_path = Path(os.environ["V4_QUALIFICATION"])
            if not qualification_path.is_file():
                raise FileNotFoundError("qualification.json was not written")
            decision = json.loads(qualification_path.read_text(encoding="utf-8"))
            print(json.dumps(decision, indent=2, sort_keys=True))
            if decision.get("qualified") is not True:
                print("DO NOT SCALE. Preserve v4 evidence and revise the intervention.")
            else:
                print("QUALIFIED. A fresh 2,000-record successor may be prepared.")
            """
        ),
        _markdown(
            """
            ## 15. Scaling guard

            This notebook never automatically starts the 2,000-record successor.
            """
        ),
        _code(
            """
            if decision.get("qualified") is not True:
                raise RuntimeError(
                    "v4-200 did not qualify; 2,000-record generation is blocked"
                )
            print("v4-200 qualified; preserve this evidence before scaling.")
            """
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "name": "qwen38_teacher_distillation_v4_colab.ipynb",
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


def main() -> None:
    OUTPUT.write_text(
        json.dumps(_notebook(), indent=1) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
