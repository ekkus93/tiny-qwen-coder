#!/usr/bin/env python3
"""Build the P9-009B semantic-contract generation Google Colab notebook."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path("scripts/teacher_distillation/qwen38_p9_009_semantic_contract_generation_colab.ipynb")
RUFF_CONFIG = REPO_ROOT / "pyproject.toml"


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


def _cell_id(index: int, cell: dict[str, object]) -> str:
    source = cell.get("source")
    if not isinstance(source, list):
        raise RuntimeError("notebook cell source is not a list")
    payload = f"{index}\0{cell.get('cell_type')}\0{''.join(str(part) for part in source)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _notebook() -> dict[str, object]:
    cells: list[dict[str, object]] = [
        _markdown(
            """
            # P9-009B — Qwen3.8 semantic-contract generation on Google Colab

            This notebook generates exactly one **candidate-hidden semantic contract** for each of
            the 1,557 repaired P9-007 accepted examples. It does not regenerate student targets,
            train a student, or execute generated contract code while Google Drive is mounted.

            Use a fresh **A100 80 GB** Colab runtime. Generation checkpoints live on Google Drive,
            so an interrupted runtime can be replaced and resumed from the top.
            """
        ),
        _markdown("## 1. Mount Google Drive"),
        _code(
            """
            from google.colab import drive

            drive.mount("/content/drive")
            """
        ),
        _markdown("## 2. Define frozen source evidence and fresh P9-009B paths"),
        _code(
            """
            import os
            from pathlib import Path

            os.environ["TQC_DRIVE"] = "/content/drive/MyDrive/tiny-qwen-coder"
            os.environ["CODE_DIR"] = f"{os.environ['TQC_DRIVE']}/code"
            os.environ["CODE_ARCHIVE"] = f"{os.environ['CODE_DIR']}/tiny-qwen-coder-p9-009b.zip"

            os.environ["SOURCE_DIR"] = (
                f"{os.environ['TQC_DRIVE']}/distillation/qwen38-27b-v4-2000-salvage-v1"
            )
            os.environ["SOURCE_FINAL_DIR"] = f"{os.environ['SOURCE_DIR']}/final"
            os.environ["SOURCE_ACCEPTED"] = f"{os.environ['SOURCE_FINAL_DIR']}/accepted.jsonl"
            os.environ["SOURCE_MANIFEST"] = f"{os.environ['SOURCE_FINAL_DIR']}/dataset-manifest.json"
            os.environ["SOURCE_IDENTITY"] = f"{os.environ['SOURCE_DIR']}/source-identity/run-identity.json"
            os.environ["SOURCE_QUALIFICATION"] = f"{os.environ['SOURCE_DIR']}/qualification.json"

            os.environ["SEMANTIC_DIR"] = (
                f"{os.environ['TQC_DRIVE']}/distillation/qwen38-27b-v4-2000-semantic-v1"
            )
            os.environ["CONTRACT_INPUT_DIR"] = f"{os.environ['SEMANTIC_DIR']}/input"
            os.environ["CONTRACT_INPUT"] = f"{os.environ['CONTRACT_INPUT_DIR']}/contract-input.jsonl"
            os.environ["CONTRACT_CHECKPOINT_DIR"] = f"{os.environ['SEMANTIC_DIR']}/generation-checkpoint"
            os.environ["CONTRACT_EXPORT_DIR"] = f"{os.environ['SEMANTIC_DIR']}/export"
            os.environ["CONTRACT_EXPORT"] = (
                f"{os.environ['CONTRACT_EXPORT_DIR']}/p9-009b-semantic-contracts.zip"
            )

            for key in (
                "CODE_DIR",
                "SEMANTIC_DIR",
                "CONTRACT_INPUT_DIR",
                "CONTRACT_CHECKPOINT_DIR",
                "CONTRACT_EXPORT_DIR",
            ):
                Path(os.environ[key]).mkdir(parents=True, exist_ok=True)

            print("Frozen repaired source:", os.environ["SOURCE_DIR"])
            print("Fresh semantic namespace:", os.environ["SEMANTIC_DIR"])
            print("Repository ZIP expected at:", os.environ["CODE_ARCHIVE"])
            """
        ),
        _markdown(
            """
            ## 3. Put the exact P9-009B repository ZIP on Drive

            Download the repository ZIP for the exact commit containing this notebook, rename it
            `tiny-qwen-coder-p9-009b.zip`, and place it under `MyDrive/tiny-qwen-coder/code/`.
            The cell can also upload it interactively.
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
        _markdown("## 4. Seal and extract the repository ZIP"),
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
                        "P9-009B repository ZIP checksum changed; use a new experiment namespace."
                    )
            else:
                checksum_path.write_text(
                    f"{archive_sha256}  {archive.name}\\n",
                    encoding="ascii",
                )

            scratch_root = Path("/content/tiny-qwen-coder-p9-009b-code")
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

            venv = Path("/content/tqc-p9-009b-venv")
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

            command = [
                uv_executable,
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
            print("$", " ".join(command), flush=True)
            subprocess.run(command, check=True)
            """
        ),
        _markdown("## 6. Verify CUDA, vLLM, Ninja, and the A100"),
        _code(
            """
            import textwrap

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
                if "A100" not in torch.cuda.get_device_name(0):
                    raise RuntimeError("P9-009B requires an A100 runtime.")
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
        _markdown("## 8. Verify the frozen repaired-corpus boundary"),
        _code(
            """
            import json

            expected_manifest_sha256 = (
                "7e299de17cbb62bff3cf5f50c61ad6ad30ca571c9297b9935ff1307cd52e0eb7"
            )
            expected_source_output_sha256 = (
                "7f07f7253e98bf8ed295b12d72ebdf03c44b2e08a8d9f74aaa02dce5b760d966"
            )

            source_manifest_path = Path(os.environ["SOURCE_MANIFEST"])
            source_identity_path = Path(os.environ["SOURCE_IDENTITY"])
            source_qualification_path = Path(os.environ["SOURCE_QUALIFICATION"])
            source_accepted_path = Path(os.environ["SOURCE_ACCEPTED"])
            for path in (
                source_manifest_path,
                source_identity_path,
                source_qualification_path,
                source_accepted_path,
            ):
                if not path.is_file():
                    raise FileNotFoundError(path)

            if sha256_file(source_manifest_path) != expected_manifest_sha256:
                raise RuntimeError("repaired source dataset manifest checksum drifted")
            source_identity = json.loads(source_identity_path.read_text(encoding="utf-8"))
            if source_identity.get("output_sha256") != expected_source_output_sha256:
                raise RuntimeError("repaired source output identity drifted")
            qualification = json.loads(source_qualification_path.read_text(encoding="utf-8"))
            if qualification.get("qualified") is not True:
                raise RuntimeError("repaired source corpus is not qualified")
            if qualification.get("contamination_status") != "clean":
                raise RuntimeError("repaired source contamination evidence is not clean")

            accepted_records = sum(1 for line in source_accepted_path.open(encoding="utf-8") if line.strip())
            if accepted_records != 1557:
                raise RuntimeError(
                    f"expected 1557 repaired accepted records, found {accepted_records}"
                )

            print("source manifest SHA-256:", expected_manifest_sha256)
            print("source output SHA-256:", expected_source_output_sha256)
            print("accepted records:", accepted_records)
            print("qualification: TRUE; contamination: CLEAN")
            """
        ),
        _markdown(
            """
            ## 9. Build and seal the 1,557 candidate-hidden verifier inputs

            The independent verifier receives the original user task and the frozen policy only.
            The repaired candidate answer is absent from model messages; its SHA-256 is retained
            only as provenance for later pairing on the self-hosted execution runner.
            """
        ),
        _code(
            """
            run_teacher(
                "scripts/teacher_distillation/prepare_semantic_contract_input.py",
                "--input",
                os.environ["SOURCE_ACCEPTED"],
                "--output",
                os.environ["CONTRACT_INPUT"],
            )
            contract_input = Path(os.environ["CONTRACT_INPUT"])
            contract_input_sha256 = sha256_file(contract_input)
            print("contract input SHA-256:", contract_input_sha256)
            """
        ),
        _markdown("## 10. Validate durable progress without loading Qwen3.8"),
        _code(
            """
            run_teacher(
                "scripts/teacher_distillation/generate_teacher_data.py",
                "--config",
                "configs/distillation/python/qwen38_27b_semantic_contract_v1.yaml",
                "--input",
                os.environ["CONTRACT_INPUT"],
                "--checkpoint-dir",
                os.environ["CONTRACT_CHECKPOINT_DIR"],
                "--work-dir",
                "/content/tqc-p9-009b-generation",
                "--status-only",
            )
            """
        ),
        _markdown(
            """
            ## 11. Generate or resume the 1,557 semantic contracts

            This is the only long-running model-generation cell. Completed shards are sealed on
            Google Drive after each shard, so rerunning this cell after a disconnect resumes from
            verified durable progress rather than starting over.
            """
        ),
        _code(
            """
            run_teacher(
                "scripts/teacher_distillation/generate_teacher_data.py",
                "--config",
                "configs/distillation/python/qwen38_27b_semantic_contract_v1.yaml",
                "--input",
                os.environ["CONTRACT_INPUT"],
                "--checkpoint-dir",
                os.environ["CONTRACT_CHECKPOINT_DIR"],
                "--work-dir",
                "/content/tqc-p9-009b-generation",
            )
            """
        ),
        _markdown("## 12. Verify the complete checkpoint without loading the teacher"),
        _code(
            """
            status = run_teacher(
                "scripts/teacher_distillation/generate_teacher_data.py",
                "--config",
                "configs/distillation/python/qwen38_27b_semantic_contract_v1.yaml",
                "--input",
                os.environ["CONTRACT_INPUT"],
                "--checkpoint-dir",
                os.environ["CONTRACT_CHECKPOINT_DIR"],
                "--work-dir",
                "/content/tqc-p9-009b-generation",
                "--status-only",
            )
            if "verified_records=1557/1557" not in status.stdout:
                raise RuntimeError("P9-009B contract generation is not complete")
            print("P9-009B generation checkpoint is complete and verified.")
            """
        ),
        _markdown(
            """
            ## 13. Create a deterministic transfer archive for P9-009C

            This archive contains the sealed verifier input, config, and complete generation
            checkpoint. It contains generated contract code but **does not execute it**. P9-009C
            must execute contracts later on the containerized self-hosted runner.
            """
        ),
        _code(
            """
            import zipfile

            export_path = Path(os.environ["CONTRACT_EXPORT"])
            export_tmp = export_path.with_suffix(export_path.suffix + ".tmp")
            if export_tmp.exists():
                export_tmp.unlink()

            repo_root = Path(os.environ["TQC_REPO"])
            members = [
                (Path(os.environ["CONTRACT_INPUT"]), Path("input/contract-input.jsonl")),
                (
                    Path(os.environ["CONTRACT_INPUT"] + ".sha256"),
                    Path("input/contract-input.jsonl.sha256"),
                ),
                (
                    repo_root / "configs/distillation/python/qwen38_27b_semantic_contract_v1.yaml",
                    Path("config/qwen38_27b_semantic_contract_v1.yaml"),
                ),
            ]
            checkpoint_dir = Path(os.environ["CONTRACT_CHECKPOINT_DIR"])
            members.extend(
                (path, Path("generation-checkpoint") / path.relative_to(checkpoint_dir))
                for path in sorted(checkpoint_dir.rglob("*"))
                if path.is_file()
            )

            with zipfile.ZipFile(
                export_tmp,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            ) as archive_file:
                for source, relative in sorted(members, key=lambda item: item[1].as_posix()):
                    info = zipfile.ZipInfo(relative.as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = 0o100644 << 16
                    archive_file.writestr(
                        info,
                        source.read_bytes(),
                        compress_type=zipfile.ZIP_DEFLATED,
                    )

            export_tmp.replace(export_path)
            export_sha256 = sha256_file(export_path)
            export_path.with_suffix(export_path.suffix + ".sha256").write_text(
                f"{export_sha256}  {export_path.name}\\n",
                encoding="ascii",
            )
            print("P9-009B export:", export_path)
            print("P9-009B export SHA-256:", export_sha256)
            """
        ),
        _markdown(
            """
            ## 14. Stop boundary

            P9-009B ends here. Do **not** run the generated reference solutions or assertions in
            Colab, do not train a student, and do not inspect benchmark outcomes. Preserve the
            export ZIP and SHA-256. P9-009C will transfer the sealed evidence to the self-hosted
            runner and execute reference-first verification there.
            """
        ),
        _markdown(
            """
            ## Recovery

            `/content` is disposable; Google Drive is durable. After a Colab disconnect, create a
            fresh A100 runtime, reuse the exact sealed repository ZIP and the same semantic namespace,
            then rerun from the top. The input reconstruction is deterministic and
            `generate_teacher_data.py` resumes only verified missing shards. Never edit a completed
            shard or its checksum sidecar.
            """
        ),
    ]

    for index, cell in enumerate(cells):
        cell["id"] = _cell_id(index, cell)

    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "name": "qwen38_p9_009_semantic_contract_generation_colab.ipynb",
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
    output.write_text(json.dumps(_notebook(), indent=1) + "\n", encoding="utf-8")

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
