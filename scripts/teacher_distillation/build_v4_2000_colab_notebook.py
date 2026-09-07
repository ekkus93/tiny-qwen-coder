#!/usr/bin/env python3
"""Build the fresh 2,000-record v4 successor Colab notebook."""

from __future__ import annotations

import copy
import json
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE / "qwen38_teacher_distillation_v4_colab.ipynb"
OUTPUT = HERE / "qwen38_teacher_distillation_v4_2000_colab.ipynb"


def lines(text: str) -> list[str]:
    return [line + "\n" for line in textwrap.dedent(text).strip().splitlines()]


def md(text: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text)}


def code(text: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(text),
    }


def text(cell: dict[str, object]) -> str:
    source = cell.get("source")
    if not isinstance(source, list):
        raise RuntimeError("notebook cell source is not a list")
    return "".join(str(part) for part in source)


def pair(base_cells: list[dict[str, object]], heading: str) -> list[dict[str, object]]:
    for index, cell in enumerate(base_cells):
        if cell.get("cell_type") == "markdown" and text(cell).startswith(heading):
            if index + 1 >= len(base_cells) or base_cells[index + 1].get("cell_type") != "code":
                raise RuntimeError(f"missing code cell after {heading!r}")
            return [copy.deepcopy(cell), copy.deepcopy(base_cells[index + 1])]
    raise RuntimeError(f"could not find notebook heading {heading!r}")


def main() -> None:
    base = json.loads(BASE.read_text(encoding="utf-8"))
    base_cells = base["cells"]
    if not isinstance(base_cells, list):
        raise RuntimeError("base notebook cells must be a list")

    cells: list[dict[str, object]] = [
        md(
            """
            # Qwen3.8-27B v4-2000 teacher-data successor on Google Colab

            This is the fresh 2,000-record successor authorized by the qualified bounded v4-200
            selective-compression study. It never modifies the v4-200 evidence.

            It generates 2,000 fresh **high-reasoning** Qwen3.8 answers under the frozen v3
            per-record answer-budget policy, then applies the qualified v4 **low-reasoning**
            compression pass only to normal-stop records that exceed both the Qwen3.5-4B
            2,048-token student envelope and their frozen answer budget.
            """
        )
    ]
    cells.extend(pair(base_cells, "## 1. Mount Google Drive"))
    cells.extend(
        [
            md("## 2. Define frozen qualification evidence and fresh v4-2000 paths"),
            code(
                """
                import os
                from pathlib import Path

                os.environ["TQC_DRIVE"] = "/content/drive/MyDrive/tiny-qwen-coder"
                os.environ["CODE_DIR"] = f"{os.environ['TQC_DRIVE']}/code"
                os.environ["CODE_ARCHIVE"] = f"{os.environ['CODE_DIR']}/tiny-qwen-coder-v4-2000.zip"
                os.environ["BASE_INPUT"] = f"{os.environ['TQC_DRIVE']}/distillation/qwen38-27b-v1/input/accepted.jsonl"
                os.environ["V4_200_QUALIFICATION"] = f"{os.environ['TQC_DRIVE']}/distillation/qwen38-27b-v4-compression-200/qualification.json"
                os.environ["V4_2000_DIR"] = f"{os.environ['TQC_DRIVE']}/distillation/qwen38-27b-v4-2000"
                os.environ["V4_2000_INPUT_DIR"] = f"{os.environ['V4_2000_DIR']}/input"
                os.environ["V4_2000_SOURCE_INPUT"] = f"{os.environ['V4_2000_INPUT_DIR']}/p0-2000.jsonl"
                os.environ["V4_2000_GENERATION_INPUT"] = f"{os.environ['V4_2000_INPUT_DIR']}/p0-2000-v3.jsonl"
                os.environ["V4_2000_GENERATION_CHECKPOINT_DIR"] = f"{os.environ['V4_2000_DIR']}/generation-checkpoint"
                os.environ["V4_2000_GENERATION_DIAG_DIR"] = f"{os.environ['V4_2000_DIR']}/generation-diagnostics"
                os.environ["V4_2000_COMPRESSION_CHECKPOINT_DIR"] = f"{os.environ['V4_2000_DIR']}/compression-checkpoint"
                os.environ["V4_2000_MERGED_DIR"] = f"{os.environ['V4_2000_DIR']}/merged"
                os.environ["V4_2000_MERGED"] = f"{os.environ['V4_2000_MERGED_DIR']}/p0-2000-v4.jsonl"
                os.environ["V4_2000_DIAG_DIR"] = f"{os.environ['V4_2000_DIR']}/diagnostics"
                os.environ["V4_2000_FINAL_DIR"] = f"{os.environ['V4_2000_DIR']}/final"
                os.environ["V4_2000_QUALIFICATION"] = f"{os.environ['V4_2000_DIR']}/qualification.json"

                for key in (
                    "CODE_DIR", "V4_2000_DIR", "V4_2000_INPUT_DIR",
                    "V4_2000_GENERATION_CHECKPOINT_DIR", "V4_2000_GENERATION_DIAG_DIR",
                    "V4_2000_COMPRESSION_CHECKPOINT_DIR", "V4_2000_MERGED_DIR",
                    "V4_2000_DIAG_DIR", "V4_2000_FINAL_DIR",
                ):
                    Path(os.environ[key]).mkdir(parents=True, exist_ok=True)

                print("Repository ZIP expected at:", os.environ["CODE_ARCHIVE"])
                print("Frozen v4-200 qualification:", os.environ["V4_200_QUALIFICATION"])
                print("Fresh v4-2000 namespace:", os.environ["V4_2000_DIR"])
                """
            ),
            md(
                """
                ## 3. Put the frozen v4-2000 repository ZIP on Drive

                Upload the exact repository ZIP for this notebook as
                `MyDrive/tiny-qwen-coder/code/tiny-qwen-coder-v4-2000.zip`.
                """
            ),
            code(
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
        ]
    )

    for heading in (
        "## 4. Verify/extract the frozen ZIP and create local provenance",
        "## 5. Create the isolated uv/vLLM CUDA 13.0 environment",
        "## 6. Verify CUDA, vLLM, Ninja, and the A100",
    ):
        section = pair(base_cells, heading)
        if heading.startswith("## 4."):
            source = text(section[1]).replace(
                'Path("/content/tiny-qwen-coder-v4-code")',
                'Path("/content/tiny-qwen-coder-v4-2000-code")',
            ).replace(
                '"v4 repository ZIP checksum changed; use a new experiment namespace."',
                '"v4-2000 repository ZIP checksum changed; use a new experiment namespace."',
            )
            section[1]["source"] = lines(source)
        cells.extend(section)

    cells.extend(
        [
            md("## 7. Define a streaming subprocess helper and enforce the v4-200 gate"),
            code(
                """
                import json
                import shlex
                import subprocess
                from pathlib import Path

                def run_teacher(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
                    command = [os.environ["TQC_PYTHON"], "-u", *args]
                    print("$", shlex.join(command), flush=True)
                    with subprocess.Popen(
                        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1,
                    ) as process:
                        assert process.stdout is not None
                        captured: list[str] = []
                        for line in process.stdout:
                            print(line, end="", flush=True)
                            captured.append(line)
                        returncode = process.wait()
                    result = subprocess.CompletedProcess(command, returncode, "".join(captured), None)
                    if check:
                        result.check_returncode()
                    return result

                qualification_path = Path(os.environ["V4_200_QUALIFICATION"])
                base_input_path = Path(os.environ["BASE_INPUT"])
                if not qualification_path.is_file():
                    raise FileNotFoundError(qualification_path)
                if not base_input_path.is_file():
                    raise FileNotFoundError(base_input_path)
                bounded = json.loads(qualification_path.read_text(encoding="utf-8"))
                required = {
                    "qualified": True,
                    "contamination_status": "clean",
                    "minimum_stop_rate": 0.9,
                    "minimum_student_length_accept_rate_given_stop": 0.85,
                }
                for key, expected in required.items():
                    if bounded.get(key) != expected:
                        raise RuntimeError(
                            f"v4-200 does not authorize scaling: {key}={bounded.get(key)!r}; "
                            f"expected {expected!r}"
                        )
                print("Frozen v4-200 qualification authorizes the successor.")
                print(json.dumps(bounded, indent=2, sort_keys=True))
                """
            ),
            md("## 8. Build and seal the deterministic 2,000-record v3-policy input"),
            code(
                """
                run_teacher(
                    "scripts/teacher_distillation/select_teacher_input.py",
                    "--input", os.environ["BASE_INPUT"],
                    "--output", os.environ["V4_2000_SOURCE_INPUT"],
                    "--count", "2000",
                )
                run_teacher(
                    "scripts/teacher_distillation/prepare_teacher_v3_input.py",
                    "--input", os.environ["V4_2000_SOURCE_INPUT"],
                    "--output", os.environ["V4_2000_GENERATION_INPUT"],
                )
                """
            ),
            md("## 9. Preflight fresh 2,000-record generation without loading Qwen3.8"),
            code(
                """
                run_teacher(
                    "scripts/teacher_distillation/generate_teacher_data.py",
                    "--config", "configs/distillation/python/qwen38_27b_v3.yaml",
                    "--input", os.environ["V4_2000_GENERATION_INPUT"],
                    "--checkpoint-dir", os.environ["V4_2000_GENERATION_CHECKPOINT_DIR"],
                    "--work-dir", "/content/tqc-distillation-v4-2000-generation",
                    "--status-only",
                )
                """
            ),
            md("## 10. Generate or resume the 2,000 high-reasoning teacher answers"),
            code(
                """
                run_teacher(
                    "scripts/teacher_distillation/generate_teacher_data.py",
                    "--config", "configs/distillation/python/qwen38_27b_v3.yaml",
                    "--input", os.environ["V4_2000_GENERATION_INPUT"],
                    "--checkpoint-dir", os.environ["V4_2000_GENERATION_CHECKPOINT_DIR"],
                    "--work-dir", "/content/tqc-distillation-v4-2000-generation",
                )
                """
            ),
            md("## 11. Verify the durable generation checkpoint without loading the teacher"),
            code(
                """
                run_teacher(
                    "scripts/teacher_distillation/generate_teacher_data.py",
                    "--config", "configs/distillation/python/qwen38_27b_v3.yaml",
                    "--input", os.environ["V4_2000_GENERATION_INPUT"],
                    "--checkpoint-dir", os.environ["V4_2000_GENERATION_CHECKPOINT_DIR"],
                    "--work-dir", "/content/tqc-distillation-v4-2000-generation",
                    "--status-only",
                )
                """
            ),
            md("## 12. Diagnose the first-pass 2,000-record generation"),
            code(
                """
                run_teacher(
                    "scripts/teacher_distillation/diagnose_teacher_data.py",
                    "--distillation-config", "configs/distillation/python/qwen38_27b_v3.yaml",
                    "--input", os.environ["V4_2000_GENERATION_INPUT"],
                    "--checkpoint-dir", os.environ["V4_2000_GENERATION_CHECKPOINT_DIR"],
                    "--output-dir", os.environ["V4_2000_GENERATION_DIAG_DIR"],
                )
                """
            ),
            md("## 13. Preflight selective v4 compression without loading Qwen3.8"),
            code(
                """
                run_teacher(
                    "scripts/teacher_distillation/compress_teacher_v4.py",
                    "--source-input", os.environ["V4_2000_GENERATION_INPUT"],
                    "--source-checkpoint-dir", os.environ["V4_2000_GENERATION_CHECKPOINT_DIR"],
                    "--checkpoint-dir", os.environ["V4_2000_COMPRESSION_CHECKPOINT_DIR"],
                    "--work-dir", "/content/tqc-distillation-v4-2000-compression",
                    "--output", os.environ["V4_2000_MERGED"],
                    "--status-only",
                )
                """
            ),
            md("## 14. Run or resume selective low-reasoning compression"),
            code(
                """
                run_teacher(
                    "scripts/teacher_distillation/compress_teacher_v4.py",
                    "--source-input", os.environ["V4_2000_GENERATION_INPUT"],
                    "--source-checkpoint-dir", os.environ["V4_2000_GENERATION_CHECKPOINT_DIR"],
                    "--checkpoint-dir", os.environ["V4_2000_COMPRESSION_CHECKPOINT_DIR"],
                    "--work-dir", "/content/tqc-distillation-v4-2000-compression",
                    "--output", os.environ["V4_2000_MERGED"],
                )
                """
            ),
            md("## 15. Verify the durable compression checkpoint without loading the teacher"),
            code(
                """
                run_teacher(
                    "scripts/teacher_distillation/compress_teacher_v4.py",
                    "--source-input", os.environ["V4_2000_GENERATION_INPUT"],
                    "--source-checkpoint-dir", os.environ["V4_2000_GENERATION_CHECKPOINT_DIR"],
                    "--checkpoint-dir", os.environ["V4_2000_COMPRESSION_CHECKPOINT_DIR"],
                    "--work-dir", "/content/tqc-distillation-v4-2000-compression",
                    "--output", os.environ["V4_2000_MERGED"],
                    "--status-only",
                )
                """
            ),
            md("## 16. Diagnose the merged 2,000-record v4 corpus"),
            code(
                """
                run_teacher(
                    "scripts/teacher_distillation/diagnose_prepared_teacher_data.py",
                    "--input", os.environ["V4_2000_MERGED"],
                    "--output-dir", os.environ["V4_2000_DIAG_DIR"],
                )
                """
            ),
            md("## 17. Finalize with Python quality, deduplication, split, and contamination checks"),
            code(
                """
                run_teacher(
                    "scripts/teacher_distillation/finalize_prepared_teacher_data.py",
                    "--input", os.environ["V4_2000_MERGED"],
                    "--output-dir", os.environ["V4_2000_FINAL_DIR"],
                )
                """
            ),
            md("## 18. Mechanically assess the final 2,000-record successor"),
            code(
                """
                qualification = run_teacher(
                    "scripts/teacher_distillation/qualify_teacher_study.py",
                    "--diagnostics-summary",
                    str(Path(os.environ["V4_2000_DIAG_DIR"]) / "teacher-length-summary.json"),
                    "--dataset-manifest",
                    str(Path(os.environ["V4_2000_FINAL_DIR"]) / "dataset-manifest.json"),
                    "--minimum-student-length-accept-rate-given-stop", "0.85",
                    "--output", os.environ["V4_2000_QUALIFICATION"],
                    check=False,
                )
                print("qualification exit code:", qualification.returncode)
                """
            ),
            md("## 19. Inspect the final corpus decision and counts"),
            code(
                """
                qualification_path = Path(os.environ["V4_2000_QUALIFICATION"])
                manifest_path = Path(os.environ["V4_2000_FINAL_DIR"]) / "dataset-manifest.json"
                finalization_path = Path(os.environ["V4_2000_FINAL_DIR"]) / "teacher-finalization.json"
                for path in (qualification_path, manifest_path, finalization_path):
                    if not path.is_file():
                        raise FileNotFoundError(path)
                decision = json.loads(qualification_path.read_text(encoding="utf-8"))
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                finalization = json.loads(finalization_path.read_text(encoding="utf-8"))
                print("QUALIFICATION")
                print(json.dumps(decision, indent=2, sort_keys=True))
                print("\\nFINALIZATION")
                print(json.dumps(finalization, indent=2, sort_keys=True))
                print("\\nMANIFEST COUNTS")
                print(json.dumps(manifest["counts"], indent=2, sort_keys=True))
                print("\\nCONTAMINATION")
                print(json.dumps(manifest["contamination"], indent=2, sort_keys=True))
                if decision.get("qualified") is not True:
                    print("DO NOT TRAIN. Preserve the v4-2000 evidence and investigate.")
                else:
                    print("V4-2000 READY. Preserve this evidence before training.")
                """
            ),
            md("## 20. Training-readiness guard"),
            code(
                """
                if decision.get("qualified") is not True:
                    raise RuntimeError("v4-2000 did not qualify; student training is blocked")
                print("v4-2000 qualified; freeze the final artifacts before training.")
                """
            ),
        ]
    )

    notebook = {
        "cells": cells,
        "metadata": copy.deepcopy(base["metadata"]),
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook["metadata"]["colab"]["name"] = OUTPUT.name
    OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
