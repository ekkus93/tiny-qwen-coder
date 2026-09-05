# Tiny Qwen Coder

Tiny Qwen Coder is a research project for turning a single shared `Qwen/Qwen3.5-4B` base model into a family of programming-language specialists by attaching interchangeable LoRA adapters.

The base model is intended to remain fixed. Language expertise is packaged separately:

```text
Qwen3.5-4B
    |
    +-- Python LoRA
    +-- TypeScript LoRA
    `-- Rust LoRA
```

The long-term runtime goal is to load the base model once and switch only the active LoRA adapter when the working language changes. Python is the first implementation target. TypeScript and Rust are planned follow-on adapters built on the same language-neutral data, training, evaluation, and runtime infrastructure.

## Project status

The repository is being implemented incrementally from [`docs/TODO.md`](docs/TODO.md). The architecture, experimental methodology, reproducibility requirements, dataset policy, evaluation strategy, and adapter compatibility rules are defined in [`docs/SPEC.md`](docs/SPEC.md).

Current scope:

- one pinned `Qwen/Qwen3.5-4B` base model;
- 4-bit NF4 QLoRA with double quantization and BF16 compute is the measured canonical training mode on the 16 GB reference GPU;
- text/code-only specialization for the multimodal 4B checkpoint, with vision components kept frozen;
- Python as the first language specialization;
- TypeScript and Rust adapters after the Python pipeline proves the architecture;
- explicit regression testing so specialization does not silently destroy general instruction-following or tool-use behavior;
- later OpenCode/tool-agent integration after language-adapter training and switching are reliable.

## Requirements

For normal development and CPU-safe tests:

- Linux, macOS, or another environment supported by the pinned Python dependencies;
- Python 3.11 or newer;
- [`uv`](https://docs.astral.sh/uv/) for dependency and environment management;
- Git.

A GPU is **not** required for repository development, linting, type checking, unit tests, dataset-pipeline tests, or other CPU-safe validation.

GPU-backed work such as loading the canonical model in BF16, measuring VRAM, LoRA training, and model evaluation will require a compatible NVIDIA GPU and CUDA/PyTorch environment. GPU work is intentionally separated from the required CPU-safe development/CI path.

## Setup

Clone the repository and enter it:

```bash
git clone https://github.com/ekkus93/tiny-qwen-coder.git
cd tiny-qwen-coder
```

Create the project environment exactly from the committed lockfile:

```bash
uv sync --frozen
```

For canonical QLoRA training on a compatible NVIDIA/CUDA host, install the locked optional runtime as well:

```bash
uv sync --frozen --extra qlora
```

Verify the package is importable:

```bash
uv run --frozen python -c "import tiny_qwen_coder"
```

## Google Colab: generate Qwen3.8-27B teacher data

The current distillation experiment keeps `Qwen/Qwen3.5-4B` as the student but
uses `Qwen/Qwen3.8-27B` as a stronger teacher. Teacher inference is intended to
run on a Google Colab A100, while QLoRA fine-tuning remains on the local RTX
4070 Ti SUPER.

The workflow assumes a Colab runtime can disappear at any time. `/content` is
scratch space; Google Drive is the durable boundary. Completed 16-record shards
are SHA-256 sealed on Drive and verified before they are accepted, so rerunning
the same command resumes from the last completed shard.

The cells below are written in **Colab notebook syntax** and can be copied into
separate Python cells. For the full checkpoint/failure-recovery contract, see
[`docs/TEACHER_DISTILLATION_COLAB.md`](docs/TEACHER_DISTILLATION_COLAB.md).

### 1. Select an A100 runtime and mount Google Drive

In Colab, select an A100 GPU runtime, then run:

```python
from google.colab import drive

drive.mount("/content/drive")
```

Define durable paths in Python so they remain available to later `!` shell
commands in the notebook:

```python
import os
from pathlib import Path

os.environ["TQC_DRIVE"] = "/content/drive/MyDrive/tiny-qwen-coder"
os.environ["RUN_ROOT"] = f"{os.environ['TQC_DRIVE']}/distillation/qwen38-27b-v1"
os.environ["INPUT_DIR"] = f"{os.environ['RUN_ROOT']}/input"
os.environ["SUBSET_DIR"] = f"{os.environ['RUN_ROOT']}/subsets"
os.environ["SMOKE_DIR"] = f"{os.environ['TQC_DRIVE']}/distillation/qwen38-27b-v1-smoke"
os.environ["PILOT_DIR"] = f"{os.environ['TQC_DRIVE']}/distillation/qwen38-27b-v1-2000"

for key in ("RUN_ROOT", "INPUT_DIR", "SUBSET_DIR", "SMOKE_DIR", "PILOT_DIR"):
    Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
```

### 2. Clone the repository and freeze the code revision

A resumable experiment must use the **same repository commit after every Colab
preemption**. The following cell records the first-run Git revision on Drive and
checks out that exact revision on later allocations:

```python
import os
import subprocess
from pathlib import Path

repo = Path("/content/tiny-qwen-coder")
if not repo.exists():
    subprocess.run(
        ["git", "clone", "https://github.com/ekkus93/tiny-qwen-coder.git", str(repo)],
        check=True,
    )

subprocess.run(["git", "-C", str(repo), "fetch", "origin", "master"], check=True)
revision_file = Path(os.environ["RUN_ROOT"]) / "repo-revision.txt"

if revision_file.exists():
    revision = revision_file.read_text(encoding="utf-8").strip()
else:
    revision = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "origin/master"],
        text=True,
    ).strip()
    revision_file.write_text(revision + "\n", encoding="utf-8")

subprocess.run(["git", "-C", str(repo), "checkout", "--detach", revision], check=True)
os.environ["TQC_CODE_REVISION"] = revision
print("frozen repository revision:", revision)
```

Move into the checkout and install the project plus the pinned A100 teacher
runtime:

```python
%cd /content/tiny-qwen-coder
!python -m pip install -e .
!python -m pip install -r requirements/colab-teacher.txt
```

After a completely fresh Colab allocation, rerun steps 1 and 2. The revision
file on Drive prevents a later `master` commit from silently changing a running
experiment.

### 3. Verify the A100 before downloading Qwen3.8

```python
!nvidia-smi

import torch

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
```

Stop here if the runtime did not receive the expected CUDA GPU.

### 4. Build and seal the immutable teacher input once

The original P0 assistant answers are not sent to the teacher. The script builds
only the canonical system/user prompt prefix and writes a checksum alongside it.

```python
!python scripts/prepare_teacher_input.py \
  --output "$INPUT_DIR/accepted.jsonl"
```

This creates:

```text
accepted.jsonl
accepted.jsonl.sha256
```

For an existing experiment, do not rebuild or edit that file in place. Reuse the
same sealed input.

### 5. Create deterministic representative pilot subsets

Do not use the first N source rows as a scientific pilot. Create source-stratified
subsets instead:

```python
!python scripts/select_teacher_input.py \
  --input "$INPUT_DIR/accepted.jsonl" \
  --output "$SUBSET_DIR/p0-16.jsonl" \
  --count 16

!python scripts/select_teacher_input.py \
  --input "$INPUT_DIR/accepted.jsonl" \
  --output "$SUBSET_DIR/p0-500.jsonl" \
  --count 500

!python scripts/select_teacher_input.py \
  --input "$INPUT_DIR/accepted.jsonl" \
  --output "$SUBSET_DIR/p0-2000.jsonl" \
  --count 2000
```

Recommended progression:

1. **16 records** — prove model loading, quantization, checkpointing, and resume.
2. **500 records** — inspect teacher output and finalization rejection rates.
3. **2,000 records** — perform the first real Qwen3.5-4B learning experiment.
4. Generate more only if the 2,000-record adapter improves the frozen base
   benchmark.

### 6. Run the 16-record A100 smoke test

```python
!python scripts/generate_teacher_data.py \
  --input "$SUBSET_DIR/p0-16.jsonl" \
  --checkpoint-dir "$SMOKE_DIR/checkpoint" \
  --work-dir /content/tqc-distillation-smoke
```

A successful smoke completes all 16 records and leaves one sealed shard on
Google Drive.

Check its status without loading the 27B teacher again:

```python
!python scripts/generate_teacher_data.py \
  --input "$SUBSET_DIR/p0-16.jsonl" \
  --checkpoint-dir "$SMOKE_DIR/checkpoint" \
  --work-dir /content/tqc-distillation-smoke \
  --status-only
```

### 7. Generate the 2,000-record pilot

Only after the smoke succeeds:

```python
!python scripts/generate_teacher_data.py \
  --input "$SUBSET_DIR/p0-2000.jsonl" \
  --checkpoint-dir "$PILOT_DIR/checkpoint" \
  --work-dir /content/tqc-distillation-2000
```

If Colab disconnects, crashes, or revokes the instance:

1. acquire another runtime;
2. remount Drive;
3. rerun steps 1-3;
4. execute the **same generation command** again.

Do not supply a row offset and do not edit checkpoint state. The preflight
verifies the frozen run identity and every completed shard before Qwen3.8 is
loaded, then skips sealed work automatically.

To inspect progress without loading Qwen3.8:

```python
!python scripts/generate_teacher_data.py \
  --input "$SUBSET_DIR/p0-2000.jsonl" \
  --checkpoint-dir "$PILOT_DIR/checkpoint" \
  --work-dir /content/tqc-distillation-2000 \
  --status-only
```

### 8. Finalize the pilot for the RTX 4070 Ti machine

After generation reaches 2,000/2,000:

```python
!python scripts/finalize_teacher_data.py \
  --input "$SUBSET_DIR/p0-2000.jsonl" \
  --checkpoint-dir "$PILOT_DIR/checkpoint" \
  --output-dir "$PILOT_DIR/final"
```

`$PILOT_DIR/final` contains the accepted corpus, deterministic train/validation
splits, checksums, dataset manifest, and finalization report. Sync or download
that directory to the RTX 4070 Ti machine. Student fine-tuning remains on the
normal Qwen3.5-4B 4-bit QLoRA path and does not require vLLM.

### Colab recovery rules

- Treat `/content` as disposable and Google Drive as durable.
- Keep each 16/500/2,000/full experiment in its own checkpoint directory.
- Reuse the exact sealed input and frozen repository revision on resume.
- Never edit `run-identity.json`, generated shards, or checksum sidecars to force
  progress.
- An uncommitted shard left by a killed runtime is regenerated automatically.
- A sealed shard whose checksum is wrong fails closed as corruption.
- Do not start the full corpus until the bounded 2,000-record experiment shows
  that the distilled data actually improves the student.

Print the resolved Python and project dependency versions:

```bash
uv run --frozen tiny-qwen-coder-versions
```

Print a standalone machine-readable runtime/GPU environment report without loading a model or starting training:

```bash
uv run --frozen tiny-qwen-coder-env
```

The project declares Python `>=3.11`; `.python-version` selects Python 3.11 when the requested interpreter is available.

## Quality gates

Run these commands from the repository root before publishing a change.

### Formatting

Check formatting without modifying files:

```bash
uv run --frozen ruff format --check .
```

To apply Ruff formatting locally:

```bash
uv run --frozen ruff format .
```

### Linting

```bash
uv run --frozen ruff check .
```

To apply Ruff's safe automatic fixes where available:

```bash
uv run --frozen ruff check --fix .
```

### Type checking

```bash
uv run --frozen mypy src scripts tests
```

The repository currently runs mypy in strict mode.

### Tests

```bash
uv run --frozen pytest
```

### Git whitespace check

```bash
git diff --check
```

A normal pre-publish validation sequence is therefore:

```bash
uv sync --frozen
uv run --frozen ruff format --check .
uv run --frozen ruff check .
uv run --frozen mypy src scripts tests
uv run --frozen pytest
git diff --check
```

## CPU development versus GPU work

The project deliberately distinguishes inexpensive repository validation from model workloads.

### CPU-safe development

The following should remain runnable without downloading the full model or requiring CUDA:

- formatting and linting;
- static type checking;
- unit tests;
- configuration parsing and validation;
- dataset-pipeline tests using bounded fixtures;
- adapter-manifest and compatibility tests using synthetic metadata;
- other small deterministic infrastructure tests.

Required GitHub Actions CI is intended to remain CPU-safe as well.

### GPU-backed work

GPU execution is reserved for work that genuinely needs the model, including:

- canonical `Qwen3.5-4B` model-load smoke tests;
- canonical NF4 QLoRA training-memory preflights and training runs;
- LoRA smoke training and canonical training runs;
- base-versus-adapter generation benchmarks;
- adapter switching/runtime performance measurements;
- later agent/OpenCode experiments.

Model checkpoints, generated adapters, datasets, benchmark sandboxes, logs, and experiment artifacts are intentionally excluded from normal Git tracking.

## Repository layout

```text
configs/
    base/       shared base-model identity/config
    languages/  programming-language definitions
    data/       dataset preparation configs
    train/      training configs
    eval/       evaluation configs
    runtime/    adapter/runtime configs

src/tiny_qwen_coder/
    model/      shared model loading/inspection
    adapters/   LoRA metadata and compatibility
    languages/  language registry/plugins
    data/       language-neutral data pipeline
    training/   generic adapter training
    evaluation/ common evaluation framework
    runtime/    inference/adapter switching/serving
    reporting/  manifests and reports

scripts/
    inspect_model.py
    prepare_data.py
    train_adapter.py
    evaluate.py
    infer.py
    serve.py

tests/
    unit/
    integration/
    fixtures/
```

The generic scripts are intentionally language-neutral. The project should add a new language through configuration/plugins rather than by creating separate entry points such as `train_python.py`, `train_typescript.py`, and `train_rust.py`.

## Adapter model

All canonical language adapters must target the same pinned base-model revision, tokenizer/chat-template identity, and compatible LoRA architecture. An adapter manifest will record those identities so an incompatible adapter can be rejected instead of silently attached to the wrong base model.

The initial target family is:

| Adapter | Status | Purpose |
| --- | --- | --- |
| Python | First target | Python generation, reasoning, repair, and later tool-driven development |
| TypeScript | Planned | TypeScript/JavaScript ecosystem specialization |
| Rust | Planned | Rust generation, compiler-guided repair, and Cargo-oriented development |

Polyglot repositories and automatic/dynamic adapter selection are later milestones. The initial system will prove that one base model can be reused while adapters are selected explicitly and independently.

## Documentation

- [`docs/SPEC.md`](docs/SPEC.md) — project specification and design contract.
- [`docs/TODO.md`](docs/TODO.md) — phased implementation plan and acceptance criteria.
- [`docs/TEACHER_DISTILLATION_COLAB.md`](docs/TEACHER_DISTILLATION_COLAB.md) — detailed
  Qwen3.8-27B A100 distillation, checkpoint, resume, and finalization runbook.

The specification is authoritative for architecture and experimental rules; the TODO is the execution checklist used to implement it.

## License

See [`LICENSE`](LICENSE).
