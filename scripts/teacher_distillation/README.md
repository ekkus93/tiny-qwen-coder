# Qwen3.8-27B teacher distillation

**Start here for generating the new Python training data on Google Colab.**

This directory intentionally keeps the runnable entry points and their operating
instructions together. The teacher is `Qwen/Qwen3.8-27B` on a Google Colab A100;
the student remains `Qwen/Qwen3.5-4B`, fine-tuned later on the RTX 4070 Ti SUPER.

## Directory contents

| File | Purpose |
| --- | --- |
| `prepare_teacher_input.py` | Build and SHA-256 seal the canonical prompt-only teacher input. |
| `select_teacher_input.py` | Create deterministic source-stratified 16/500/2,000-record pilot inputs. |
| `generate_teacher_data.py` | Run resumable Qwen3.8 generation with durable Google Drive checkpoints. |
| `finalize_teacher_data.py` | Validate/filter completed shards and emit the Qwen3.5-4B training corpus and manifest. |
| `README.md` | This Google Colab runbook and recovery contract. |

All commands below are run from the **repository root** after cloning
`tiny-qwen-coder`. The scripts live here so there is one obvious place to find
both the workflow and its documentation.

## Frozen v1 contract

- Teacher: `Qwen/Qwen3.8-27B`
- Teacher revision: `72a217afab8029b39e4af1c7273a829995a3dbaf`
- Runtime: vLLM 0.28.0, text-only mode
- BitsAndBytes plugin: vllm-bnb-plugin 0.0.3
- BitsAndBytes: 0.50.2
- Weight loading: in-flight 4-bit BitsAndBytes quantization
- Thinking: enabled
- Reasoning effort: `xhigh` (explicitly frozen rather than relying on a default)
- Preserved historical thinking: disabled
- Sampling: Qwen's recommended thinking-mode values
- Maximum model context: 16,384 tokens
- Maximum generated completion: 8,192 tokens
- Generation seed: 1729 plus the canonical input-record index
- Durable shard size: 16 records
- Student: unchanged `Qwen/Qwen3.5-4B`
- Student preparation boundary: 2,048 tokens with reject-on-overlength

The original P0 assistant response is never sent to the teacher. Only the
system/user prompt prefix is sent. The teacher's `<think>...</think>` content is
not written to the training corpus or checkpoint JSONL. Checkpoints retain only
its SHA-256 digest and character count for bounded audit evidence.

## Why checkpoints are safe to resume

Every durable shard is bound to all of the following:

1. the semantic distillation-config SHA-256, including pinned inference-package versions;
2. the SHA-256 of the generator/config/backend source implementation;
3. the byte-level SHA-256 of the complete input JSONL;
4. the exact source-record index and normalized record fingerprint;
5. the exact prompt fingerprint;
6. the pinned teacher repository and revision; and
7. a SHA-256 sidecar for the completed shard itself.

A shard is generated into local Colab scratch first. Only after it is complete
is it copied to Drive, with its checksum sidecar. The Drive copy is then read
back and validated before progress advances. A killed runtime can therefore
lose at most the currently unsealed 16-record shard. Re-running the same command
verifies and skips every already sealed shard.

`run-identity.json` deliberately binds the requested record count. Never reuse a
smoke-run checkpoint directory for a larger run.

## Google Colab runbook

The current distillation experiment keeps `Qwen/Qwen3.5-4B` as the student but
uses `Qwen/Qwen3.8-27B` as a stronger teacher. Teacher inference is intended to
run on a Google Colab A100, while QLoRA fine-tuning remains on the local RTX
4070 Ti SUPER.

The workflow assumes a Colab runtime can disappear at any time. `/content` is
scratch space; Google Drive is the durable boundary. Completed 16-record shards
are SHA-256 sealed on Drive and verified before they are accepted, so rerunning
the same command resumes from the last completed shard.

The cells below are written in **Colab notebook syntax** and can be copied into
separate Python cells. This README is the canonical checkpoint and recovery contract.

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
!python scripts/teacher_distillation/prepare_teacher_input.py \
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
!python scripts/teacher_distillation/select_teacher_input.py \
  --input "$INPUT_DIR/accepted.jsonl" \
  --output "$SUBSET_DIR/p0-16.jsonl" \
  --count 16

!python scripts/teacher_distillation/select_teacher_input.py \
  --input "$INPUT_DIR/accepted.jsonl" \
  --output "$SUBSET_DIR/p0-500.jsonl" \
  --count 500

!python scripts/teacher_distillation/select_teacher_input.py \
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
!python scripts/teacher_distillation/generate_teacher_data.py \
  --input "$SUBSET_DIR/p0-16.jsonl" \
  --checkpoint-dir "$SMOKE_DIR/checkpoint" \
  --work-dir /content/tqc-distillation-smoke
```

A successful smoke completes all 16 records and leaves one sealed shard on
Google Drive.

Check its status without loading the 27B teacher again:

```python
!python scripts/teacher_distillation/generate_teacher_data.py \
  --input "$SUBSET_DIR/p0-16.jsonl" \
  --checkpoint-dir "$SMOKE_DIR/checkpoint" \
  --work-dir /content/tqc-distillation-smoke \
  --status-only
```

### 7. Generate the 2,000-record pilot

Only after the smoke succeeds:

```python
!python scripts/teacher_distillation/generate_teacher_data.py \
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
!python scripts/teacher_distillation/generate_teacher_data.py \
  --input "$SUBSET_DIR/p0-2000.jsonl" \
  --checkpoint-dir "$PILOT_DIR/checkpoint" \
  --work-dir /content/tqc-distillation-2000 \
  --status-only
```

### 8. Finalize the pilot for the RTX 4070 Ti machine

After generation reaches 2,000/2,000:

```python
!python scripts/teacher_distillation/finalize_teacher_data.py \
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

## What v1 does and does not prove

This v1 corpus uses a substantially stronger teacher and deterministic,
preemption-safe provenance. It filters truncation, malformed content, obvious
Python syntax/Python-2 failures, overlength records, and exact duplicates.

It does **not** claim semantic execution verification for every Magicoder/OLMo
prompt because those source prompts do not uniformly provide an executable
reference-test oracle. Teacher output is therefore *distilled*, not
*mechanically verified*. The frozen downstream benchmark remains the decisive
measurement of whether the new corpus improves the student.
