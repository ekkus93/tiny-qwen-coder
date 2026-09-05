# Qwen3.8-27B teacher distillation

**Start here for generating the new Python training data on Google Colab.**

This directory keeps the runnable entry points and their operating instructions
together. The teacher is `Qwen/Qwen3.8-27B` on a Google Colab A100; the student
remains `Qwen/Qwen3.5-4B`, fine-tuned later on the RTX 4070 Ti SUPER.

Google Colab does **not** need GitHub credentials, SSH keys, `git clone`,
`git pull`, or `git push` for this workflow. Upload a frozen ZIP of the repository
to Google Drive and run that exact archive on every Colab allocation.

## Directory contents

| File | Purpose |
| --- | --- |
| `prepare_teacher_input.py` | Build and SHA-256 seal the canonical prompt-only teacher input. |
| `select_teacher_input.py` | Create deterministic source-stratified 16/500/2,000-record pilot inputs. |
| `generate_teacher_data.py` | Run resumable Qwen3.8 generation with durable Google Drive checkpoints. |
| `finalize_teacher_data.py` | Validate/filter completed shards and emit the Qwen3.5-4B training corpus and manifest. |
| `README.md` | This Google Colab runbook and recovery contract. |

All commands below are run from the repository root after the frozen ZIP has
been extracted into Colab scratch space.

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

The repository ZIP is also SHA-256 sealed on Google Drive. A new Colab allocation
must use the same ZIP bytes before resuming an existing run.

A shard is generated into local Colab scratch first. Only after it is complete
is it copied to Drive, with its checksum sidecar. The Drive copy is then read
back and validated before progress advances. A killed runtime can therefore
lose at most the currently unsealed 16-record shard. Re-running the same command
verifies and skips every already sealed shard.

`run-identity.json` deliberately binds the requested record count. Never reuse a
smoke-run checkpoint directory for a larger run.

# Google Colab runbook

The workflow assumes a Colab runtime can disappear at any time. `/content` is
scratch space; Google Drive is the durable boundary for the frozen code archive,
input corpus, subsets, and generated checkpoints.

The cells below are written in **Colab notebook syntax** and can be copied into
separate Python cells.

## 0. Before opening Colab: upload a frozen repository ZIP

On your normal development machine, make or download a ZIP containing the exact
`tiny-qwen-coder` repository version you want to run.

The ZIP does not need a `.git` directory. It only needs the repository files.

Upload it to Google Drive as:

```text
MyDrive/
└── tiny-qwen-coder/
    └── code/
        └── tiny-qwen-coder.zip
```

Once a real generation run has started, **do not overwrite that ZIP with newer
code**. A different code version should use a new archive and a new checkpoint
directory.

You do not need to configure GitHub authentication or SSH keys in Colab.

## 1. Select an A100 runtime and mount Google Drive

In Colab, select an A100 GPU runtime, then run:

```python
from google.colab import drive

drive.mount("/content/drive")
```

Define the durable paths:

```python
import os
from pathlib import Path

os.environ["TQC_DRIVE"] = "/content/drive/MyDrive/tiny-qwen-coder"
os.environ["CODE_DIR"] = f"{os.environ['TQC_DRIVE']}/code"
os.environ["CODE_ARCHIVE"] = f"{os.environ['CODE_DIR']}/tiny-qwen-coder.zip"
os.environ["RUN_ROOT"] = f"{os.environ['TQC_DRIVE']}/distillation/qwen38-27b-v1"
os.environ["INPUT_DIR"] = f"{os.environ['RUN_ROOT']}/input"
os.environ["SUBSET_DIR"] = f"{os.environ['RUN_ROOT']}/subsets"
os.environ["SMOKE_DIR"] = f"{os.environ['TQC_DRIVE']}/distillation/qwen38-27b-v1-smoke"
os.environ["PILOT_DIR"] = f"{os.environ['TQC_DRIVE']}/distillation/qwen38-27b-v1-2000"

for key in (
    "CODE_DIR",
    "RUN_ROOT",
    "INPUT_DIR",
    "SUBSET_DIR",
    "SMOKE_DIR",
    "PILOT_DIR",
):
    Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
```

## 2. Verify and extract the frozen repository ZIP

This replaces all Git clone/pull/revision-freezing steps.

The first time this cell sees the ZIP, it writes a SHA-256 sidecar next to it on
Drive. Every later Colab allocation verifies the archive against that checksum
before extracting it.

```python
import hashlib
import os
import shutil
from pathlib import Path
from zipfile import ZipFile

archive = Path(os.environ["CODE_ARCHIVE"])
if not archive.is_file():
    raise FileNotFoundError(
        f"Repository ZIP not found: {archive}\n"
        "Upload tiny-qwen-coder.zip to the Google Drive code directory first."
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


archive_sha256 = sha256_file(archive)
checksum_path = archive.with_suffix(archive.suffix + ".sha256")

if checksum_path.exists():
    expected_sha256 = checksum_path.read_text(encoding="ascii").split()[0]
    if archive_sha256 != expected_sha256:
        raise RuntimeError(
            "Repository ZIP checksum changed. Do not resume this experiment "
            "with different code."
        )
else:
    checksum_path.write_text(
        f"{archive_sha256}  {archive.name}\n",
        encoding="ascii",
    )

scratch_root = Path("/content/tiny-qwen-coder-code")
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
        "Expected exactly one tiny-qwen-coder repository in the ZIP; "
        f"found {len(repo_candidates)} candidates."
    )

repo = repo_candidates[0]
os.environ["TQC_REPO"] = str(repo)
os.environ["TQC_CODE_ARCHIVE_SHA256"] = archive_sha256

print("repository:", repo)
print("archive SHA-256:", archive_sha256)
print("checksum:", checksum_path)
```

This works whether the ZIP contains the repository files directly or wraps them
inside a directory such as `tiny-qwen-coder-master/`.

Now enter the extracted repository and install the project plus the pinned A100
teacher runtime:

```python
import os

os.chdir(os.environ["TQC_REPO"])
print("working directory:", os.getcwd())

!python -m pip install -e .
!python -m pip install -r requirements/colab-teacher.txt
```

After a completely fresh Colab allocation, rerun steps 1 and 2. The ZIP and its
checksum are on Drive, so no GitHub access is required to reconstruct the exact
code environment.

## 3. Verify the A100 before downloading Qwen3.8

```python
!nvidia-smi

import torch

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
```

Stop here if the runtime did not receive the expected CUDA GPU.

## 4. Build and seal the immutable teacher input once

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

Do this once for an experiment. Do not rebuild or edit the file in place after
generation has started.

## 5. Create deterministic representative pilot subsets

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
4. Generate more only if the 2,000-record adapter improves the frozen base benchmark.

## 6. Run the 16-record A100 smoke test

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

## 7. Generate the 2,000-record pilot

Only after the smoke succeeds:

```python
!python scripts/teacher_distillation/generate_teacher_data.py \
  --input "$SUBSET_DIR/p0-2000.jsonl" \
  --checkpoint-dir "$PILOT_DIR/checkpoint" \
  --work-dir /content/tqc-distillation-2000
```

If Colab disconnects, crashes, or revokes the instance:

1. acquire another A100 runtime;
2. remount the same Google Drive;
3. rerun steps 1 through 3 using the same frozen ZIP;
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

## 8. Finalize the pilot for the RTX 4070 Ti machine

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

## Colab recovery rules

- Treat `/content` as disposable and Google Drive as durable.
- Keep the frozen repository ZIP and its `.sha256` sidecar on Drive.
- Keep each 16/500/2,000/full experiment in its own checkpoint directory.
- Reuse the exact same sealed input and exact same repository ZIP on resume.
- Do not overwrite the repository ZIP after a real generation run has started.
- Never edit `run-identity.json`, generated shards, or checksum sidecars to force progress.
- An uncommitted shard left by a killed runtime is regenerated automatically.
- A sealed shard whose checksum is wrong fails closed as corruption.
- Do not start the full corpus until the bounded 2,000-record experiment shows that the distilled data actually improves the student.

There is intentionally **no GitHub write workflow in Colab**. Development,
commits, pushes, and pulls happen on the normal development machine. Colab is
only a disposable GPU worker consuming a frozen code archive.

## What v1 does and does not prove

This v1 corpus uses a substantially stronger teacher and deterministic,
preemption-safe provenance. It filters truncation, malformed content, obvious
Python syntax/Python-2 failures, overlength records, and exact duplicates.

It does **not** claim semantic execution verification for every Magicoder/OLMo
prompt because those source prompts do not uniformly provide an executable
reference-test oracle. Teacher output is therefore *distilled*, not
*mechanically verified*. The frozen downstream benchmark remains the decisive
measurement of whether the new corpus improves the student.
