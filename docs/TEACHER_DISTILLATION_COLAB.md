# Qwen3.8-27B teacher distillation on Google Colab

This runbook generates a replacement Python training corpus for the existing
Qwen3.5-4B student. Teacher inference runs on a Google Colab A100; LoRA/QLoRA
training remains on the local RTX 4070 Ti SUPER.

The generation workflow is designed for preemptible Colab runtimes. Google
Drive is the durable boundary. The local Colab filesystem is only scratch.

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

## 1. Start an A100 Colab runtime and mount Drive

In a Colab Python cell:

```python
from google.colab import drive

drive.mount("/content/drive")
```

Use a stable project root in Drive:

```bash
export TQC_DRIVE="/content/drive/MyDrive/tiny-qwen-coder"
mkdir -p "$TQC_DRIVE/distillation/qwen38-27b-v1"
```

## 2. Get the repository and install the generation runtime

```bash
cd /content
git clone https://github.com/ekkus93/tiny-qwen-coder.git
cd tiny-qwen-coder

python -m pip install -e .
python -m pip install -r requirements/colab-teacher.txt
```

The teacher runtime versions are pinned in both the config and the requirements file.
Generation fails before loading the model if those installed versions drift. After a fresh
Colab allocation, repeat installation if `/content` was lost.

Do not copy the Hugging Face model cache to Drive: the model is large and Drive
is a poor model-cache filesystem. Re-downloading model weights after a truly
fresh runtime is preferable to making every inference access go through Drive.
The irreplaceable generated shards are what must live on Drive.

Confirm the accelerator before generation:

```bash
nvidia-smi
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
PY
```

## 3. Build and seal the immutable teacher input once

The canonical P0 input is generated from the project's pinned source configs and
Qwen3.5-4B tokenizer. Write it directly into Drive:

```bash
export INPUT_DIR="$TQC_DRIVE/distillation/qwen38-27b-v1/input"
mkdir -p "$INPUT_DIR"

python scripts/prepare_teacher_input.py \
  --output "$INPUT_DIR/accepted.jsonl"
```

This writes both:

- `accepted.jsonl`
- `accepted.jsonl.sha256`

Do this once. All production generation commands must use the same file.

## 4. Create representative bounded inputs

Do not use the first N rows for learning experiments. The canonical P0 file can be
ordered by upstream source, so prefix sampling can badly bias a pilot. Build sealed,
source-stratified subsets instead:

```bash
export SUBSET_DIR="$TQC_DRIVE/distillation/qwen38-27b-v1/subsets"
mkdir -p "$SUBSET_DIR"

python scripts/select_teacher_input.py \
  --input "$INPUT_DIR/accepted.jsonl" \
  --output "$SUBSET_DIR/p0-16.jsonl" \
  --count 16

python scripts/select_teacher_input.py \
  --input "$INPUT_DIR/accepted.jsonl" \
  --output "$SUBSET_DIR/p0-500.jsonl" \
  --count 500

python scripts/select_teacher_input.py \
  --input "$INPUT_DIR/accepted.jsonl" \
  --output "$SUBSET_DIR/p0-2000.jsonl" \
  --count 2000
```

The selector preserves the population proportions of each exact source/revision and
ranks records deterministically within each source. For a fixed seed, smaller
proportional subsets are nested inside larger ones when their per-source quotas nest.
Each subset receives its own SHA-256 sidecar and summary JSON.

## 5. Run a 16-record smoke generation first

Use a completely separate checkpoint directory:

```bash
export SMOKE_DIR="$TQC_DRIVE/distillation/qwen38-27b-v1-smoke"

python scripts/generate_teacher_data.py \
  --input "$SUBSET_DIR/p0-16.jsonl" \
  --checkpoint-dir "$SMOKE_DIR/checkpoint" \
  --work-dir /content/tqc-distillation-smoke
```

A successful smoke should produce one sealed shard and report 16/16 records.
It does not modify the production checkpoint directory.

## 6. Start or resume production generation

```bash
export RUN_DIR="$TQC_DRIVE/distillation/qwen38-27b-v1"

python scripts/generate_teacher_data.py \
  --input "$INPUT_DIR/accepted.jsonl" \
  --checkpoint-dir "$RUN_DIR/checkpoint" \
  --work-dir /content/tqc-distillation
```

If Colab disconnects, crashes, or revokes the instance, reconnect, remount
Drive, reinstall the environment as necessary, and run the exact same command.
The preflight executes before Qwen3.8 is loaded. Existing sealed shards are
verified and skipped.

To inspect progress without loading the 27B model:

```bash
python scripts/generate_teacher_data.py \
  --input "$INPUT_DIR/accepted.jsonl" \
  --checkpoint-dir "$RUN_DIR/checkpoint" \
  --work-dir /content/tqc-distillation \
  --status-only
```

Human-readable progress is also written to `checkpoint/progress.json`, but that
file is informational only. Resume correctness comes from the sealed shards,
not from trusting the progress counter.

## 7. Do not generate all 40,000 records before the first learning test

Use the sealed stratified input files above for bounded learning experiments. Each
subset naturally has a different input SHA-256 and therefore requires its own
checkpoint directory. `--limit` remains available only as a diagnostic convenience;
it should not be used to define a scientific pilot corpus.

A recommended sequence is:

- 16 records: runtime/smoke validation;
- 500 records: inspect output distribution and finalization losses;
- 2,000 records: first Qwen3.5-4B low-LR fine-tuning experiment;
- scale further only if the 2,000-record distilled run improves the frozen base
  benchmark.

For the 2,000-record experiment, point `--input` at `p0-2000.jsonl` and use a
dedicated `qwen38-27b-v1-2000/checkpoint`. Do not reuse that directory for the
40,000-record production run.

## 8. Finalize a completed run for the 4070 Ti training machine

Finalization validates every expected durable shard, rejects non-`stop`
completions and clearly invalid Python, applies the existing generic content,
student-tokenizer length, exact-deduplication, and deterministic split pipeline,
and writes a manifest.

For a full production run:

```bash
python scripts/finalize_teacher_data.py \
  --input "$INPUT_DIR/accepted.jsonl" \
  --checkpoint-dir "$RUN_DIR/checkpoint" \
  --output-dir "$RUN_DIR/final"
```

For a bounded run, use the exact same sealed subset file that was used during
generation and point at that bounded run's checkpoint directory.

The final directory contains:

- `accepted.jsonl`
- `train.jsonl`
- `validation.jsonl`
- `dataset-manifest.json`
- `dataset-manifest.sha256`
- `teacher-finalization.json`

Copy/download that final directory to the RTX 4070 Ti machine. Fine-tuning still
uses Qwen3.5-4B and the project's normal 4-bit QLoRA path; no A100-specific
training dependency is introduced.

## Failure rules

Do not delete a checksum sidecar, edit a shard, change the input file in place,
or modify the distillation config and then continue in the same checkpoint
directory. The code intentionally fails closed in those cases.

If a shard is corrupt, preserve the checkpoint directory for diagnosis and move
or remove only the corrupt shard plus its sidecar before deliberately
regenerating that shard. If `run-identity.json` disagrees with the intended run,
use a new checkpoint directory rather than overriding the identity.

## What v1 does and does not prove

This v1 corpus uses a substantially stronger teacher and deterministic,
preemption-safe provenance. It filters truncation, malformed content, obvious
Python syntax/Python-2 failures, overlength records, and exact duplicates.

It does **not** claim semantic execution verification for every Magicoder/OLMo
prompt because those source prompts do not uniformly provide an executable
reference-test oracle. Teacher output is therefore *distilled*, not
*mechanically verified*. The frozen downstream benchmark remains the decisive
measurement of whether the new corpus improves the student.
