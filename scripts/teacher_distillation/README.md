# Qwen3.8-27B teacher distillation

**Start here for generating the new Python training data on Google Colab.**

The easiest and canonical way to run this workflow is the executable notebook:

- [`qwen38_teacher_distillation_colab.ipynb`](qwen38_teacher_distillation_colab.ipynb)

Open that notebook in Google Colab, select an **A100 80 GB** runtime, and run it from
the beginning. The notebook is the executable source of truth for environment setup,
Drive paths, smoke testing, resumable generation, and finalization.

Google Colab is only a disposable GPU worker. It does **not** need GitHub credentials,
SSH keys, `git clone`, `git pull`, or `git push`. Upload a frozen ZIP of the repository
to Google Drive and run that exact archive on every Colab allocation.

## Directory contents

| File | Purpose |
| --- | --- |
| `qwen38_teacher_distillation_colab.ipynb` | Canonical executable Google Colab workflow. |
| `prepare_teacher_input.py` | Build and SHA-256 seal the canonical prompt-only teacher input. |
| `select_teacher_input.py` | Create deterministic source-stratified 16/500/2,000-record pilot inputs. |
| `generate_teacher_data.py` | Run resumable Qwen3.8 generation with durable Google Drive checkpoints. |
| `finalize_teacher_data.py` | Validate/filter completed shards and emit the Qwen3.5-4B training corpus and manifest. |
| `README.md` | Architecture, experiment contract, and recovery notes. |

## Frozen v1 contract

- Teacher: `Qwen/Qwen3.8-27B`
- Teacher revision: `72a217afab8029b39e4af1c7273a829995a3dbaf`
- Runtime: vLLM 0.28.0, text-only mode
- Weight loading: native BF16 on the 80 GB A100
- Weight quantization: disabled (`quantization: none`)
- Thinking: enabled
- Reasoning effort: `xhigh`
- Preserved historical thinking: disabled
- Sampling: Qwen's recommended thinking-mode values
- Maximum model context: 16,384 tokens
- Maximum generated completion: 8,192 tokens
- Generation seed: 1729 plus the canonical input-record index
- Durable shard size: 16 records
- Student: unchanged `Qwen/Qwen3.5-4B`
- Student preparation boundary: 2,048 tokens with reject-on-overlength

The original P0 assistant response is never sent to the teacher. Only the system/user
prompt prefix is sent. The teacher's `<think>...</think>` content is not written to
the training corpus or checkpoint JSONL. Checkpoints retain only its SHA-256 digest
and character count for bounded audit evidence.

## Colab runtime isolation

Do not use Colab's preinstalled PyTorch/TorchAudio environment for teacher inference.
Colab images can contain packages built against different CUDA versions.

The notebook therefore:

1. installs/updates `uv`;
2. creates `/content/tqc-teacher-venv` with `uv venv`;
3. installs `vllm==0.28.0` and this repository with
   `uv pip install --torch-backend=cu129`;
4. lets vLLM own its compatible Torch/TorchAudio/TorchVision constraints while uv
   selects the CUDA 12.9 wheel backend; and
5. runs all teacher scripts through `/content/tqc-teacher-venv/bin/python`.

`requirements/colab-teacher.txt` intentionally pins only the top-level teacher runtime:

```text
vllm==0.28.0
```

Do not manually add `torch==...+cu129`, `torchaudio==...+cu129`, or
`torchvision==...+cu129` pins. The vLLM wheel and uv backend selection should resolve
the compatible CUDA stack together.

The NVIDIA driver may still report CUDA 13.0 in `nvidia-smi`. That is expected. The
notebook verifies that the isolated PyTorch runtime itself is using CUDA 12.9.

## Frozen repository ZIP

Before opening Colab, make or download a ZIP containing the exact repository version
you want to run. The ZIP does not need a `.git` directory.

Place it at:

```text
MyDrive/
└── tiny-qwen-coder/
    └── code/
        └── tiny-qwen-coder.zip
```

The notebook creates and verifies:

```text
MyDrive/tiny-qwen-coder/code/tiny-qwen-coder.zip.sha256
```

Once a real generation run has started, do not overwrite that ZIP with newer code.
A different code version should use a new archive and new checkpoint directories.

## Recommended progression

1. **16 records** — prove model loading, runtime compatibility, checkpointing, and resume.
2. **500 records** — inspect teacher output and finalization rejection rates.
3. **2,000 records** — perform the first real Qwen3.5-4B learning experiment.
4. Generate more only if the 2,000-record adapter improves the frozen base benchmark.

Do not jump directly to the full ~40k corpus.

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

A shard is generated into local Colab scratch first. Only after it is complete is it
copied to Drive with its checksum sidecar. The Drive copy is then read back and
validated before progress advances. A killed runtime can therefore lose at most the
currently unsealed 16-record shard. Re-running the same command verifies and skips
every already sealed shard.

`run-identity.json` deliberately binds the requested record count. Never reuse a
smoke-run checkpoint directory for a larger run.

## Recovery rules

- Treat `/content` as disposable and Google Drive as durable.
- Keep the frozen repository ZIP and its `.sha256` sidecar on Drive.
- Reuse the exact same sealed input and exact same repository ZIP on resume.
- Recreate `/content/tqc-teacher-venv` by rerunning notebook Step 5 after a fresh Colab allocation.
- Always run teacher scripts with the notebook's `$TQC_PYTHON`, never Colab's system `python`.
- Keep 16/500/2,000/full experiments in separate checkpoint directories.
- Never edit `run-identity.json`, generated shards, or checksum sidecars to force progress.
- An uncommitted shard left by a killed runtime is regenerated automatically.
- A sealed shard whose checksum is wrong fails closed as corruption.
- Do not start the full corpus until the bounded 2,000-record experiment shows that the distilled data actually improves the student.

There is intentionally no GitHub write workflow in Colab. Development, commits,
pushes, and pulls happen on the normal development machine.

## What v1 does and does not prove

This v1 corpus uses a substantially stronger teacher and deterministic,
preemption-safe provenance. It filters truncation, malformed content, obvious Python
syntax/Python-2 failures, overlength records, and exact duplicates.

It does **not** claim semantic execution verification for every Magicoder/OLMo prompt
because those source prompts do not uniformly provide an executable reference-test
oracle. Teacher output is therefore *distilled*, not *mechanically verified*. The
frozen downstream benchmark remains the decisive measurement of whether the new
corpus improves the student.
