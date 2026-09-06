# Qwen3.8-27B teacher distillation

**Start here for generating replacement Python training data with Qwen3.8-27B on Google Colab.**

There are now three executable notebooks:

- [`qwen38_teacher_distillation_colab.ipynb`](qwen38_teacher_distillation_colab.ipynb) — preserved v1 workflow and historical 16/500/2,000 progression.
- [`qwen38_teacher_distillation_v2_colab.ipynb`](qwen38_teacher_distillation_v2_colab.ipynb) — preserved bounded v2 high-reasoning study.
- [`qwen38_teacher_distillation_v3_colab.ipynb`](qwen38_teacher_distillation_v3_colab.ipynb) — current bounded v3 per-record answer-budget study.

**Open the current v3 workflow in Colab:** [qwen38_teacher_distillation_v3_colab.ipynb](https://colab.research.google.com/github/ekkus93/tiny-qwen-coder/blob/master/scripts/teacher_distillation/qwen38_teacher_distillation_v3_colab.ipynb)

For new teacher-data work, use the **v3 notebook**. Select an **A100 80 GB** runtime. Colab is only a disposable GPU worker: it does not need GitHub credentials, SSH keys, `git clone`, `git pull`, or `git push`.

## Directory contents

| File | Purpose |
| --- | --- |
| `qwen38_teacher_distillation_colab.ipynb` | Preserved v1 executable Colab workflow. |
| `qwen38_teacher_distillation_v2_colab.ipynb` | Preserved bounded v2 high-reasoning study. |
| `qwen38_teacher_distillation_v3_colab.ipynb` | Current bounded v3 per-record answer-budget study. |
| `prepare_teacher_input.py` | Build and SHA-256 seal the canonical prompt-only teacher input. |
| `select_teacher_input.py` | Create deterministic source-stratified subsets. |
| `prepare_teacher_v2_input.py` | Add and SHA-256 bind the teacher-only concise-answer v2 policy. |
| `prepare_teacher_v3_input.py` | Compute, inject, and SHA-256 bind the canonical Qwen3.5-4B per-record final-answer budget. |
| `generate_teacher_data.py` | Run resumable Qwen3.8 generation with durable Google Drive checkpoints. |
| `diagnose_teacher_data.py` | Measure teacher/runtime and student-tokenizer length behavior without persisting hidden reasoning. |
| `finalize_teacher_data.py` | Filter, contamination-check, and emit the Qwen3.5-4B training corpus. |
| `qualify_teacher_study.py` | Mechanically decide whether a bounded teacher study may scale. |
| `README.md` | Architecture, experiment contract, recovery, and scaling rules. |

## Preserved v1 contract

- Teacher: `Qwen/Qwen3.8-27B`
- Teacher revision: `72a217afab8029b39e4af1c7273a829995a3dbaf`
- Runtime: vLLM 0.28.0, text-only mode
- Weight loading: native BF16 on the 80 GB A100
- Weight quantization: disabled (`quantization: none`)
- Thinking: enabled
- Reasoning effort: `xhigh`
- Preserved historical thinking: disabled
- Sampling: Qwen thinking-mode values already frozen in the v1 config
- Maximum model context: 16,384 tokens
- Maximum generated completion: 8,192 tokens
- Generation seed: 1729 plus the canonical input-record index
- Durable shard size: 16 records
- Student: unchanged `Qwen/Qwen3.5-4B`
- Student preparation boundary: 2,048 tokens with reject-on-overlength

The original source assistant response is never sent to the teacher. The teacher's `<think>...</think>` content is not written to checkpoint JSONL or the training corpus; checkpoints retain only a digest and character count for bounded audit evidence.

## What the v1 2,000-candidate run taught us

The first native-BF16 v1 2,000-candidate run is preserved as scientific evidence, but it is **not** the canonical 2,000-record learning experiment:

- 2,000 durable teacher candidates were produced;
- 1,297 ended with `finish_reason=length`;
- only 703 reached a normal `stop`;
- of those 703, another 361 exceeded the student's 2,048-token full-record boundary; and
- only 342 unique records survived final preparation.

The failure mode is therefore dominated by generation length and student-envelope mismatch. Generating more v1 samples would mostly spend GPU time producing records we already know are unlikely to survive.

## Preserved bounded v2 study

`configs/distillation/python/qwen38_27b_v2.yaml` kept the expensive/runtime variables fixed and changed only the generation policy implicated by the v1 evidence:

- same pinned Qwen3.8-27B teacher and revision;
- same native BF16 weights;
- same vLLM 0.28.0 runtime;
- same 16,384-token model context;
- same 8,192-token generation cap;
- same sampling parameters, seed, and durable shard size;
- reasoning effort reduced from `xhigh` to **`high`**; and
- a deterministic teacher-only concise-answer instruction injected by `prepare_teacher_v2_input.py`.

The bounded v2 run used 200 deterministic source records. It achieved:

- **200/200 normal stops (100%)**;
- protected-benchmark contamination status **`clean`**; and
- **153/200 student-length accepted records (76.5%)**.

It therefore failed only the predeclared 80% student-length acceptance gate. Diagnostics showed that prompt length was not the driver: accepted prompts had a median of 148 student tokens and rejected prompts a median of 141, while rejected final answers had a median of 2,650 student tokens versus 813 for accepted answers.

The v2 teacher-only instruction is **not student training data**. It is removed again before Python quality checking, student tokenization, contamination checking, final corpus writing, and student-length qualification. Its policy ID and SHA-256 remain only as provenance metadata.

## Bounded v3 per-record answer-budget contract

v3 keeps the successful v2 teacher/runtime/generation contract fixed and changes only the teacher-only input policy:

- Qwen3.8-27B at the same pinned revision, native BF16, vLLM 0.28.0;
- reasoning effort remains `high`;
- sampling parameters, seed, 16-record shard size, 16,384-token context, and 8,192-token teacher generation cap remain unchanged;
- the same deterministic 200 source records are used for the bounded study;
- the Qwen3.5-4B canonical tokenizer measures each source prompt before teacher-only policy injection;
- each record receives `min(1792, 2048 - student_prompt_tokens - 128)` as its approximate final-answer budget;
- the 128-token reserve protects chat-template boundaries and imperfect token-budget adherence;
- the 8,192 teacher generation cap is deliberately not reduced because it includes hidden reasoning, while the new budget applies only to the final answer; and
- the dynamic instruction is stripped before Python quality checks, student tokenization, contamination checks, or corpus writing.

The generic input-policy stripper fails closed on missing, incomplete, unknown, or corrupted policy metadata so a teacher-only instruction cannot silently leak into student training data.

The historical formal floor remains >=90% normal stops, >=80% student-length acceptance among normal stops, and contamination `clean`. For the v3 scaling decision, the notebook deliberately uses the stricter **85% student-length gate** to leave margin before spending on a fresh 2,000-record run. Do not lower the gate after observing the result.

After generation, `diagnose_teacher_data.py` records per-record and aggregate evidence for:

- teacher prompt tokens;
- teacher completion tokens;
- retained reasoning character count (never reasoning text);
- teacher final-answer tokens;
- student prompt tokens after stripping the teacher-only policy;
- student final-answer tokens;
- student full-record tokens;
- finish reason; and
- whether the student record fits the 2,048-token boundary.

A failed gate means revise the policy and run another bounded study. It is not permission to compensate by blindly generating more samples.

## Protected benchmark contamination is fail-closed

Teacher generation never uses protected benchmarks as input. Finalization loads protected examples only after generation and runs the existing exact/high-overlap contamination checker against the retained training corpus.

For Python, coverage must include every registered protected benchmark:

- HumanEval prompts and canonical solutions;
- MBPP prompts and canonical solutions; and
- the repository-owned holdout prompts.

Benchmark text is used only in memory by the checker and is not copied into the distilled corpus or manifest. Finalization refuses to write a training corpus if contamination checks did not run or if any finding is present.

## Colab runtime isolation

Do not use Colab's preinstalled PyTorch/TorchAudio environment for teacher inference. Colab images can contain packages built against different CUDA versions.

The notebooks therefore:

1. install/update `uv`;
2. create `/content/tqc-teacher-venv` with `uv venv`;
3. install `ninja==1.13.2`, `vllm==0.28.0`, and this repository with `uv pip install --torch-backend=cu130`;
4. let vLLM own its compatible PyTorch constraint while uv selects the CUDA 13.0 wheel backend;
5. prepend the uv environment `bin/` directory to `PATH` so FlashInfer JIT builds can find Ninja; and
6. run teacher scripts through `/content/tqc-teacher-venv/bin/python`.

`requirements/colab-teacher.txt` pins the required top-level runtime tools:

```text
ninja==1.13.2
vllm==0.28.0
```

Do not manually add PyTorch/TorchAudio/TorchVision CUDA pins.

## Frozen repository ZIPs

Each experimental generation must use its own immutable source archive and checkpoint namespace. Keep prior archives untouched. The current v3 notebook expects:

```text
MyDrive/
└── tiny-qwen-coder/
    └── code/
        ├── tiny-qwen-coder.zip
        ├── tiny-qwen-coder.zip.sha256
        ├── tiny-qwen-coder-v2.zip
        ├── tiny-qwen-coder-v2.zip.sha256
        ├── tiny-qwen-coder-v3.zip
        └── tiny-qwen-coder-v3.zip.sha256
```

On first use, the v3 notebook seals the exact `tiny-qwen-coder-v3.zip` bytes with the adjacent `.sha256` file; later runs in that experiment namespace must match that checksum.

GitHub source ZIPs contain no `.git` directory. After extraction, the notebook creates a deterministic local Git commit solely so generic dataset-manifest provenance can record an exact source-tree identity. No remote is configured and no GitHub credentials are required.

## Why old checkpoints require their frozen source archive

Durable generation shards bind the exact source of:

- `generation.py`;
- `config.py`; and
- `vllm_backend.py`;

plus the semantic distillation config, input SHA-256, record identity, prompt identity, and shard checksum.

The v2 reasoning-effort correction changed `config.py`, which is part of the generation implementation identity. Inspect preserved v1 checkpoints with the frozen v1 repository archive that created them; do not weaken or rewrite their run identity to make newer code accept them. The same rule applies to every later experiment generation.

Never edit `run-identity.json`, shard payloads, or checksum sidecars to force compatibility.

## Current recommended progression

1. Preserve the completed v1 and v2 checkpoints and diagnostics as evidence.
2. Start the **v3 notebook** from a fresh A100 80 GB Colab allocation using a frozen v3 repository ZIP.
3. Recreate the same deterministic 200-record source subset and transform it with the v3 per-record answer-budget policy.
4. Generate/resume the bounded 200-record v3 checkpoint.
5. Diagnose, finalize, run fail-closed contamination checks, and mechanically qualify the v3 study.
6. Scale only if normal stops are >=90%, student-length acceptance is >=85%, and contamination is `clean`.
7. If v3 qualifies, prepare a fresh **2,000-record v3** experiment rather than reusing the 200-record checkpoint namespace.
8. Train/evaluate the Qwen3.5-4B student on the qualified 2,000-record corpus.
9. Generate more than 2,000 records only if the v3 2,000-record adapter improves the frozen base benchmark.

Do not jump directly to the full ~40k corpus.

## Recovery rules

- Treat `/content` as disposable and Google Drive as durable.
- Keep each frozen repository ZIP and its `.sha256` sidecar on Drive.
- Reuse the exact same sealed input and repository ZIP when resuming a checkpoint.
- Recreate `/content/tqc-teacher-venv` after a fresh Colab allocation.
- Always run teacher scripts with the notebook's `$TQC_PYTHON`.
- Keep v1, v2, v3-200, v3-2000, and later experiments in separate checkpoint directories.
- Never replace a code ZIP after generation has started in the checkpoint namespace it governs.
- Never edit `run-identity.json`, generated shards, or checksum sidecars.
- An incomplete/unsealed shard is regenerated automatically.
- A sealed shard with a bad checksum fails closed as corruption.

There is intentionally no GitHub write workflow in Colab. Development, commits, pushes, and pulls happen outside the disposable GPU runtime.
