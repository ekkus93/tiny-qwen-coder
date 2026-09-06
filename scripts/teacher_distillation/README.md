# Qwen3.8-27B teacher distillation

**Start here for generating replacement Python training data with Qwen3.8-27B on Google Colab.**

There are now three executable notebooks:

- [`qwen38_teacher_distillation_colab.ipynb`](qwen38_teacher_distillation_colab.ipynb) — preserved v1 workflow and historical 16/500/2,000 progression.
- [`qwen38_teacher_distillation_v2_colab.ipynb`](qwen38_teacher_distillation_v2_colab.ipynb) — preserved bounded v2 high-reasoning study.
- [`qwen38_teacher_distillation_v3_colab.ipynb`](qwen38_teacher_distillation_v3_colab.ipynb) — current bounded v3 per-record answer-budget study.

**Open the current v3 workflow in Colab:** [qwen38_teacher_distillation_v3_colab.ipynb](https://colab.research.google.com/github/ekkus93/tiny-qwen-coder/blob/master/scripts/teacher_distillation/qwen38_teacher_distillation_v3_colab.ipynb)

For new work after the v1 pilot, use the **v2 notebook**. Select an **A100 80 GB** runtime. Colab is only a disposable GPU worker: it does not need GitHub credentials, SSH keys, `git clone`, `git pull`, or `git push`.

## Directory contents

| File | Purpose |
| --- | --- |
| `qwen38_teacher_distillation_colab.ipynb` | Preserved v1 executable Colab workflow. |
| `qwen38_teacher_distillation_v2_colab.ipynb` | Current bounded v2 diagnostic/generation/qualification workflow. |
| `prepare_teacher_input.py` | Build and SHA-256 seal the canonical prompt-only teacher input. |
| `select_teacher_input.py` | Create deterministic source-stratified subsets. |
| `prepare_teacher_v2_input.py` | Add and SHA-256 bind the teacher-only concise-answer v2 policy. |
| `prepare_teacher_v3_input.py` | Compute, inject, and SHA-256 bind the canonical Qwen3.5-4B per-record final-answer budget. |
| `generate_teacher_data.py` | Run resumable Qwen3.8 generation with durable Google Drive checkpoints. |
| `diagnose_teacher_data.py` | Measure teacher/runtime and student-tokenizer length behavior without persisting hidden reasoning. |
| `finalize_teacher_data.py` | Filter, contamination-check, and emit the Qwen3.5-4B training corpus. |
| `qualify_teacher_study.py` | Mechanically decide whether the bounded v2 study may scale. |
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

## Bounded v2 successor contract

`configs/distillation/python/qwen38_27b_v2.yaml` intentionally keeps the expensive/runtime variables fixed and changes only the generation policy implicated by the v1 evidence:

- same pinned Qwen3.8-27B teacher and revision;
- same native BF16 weights;
- same vLLM 0.28.0 runtime;
- same 16,384-token model context;
- same 8,192-token generation cap;
- same sampling parameters, seed, and durable shard size;
- reasoning effort reduced from `xhigh` to **`high`**; and
- a deterministic teacher-only concise-answer instruction is injected by `prepare_teacher_v2_input.py`.

The v2 instruction asks the teacher to solve the user's task completely while keeping the final answer concise enough for the Qwen3.5-4B 2,048-token full-conversation envelope. The original user request is not rewritten.

The teacher-only instruction is **not student training data**. It is removed again before Python quality checking, student tokenization, contamination checking, final corpus writing, and student-length qualification. Its policy ID and SHA-256 remain only as provenance metadata.

## v2 qualification study

Do **not** jump directly to another 2,000-record generation. The first v2 generation is a deterministic **200-record bounded study** using a new checkpoint namespace.

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

The study may scale only when `qualify_teacher_study.py` proves all three gates:

1. normal-stop rate is at least **0.90**;
2. student-length acceptance among normal-stop records is at least **0.80**; and
3. protected-benchmark contamination status is **`clean`**.

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

The v1 and v2 runs must not silently share mutable source archives. Keep the preserved v1 archive untouched and give the successor code a new filename, for example:

```text
MyDrive/
└── tiny-qwen-coder/
    └── code/
        ├── tiny-qwen-coder.zip
        ├── tiny-qwen-coder.zip.sha256
        ├── tiny-qwen-coder-v2.zip
        └── tiny-qwen-coder-v2.zip.sha256
```

The v2 notebook expects `tiny-qwen-coder-v2.zip`. On first use it seals the exact ZIP bytes with the adjacent `.sha256` file; later runs must match that checksum.

GitHub source ZIPs contain no `.git` directory. After extraction, the notebook creates a deterministic local Git commit solely so generic dataset-manifest provenance can record an exact source-tree identity. No remote is configured and no GitHub credentials are required.

## Why the preserved v1 checkpoint remains readable

Durable generation shards bind only the exact source of:

- `generation.py`;
- `config.py`; and
- `vllm_backend.py`;

plus the semantic distillation config, input SHA-256, record identity, prompt identity, and shard checksum.

The v2 reasoning-effort correction changes `config.py`, which is part of the generation implementation identity. Inspect preserved v1 checkpoints with the frozen v1 repository archive that created them; do not weaken or rewrite their run identity to make newer code accept them.

Never edit `run-identity.json`, shard payloads, or checksum sidecars to force compatibility.

## Current recommended progression

1. Preserve the completed v1 2,000-candidate checkpoint as evidence.
2. Run the v2 notebook's v1 diagnostic cell to record the failure distribution with the new tooling.
3. Generate the deterministic 200-record v2 study in a fresh checkpoint directory.
4. Diagnose, finalize, run contamination checks, and mechanically qualify the 200 records.
5. Only if all qualification gates pass, generate a fresh **2,000-record v2** experiment.
6. Train/evaluate the Qwen3.5-4B student on that qualified corpus.
7. Generate more than 2,000 records only if the v2 2,000-record adapter improves the frozen base benchmark.

Do not jump directly to the full ~40k corpus.

## Recovery rules

- Treat `/content` as disposable and Google Drive as durable.
- Keep each frozen repository ZIP and its `.sha256` sidecar on Drive.
- Reuse the exact same sealed input and repository ZIP when resuming a checkpoint.
- Recreate `/content/tqc-teacher-venv` after a fresh Colab allocation.
- Always run teacher scripts with the notebook's `$TQC_PYTHON`.
- Keep v1, v2-200, v2-2000, and any later experiment in separate checkpoint directories.
- Never replace a code ZIP after generation has started in the checkpoint namespace it governs.
- Never edit `run-identity.json`, generated shards, or checksum sidecars.
- An incomplete/unsealed shard is regenerated automatically.
- A sealed shard with a bad checksum fails closed as corruption.

There is intentionally no GitHub write workflow in Colab. Development, commits, pushes, and pulls happen outside the disposable GPU runtime.


## Bounded v3 per-record answer-budget contract

The 200-record v2 high-reasoning study fixed the v1 runaway-generation problem: all 200 candidates stopped normally and protected-benchmark contamination was clean. Its remaining failure was narrow and specifically student-envelope related: 153/200 records (76.5%) fit the 2,048-token student boundary, below the predeclared 80% floor. Prompt lengths were not the driver; rejected final answers were substantially longer than accepted answers.

v3 therefore keeps the successful v2 teacher/runtime/generation contract fixed and changes only the teacher-only input policy:

- Qwen3.8-27B at the same pinned revision, native BF16, vLLM 0.28.0;
- reasoning effort remains `high`;
- sampling parameters, seed, 16-record shard size, 16,384-token context, and 8,192-token teacher generation cap remain unchanged;
- the same deterministic 200 source records are used for the bounded study;
- the Qwen3.5-4B canonical tokenizer measures each source prompt before teacher-only policy injection;
- each record receives `min(1792, 2048 - student_prompt_tokens - 128)` as its approximate final-answer budget;
- the 128-token reserve protects chat-template boundaries and imperfect token-budget adherence;
- the 8,192 teacher generation cap is deliberately not reduced because it includes hidden reasoning, while the new budget applies only to the final answer;
- the dynamic instruction is stripped before Python quality checks, student tokenization, contamination checks, or corpus writing.

The existing formal scale floor remains >=90% normal stops, >=80% student-length acceptance among normal stops, and contamination `clean`. For v3, the notebook uses a stricter **85% student-length scale gate** to provide margin before spending on a fresh 2,000-record run. Do not lower either gate after observing the result.
