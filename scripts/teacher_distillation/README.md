# Qwen3.8-27B teacher distillation

**Start here for generating or repairing replacement Python training data with Qwen3.8-27B on Google Colab.**

There are now six executable notebooks:

- [`qwen38_teacher_distillation_colab.ipynb`](qwen38_teacher_distillation_colab.ipynb) — preserved v1 workflow and historical 16/500/2,000 progression.
- [`qwen38_teacher_distillation_v2_colab.ipynb`](qwen38_teacher_distillation_v2_colab.ipynb) — preserved bounded v2 high-reasoning study.
- [`qwen38_teacher_distillation_v3_colab.ipynb`](qwen38_teacher_distillation_v3_colab.ipynb) — preserved bounded v3 per-record answer-budget study.
- [`qwen38_teacher_distillation_v4_colab.ipynb`](qwen38_teacher_distillation_v4_colab.ipynb) — preserved bounded v4 selective-compression study.
- [`qwen38_teacher_distillation_v4_2000_colab.ipynb`](qwen38_teacher_distillation_v4_2000_colab.ipynb) — preserved first v4-2000 successor runbook; its completed output is affected by the reasoning-leak parser defect and must not be used for training.
- [`qwen38_teacher_distillation_v4_2000_salvage_colab.ipynb`](qwen38_teacher_distillation_v4_2000_salvage_colab.ipynb) — **current authoritative repair workflow for the completed v4-2000 evidence**.

**Open the current v4-2000 salvage workflow in Colab:** [qwen38_teacher_distillation_v4_2000_salvage_colab.ipynb](https://colab.research.google.com/github/ekkus93/tiny-qwen-coder/blob/master/scripts/teacher_distillation/qwen38_teacher_distillation_v4_2000_salvage_colab.ipynb)

For the already completed `qwen38-27b-v4-2000` run, use the **salvage notebook only**. Do not rerun the ordinary v4-2000 notebook against that affected evidence and do not train from its old `final/` directory. Select an **A100 80 GB** runtime. Colab is only a disposable GPU worker: it does not need GitHub credentials, SSH keys, `git clone`, `git pull`, or `git push`.

For unrelated future bounded teacher-data experiments, the preserved v4 notebook remains the reference selective-compression design, but every new experiment must use a fresh immutable repository ZIP and fresh checkpoint namespace.

## Directory contents

| File | Purpose |
| --- | --- |
| `qwen38_teacher_distillation_colab.ipynb` | Preserved v1 executable Colab workflow. |
| `qwen38_teacher_distillation_v2_colab.ipynb` | Preserved bounded v2 high-reasoning study. |
| `qwen38_teacher_distillation_v3_colab.ipynb` | Preserved bounded v3 per-record answer-budget study. |
| `qwen38_teacher_distillation_v4_colab.ipynb` | Preserved bounded v4 selective-compression study. |
| `qwen38_teacher_distillation_v4_2000_colab.ipynb` | Preserved first v4-2000 successor runbook; affected completed output is frozen evidence only. |
| `qwen38_teacher_distillation_v4_2000_salvage_colab.ipynb` | Current executable repair workflow for the affected completed v4-2000 generation evidence. |
| `build_v4_colab_notebook.py` | Deterministically rebuild the bounded v4 notebook from stdlib-only source. |
| `build_v4_2000_colab_notebook.py` | Deterministically rebuild the preserved first v4-2000 successor notebook. |
| `build_v4_2000_salvage_colab_notebook.py` | Deterministically rebuild the v4-2000 salvage notebook. |
| `prepare_teacher_input.py` | Build and SHA-256 seal the canonical prompt-only teacher input. |
| `select_teacher_input.py` | Create deterministic source-stratified subsets. |
| `prepare_teacher_v2_input.py` | Add and SHA-256 bind the teacher-only concise-answer v2 policy. |
| `prepare_teacher_v3_input.py` | Compute, inject, and SHA-256 bind the canonical Qwen3.5-4B per-record final-answer budget. |
| `generate_teacher_data.py` | Run resumable first-pass Qwen3.8 generation with durable Google Drive checkpoints. |
| `salvage_teacher_reasoning.py` | Validate the affected legacy generation checkpoint and reconstruct reasoning-free first-pass records without regenerating the 2,000 expensive answers. |
| `compress_teacher_v4.py` | Select and durably rewrite only normal-stop answers that remain over the student envelope and their frozen answer budget. |
| `diagnose_teacher_data.py` | Diagnose a first-pass generation checkpoint. |
| `diagnose_prepared_teacher_data.py` | Diagnose an already student-shaped merged v4 corpus. |
| `finalize_teacher_data.py` | Finalize a first-pass generation checkpoint. |
| `finalize_prepared_teacher_data.py` | Finalize a merged v4 corpus without reinterpreting teacher-only prompt policies. |
| `qualify_teacher_study.py` | Mechanically decide whether a bounded teacher study may scale or train. |
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

The original source assistant response is never sent to the teacher. The corrected runtime strips Qwen thinking content before trainable records are written, and finalization now fails closed if any `<think>` or `</think>` marker survives in an assistant training message. Checkpoints retain only bounded reasoning digest/character-count evidence rather than hidden reasoning text.

## What the v1 2,000-candidate run taught us

The first native-BF16 v1 2,000-candidate run is preserved as scientific evidence, but it is **not** the canonical 2,000-record learning experiment:

- 2,000 durable teacher candidates were produced;
- 1,297 ended with `finish_reason=length`;
- only 703 reached a normal `stop`;
- of those 703, another 361 exceeded the student's 2,048-token full-record boundary; and
- only 342 unique records survived final preparation.

The failure mode was dominated by generation length and student-envelope mismatch. Generating more v1 samples would mostly spend GPU time producing records we already know are unlikely to survive.

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

## Preserved bounded v3 study

v3 kept the successful v2 teacher/runtime/generation contract fixed and changed only the teacher-only input policy:

- Qwen3.8-27B at the same pinned revision, native BF16, vLLM 0.28.0;
- reasoning effort remained `high`;
- sampling parameters, seed, 16-record shard size, 16,384-token context, and 8,192-token teacher generation cap stayed unchanged;
- the same deterministic 200 source records were used;
- the Qwen3.5-4B tokenizer measured each source prompt before teacher-only policy injection; and
- each record received `min(1792, 2048 - student_prompt_tokens - 128)` as its approximate final-answer budget.

The live v3 run achieved:

- **199/200 normal stops (99.5%)**;
- protected-benchmark contamination status **`clean`**;
- **159/199 student-length accepted normal-stop records (79.899%)**, below the frozen 85% scale gate;
- **147/199** normally stopped answers obeyed their explicit v3 answer budget;
- **52/199** exceeded that budget; and
- **0/40** student-overlength normal-stop records actually obeyed their budget.

The v3 causal conclusion is therefore stronger than a generic length correlation: the remaining student-envelope failures are budget-instruction compliance failures. The per-record budget arithmetic itself is not the blocker. Lowering the qualification gate or merely asking for a slightly smaller budget would move the target rather than fix the mechanism.

## Bounded v4 selective-compression contract

v4 preserves every v3 answer that already fits the student's 2,048-token envelope. It performs a second Qwen3.8 pass **only** for a v3 record when all of the following are true:

1. the v3 generation ended with `finish_reason=stop`;
2. the actual student-formatted record exceeds 2,048 tokens; and
3. the v3 final answer exceeded its frozen per-record answer budget.

For the observed v3-200 evidence this selects **40 records**, not all 200 and not the 12 budget violators that happened to fit because of the v3 reserve.

The v4 compression pass is intentionally a different task from first-pass solving:

- same pinned Qwen3.8-27B teacher and native BF16 runtime;
- same 16,384-token teacher context and 8,192-token generation cap;
- same sampling parameters and deterministic seed base;
- reasoning effort reduced to **`low`** because the problem has already been solved;
- the compression model receives the original student conversation, the already-generated v3 final answer, and the frozen target answer budget;
- it never receives hidden reasoning;
- it is instructed to preserve correctness, code behavior, required edge cases, and necessary instructions while removing repetition, unrequested alternatives, and nonessential commentary; and
- compressed outputs live in a new durable shard namespace and never overwrite first-pass evidence.

The v4 run identity binds the exact compression config, compression implementation, first-pass source run identity, selected input indices, and source final-answer SHA-256 values. Each compression row records source/target identity, original and compressed answer lengths, original and compressed full-record lengths, prompt/completion counts, finish reason, reasoning digest/character count, and original/compressed response SHA-256 values. Hidden reasoning text is never persisted.

After all selective compression shards are sealed, v4 writes a deterministic merged corpus. Teacher-only v3 prompt-policy metadata is removed from its active namespace before that corpus is diagnosed or finalized; historical policy identity and budget provenance remain under v4 metadata keys. The final pipeline still runs Python-quality filtering, the 2,048-token student filter, exact deduplication, deterministic splitting, and fail-closed protected-benchmark contamination checks.

The v4 scale gate remains deliberately strict:

1. normal-stop rate >= **0.90**;
2. student-length acceptance among normal stops >= **0.85**; and
3. protected-benchmark contamination status is **`clean`**.

Do not lower these gates after observing v4.

## Affected first v4-2000 successor and current salvage

The first completed v4-2000 successor is frozen scientific evidence but its old merged/final corpus is not valid student training data. The original parser assumed returned thinking text began with `<think>`, while the Qwen chat template could prefill that opening marker and vLLM could return `reasoning ... </think> final answer`. That allowed hidden reasoning plus the closing marker into the trainable assistant response.

The current salvage workflow does **not** regenerate the expensive 2,000 first-pass answers. It independently validates the sealed legacy generation evidence, strips reasoning using the corrected Qwen boundary parser, preserves only bounded reasoning provenance, recomputes selective-compression targets from the sanitized first-pass answers, and uses a fresh low-reasoning compression namespace for only the records that still require compression.

The original affected namespace remains immutable. The repaired output must live under:

```text
MyDrive/tiny-qwen-coder/distillation/qwen38-27b-v4-2000-salvage-v1/
```

See [`docs/V4_2000_REASONING_SALVAGE.md`](../../docs/V4_2000_REASONING_SALVAGE.md) for the frozen evidence and repair contract.

## Protected benchmark contamination is fail-closed

Teacher generation and compression never use protected benchmarks as input. Finalization loads protected examples only after generation and runs the existing exact/high-overlap contamination checker against the retained training corpus.

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

Each experimental generation, compression, or salvage pass must use its own immutable source archive and checkpoint namespace. Keep prior archives untouched. Historical notebooks use their own frozen ZIP names. The current v4-2000 salvage notebook expects its own salvage ZIP name and seals those exact bytes with an adjacent `.sha256` file on first use.

GitHub source ZIPs contain no `.git` directory. After extraction, the notebooks create a deterministic local Git commit solely so generic dataset-manifest provenance can record an exact source-tree identity. No remote is configured and no GitHub credentials are required.

## Why preserved checkpoints require identity-aware readers

Durable generation and compression shards bind exact implementation/config/input identity in addition to per-record fingerprints and checksum sidecars. Code repairs therefore must not rewrite old run identities or pretend a changed parser is the same implementation.

The salvage path intentionally uses an independent legacy-evidence validator rather than forcing the affected generation checkpoint through the corrected ordinary loader. It then creates a new sanitized source identity and new compression checkpoint identity.

Never edit `run-identity.json`, shard payloads, or checksum sidecars to force compatibility.

## Current recommended progression

1. Preserve v1, v2, v3, bounded-v4, and the affected first v4-2000 directories as immutable evidence.
2. **Do not train from the old `qwen38-27b-v4-2000/final/` directory.**
3. Freeze the exact repository ZIP for the current salvage revision under the filename expected by `qwen38_teacher_distillation_v4_2000_salvage_colab.ipynb`.
4. Start the **v4-2000 salvage notebook** from a fresh A100 80 GB allocation.
5. Validate every sealed legacy first-pass generation shard and reconstruct the deterministic reasoning-free 2,000-record source without loading Qwen3.8.
6. Confirm the frozen affected-run accounting before any new GPU generation.
7. Recompute the selective-compression target set from the sanitized answers; this computed target count is authoritative.
8. Generate/resume only the missing fresh low-reasoning compression shards in the new salvage namespace.
9. Diagnose, finalize, run fail-closed protected-benchmark contamination checks, and mechanically qualify the repaired merged corpus.
10. Require normal stops >=90%, student-length acceptance among normal stops >=85%, contamination `clean`, and zero surviving `<think>`/`</think>` markers.
11. Train/evaluate the Qwen3.5-4B student only if the repaired v4-2000 corpus qualifies.
12. Generate more than 2,000 teacher records only if that adapter improves the frozen base benchmark.

Do not jump directly to the full ~40k corpus.

## Recovery rules

- Treat `/content` as disposable and Google Drive as durable.
- Keep each frozen repository ZIP and its `.sha256` sidecar on Drive.
- Reuse the exact same sealed archive when resuming a checkpoint namespace.
- Recreate `/content/tqc-teacher-venv` after a fresh Colab allocation.
- Always run teacher scripts with the notebook's `$TQC_PYTHON`.
- Keep v1, v2, v3, bounded v4, affected v4-2000, repaired v4-2000 salvage, and later experiments in separate checkpoint directories.
- Never replace a code ZIP after generation, compression, or salvage has started in the namespace it governs.
- Never edit run identities, generated shard payloads, or checksum sidecars.
- An incomplete/unsealed shard is regenerated automatically when that workflow owns the shard.
- A sealed shard with a bad checksum fails closed as corruption.

There is intentionally no GitHub write workflow in Colab. Development, commits, pushes, and pulls happen outside the disposable GPU runtime.
