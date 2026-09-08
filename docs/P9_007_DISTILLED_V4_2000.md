# P9-007 — Repaired Qwen3.8 V4-2000 distilled-data student study

## Status

P9-007A is complete: the repaired V4-2000 teacher corpus has been exported from Google
Drive, independently revalidated, and wired into a fail-closed local import path. GPU smoke and
student training have not started yet.

P9-004 established that additional rank/LR/epoch search on the original P0 corpus is not a
productive direction. The unchanged Qwen3.5-4B base scored `103/175` on the frozen development
slice, while the least-damaging P9-004 checkpoint scored `100/175`. P9-007 therefore changes the
training distribution rather than adding another P0 hyperparameter point.

## Frozen repaired corpus

Canonical durable source namespace:

```text
MyDrive/tiny-qwen-coder/distillation/qwen38-27b-v4-2000-salvage-v1/
```

The Google Drive export inspected for P9-007A has:

```text
export archive sha256   = 6628a83f2f0ae87d11610d28edd177d6c58520c00ebe1bc987771de10205029b
source output sha256    = 7f07f7253e98bf8ed295b12d72ebdf03c44b2e08a8d9f74aaa02dce5b760d966
dataset manifest sha256 = 7e299de17cbb62bff3cf5f50c61ad6ad30ca571c9297b9935ff1307cd52e0eb7
qualification sha256    = 6c9ca0737ada462d408d3fb001bc964f37bf7125c78fd504f727d63d2aeeedef
finalization sha256     = 004551219fa2b16c503152ee0936bd7aa2da19e3e18ee5a2b30e1353a40b3217
```

The export passed the following independent checks outside Colab:

- every checksum sidecar in the bundle matched its payload;
- the final dataset-manifest sidecar matched the exact manifest bytes;
- the manifest's unique-corpus, train, validation, and split-membership hashes were
  independently recomputed from the JSONL content;
- all `1,557` accepted normalized-record hashes are unique;
- train and validation have zero exact normalized-record overlap;
- accepted is exactly the union of train and validation;
- the qualification decision is `qualified: true` with contamination `clean`;
- no literal `<think>` or `</think>` marker occurs in accepted/train/validation.

Final attrition is:

| Boundary | Records |
| --- | ---: |
| repaired merged candidates | 2,000 |
| normal-stop candidates after selective compression | 1,982 |
| finish-reason rejected | 18 |
| Python-quality rejected | 392 |
| pre-pipeline accepted | 1,590 |
| student-length rejected | 33 |
| prepared unique | 1,557 |
| train | 1,479 |
| validation | 78 |

The source salvage first pass had `1,983` normal stops and `17` length completions. One of the
`234` selective-compression targets completed by length, so the repaired merged corpus has
`1,982` normal stops and `18` length completions. This remains far above the frozen 90% normal-stop
qualification gate.

## Local import contract

The generated data stays outside Git. Import the exact exported bundle with:

```bash
uv run --frozen python scripts/import_v4_2000_salvage.py \
  --archive /path/to/qwen38-27b-v4-2000-salvage-v1-20260908T103445Z-1-001.zip
```

The default immutable destination is:

```text
data/python/qwen38-27b-v4-2000-salvage-v1/
```

The importer validates the full bundle before copying the training-critical evidence. It refuses
to overwrite an existing destination. The imported directory contains the frozen manifest,
accepted/train/validation JSONL, qualification evidence, source-run identity, teacher-finalization
summary, and an `import-receipt.json` recording the exact source archive SHA-256 used locally.

Do not import or train from the old `qwen38-27b-v4-2000/final/` directory.

## Student structural control

P9-007 uses the least-damaging P9-004 structure as the initial student control:

```text
base                 Qwen/Qwen3.5-4B (frozen repository revision)
training mode        4-bit QLoRA
LoRA rank            8
LoRA alpha           32
LoRA dropout         0.05
LoRA targets         frozen selective target profile
sequence length      2,048
micro batch          1
grad accumulation    8
effective batch      8
loss                  assistant-only SFT
learning rate        1e-5
scheduler             cosine
warmup ratio          0.03
seed                   1729
```

Canonical config:

```text
configs/train/python/p9_distilled_v4_2000_r8_lr1e5.yaml
```

This keeps the initial optimizer/adapter intervention conservative so the dominant changed
variable relative to the P9-004 low-LR control is the repaired teacher-distilled training
distribution.

## Frozen execution sequence

### P9-007A — Freeze and import repaired corpus

Complete.

### P9-007B — 4070 Ti smoke

Run the existing bounded GPU smoke machinery against:

```text
configs/train/python/p9_distilled_v4_2000_r8_lr1e5_smoke.yaml
```

Acceptance requires the normal memory/preflight invariants, one completed optimizer step, valid
adapter-only output, and no dataset-integrity failure. Smoke artifacts are non-promotable.

### P9-007C — Checkpointed one-epoch student trajectory

Do not run an uncheckpointed full experiment first. Freeze one r8 / `1e-5` trajectory with a
`185` optimizer-step horizon, corresponding to one pass over `1,479` training records at effective
batch size `8`. Preserve development candidates at steps:

```text
25, 50, 100, 185
```

The exact training harness for this trajectory must be committed and CPU-green before the GPU run.
Training loss and validation loss are diagnostic only and cannot select a checkpoint.

### P9-007D — Development-only executable evaluation

Reuse the already-frozen P9-004 HumanEval/MBPP development membership. Do not repartition the
protected benchmarks.

Unchanged-base development control:

```text
HumanEval  33/45
MBPP       70/130
combined  103/175
```

A P9-007 checkpoint is eligible for one-shot qualification only when all are true:

1. combined development passes are at least `104/175`;
2. HumanEval is at least `33/45`;
3. MBPP is at least `70/130`.

Select mechanically by combined passes, then fewer optimizer steps as the first tie-breaker. If no
checkpoint clears the gate, stop without qualification and diagnose the dataset/objective.

### P9-007E — Winner-only one-shot qualification

Only the single development-selected checkpoint may consume the frozen P9-004 qualification slice,
including the complete repository holdout. Reuse the Phase 8 promotion/preservation requirements;
do not try a runner-up after observing qualification results.

## Scientific question

P9-007 asks one narrow question:

> Does replacing the harmful P0 training distribution with the repaired, filtered Qwen3.8-27B
> teacher corpus make a conservative Qwen3.5-4B LoRA update improve executable Python behavior?

A positive result justifies scaling teacher data beyond 2,000 generated examples. A negative result
means the next redesign should change the distillation/verification objective rather than simply
produce more samples under the same recipe.
