# P9-007 — Repaired Qwen3.8 V4-2000 distilled-data student study

## Status

P9-007A through P9-007D are complete. The repaired V4-2000 corpus was imported fail closed,
the 4070 Ti one-step smoke passed, one common 185-step trajectory completed, and the exact
four-checkpoint development evaluation was replayed cleanly after execution-capture hardening.

The clean replay is GitHub Actions run `34255423595`, attempt `2`, job `102160760346`, at source
Git SHA `00ef8ad3e2edf2e8e97636026eb66bc2fb7fd8f5`. It reproduced the provisional scores exactly
and mechanically selected **step 185**:

```text
step  25  HumanEval 32/45  MBPP 63/130  combined  95/175  ineligible
step  50  HumanEval 30/45  MBPP 66/130  combined  96/175  ineligible
step 100  HumanEval 31/45  MBPP 70/130  combined 101/175  ineligible
step 185  HumanEval 34/45  MBPP 70/130  combined 104/175  ELIGIBLE
```

The repository holdout remained untouched. Step 185 is the **only** checkpoint authorized to
consume the qualification slice. The clean replay is frozen in
`docs/evidence/P9_007_DEVELOPMENT_RUN_34255423595.json`.

P9-007E qualification has not yet been consumed. Its winner-only harness and workflow are being
frozen before execution. No runner-up may be evaluated after qualification is observed.

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
loss                 assistant-only SFT
learning rate        1e-5
scheduler            cosine
warmup ratio         0.03
seed                 1729
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

Complete. The exact repaired archive was independently validated and imported through the
fail-closed importer. The train/validation split is `1,479 / 78` and contamination is clean.

### P9-007B — 4070 Ti smoke

Complete. The one-step 4-bit QLoRA smoke completed with valid adapter-only output, finite training
and validation loss, clean corpus identity, and ample 16 GiB GPU headroom. Smoke evidence remained
non-promotable as required.

### P9-007C — Checkpointed one-epoch student trajectory

Complete in GitHub Actions run `34230186622`, job `102074065676`, source SHA
`d434ed274f2bcf56ca8e37863076c8937d30c83f`. The trajectory completed exactly `185` optimizer
steps and preserved adapter-only candidates at:

```text
25, 50, 100, 185
```

Final diagnostic training loss was `0.4845782418508787`; validation loss was
`0.48143240809440613`. Loss did not participate in checkpoint selection.

The exact step-185 identity is:

```text
artifact-set sha256 = 0462ed5e7dc08df76c1e72c58bf53d153ca9d74cc072a0724c62537307a476da
adapter-model sha256 = 2ee57b8c6fd10237e6e2faf11a9ff376fe7115f58062a8ed54eb962c3be6478a
```

### P9-007D — Development-only executable evaluation

Complete. The frozen P9-004 HumanEval/MBPP development membership was reused without
repartitioning:

```text
unchanged base  HumanEval 33/45  MBPP 70/130  combined 103/175
```

The precommitted gate remained:

1. combined development passes at least `104/175`;
2. HumanEval at least `33/45`;
3. MBPP at least `70/130`;
4. highest combined score wins, then fewer optimizer steps.

The first development run, `34243621029`, selected step 185 but emitted an uncaught background
output-capture exception. That run is preserved as provisional evidence rather than silently
accepted or discarded. The execution harness was then hardened to drain capture threads before
closing their streams and to propagate capture-thread errors fail closed.

The exact replay after that fix, run `34255423595` attempt `2` / job `102160760346`, completed every
verification step without the capture exception and reproduced all four scores exactly. Step 185
is the only eligible checkpoint. The complete replay artifact is:

```text
artifact id     = 10070646552
artifact digest = sha256:514c6be5de40d881a42c31bf7574ae61730254a036051ddf7e8603a4206ec286
```

The repository holdout was not evaluated in either development run.

### P9-007E — Winner-only one-shot qualification

Authorized but not yet consumed.

Only **step 185** may consume the untouched qualification membership:

```text
HumanEval qualification        119
MBPP qualification             370
repository holdout              11
combined qualification         500
```

The qualification harness must validate the clean P9-007D replay, the exact trajectory artifact,
and the exact step-185 adapter hashes before generating any protected response. It generates and
scores only those 500 qualification tasks. Missing transported responses, harness errors,
source/provenance drift, membership drift, or any runner-up request fail closed.

The result is combined with the frozen development score (`34/45`, `70/130`, `104/175`) to
reconstruct the full protected Python score. The frozen Phase-8 target-language promotion boundary
remains authoritative:

```text
HumanEval full          >= 128/164
MBPP full               >= 290/500
repository holdout      >=   6/11
combined full           >= 438/675
```

If the winner fails this target-language boundary, P9-007 stops and no runner-up is tested. If it
passes, only this same fixed winner may proceed through the remaining Phase-8 general-tool and
cross-language preservation gates before any final promotion decision.

## Scientific question

P9-007 asks one narrow question:

> Does replacing the harmful P0 training distribution with the repaired, filtered Qwen3.8-27B
> teacher corpus make a conservative Qwen3.5-4B LoRA update improve executable Python behavior?

The development result is the first positive signal in this project: the selected repaired-corpus
adapter improves the frozen development combined score by one pass while preserving the MBPP
baseline. That is not yet a final promotion result. The untouched one-shot qualification slice is
the next decision boundary.

A positive protected qualification plus preservation result would justify scaling teacher data
beyond 2,000 generated examples. A negative result means the next redesign should change the
distillation/verification objective rather than simply produce more samples under the same recipe.
