# P9-001 — Python LoRA rank sweep

## Status

Complete. CPU protocol validation, GPU smoke, full training, protected generation, direct scoring, independent evidence verification, and the comparative report are complete.

**Result: no fine-tuned rank improved over the unchanged Qwen3.5-4B base.** Rank 8 was the best fine-tuned candidate at `231/675` (`0.3422222222222222`) versus base `424/675` (`0.6281481481481481`), a deficit of `193` passes / `0.2859259259259259` micro pass rate.

Machine-readable closure evidence is frozen at `docs/evidence/P9_001_RANK_SWEEP_COMPARISON.json`.

## Objective

P9-001 isolates LoRA rank as the explicit configured experimental variable after Python P0 failed the Phase 8 promotion decision. The sweep compares:

- `r=8` — `language/python/p9-rank-r8`;
- `r=16` — the already-trained canonical P0 control, `language/python/p0`;
- `r=32` — `language/python/p9-rank-r32`; and
- `r=64` — `language/python/p9-rank-r64`.

The machine-readable protocol is `configs/train/python/p9_rank_sweep_v1.yaml`. The CPU-only validator in `tiny_qwen_coder.training.rank_sweep` fails closed unless the four ranks are exactly `8, 16, 32, 64` and every training field other than LoRA rank, adapter ID, and output path remains identical to P0.

## Frozen control variables

The following remain unchanged from `configs/train/python/p0.yaml` for every newly trained rank:

- base model and tokenizer identity;
- frozen Python P0 dataset manifest and exact 38,000/2,000 train/validation files;
- seed `1729`;
- 4-bit QLoRA with NF4, double quantization, and BF16 compute;
- sequence length `2048`;
- micro-batch size `1`;
- gradient accumulation `8`;
- one epoch / `4,750` expected optimizer steps;
- learning rate `2e-4`;
- cosine scheduler and `0.03` warmup ratio;
- gradient checkpointing;
- assistant-only loss;
- LoRA alpha `32`;
- LoRA dropout `0.05`;
- LoRA bias `none`; and
- the exact P7-001 12-module selective target set.

**Alpha is intentionally not scaled with rank.** This keeps the configured alpha field fixed, but it also means the effective `alpha/r` scaling changes with rank: `4` at r8, `2` at r16, `1` at r32, and `0.5` at r64. Consequently the observed quality trend must not be over-interpreted as a pure adapter-capacity effect.

Only these configured fields differ:

1. `lora.rank` — the explicit experimental variable;
2. `adapter_id` — required to keep artifacts distinct; and
3. `output_dir` — required to prevent artifact collisions.

## Why the existing `r=16` P0 run is reused

The canonical rank-16 control is P7-006 run `33422910444`, trained from source `02df92a9c2d347b9fb013dc25714fe066c6bcafe` with adapter ID `language/python/p0`.

A repository comparison from that source through the Phase 8 master state found no changes to the generic training implementation, `uv.lock`, the P0 training config, base-model config, Python P0 dataset pipeline/configuration, or the frozen selective LoRA target profile. Changes after P7-006 were evaluation, runtime-validation, documentation, and workflow/evidence plumbing. Re-running rank 16 would therefore spend roughly another full P0 training run without changing a training variable relevant to this sweep.

The rank-16 measured training reference remains:

- trainable parameters: `32,464,896`;
- training loss: `0.6934559548340345`;
- validation loss: `0.7679226398468018`;
- trainer runtime: `45,979.9559` seconds;
- total runtime: `47,050.625130466186` seconds;
- throughput: `0.826` samples/s and `0.103` steps/s;
- peak allocated VRAM: `9,119,821,824` bytes; and
- peak reserved VRAM: `13,413,384,192` bytes.

## Completed GPU training evidence

The one-shot full-training acceptance run `33587975474` completed successfully from source `1aca2f2a778ade4cc300d66353a26206900fc84b`. All three jobs passed rank-only protocol validation, mandatory training preflight, exactly `4,750` optimizer steps, fail-closed artifact verification, and compact evidence upload.

| Rank | Trainable parameters | Train loss | Validation loss | Trainer runtime (s) | Samples/s | Steps/s | Peak allocated VRAM (bytes) | Peak reserved VRAM (bytes) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 16,232,448 | 0.6962825055122376 | 0.772124171257019 | 46,202.2761 | 0.822 | 0.103 | 9,022,427,136 | 13,270,777,856 |
| 16 | 32,464,896 | 0.6934559548340345 | 0.7679226398468018 | 45,979.9559 | 0.826 | 0.103 | 9,119,821,824 | 13,413,384,192 |
| 32 | 64,929,792 | 0.6916291663803552 | 0.7658730745315552 | 46,955.3457 | 0.809 | 0.101 | 9,314,611,200 | 13,675,528,192 |
| 64 | 129,859,584 | 0.6906089604214618 | 0.7644960880279541 | 47,539.25 | 0.799 | 0.100 | 9,713,620,992 | 11,198,791,680 |

The rank-64 lower *reserved* peak than rank 32 is an allocator observation, not evidence that rank 64 uses less model memory: peak allocated memory rises monotonically with rank. Quality selection must not be based on loss or allocator reservation alone.

The completed adapter identities were frozen separately in `configs/eval/python/p9_rank_sweep_candidates_v1.yaml` before protected evaluation. The new runs have a different raw dataset-manifest file SHA because the manifest records the new preparation Git SHA, but the actual corpus identity is unchanged: dataset-config SHA `4f9663e72b22d81ce8975e6f6ed87ee7457d3bef0a08fe211e700dd5ea12fbff` and split-membership SHA `78559765eac305528e5ba96ae3dae04feffda39fc4727d450d602f6e68697428` match the P0 control, as do source revisions and accepted/rejected counts.

## Execution sequence

### 1. CPU protocol validation

`python -m tiny_qwen_coder.training.rank_sweep --repo-root .` verifies the frozen protocol SHA, frozen fixed-training payload SHA, candidate identities, smoke bounds, and rank-only invariants before any GPU work starts.

### 2. One-step GPU smoke

`.github/workflows/python-p9-rank-sweep-smoke.yml` runs one optimizer step for ranks 8, 32, and 64 with `max-parallel: 1` on the self-hosted GPU. Rank 16 is not rerun. Each smoke must produce finite train/validation loss and positive, internally consistent VRAM measurements.

The rank-64 smoke proved that the largest candidate fit the reference GPU before full training.

### 3. Full training

`.github/workflows/python-p9-rank-sweep-training.yml` trained ranks 8, 32, and 64 from one exact source commit, again with `max-parallel: 1`. Each job:

- used the frozen P0 corpus;
- completed exactly `4,750` optimizer steps;
- preserved the frozen rank protocol;
- emitted an unmerged PEFT LoRA adapter;
- recorded finite loss/runtime/throughput metrics;
- recorded positive allocated/reserved VRAM;
- recorded the measured trainable parameter count; and
- independently rehashed the persisted evidence inventory.

This stage is complete via run `33587975474`.

### 4. Protected Python generation and scoring

Protected GPU generation run `33809220955` generated exactly `675` responses for each of ranks 8, 32, and 64 from source `0d764300ebb76dbd1273f6f9f831ac9b57c59d0c`. All three generation artifacts were sealed before scoring.

The first scoring attempt correctly failed closed because a GitHub-hosted runner pulled the pinned execution image into Docker while evaluator runtime discovery selected Podman. No model score was accepted from that failed attempt.

At the user's explicit infrastructure boundary, P9 scoring was then changed to execute generated benchmark programs directly inside the already-containerized self-hosted runner rather than nesting Docker or Podman. The direct backend is explicit, never an automatic fallback. It uses a fresh temporary working directory, a minimal credential-free child environment, bounded input/output, wall-clock timeout, process-group cleanup, and Linux resource limits. It **does not** claim OCI-equivalent filesystem or network isolation; the harness does not enforce network isolation.

Direct scoring run `33842212951`, source `7bff3afa49c37be9f235db1ed64c87d679d1c378`, reused the exact sealed generation artifacts and the accepted P6-005 unchanged-base evidence. All three score jobs completed, independently verified persisted evidence, checked generation/scoring provenance, checked the `675`-problem cardinality and frozen `424/675` base reference, and uploaded final evaluation artifacts.

Rank 16 reuses canonical P8-001 evaluation run `33538724658`. The unchanged-base and rank-16 reference scores are the previously accepted canonical OCI-scored evidence; the direct-vs-OCI execution-backend difference is recorded in the machine-readable P9 evidence and is not represented as equivalent isolation.

## Comparative protected results

| Model | HumanEval | MBPP | Repo holdout | Combined | Combined rate | Delta vs base |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| unchanged base | **128/164** | **290/500** | **6/11** | **424/675** | **62.81%** | — |
| r8 | 91/164 | 139/500 | 1/11 | **231/675** | **34.22%** | -193 / -28.59 pp |
| r16 / P0 | 88/164 | 97/500 | 2/11 | 187/675 | 27.70% | -237 / -35.11 pp |
| r32 | **94/164** | 34/500 | 1/11 | 129/675 | 19.11% | -295 / -43.70 pp |
| r64 | 88/164 | 36/500 | **3/11** | 127/675 | 18.81% | -297 / -44.00 pp |

Bold values among fine-tuned rows indicate the best observed fine-tuned result in that column; the unchanged base still leads every protected suite and the combined score.

The fine-tuned ranking by combined protected passes is:

1. `r8`: `231/675`;
2. `r16/P0`: `187/675`;
3. `r32`: `129/675`;
4. `r64`: `127/675`.

The unchanged base remains far ahead at `424/675`.

Notably, validation loss improves monotonically from r8 through r64 while protected benchmark quality does not. Rank 32 has the best fine-tuned HumanEval score (`94/164`) and rank 64 has the best fine-tuned holdout score (`3/11`), yet both collapse on MBPP (`34/500` and `36/500`). This is direct evidence that training/validation loss is not a suitable selection metric for this experiment.

## Interpretation

P9-001 rules out “increase LoRA rank” as the next remedy for this frozen recipe. The best fine-tuned candidate, r8, still loses `193` of `675` protected problems relative to base and regresses on every protected suite. Larger ranks become worse overall, driven particularly by MBPP degradation.

This does **not** prove that rank alone causes the degradation because fixed alpha means `alpha/r` co-varies with rank. More importantly, every tested rank remains substantially below base, which points toward the training recipe/data interaction rather than insufficient adapter capacity as the primary issue to investigate.

The next high-value experiments should therefore focus on variables such as learning rate/training length, target-module scope, and training-data quality/mixing rather than simply allocating more LoRA rank. P9-004 is especially relevant because the current `2e-4`, one-epoch recipe may be too aggressive for an already strong base model.

## Promotion boundary

P9-001 is an experiment, not a promotion decision. The Phase 8 `python-promotion-v1` policy remains authoritative. The reused P0 dataset manifest records contamination status `not_run`; consequently no rank-sweep candidate can be marked recommended merely because it wins this rank comparison. Promotion eligibility still requires clean contamination evidence plus all target-language and preservation gates.

No P9-001 candidate reaches even the simpler unchanged-base comparison boundary of `>424/675`, much less the frozen promotion requirement of at least `438/675` combined with suite preservation and the remaining gates.
