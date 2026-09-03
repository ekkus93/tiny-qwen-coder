# P9-001 — Python LoRA rank sweep

## Status

CPU protocol validation, GPU smoke, and full training are complete. Protected Python evaluation and the final comparative report remain pending.

## Objective

P9-001 isolates LoRA rank as a single experimental variable after Python P0 failed the Phase 8 promotion decision. The sweep compares:

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

**Alpha is intentionally not scaled with rank.** Scaling alpha would change a second optimization variable and make the experiment no longer a rank-only sweep.

Only these fields may differ:

1. `lora.rank` — the experimental variable;
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

The completed adapter identities are frozen separately in `configs/eval/python/p9_rank_sweep_candidates_v1.yaml` before protected evaluation. The new runs have a different raw dataset-manifest file SHA because the manifest records the new preparation Git SHA, but the actual corpus identity is unchanged: dataset-config SHA `4f9663e72b22d81ce8975e6f6ed87ee7457d3bef0a08fe211e700dd5ea12fbff` and split-membership SHA `78559765eac305528e5ba96ae3dae04feffda39fc4727d450d602f6e68697428` match the P0 control, as do source revisions and accepted/rejected counts.

## Execution sequence

### 1. CPU protocol validation

`python -m tiny_qwen_coder.training.rank_sweep --repo-root .` verifies the frozen protocol SHA, frozen fixed-training payload SHA, candidate identities, smoke bounds, and rank-only invariants before any GPU work starts.

### 2. One-step GPU smoke

`.github/workflows/python-p9-rank-sweep-smoke.yml` runs one optimizer step for ranks 8, 32, and 64 with `max-parallel: 1` on the self-hosted GPU. Rank 16 is not rerun. Each smoke must produce finite train/validation loss and positive, internally consistent VRAM measurements.

The rank-64 smoke is particularly important because it proves that the largest candidate fits the reference GPU before a full run is authorized.

### 3. Full training

`.github/workflows/python-p9-rank-sweep-training.yml` trains ranks 8, 32, and 64 from one exact source commit, again with `max-parallel: 1`. Each job must:

- use the frozen P0 corpus;
- complete exactly `4,750` optimizer steps;
- preserve the exact rank-only protocol;
- emit an unmerged PEFT LoRA adapter;
- record finite loss/runtime/throughput metrics;
- record positive allocated/reserved VRAM;
- record the measured trainable parameter count; and
- independently rehash the persisted evidence inventory.

This stage is complete via run `33587975474`.

### 4. Protected Python evaluation

The exact completed adapter hashes, sizes, run IDs, training SHAs, artifact IDs, and artifact digests are frozen in `configs/eval/python/p9_rank_sweep_candidates_v1.yaml`. Ranks 8, 32, and 64 will be evaluated with the same protected HumanEval, MBPP, and repository-holdout settings used for P0 and the unchanged base. Rank 16 reuses canonical P8-001 evaluation run `33538724658`.

The sweep comparison must not change scoring after observing outputs.

### 5. Comparative report

P9-001 closes only after a report compares, at minimum, for ranks 8/16/32/64:

- trainable parameters;
- training and validation loss;
- training runtime and throughput;
- peak allocated/reserved VRAM;
- HumanEval pass@1;
- MBPP pass@1;
- repository-holdout pass@1; and
- combined protected Python pass rate and delta versus unchanged base/P0.

Training loss alone cannot select a winner.

## Promotion boundary

P9-001 is an experiment, not a promotion decision. The Phase 8 `python-promotion-v1` policy remains authoritative. The reused P0 dataset manifest records contamination status `not_run`; consequently no rank-sweep candidate can be marked recommended merely because it wins this rank comparison. Promotion eligibility still requires clean contamination evidence plus all target-language and preservation gates.
