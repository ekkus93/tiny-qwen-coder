# P9-004 — Checkpointed low-LR minimum-intervention study

## Status

Complete as a negative experiment. The frozen protocol, five GPU training trajectories, exact 25-checkpoint registry, deterministic development/qualification membership, development-only generation/scoring, and mechanical checkpoint selection all completed successfully.

No checkpoint cleared the precommitted development gate. Therefore no qualification checkpoint was selected, the qualification-only repository holdout was not evaluated, and P9-004 authorizes no P9-004E qualification run.

Canonical development-selection evidence is frozen in `docs/evidence/P9_004_MINIMUM_INTERVENTION_DEVELOPMENT.json`.

## Why P9-004 changed

P9-001 showed that increasing LoRA rank does not repair Python quality. The unchanged Qwen3.5-4B base scored `424/675` (`62.81%`), while r8/r16/r32/r64 scored `231/675`, `187/675`, `129/675`, and `127/675`. At the same time, validation loss improved monotonically as protected coding quality deteriorated.

That pattern is evidence that the optimizer is successfully fitting the P0 training distribution while moving the model away from the executable-coding behavior we care about. P9-004 therefore asked a narrower question:

> Is there a small-update regime in which the Python adapter changes the model just enough to improve coding behavior without displacing the strong base distribution?

The answer from the completed study is **no for the frozen P0 corpus/objective under the tested r8 regime**. Even the lowest LR and earliest checkpoint was already below base on the development benchmark.

Training/validation loss remains descriptive evidence only. It did not select the winner.

## Fixed training lineage

P9-004 uses the completed P9-001 r8 configuration as the structural control because r8 was the least damaging fine-tuned rank. Every trajectory kept the following fixed:

- exact `Qwen/Qwen3.5-4B` model/tokenizer revision;
- frozen Python P0 dataset and exact train/validation membership;
- seed `1729`;
- 4-bit QLoRA with NF4, double quantization, and BF16 compute;
- sequence length `2048`;
- micro-batch `1`, gradient accumulation `8`, effective batch `8`;
- cosine scheduler;
- `0.03` warmup ratio under the pinned Transformers v5 unified warmup API;
- gradient checkpointing;
- assistant-only loss;
- LoRA rank `8`, alpha `32`, dropout `0.05`, bias `none`;
- the exact 12-module selective target set; and
- training-record order and seed.

Only learning rate, adapter identity, and output path differed across trajectories. CPU validation fails closed on any other config drift.

## Training grid

Five trajectories started independently from the exact unchanged base and ran to a common horizon of exactly `1,000` optimizer steps:

| Trajectory | Learning rate |
| --- | ---: |
| `lr-1e-5` | `1e-5` |
| `lr-2e-5` | `2e-5` |
| `lr-5e-5` | `5e-5` |
| `lr-1e-4` | `1e-4` |
| `lr-2e-4` | `2e-4` (P9-001 recipe control) |

Each trajectory exposed adapter-only snapshots at exactly:

`50, 100, 250, 500, 1000` optimizer steps.

This produced 25 candidate checkpoints from five training trajectories while keeping the scheduler horizon identical. We intentionally did not implement the step curve as separate max-step runs because changing `max_steps` would also change the cosine schedule horizon and confound the step comparison.

## Benchmark hygiene: development versus qualification

Repeatedly selecting checkpoints on all 675 protected problems would turn the protected benchmark into a tuning set. P9-004 therefore froze a deterministic partition before any checkpoint score was inspected.

For HumanEval and MBPP, task membership is computed from:

`sha256("python-p9-004-development-v1\\0" + suite + "\\0" + task_id)`

The first 64 digest bits are interpreted as an unsigned integer. A task is in development when `value mod 4 == 0`; otherwise it remains qualification.

The resulting development slice is exactly:

- HumanEval: `45` tasks;
- MBPP: `130` tasks;
- combined: `175` tasks.

The qualification slice keeps the remaining `119` HumanEval and `370` MBPP tasks. The repository holdout remains entirely qualification-only: `0` development / `11` qualification.

Frozen identities:

- development manifest SHA-256: `260682d773640b28673357c7a441474656bbaffc67f0b5fcbe45eca3a07283de`;
- membership SHA-256: `c8765b7a3a69f134be4066e274a64640a15f2d05858374035432207b3521498b`;
- 25-checkpoint registry SHA-256: `31362b22f3f84fe05a6c499accf63a23b65295b337c3c9d614568732c0643196`.

## Development checkpoint selection

The unchanged base scored:

- HumanEval development: `33/45`;
- MBPP development: `70/130`;
- combined development: `103/175`.

Selection was precommitted:

1. Primary metric: total development problems passed across HumanEval + MBPP.
2. A candidate is eligible only if it strictly improves combined development passes over base, therefore at least `104/175`.
3. A candidate is ineligible if HumanEval falls below `33/45` or MBPP falls below `70/130`.
4. Ties are broken by fewer optimizer steps, then lower learning rate.
5. Validation loss, training loss, VRAM, or throughput cannot override executable pass/fail results.

### Completed development results

| LR | Step 50 | Step 100 | Step 250 | Step 500 | Step 1000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `1e-5` | **100** | 95 | 94 | 88 | 89 |
| `2e-5` | 91 | 88 | 79 | 79 | 84 |
| `5e-5` | 88 | 83 | 80 | 76 | 80 |
| `1e-4` | 82 | 80 | 78 | 78 | 77 |
| `2e-4` | 78 | 79 | 72 | 78 | 71 |

Values are combined passes out of `175`.

The best observed checkpoint was `lr-1e-5 @ step 50`:

- HumanEval: `33/45`, delta `0` vs base;
- MBPP: `67/130`, delta `-3` vs base;
- combined: `100/175`, delta `-3` vs base.

It therefore failed both the strict-combined-improvement gate and the MBPP-preservation gate. Every other checkpoint scored still lower on combined development passes.

The mechanical selector produced:

- `selected: null`;
- `qualification_authorized: false`;
- `repository_holdout_evaluated: false`.

Canonical development evaluation run: `33950802529`, source SHA `e524f6b3a8e60440b8869d3800de4e01188e7240`.

Selection evidence:

- selection SHA-256: `cc4bd3d89f8b9e9e1ebf481509c6b97ec50de7263e954247b68c30899941bc6c`;
- GitHub artifact ID: `9967947428`;
- artifact archive SHA-256: `94975fe9211ef63f807f8f5980f08280246fe070b14a33ed1bd89363c2dc0080`.

All five trajectory jobs and the final 25-way selection job completed successfully. Exact-head CPU CI `33950804231` also passed formatting, Ruff, mypy, pytest, and whitespace checks.

## One-shot qualification

One-shot qualification is **not authorized** because there is no development-selected checkpoint.

The untouched qualification slice, including the complete 11-task repository holdout, must remain unopened for P9-004. Running a runner-up or the least-bad checkpoint on qualification after observing these development results would violate the precommitted protocol.

## Execution closure

### P9-004A — Freeze/validate protocol

Complete.

### P9-004B — Implement trajectory snapshot training

Complete. Five r8 trajectories were trained to the common 1,000-step horizon and exact adapter-only snapshots were preserved at steps 50/100/250/500/1000.

### P9-004C — Materialize development/qualification membership

Complete. Exact HumanEval/MBPP membership is frozen and the repository holdout remains qualification-only.

### P9-004D — Development evaluation and preselection

Complete. All 25 snapshots were generated and scored on development only, all score identities were verified, and the frozen selector selected no checkpoint.

### P9-004E — One-shot qualification

Not run and not authorized. This is the correct protocol outcome, not unfinished work.

## Interpretation

The study isolates intervention strength much more tightly than P9-001. The lowest tested learning rate was `20x` below the P9-001 control LR, and the earliest checkpoint was only 50 optimizer steps into a common 1,000-step schedule. Yet that checkpoint was already below base, with the loss concentrated in MBPP.

The broader trajectory is also directionally consistent: increasing LR and/or training duration generally worsens executable development performance. There are small non-monotonic fluctuations, but no checkpoint approaches the `104/175` eligibility threshold, and none improves the unchanged base.

This is strong evidence that the dominant problem is not simply rank, learning rate, or excessive training duration. Under the frozen P0 corpus and assistant-only SFT objective, the update direction itself is harmful to the target executable-coding distribution.

Per the precommitted protocol, P9 should therefore **stop additional P0 rank/LR/epoch searching** and prioritize dataset/objective redesign. The next useful experiment should improve semantic training-sample quality and alignment with executable correctness—for example verified problem/solution/test records, capability-gap mining against base failures, rejection-sampled teacher distillation, or another execution-aligned objective—rather than adding more hyperparameter points around the same P0 data distribution.

The P0 corpus contamination state is still `not_run`; this study does not change that status.
