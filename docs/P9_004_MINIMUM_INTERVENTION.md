# P9-004 — Checkpointed low-LR minimum-intervention study

## Status

Protocol redesigned and frozen. GPU training, development scoring, checkpoint selection, and one-shot qualification remain pending.

The machine-readable protocol is `configs/train/python/p9_minimum_intervention_v1.yaml`. CPU validation is provided by `python -m tiny_qwen_coder.training.minimum_intervention`.

## Why P9-004 changed

P9-001 showed that increasing LoRA rank does not repair Python quality. The unchanged Qwen3.5-4B base scored `424/675` (`62.81%`), while r8/r16/r32/r64 scored `231/675`, `187/675`, `129/675`, and `127/675`. At the same time, validation loss improved monotonically as protected coding quality deteriorated.

That pattern is evidence that the optimizer is successfully fitting the P0 training distribution while moving the model away from the executable-coding behavior we care about. P9-004 therefore asks a narrower question than the original TODO item:

> Is there a small-update regime in which the Python adapter changes the model just enough to improve coding behavior without displacing the strong base distribution?

Training/validation loss is descriptive evidence only. It cannot select the winner.

## Fixed training lineage

P9-004 uses the completed P9-001 r8 configuration as the structural control because r8 was the least damaging fine-tuned rank. Every trajectory keeps the following fixed:

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

Only learning rate, adapter identity, and output path may differ across trajectories. The CPU validator fails closed on any other config drift.

## Training grid

Five trajectories start independently from the exact unchanged base and run to a common horizon of exactly `1,000` optimizer steps:

| Trajectory | Learning rate |
| --- | ---: |
| `lr-1e-5` | `1e-5` |
| `lr-2e-5` | `2e-5` |
| `lr-5e-5` | `5e-5` |
| `lr-1e-4` | `1e-4` |
| `lr-2e-4` | `2e-4` (current recipe control) |

Each trajectory exposes adapter snapshots at exactly:

`50, 100, 250, 500, 1000` optimizer steps.

This produces 25 candidate checkpoints from five training trajectories while keeping the scheduler horizon identical. We intentionally do **not** implement the step curve as separate max-step runs because changing `max_steps` would also change the cosine schedule horizon and confound the step comparison.

If none of the first 1,000-step checkpoints clears the development gate, the study stops. Longer training is not automatically authorized.

## Benchmark hygiene: development versus qualification

Repeatedly selecting checkpoints on all 675 protected problems would turn the protected benchmark into a tuning set. P9-004 therefore freezes a deterministic partition before any checkpoint score is inspected.

For HumanEval and MBPP, task membership is computed from:

`sha256("python-p9-004-development-v1\\0" + suite + "\\0" + task_id)`

The first 64 digest bits are interpreted as an unsigned integer. A task is in development when `value mod 4 == 0`; otherwise it remains qualification. This consumes approximately 25% of HumanEval/MBPP for development and leaves approximately 75% untouched for qualification.

The repository holdout is **qualification-only**. None of its 11 tasks may be used for checkpoint or learning-rate selection.

The split is deterministic and score-independent. Changing the salt, modulus, remainder, suite membership, or task IDs after scoring is protocol drift.

## Development checkpoint selection

All 25 snapshots may be evaluated on the frozen development slice. The unchanged base is evaluated once on that same slice.

Selection is precommitted:

1. Primary metric: total development problems passed across the development HumanEval + MBPP slice.
2. A candidate is eligible only if it strictly improves combined development passes over base.
3. A candidate is ineligible if either development suite regresses versus base.
4. Ties are broken by fewer optimizer steps, then lower learning rate.
5. Validation loss, training loss, VRAM, or throughput cannot override executable pass/fail results.

If no checkpoint is eligible, P9-004 stops as a negative experiment and the qualification slice remains unopened.

## One-shot qualification

Only the single development-selected checkpoint may be evaluated on the untouched qualification slice. Qualification is one-shot.

The selected checkpoint passes the P9-004 quality gate only if:

- combined qualification passes strictly exceed the unchanged base on the same tasks;
- HumanEval qualification does not regress;
- MBPP qualification does not regress; and
- the full repository holdout does not regress.

A qualification failure is preserved as negative evidence. We do not evaluate the runner-up checkpoint on qualification after observing failure.

A qualification success still does **not** automatically promote the adapter. The P0 corpus contamination state remains `not_run`; general/tool preservation, cross-language behavior, contamination, adapter integrity, and an updated promotion protocol remain separate gates.

## Execution plan

### P9-004A — Freeze/validate protocol

- freeze the five LR configs and 1,000-step common horizon;
- freeze the five snapshot steps;
- freeze the development/qualification partition rule;
- freeze selection and qualification gates;
- CPU-validate that only learning rate/identity/output vary.

### P9-004B — Implement trajectory snapshot training

- train one r8 trajectory per learning rate on the self-hosted GPU runner;
- use a common 1,000-step scheduler horizon;
- persist adapter-only snapshots at steps 50/100/250/500/1000;
- record exact source/config/dataset/runtime identities and artifact hashes;
- do not create a promotable/recommended adapter from this stage.

### P9-004C — Materialize development/qualification membership

- enumerate exact HumanEval and MBPP task IDs under the frozen hash rule;
- preserve the complete 11-task repository holdout for qualification;
- freeze membership manifests and counts before model scoring.

### P9-004D — Development evaluation and preselection

- evaluate unchanged base once on development;
- evaluate all 25 snapshots on development only;
- apply the frozen selection rule mechanically;
- stop without qualification if no candidate beats/preserves base.

### P9-004E — One-shot qualification

- evaluate only the selected checkpoint against unchanged base on qualification;
- independently verify evidence and exact membership;
- record pass/fail without trying another checkpoint after seeing qualification results.

## What this experiment can tell us

If a very early/low-LR checkpoint matches or beats base while later or higher-LR checkpoints degrade, the primary failure mode is excessive intervention and P9 can focus on preserving that narrow useful regime.

If all 25 development checkpoints are below base, even at `1e-5` and 50 steps, that is strong evidence that the P0 data/objective itself is directionally harmful. The next experiment should then prioritize dataset quality/semantic verification rather than more LR, rank, or epoch sweeps.

If development improves but one-shot qualification fails, the development signal did not generalize. That result argues for a better development corpus and prevents us from silently overfitting the existing benchmark.
