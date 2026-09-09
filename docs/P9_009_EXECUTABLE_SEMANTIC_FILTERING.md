# P9-009 — Executable semantic filtering of repaired teacher data

## Status

P9-009 begins after P9-007 failed winner-only qualification and P9-008 localized the loss to
semantic MBPP regressions. P9-009 changes the **training-data acceptance objective**, not the
student rank/learning-rate/epoch search space.

No P9-009 training is authorized by this implementation alone. The first live result is a survivor
census over the already-repaired 1,557 examples.

## Why this experiment exists

P9-007 step 185 preserved full HumanEval (`128/164`) and repository holdout (`6/11`) but reduced
MBPP from the unchanged base's `290/500` to `266/500`. P9-008 compared the frozen qualification
results task-by-task:

```text
base pass -> step-185 pass     179
base fail -> step-185 fail     133
base pass -> step-185 fail      41
base fail -> step-185 pass      17
net MBPP change                -24
```

Forty of the 41 regressions reached executable tests and failed semantically; only one was a
parse/truncation failure. Another rank/LR sweep around the same teacher-answer acceptance rule is
therefore not justified.

## Frozen source boundary

P9-009 starts only from the repaired P9-007 corpus:

```text
source manifest sha256  7e299de17cbb62bff3cf5f50c61ad6ad30ca571c9297b9935ff1307cd52e0eb7
source output sha256    7f07f7253e98bf8ed295b12d72ebdf03c44b2e08a8d9f74aaa02dce5b760d966
accepted                1557
train                   1479
validation                78
```

The original train/validation membership is preserved. Filtering may remove records but may not
reshuffle survivors between partitions.

The teacher-answer ceiling remains the same repaired 2,000-example study. P9-009 does **not**
generate another 2,000 candidate answers and does not scale teacher data.

## Independent contract design

For every repaired candidate answer, a second teacher call creates an executable behavioral
contract. It uses the same pinned Qwen3.8-27B model but a separate verifier policy and seed `2718`.
The candidate answer is **never included in the verifier messages**. Only its SHA-256 is retained in
provenance so the resulting contract can be bound to the frozen candidate later.

A verifier response is either explicitly unverifiable or supplies:

- one exact public function/class entry point;
- a deterministic reference implementation;
- at least eight explicit assertions, including edge cases;
- no file, network, subprocess, interactive input, randomness, or wall-clock dependencies.

The verifier contract is not trusted merely because it is valid Python. On the self-hosted runner,
the reference implementation must first pass its own assertions. Only then are those same tests
run against the frozen candidate answer. Unverifiable tasks, invalid contracts, failing reference
solutions, ambiguous candidate code extraction, and failing candidates are rejected.

The candidate and contract are separate generations: the contract generator never grades the
response it produced during P9-007.

## Execution boundary

Contract **generation** is intended for the A100 Colab workflow and may be resumed from durable
shards. Generated code is not executed there while Google Drive is mounted.

Contract **execution** occurs later on the containerized self-hosted runner with the repository's
explicit reduced-isolation execution harness. Each process receives a fresh workspace, stripped
environment, resource limits, output bounds, and a timeout. Harness failures fail the whole census;
they are not silently counted as bad examples.

## Sequence

### P9-009A — Freeze verifier implementation and protocol

- add candidate-hidden contract-input construction;
- freeze `python-semantic-contract-v1` and seed `2718`;
- add strict JSON/static validation and candidate extraction;
- add reference-first executable verification;
- freeze source/split provenance requirements and tests.

This stage does not generate contracts or train a model.

### P9-009B — Generate the 1,557 independent contracts

Run the separately seeded verifier over the fixed repaired accepted corpus. Generation is durable
and resumable. This does not increase the 2,000 candidate-answer ceiling because the outputs are
verification evidence, not new student targets.

### P9-009C — Execute contracts and freeze survivor census

Execute every statically valid contract reference first, then the corresponding frozen candidate.
Freeze exact counts and source composition before making a training decision. Do not choose a
survivor threshold after seeing benchmark scores.

### P9-009D — One bounded student run, if the census is usable

If the census leaves both train and validation partitions usable, keep the initial student control
fixed: QLoRA 4-bit, rank 8, alpha 32, dropout 0.05, sequence length 2048, micro-batch 1,
grad-accumulation 8, LR `1e-5`, cosine schedule, warmup `0.03`, seed `1729`.

Do not start a rank/LR/epoch sweep.

## Evaluation boundary

P9-009 may reuse the frozen P9-004 development slice for development-only model selection because
that slice is already part of the project's iterative development process.

The P9-007 qualification set is **not fresh for P9-009**. Its outcomes were inspected in P9-008 and
directly motivated this redesign. It may be used only as historical diagnostic evidence. If a
P9-009 candidate clears development, final qualification requires a newly frozen untouched
benchmark/holdout before the candidate is evaluated on it.

No protected benchmark prompt or test may become contract-generation input or training data.

## Decision rule

P9-009 is successful only if the changed data-acceptance objective produces a development candidate
that clears a precommitted development gate and subsequently survives a new untouched qualification
and the existing preservation gates. Lower training loss is not evidence of success.
