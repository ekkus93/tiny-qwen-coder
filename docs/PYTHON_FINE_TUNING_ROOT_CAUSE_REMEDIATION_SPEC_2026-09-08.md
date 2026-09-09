# Python Fine-Tuning Root-Cause Remediation Specification

Date: 2026-09-08

## 1. Purpose

This document defines a controlled remediation program for the repeated finding that Python fine-tuning of `Qwen/Qwen3.5-4B` has produced models that are worse than the unchanged base model on protected coding evaluation.

The goal is not to run another speculative hyperparameter sweep. The goal is to identify, with falsifiable experiments, whether the degradation comes from:

1. teacher/model quality or teacher decoding policy;
2. bad or weakly verified training labels;
3. training methodology or implementation defects;
4. evaluation/inference plumbing defects; or
5. an interaction among those factors.

No additional large-scale student fine-tune is scientifically justified until the gates in this specification isolate the failure mode.

## 2. Current empirical evidence

### 2.1 Frozen unchanged-base performance

The current promotion baseline for `Qwen/Qwen3.5-4B` is:

| Suite | Passed | Total |
| --- | ---: | ---: |
| HumanEval | 128 | 164 |
| MBPP | 290 | 500 |
| Repository holdout | 6 | 11 |
| Combined | 424 | 675 |

### 2.2 P9-007 distilled-model result

The winner selected from the repaired v4-2000 distillation trajectory produced:

| Suite | Base | Fine-tuned | Delta |
| --- | ---: | ---: | ---: |
| HumanEval | 128 | 128 | 0 |
| MBPP | 290 | 266 | -24 |
| Repository holdout | 6 | 6 | 0 |
| Combined | 424 | 400 | -24 |

The model therefore failed the frozen target-language promotion gate. The operational workflow completed successfully; the model did not qualify.

### 2.3 P9-008 MBPP regression forensics

Task-by-task comparison over the 370 MBPP qualification tasks found:

| Outcome | Count |
| --- | ---: |
| base pass -> adapter pass | 179 |
| base fail -> adapter fail | 133 |
| base pass -> adapter fail | 41 |
| base fail -> adapter pass | 17 |
| net change | -24 |

Forty of the 41 regressions reached executable tests and failed semantically. Only one was primarily a parse/truncation failure. This is strong evidence of genuine negative transfer rather than a generic harness failure.

Observed failure patterns included wrong formulas, inverted semantics, wrong entry-point names, wrong return shapes, missing imports, off-by-one behavior, index/value confusion, and plausible but incorrect algorithm templates.

### 2.4 Repaired v4-2000 corpus status

The repaired teacher corpus is provenance-clean and reasoning-leak-free:

- source candidates: 2,000;
- accepted: 1,557;
- train: 1,479;
- validation: 78;
- contamination status: clean;
- train/validation membership frozen;
- source manifest and output identities frozen.

The reasoning-leak defect was real and was repaired. That repair does not imply semantic correctness of the retained teacher answers.

### 2.5 Training-data quality review finding

The current Python-quality validator is conservative and treats several categories as `passed=True` while declaring detailed Python validation not applicable. In particular, mixed prose plus fenced code may be accepted as `not_applicable_mixed_content` instead of having the Python blocks compiled and semantically checked.

Inspection of the repaired export found that the overwhelming majority of accepted records fall into a mixed-content path rather than the strict standalone-Python validation path. Independent review also found accepted records containing Python blocks that do not compile and records that compile but appear semantically wrong.

Therefore, the phrase "qualified corpus" must not be interpreted as "every answer is a correct solution to its prompt." Existing qualification primarily established provenance, contamination, length/budget, deduplication, and structural quality.

### 2.6 Teacher-generation policy

The current Qwen3.8 teacher generation configuration uses the pinned teacher:

- `Qwen/Qwen3.8-27B`;
- revision `72a217afab8029b39e4af1c7273a829995a3dbaf`;
- BF16, no weight quantization;
- thinking enabled;
- temperature `1.0`;
- top-p `0.95`;
- top-k `20`;
- one generated candidate per requested training record in the distillation path.

Those sampling parameters are reasonable inference defaults for a thinking model, but they have not been demonstrated to be optimal for producing single-shot SFT labels that are later treated as ground truth.

### 2.7 Current student training recipe

The failed P9-007 run used a deliberately conservative recipe:

- QLoRA 4-bit NF4;
- BF16 compute;
- sequence length 2048;
- micro-batch 1;
- gradient accumulation 8;
- one epoch;
- learning rate `1e-5`;
- cosine schedule;
- warmup ratio `0.03`;
- gradient checkpointing;
- assistant-only loss;
- LoRA rank 8, alpha 32, dropout 0.05;
- selective LoRA target modules.

The methodology may still be wrong, but the evidence does not justify assuming that learning rate or rank is the primary cause. Previous broad rank/LR searches did not solve the problem.

## 3. Governing hypotheses

The remediation program will test the following hypotheses independently.

### H1 — Evaluation/inference plumbing defect

The fine-tuned model may only appear worse because base and adapter evaluation are not behaviorally equivalent outside the adapter weights. Possible defects include:

- different chat templates;
- tokenizer/revision drift;
- adapter loading mistakes;
- different generation parameters;
- different stop conditions;
- prompt formatting drift;
- result reconstruction bugs;
- stochastic evaluation variance;
- stale or mismatched artifacts.

### H2 — Teacher is not sufficiently superior

The exact pinned Qwen3.8-27B teacher/configuration may fail to materially outperform Qwen3.5-4B on the capability being distilled. If so, distillation from it is not justified.

### H3 — Teacher decoding produces noisy labels

Qwen3.8-27B may be a stronger model while the current one-sample, temperature-1.0 generation policy produces too many inferior samples for gold-label SFT.

### H4 — Training labels are insufficiently verified

The teacher may be stronger on average, but individual generated answers may contain semantic mistakes. Treating all accepted generations as positive SFT targets can overwrite correct behavior already present in the 4B base.

### H5 — Task distribution is misaligned

The StarCoder/Magicoder-derived training distribution contains many broad OSS, framework, repository, API, class, and application tasks. It may not supply the exact-spec, compact function-level behavior that MBPP strongly measures.

### H6 — Fine-tuning methodology damages a strong prior

Even with correct data, the current SFT/QLoRA objective may move the model away from useful base behavior. Candidate causes include:

- too many updates for the information content of the corpus;
- LoRA target-module selection;
- optimizer/scheduler behavior;
- quantization effects;
- loss masking defects;
- sequence truncation;
- excessive emphasis on stylistic imitation;
- lack of preservation/replay signal for correct base behavior.

## 4. Required diagnostic sequence

The order in this section is mandatory. Later stages must not be used to explain away a failure in an earlier stage.

## 4.1 Stage A — Re-establish measurement integrity

### A1. Exact base reproduction

Re-run the unchanged, pinned Qwen3.5-4B base through the current evaluation stack.

Required result:

- exact or explicitly tolerance-bounded reproduction of the frozen baseline;
- exact prompt/template/tokenizer/model revision recorded;
- exact generation configuration recorded;
- exact result artifact checksums recorded.

If baseline reproduction fails, stop all distillation work and repair evaluation first.

### A2. No-op adapter equivalence

Evaluate the base through the adapter-loading pathway using an adapter that is mathematically behavior-neutral, or an equivalent explicit no-adapter control that exercises the same runtime path.

Required result:

- no material score change;
- no prompt/template/generation drift;
- no artifact-load ambiguity.

If this fails, stop and repair adapter/evaluation plumbing.

### A3. Determinism/variance characterization

For any stochastic evaluation setting, run enough repeated generations to quantify variance. Prefer deterministic evaluation when the benchmark protocol permits it.

The project must not treat a one-problem movement on a small development slice as strong evidence of improvement without a variance argument.

## 4.2 Stage B — Prove teacher superiority before distillation

### B1. Direct teacher benchmark

Evaluate the exact pinned Qwen3.8-27B teacher directly on the same target-language benchmark harness used for the 4B base.

At minimum compare:

- HumanEval;
- MBPP;
- repository holdout where operationally appropriate and scientifically uncontaminated;
- combined score.

This evaluation is a diagnostic of teacher capability, not student training.

### B2. Teacher-superiority gate

Before new distillation is authorized, the teacher/configuration must materially outperform the 4B base on the capability intended for distillation.

A merely equal or slightly different result is insufficient because distillation introduces approximation error and label noise.

If teacher <= base on the target capability, stop this teacher-distillation track or change teacher/configuration before any student training.

## 4.3 Stage C — Measure teacher decoding policy

Use a new, non-protected, independently testable diagnostic set. Do not use HumanEval, MBPP, repository holdout prompts, or any future qualification set as training data.

Evaluate the same teacher with a small precommitted decoding matrix, for example:

- temperature 0 / deterministic-equivalent if supported;
- temperature 0.2;
- temperature 0.5;
- temperature 1.0 using the current policy.

Hold everything else fixed where possible.

Primary metric:

- executable correctness rate.

Secondary metrics:

- parse/compile rate;
- timeout rate;
- variance across repeated samples;
- answer length;
- rate of unusable/unverifiable output.

Do not select the policy based on prose style or training loss.

## 4.4 Stage D — Pairwise teacher-vs-base capability-gap census

Create a representative diagnostic set of self-contained or otherwise independently verifiable Python tasks that are disjoint from protected evaluation.

For every task, generate both:

- unchanged Qwen3.5-4B base answer;
- Qwen3.8-27B teacher answer under the selected teacher-generation policy.

Verify both with the same independent oracle and classify each task into exactly one cell:

1. teacher correct / base correct;
2. teacher correct / base wrong;
3. teacher wrong / base correct;
4. teacher wrong / base wrong.

The most important risk statistic is cell 3: `teacher wrong / base correct`. Those are examples for which blind teacher imitation can directly erase an existing capability.

The most valuable distillation examples are cell 2: `teacher correct / base wrong`.

## 4.5 Stage E — Repair training-data quality semantics

### E1. Replace "not applicable == pass" as a training authorization concept

Validator APIs may retain tri-state semantics internally, but the training-data gate must distinguish:

- positively validated;
- explicitly rejected;
- not validated / unverifiable.

"Not applicable" must never be interpreted as evidence of correctness.

### E2. Parse fenced Python inside mixed-content answers

For responses containing prose and one or more Python fences:

- extract Python-labeled code blocks;
- compile each block where compilation is meaningful;
- identify the block(s) that purport to implement the requested entry point;
- reject malformed primary code;
- record ambiguity explicitly.

Secondary illustrative/doctest blocks may be handled separately, but they must not hide a broken primary implementation.

### E3. Semantic correctness gate

A generated answer is a candidate, not a gold label.

Where a task is executable, positive SFT authorization requires an independent oracle such as:

- existing source/repository tests;
- deterministic tests produced from a known formal/specification source;
- property-based checks generated independently of the candidate;
- a separately frozen behavioral contract whose reference implementation passes first.

Unverifiable tasks must not silently become positive SFT targets.

### E4. Independent-oracle requirement

Candidate-hidden same-model verification is better than self-grading, but it does not eliminate correlated interpretation errors.

Prefer, in descending order:

1. existing human/source tests;
2. deterministic tests derived programmatically from formal task data;
3. repository tests with pinned dependencies/environment;
4. independently generated contracts with reference-first execution and explicit correlation caveats.

If the same model family creates both solution and oracle, that fact must be recorded and the evidence must not be described as fully independent.

## 4.6 Stage F — Build a small verified-gold corpus

Construct a deliberately small corpus before attempting another 1,500-example training run.

Target order of magnitude:

- initially 100–300 examples;
- every example independently verified;
- no protected benchmark leakage;
- exact source and oracle provenance;
- frozen train/validation split;
- reproducible checksums.

Prioritize examples in this order:

1. teacher correct / base wrong — highest value;
2. teacher correct / base correct — optional preservation/style data;
3. teacher wrong / base correct — forbidden teacher target;
4. both wrong — forbidden unless a separately verified correct solution is supplied.

This corpus is a diagnostic instrument, not a scaling target.

## 4.7 Stage G — Gold-data training sanity test

Train one bounded student adapter on the verified-gold corpus using the existing conservative recipe first.

Hold fixed:

- base model/revision;
- tokenizer/template;
- sequence length;
- QLoRA mode;
- rank;
- LoRA target modules;
- LR/scheduler;
- optimizer;
- seed;
- loss mode;
- evaluation protocol.

The purpose is to answer:

> Does unquestionably correct supervision behave sensibly under the current training implementation?

Interpretation:

- gold SFT improves/preserves while old corpus hurts -> data quality/objective problem strongly supported;
- gold SFT also hurts materially -> training methodology/implementation becomes primary suspect;
- result unstable across seeds -> training variance must be characterized before further conclusions.

## 4.8 Stage H — Training-methodology isolation, only if needed

Only if Stage G fails with verified data should we begin methodology ablations.

One-factor-at-a-time candidates include:

- fewer optimizer steps / fractional epoch;
- lower learning rate;
- LoRA target-module subsets;
- lower effective update magnitude via alpha/rank changes;
- QLoRA versus a higher-precision small control if hardware permits;
- explicit truncation audits;
- assistant-only loss-mask token audits;
- preservation/replay examples for base-correct behavior;
- alternative objectives that do not treat one teacher sequence as the sole gold target.

Do not launch another broad Cartesian rank/LR/epoch sweep.

## 5. Training implementation audits

The following implementation properties require explicit mechanical tests before blaming data alone.

### 5.1 Loss masking

For representative records, persist or reconstruct token-level audit evidence proving:

- user/system tokens are masked when `assistant_only` is selected;
- assistant target tokens are unmasked;
- special tokens/template boundaries are handled as intended;
- no examples have zero trainable target tokens;
- no unexpected prompt tokens contribute to loss.

### 5.2 Sequence truncation

Measure:

- fraction of train/validation records truncated at 2048 tokens;
- whether truncation removes required prompt context;
- whether truncation removes the end of the assistant solution;
- whether train and evaluation template lengths are consistent.

Training on incomplete code can create harmful gradients even when the original record was correct.

### 5.3 LoRA target modules

Confirm every configured target module exists in the pinned Qwen3.5-4B architecture and record:

- number of adapted tensors;
- trainable parameter count;
- module-wise coverage;
- whether target choices match intended attention/MLP adaptation behavior.

### 5.4 Quantization sanity

Confirm that NF4 QLoRA:

- loads the exact base revision;
- keeps intended compute dtype;
- does not alter tokenizer/template behavior;
- can reproduce a no-training or zero-update inference control.

### 5.5 Checkpoint/adaptor identity

Every evaluated adapter must be bound to:

- exact source commit;
- exact training config SHA;
- exact dataset manifest SHA;
- exact base revision;
- exact tokenizer/template identity;
- exact training trajectory/checkpoint step.

Stale or mislabeled adapters must fail closed.

## 6. Distribution alignment requirements

The project must explicitly distinguish at least two capabilities:

1. exact-spec, self-contained Python problem solving;
2. broad conversational/OSS coding assistance.

A corpus dominated by speculative framework/repository/API answers must not be assumed to improve MBPP-style exact-spec behavior.

Future corpora should record task categories and report survivor/training composition by category. Promotion conclusions must state which capability improved and which remained unchanged or regressed.

## 7. Promotion and statistical policy changes

### 7.1 Development selection

A `+1` movement on a 175-task development slice is not sufficient evidence by itself after selecting among multiple checkpoints.

Future development gates should include:

- a precommitted minimum effect size;
- suite-level non-regression constraints;
- paired task-level analysis;
- explicit handling of multiple checkpoint comparisons;
- replication across at least two training seeds when the observed margin is small.

### 7.2 Qualification

A fresh qualification set must not be inspected during model development. Once its outcomes influence redesign, it becomes diagnostic/history for later experiments and cannot be described as untouched.

### 7.3 Stop rules

Stop immediately when a precommitted prerequisite fails. Do not compensate for a failed teacher-superiority gate by tuning LoRA hyperparameters, and do not compensate for broken baseline reproduction by generating more data.

## 8. Relationship to P9-009

P9-009 remains useful because it introduced candidate-hidden semantic-contract generation and reference-first executable verification.

However, this remediation track is broader and supersedes any assumption that P9-009 alone is sufficient to authorize another student run.

Before P9-009-derived survivors are used for student training, this track must determine:

- whether the teacher/configuration materially beats the base;
- whether the selected decoding policy produces reliably correct labels;
- whether the oracle is independent enough for the intended claim;
- whether verified-gold SFT behaves correctly under the existing trainer.

P9-009B contract generation may be preserved as evidence/work product. P9-009C execution may contribute to data-quality analysis. P9-009D student training must not bypass the gates defined here.

## 9. Decision tree

Use the following interpretation hierarchy.

### Case 1 — Base baseline or no-op adapter cannot be reproduced

Conclusion: evaluation/inference plumbing defect.

Action: stop teacher/data/training experiments and repair measurement.

### Case 2 — Teacher does not materially outperform base

Conclusion: teacher/configuration is unsuitable for this distillation objective.

Action: change teacher, decoding/configuration, or abandon this teacher-distillation path.

### Case 3 — Teacher is superior, but individual labels frequently fail independent verification

Conclusion: label-generation/selection pipeline is the primary defect.

Action: require executable verification and capability-gap selection before SFT.

### Case 4 — Verified-gold SFT works, old corpus SFT hurts

Conclusion: current training-data objective/corpus is harmful.

Action: redesign data selection; do not hyperparameter-sweep the old corpus.

### Case 5 — Verified-gold SFT also hurts

Conclusion: training methodology/implementation is the primary suspect.

Action: run bounded methodology audits/ablations.

### Case 6 — Gold SFT helps only weakly or inconsistently

Conclusion: signal may be too small relative to training/evaluation variance.

Action: increase verified capability-gap signal, replicate seeds, and quantify confidence before promotion.

## 10. New non-negotiable rules

1. No LLM-generated answer is a gold SFT target solely because the teacher is larger.
2. The proposed teacher/configuration must demonstrate superiority on the intended capability before student distillation is authorized.
3. `not applicable` validation is not positive correctness evidence.
4. For executable coding tasks, independently demonstrated behavior outranks textual plausibility.
5. Protected evaluation prompts/tests never become training or contract-generation inputs.
6. Candidate-generation and correctness-oracle provenance must be recorded separately.
7. Prefer training on demonstrated teacher capability gaps over indiscriminate imitation.
8. Do not run broad hyperparameter sweeps until clean-data training has demonstrated that methodology is the limiting factor.
9. Do not call a model improved because training loss decreased.
10. Do not consume a fresh qualification set on a candidate that has not cleared a meaningful precommitted development gate.

## 11. Deliverables

This remediation track is complete only when the repository contains durable evidence for:

- exact base/no-op reproducibility;
- direct Qwen3.8-27B teacher benchmark;
- teacher decoding-policy comparison;
- pairwise teacher/base capability-gap census;
- repaired mixed-content Python validation semantics;
- verified-gold corpus manifest and oracle provenance;
- bounded gold-data SFT result;
- training-methodology audit if gold SFT fails;
- final root-cause report identifying the supported failure mode(s);
- updated promotion policy reflecting the findings.

## 12. Success criterion

Success is not another fine-tuned checkpoint.

Success is a reproducible explanation for why previous fine-tunes degraded the base model, followed by a training pipeline in which every expensive student experiment has a justified teacher, justified labels, mechanically audited training behavior, and a statistically meaningful evaluation gate.
