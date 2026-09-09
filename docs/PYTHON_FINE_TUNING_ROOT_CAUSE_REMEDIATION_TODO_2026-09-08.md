# Python Fine-Tuning Root-Cause Remediation TODO

Date: 2026-09-08

Companion specification:

`docs/PYTHON_FINE_TUNING_ROOT_CAUSE_REMEDIATION_SPEC_2026-09-08.md`

## Status

This track exists because repeated Python fine-tuning has failed to beat the unchanged `Qwen/Qwen3.5-4B` base and the P9-007 winner regressed MBPP by 24 correct tasks while preserving HumanEval and the repository holdout.

The objective is to determine whether the failure is caused by measurement plumbing, teacher quality/decoding, label quality, task-distribution mismatch, training methodology, or an interaction among them.

Do not skip ahead to another large student fine-tune. Complete the dependency-ordered gates below.

---

# FTR-000 — Freeze the remediation boundary

- [ ] Record the exact starting master SHA for this remediation track.
- [ ] Record the frozen Qwen3.5-4B base revision.
- [ ] Record the frozen Qwen3.8-27B teacher revision.
- [ ] Record the current base benchmark evidence and artifact identities.
- [ ] Record the P9-007 qualification evidence and P9-008 forensic evidence.
- [ ] Record the repaired v4-2000 corpus manifest/output identities.
- [ ] State explicitly that P9-007 is a failed model qualification even though its workflow completed successfully.
- [ ] State explicitly that the existing P9-007 qualification set is not fresh for later experiments whose design was influenced by those outcomes.

Acceptance criteria:

- Exact hashes and artifact identities are durable in repository evidence.
- No later experiment can silently substitute a different base, teacher, corpus, or evaluation artifact.

---

# FTR-100 — Re-establish evaluation integrity

## FTR-101 — Reproduce the unchanged base exactly

- [ ] Run the exact pinned Qwen3.5-4B base through the current target-language evaluation stack.
- [ ] Capture exact model revision, tokenizer revision, chat-template identity, generation parameters, stop conditions, seeds, and harness version.
- [ ] Reconstruct HumanEval, MBPP, repository holdout, and combined scores.
- [ ] Compare task-by-task with the frozen baseline.
- [ ] Fail closed on unexplained drift.

Expected frozen scores:

- HumanEval `128/164`;
- MBPP `290/500`;
- repository holdout `6/11`;
- combined `424/675`.

Acceptance criteria:

- Exact or explicitly justified tolerance-bounded reproduction.
- Result artifact checksums are frozen.
- Any deviation has a documented root cause before FTR-102 begins.

## FTR-102 — Add a no-op adapter equivalence test

- [ ] Implement a control that exercises the adapter-loading runtime without changing model behavior.
- [ ] Verify model/tokenizer/template identity before and after adapter-path activation.
- [ ] Evaluate the no-op control on the same benchmark harness.
- [ ] Compare task-by-task with unchanged-base outputs.

Acceptance criteria:

- No material score change.
- No prompt/template/generation drift.
- Adapter plumbing cannot silently alter base behavior.

## FTR-103 — Audit generation/evaluation parity

- [ ] Diff base and adapter inference configuration mechanically.
- [ ] Verify identical chat-template application.
- [ ] Verify identical tokenization boundaries.
- [ ] Verify identical max-new-token limits.
- [ ] Verify identical stop strings/token IDs.
- [ ] Verify identical temperature/top-p/top-k settings where deterministic evaluation is not used.
- [ ] Verify identical code-extraction and execution logic.

Acceptance criteria:

- A machine-readable parity report is emitted.
- Any intentional difference is explicitly justified.

## FTR-104 — Characterize evaluation variance

- [ ] Identify every stochastic element in benchmark inference/execution.
- [ ] Prefer deterministic evaluation when protocol-compatible.
- [ ] Where stochasticity remains, perform repeated runs on a bounded diagnostic subset.
- [ ] Quantify score and task-level variance.

Acceptance criteria:

- Future development gates include a variance-aware minimum effect size.
- `+1/175` is not treated as strong standalone evidence after checkpoint selection.

Gate FTR-G1:

- [ ] Evaluation integrity established.

Stop rule:

- If FTR-101 or FTR-102 fails, stop all teacher/data/training experiments and repair evaluation first.

---

# FTR-200 — Prove teacher superiority

## FTR-201 — Build direct Qwen3.8-27B evaluation support

- [ ] Reuse the same canonical benchmark prompt construction used for Qwen3.5-4B.
- [ ] Add a direct inference path for the pinned Qwen3.8-27B teacher.
- [ ] Preserve exact response/execution evidence.
- [ ] Keep benchmark results outside training inputs.

Acceptance criteria:

- Teacher and base are evaluated under comparable prompt and scoring semantics.

## FTR-202 — Benchmark the exact pinned teacher

- [ ] Evaluate Qwen3.8-27B on HumanEval.
- [ ] Evaluate Qwen3.8-27B on MBPP.
- [ ] Evaluate repository holdout only if its scientific status permits this diagnostic use.
- [ ] Freeze exact task-level results and aggregate scores.
- [ ] Compare teacher vs base pairwise.

Acceptance criteria:

- Direct teacher capability is known rather than assumed.

## FTR-203 — Define and enforce the teacher-superiority gate

- [ ] Precommit a meaningful superiority threshold before using the result to authorize new distillation.
- [ ] Require suite-level non-catastrophic-regression constraints.
- [ ] Document why the margin is large enough to tolerate distillation approximation/noise.

Gate FTR-G2:

- [ ] Qwen3.8-27B materially outperforms Qwen3.5-4B on the intended capability.

Stop rule:

- If teacher superiority is not demonstrated, do not fine-tune the student from this teacher/configuration.

---

# FTR-300 — Isolate teacher decoding quality

## FTR-301 — Build a clean decoding-policy diagnostic set

- [ ] Create a new non-protected set of independently executable Python tasks.
- [ ] Exclude HumanEval, MBPP, repository holdout, and future qualification tasks.
- [ ] Prefer self-contained functions/classes with deterministic tests.
- [ ] Freeze task IDs, prompts, tests, and checksums before generation.

Acceptance criteria:

- Every diagnostic task has an oracle that does not depend on the generated candidate answer.

## FTR-302 — Freeze the decoding matrix

Evaluate the same pinned teacher with a small precommitted matrix:

- [ ] deterministic/temperature-0-equivalent if supported;
- [ ] temperature `0.2`;
- [ ] temperature `0.5`;
- [ ] temperature `1.0`, top-p `0.95`, top-k `20` current policy.

- [ ] Hold model revision, prompt, reasoning setting, max tokens, and other variables fixed where possible.
- [ ] If deterministic decoding is unsupported or unsuitable for the model, record the exact substitute rather than silently changing semantics.

## FTR-303 — Score decoding policies by executable correctness

For each policy record:

- [ ] pass/fail rate;
- [ ] compile/parse rate;
- [ ] timeout rate;
- [ ] answer length distribution;
- [ ] finish-reason distribution;
- [ ] repeated-sample variance on a bounded subset;
- [ ] rate of unverifiable or malformed output.

Acceptance criteria:

- Teacher-label decoding policy is selected by correctness evidence, not vendor inference defaults or prose quality.

## FTR-304 — Freeze selected label-generation policy

- [ ] Select a policy according to a precommitted ranking rule.
- [ ] Freeze config and SHA.
- [ ] Document whether the original temperature-1.0 policy remains justified.

Gate FTR-G3:

- [ ] Label-generation decoding policy selected with executable evidence.

---

# FTR-400 — Measure teacher-vs-base capability gaps

## FTR-401 — Build a representative pairwise task set

- [ ] Create 100–200 independently verifiable tasks representative of the intended training domain.
- [ ] Categorize tasks by exact-spec algorithms, data handling, classes, APIs/frameworks, repository-context requirements, and other relevant strata.
- [ ] Exclude protected evaluation data.
- [ ] Freeze the task distribution before model generation.

## FTR-402 — Generate matched base and teacher answers

For every task:

- [ ] generate Qwen3.5-4B base answer;
- [ ] generate Qwen3.8-27B answer under FTR-304 policy;
- [ ] preserve exact model/prompt/seed/response identities;
- [ ] execute both against the same oracle.

## FTR-403 — Freeze the four-cell capability-gap matrix

Classify every task into exactly one outcome:

- [ ] teacher correct / base correct;
- [ ] teacher correct / base wrong;
- [ ] teacher wrong / base correct;
- [ ] teacher wrong / base wrong.

Also report by task category.

Acceptance criteria:

- The rate of `teacher wrong / base correct` is explicitly known.
- The pool of `teacher correct / base wrong` examples is explicitly known.

## FTR-404 — Define distillation value rules

- [ ] Mark teacher-correct/base-wrong as highest-value training candidates.
- [ ] Mark teacher-wrong/base-correct as forbidden teacher targets.
- [ ] Define whether teacher-correct/base-correct examples are used for preservation/style support.
- [ ] Reject both-wrong examples unless a separately verified correct solution is available.

Gate FTR-G4:

- [ ] Capability-gap census supports a plausible positive distillation signal.

---

# FTR-500 — Repair Python training-data validation

## FTR-501 — Introduce explicit validation status semantics

- [ ] Stop treating `not applicable` as positive correctness evidence for training authorization.
- [ ] Represent at least:
  - positively validated;
  - rejected;
  - not validated/unverifiable.
- [ ] Update manifest/finalization summaries to report those states separately.
- [ ] Preserve backward-readable evidence where needed.

Acceptance criteria:

- A record cannot become semantically "good" solely because the validator declined to inspect it.

## FTR-502 — Parse Python fences inside mixed-content responses

- [ ] Extract Python/Python3/py fenced blocks from prose answers.
- [ ] Distinguish primary implementation blocks from secondary examples/tests where possible.
- [ ] Compile primary blocks.
- [ ] Record all syntax failures.
- [ ] Fail closed on ambiguous primary implementation when training authorization requires executable code.
- [ ] Add regression tests for malformed fenced Python embedded in otherwise polished prose.

Acceptance criteria:

- The current `not_applicable_mixed_content` loophole cannot hide broken primary Python code.

## FTR-503 — Audit all 1,557 repaired accepted records with the new validator

- [ ] Run the revised structural/syntax validator over the frozen repaired corpus without mutating the old evidence.
- [ ] Freeze counts by validation outcome.
- [ ] Freeze examples/reason codes for rejected/unverifiable categories without exposing protected data.
- [ ] Compare with the legacy quality result distribution.

Acceptance criteria:

- We know exactly how much of the old corpus was positively validated versus merely not rejected.

## FTR-504 — Add semantic oracle interfaces

- [ ] Define a stable representation for executable behavioral tests/contracts.
- [ ] Record oracle provenance separately from candidate provenance.
- [ ] Require reference-first execution when the oracle includes a generated reference implementation.
- [ ] Propagate harness infrastructure errors rather than counting them as ordinary candidate failures.

## FTR-505 — Rank oracle independence

Implement/report oracle source categories:

- [ ] existing human/source tests;
- [ ] deterministic formal/programmatic tests;
- [ ] repository tests;
- [ ] independently generated contract/reference;
- [ ] same-family generated contract/reference with correlation caveat.

Acceptance criteria:

- Results never describe a same-family generated oracle as fully independent.

Gate FTR-G5:

- [ ] Training-data authorization semantics repaired.

---

# FTR-600 — Integrate and reassess P9-009

## FTR-601 — Preserve P9-009B work product

- [ ] Preserve any candidate-hidden contract-generation artifacts and provenance.
- [ ] Do not execute generated code on Colab with Google Drive mounted.
- [ ] Do not treat generated contracts as student targets.

## FTR-602 — Evaluate P9-009 oracle correlation risk

- [ ] Quantify how often the same Qwen3.8 family is responsible for both candidate-generation lineage and contract-generation lineage.
- [ ] Mark those cases with explicit oracle-independence grade.
- [ ] Identify subsets with stronger external oracles.

## FTR-603 — Execute P9-009C only as a data-quality experiment

- [ ] Run reference-first contract verification on the containerized self-hosted runner.
- [ ] Freeze survivor/rejection counts and reason codes.
- [ ] Do not authorize P9-009D training solely from survivor count.

Acceptance criteria:

- P9-009 evidence contributes to FTR-500/FTR-700 but cannot bypass FTR-G2 through FTR-G6.

---

# FTR-700 — Build a verified-gold diagnostic corpus

## FTR-701 — Freeze corpus admission rules

A positive training record must satisfy:

- [ ] non-protected source;
- [ ] exact source provenance;
- [ ] candidate answer bound by SHA;
- [ ] parse/compile success where applicable;
- [ ] independent behavioral verification;
- [ ] no hidden reasoning leakage;
- [ ] student tokenizer/template preflight;
- [ ] no unresolved truncation;
- [ ] no contamination;
- [ ] exact split membership.

## FTR-702 — Prioritize teacher capability gaps

Populate the initial corpus in this order:

- [ ] teacher correct / base wrong;
- [ ] teacher correct / base correct only as needed;
- [ ] never teacher wrong / base correct;
- [ ] never both wrong without a separately verified correct answer.

## FTR-703 — Build 100–300 example gold corpus

- [ ] Create immutable corpus namespace.
- [ ] Freeze train/validation split before training.
- [ ] Freeze manifest, checksums, source/oracle composition, and counts.
- [ ] Report task-category distribution.
- [ ] Keep corpus intentionally small for diagnosis.

## FTR-704 — Add gold-corpus preflight tests

- [ ] Every admitted executable record re-runs successfully from manifest evidence.
- [ ] No protected benchmark overlap.
- [ ] No duplicate train/validation membership.
- [ ] No missing oracle evidence.
- [ ] No reasoning markers.
- [ ] No invalid primary Python block.

Gate FTR-G6:

- [ ] Small verified-gold corpus frozen and mechanically reproducible.

---

# FTR-800 — Audit training implementation before gold SFT

## FTR-801 — Token-level assistant-only loss-mask audit

For representative records:

- [ ] persist token IDs and mask summary without leaking unnecessary data;
- [ ] prove system/user tokens are masked;
- [ ] prove assistant target tokens are unmasked;
- [ ] verify template control/special tokens behave as intended;
- [ ] detect zero-target examples;
- [ ] add unit/regression tests.

Acceptance criteria:

- Loss is applied exactly where the training contract says it is.

## FTR-802 — Sequence truncation audit

- [ ] Measure train/validation token-length distributions.
- [ ] Count records truncated at sequence length 2048.
- [ ] Determine whether prompt context is truncated.
- [ ] Determine whether assistant code endings are truncated.
- [ ] Fail training preflight for records whose required solution is cut off.

## FTR-803 — LoRA target-module audit

- [ ] Resolve every configured target module against the exact base architecture.
- [ ] Record adapted tensor count.
- [ ] Record trainable parameter count.
- [ ] Record attention/MLP coverage.
- [ ] Add fail-closed checks for unexpected architecture drift.

## FTR-804 — QLoRA zero-update/no-training control

- [ ] Load the base through the QLoRA preparation path without applying optimizer updates.
- [ ] Compare inference with unchanged base.
- [ ] Verify NF4/compute dtype and adapter preparation do not create unexplained behavior drift.

## FTR-805 — Checkpoint identity hardening

- [ ] Bind evaluated adapter to exact base revision.
- [ ] Bind to tokenizer/template identity.
- [ ] Bind to dataset manifest SHA.
- [ ] Bind to training config SHA.
- [ ] Bind to source commit and trajectory step.
- [ ] Fail closed on stale/mismatched artifacts.

Gate FTR-G7:

- [ ] Training implementation passes masking, truncation, LoRA, QLoRA, and identity controls.

---

# FTR-900 — Run the bounded verified-gold training experiment

## FTR-901 — Freeze the gold-SFT protocol

Initially keep the current conservative control fixed unless an earlier mechanical audit proves it invalid:

- [ ] QLoRA 4-bit NF4;
- [ ] BF16 compute;
- [ ] sequence length 2048, subject to FTR-802 safety;
- [ ] micro-batch 1;
- [ ] gradient accumulation 8;
- [ ] rank 8;
- [ ] alpha 32;
- [ ] dropout 0.05;
- [ ] LR `1e-5`;
- [ ] cosine schedule;
- [ ] warmup `0.03`;
- [ ] assistant-only loss;
- [ ] seed 1729;
- [ ] one bounded trajectory with precommitted checkpoints.

## FTR-902 — Define development gate before training

- [ ] Precommit target development tasks.
- [ ] Exclude protected/future qualification data.
- [ ] Define meaningful minimum paired improvement.
- [ ] Define suite-level non-regression constraints.
- [ ] Define selection/tie-break rule.
- [ ] Define when a second training seed is mandatory.

Acceptance criteria:

- A marginal `+1/175` cannot alone authorize fresh qualification after selecting among several checkpoints.

## FTR-903 — Execute one gold-data trajectory

- [ ] Train once under FTR-901.
- [ ] Preserve checkpoints and exact artifacts.
- [ ] Record loss/validation-loss trajectories only as operational evidence, not success evidence.

## FTR-904 — Evaluate development-only

- [ ] Compare every authorized checkpoint to unchanged base task-by-task.
- [ ] Report gains, regressions, and net changes.
- [ ] Apply the precommitted gate.
- [ ] Do not inspect fresh qualification data if the gate fails.

Gate FTR-G8:

- [ ] Verified-gold SFT behavior classified as improve/preserve/harm/unstable.

Diagnostic interpretation:

- [ ] gold SFT improves/preserves while old corpus hurt -> mark data/objective problem strongly supported;
- [ ] gold SFT also hurts -> activate FTR-1000 methodology isolation;
- [ ] gold SFT unstable -> characterize seed/update variance before proceeding.

---

# FTR-1000 — Training methodology isolation, conditional

Run only if verified-gold SFT still harms the base or behaves inconsistently.

## FTR-1001 — Freeze one-factor-at-a-time methodology matrix

Possible bounded factors:

- [ ] fewer optimizer steps / fractional epoch;
- [ ] lower LR;
- [ ] LoRA target-module subset;
- [ ] alpha/rank update-magnitude changes;
- [ ] higher-precision control if hardware permits;
- [ ] preservation/replay examples;
- [ ] alternative distillation objective.

Do not launch a broad Cartesian sweep.

## FTR-1002 — Add explicit update-magnitude measurements

- [ ] Record trainable parameter count.
- [ ] Record adapter norm statistics by checkpoint.
- [ ] Record gradient/update diagnostics appropriate to the framework.
- [ ] Correlate update magnitude with task-level regressions.

## FTR-1003 — Test truncation/masking counterfactuals if implicated

- [ ] Re-run only the smallest experiment needed to validate the suspected defect.
- [ ] Preserve exact before/after evidence.

## FTR-1004 — Test preservation/replay strategy if necessary

- [ ] Build a small set of base-correct verified examples disjoint from protected evaluation.
- [ ] Add only with a precommitted weighting rule.
- [ ] Measure whether known base competence is preserved without suppressing capability-gap learning.

Gate FTR-G9:

- [ ] Methodology root cause isolated or bounded enough to justify a revised protocol.

---

# FTR-1100 — Distribution alignment

## FTR-1101 — Categorize current and future training records

- [ ] exact-spec algorithm/function tasks;
- [ ] classes/data structures;
- [ ] data processing/numerical tasks;
- [ ] framework/API tasks;
- [ ] repository-context tasks;
- [ ] scripts/CLI;
- [ ] tests/debugging;
- [ ] other.

## FTR-1102 — Report performance by capability category

- [ ] Development/evaluation reports distinguish exact-spec correctness from broad conversational coding.
- [ ] Training manifests report category composition.
- [ ] Do not infer MBPP improvement from generic OSS training volume.

## FTR-1103 — Define target distribution for future corpus scaling

- [ ] Base the mix on desired product capability and demonstrated teacher advantage.
- [ ] Prefer verifiable tasks where correctness can be established.
- [ ] Treat unverifiable framework/repository answers as separate evidence, not default gold labels.

---

# FTR-1200 — Promotion-policy remediation

## FTR-1201 — Replace weak development promotion evidence

- [ ] Define precommitted effect-size minimums.
- [ ] Add paired regression/improvement counts.
- [ ] Add per-suite preservation constraints.
- [ ] Add small-margin seed replication requirement.
- [ ] Account for multiple checkpoint selection.

## FTR-1202 — Preserve qualification freshness

- [ ] Explicitly mark every benchmark/holdout as development, diagnostic, or untouched qualification.
- [ ] Once results influence redesign, automatically prevent later "fresh" qualification claims for that set.
- [ ] Freeze a new qualification set only after development succeeds.

## FTR-1203 — Add stop-rule enforcement

- [ ] Teacher-superiority failure blocks distillation.
- [ ] Baseline/no-op failure blocks all model conclusions.
- [ ] Gold-data failure blocks corpus scaling.
- [ ] Development-gate failure blocks qualification.

---

# FTR-1300 — Root-cause closure report

## FTR-1301 — Produce final causal evidence matrix

For each hypothesis, classify:

- [ ] supported;
- [ ] weakened;
- [ ] falsified;
- [ ] unresolved.

Hypotheses:

- [ ] H1 evaluation/inference plumbing defect;
- [ ] H2 teacher insufficiently superior;
- [ ] H3 noisy teacher decoding;
- [ ] H4 insufficient label verification;
- [ ] H5 task-distribution mismatch;
- [ ] H6 harmful training methodology.

## FTR-1302 — Freeze final recommendations

- [ ] approved teacher/configuration;
- [ ] approved label-generation policy;
- [ ] approved oracle requirements;
- [ ] approved corpus admission rules;
- [ ] approved training recipe;
- [ ] approved development/promotion policy;
- [ ] explicit forbidden shortcuts.

## FTR-1303 — Decide future P9 direction

Based on evidence:

- [ ] continue P9-009 with revised gates;
- [ ] supersede P9-009D with a capability-gap/gold-data successor;
- [ ] change teacher;
- [ ] change training methodology;
- [ ] or stop teacher distillation if the premise is unsupported.

Acceptance criteria:

- The next expensive student training run has a traceable causal justification.

---

# Global acceptance criteria

This remediation track is not complete until all of the following are true:

- [ ] unchanged-base evaluation is reproducible;
- [ ] adapter/no-op inference path is behaviorally controlled;
- [ ] Qwen3.8-27B superiority over Qwen3.5-4B is measured directly rather than assumed;
- [ ] teacher decoding policy is chosen using executable correctness;
- [ ] teacher-vs-base pairwise capability gaps are quantified;
- [ ] `not applicable` validation no longer masquerades as positive correctness evidence;
- [ ] mixed-content Python primary implementations are mechanically checked;
- [ ] semantic oracle provenance is explicit;
- [ ] a small verified-gold corpus exists;
- [ ] loss masking and truncation are mechanically audited;
- [ ] QLoRA/no-update and LoRA-target controls pass;
- [ ] one gold-data SFT experiment determines whether the current trainer behaves sensibly;
- [ ] methodology ablation occurs only if gold-data SFT implicates methodology;
- [ ] development selection requires meaningful evidence;
- [ ] fresh qualification is protected from iterative development leakage;
- [ ] a final root-cause report states what actually caused the degradation.

# Non-negotiable operating rules

1. Do not equate a larger teacher with a correct individual label.
2. Do not train on a generated coding answer solely because it passed provenance/length/format gates.
3. Do not treat `not applicable` as semantic validation.
4. Do not use protected benchmark prompts/tests as training or verifier-generation data.
5. Do not allow same-model candidate and oracle generation to be described as fully independent.
6. Do not launch another broad rank/LR/epoch sweep before verified-gold SFT isolates methodology.
7. Do not promote on training loss.
8. Do not consume a fresh qualification set after a marginal development result.
9. Do not overwrite frozen failed/provisional evidence; create successor namespaces.
10. Do not authorize large-scale fine-tuning until the teacher-superiority, label-quality, and training-sanity gates are green.
