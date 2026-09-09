# Python Verifiable Reinforcement Learning TODO

Date: 2026-09-09

Companion specification:

`docs/PYTHON_VERIFIABLE_RL_SPEC_2026-09-09.md`

## Status

This track defines a successor Python training architecture in which executable correctness, rather than teacher-answer imitation, is the primary learning signal.

The initial policy is the pinned `Qwen/Qwen3.5-4B`. `Qwen/Qwen3.8-27B` may author candidate tasks/tests/reference implementations, but larger-model output is not automatically a gold student target.

This TODO is dependency ordered. Do not skip gates.

### Hard relationship to FTR

- Infrastructure/specification work may proceed while FTR continues.
- No PVRL model-training experiment may be interpreted before FTR-G1 establishes evaluation integrity.
- Any FTR defect that materially affects PVRL becomes a blocking prerequisite for the affected PVRL stage.
- PVRL must not be used to abandon or bypass the root-cause investigation of the old SFT/distillation pipeline.

---

# PVRL-000 — Freeze program boundary and scientific claims

- [ ] Record the PVRL starting master SHA.
- [ ] Record the pinned Qwen3.5-4B policy revision.
- [ ] Record the pinned Qwen3.8-27B environment-author revision.
- [ ] Record the frozen unchanged-base benchmark reference.
- [ ] Record the status of FTR-G1 at PVRL track creation.
- [ ] Freeze the statement that Qwen3.8-27B is an environment author, not an automatic correctness oracle.
- [ ] Freeze the rule that protected HumanEval/MBPP/repository-holdout material cannot enter PVRL training/calibration/task-generation examples.
- [ ] Freeze the initial causal question and null/alternative hypotheses.
- [ ] Freeze the v0 scope: self-contained executable Python before repository-scale SWE.
- [ ] Freeze explicit non-goals and stop rules.

Acceptance criteria:

- A durable machine-readable evidence file binds the track to exact identities and dependencies.
- The spec/TODO cannot be interpreted as authorization to begin model training before required FTR gates.

---

# PVRL-100 — Executable environment integrity

## PVRL-101 — Define immutable environment schema

- [ ] Add a versioned environment-manifest schema.
- [ ] Require immutable environment ID/generation.
- [ ] Record task category and user-visible specification hash.
- [ ] Record initial workspace file inventory and hashes.
- [ ] Record public-vs-hidden artifact classification.
- [ ] Record dependency/runtime image identity.
- [ ] Record author provenance separately from oracle provenance.
- [ ] Record resource and network policy.
- [ ] Record contamination-check status.
- [ ] Record reference-first validation status/evidence identities.
- [ ] Add schema validation and round-trip tests.

Acceptance criteria:

- A task cannot silently mutate while retaining the same environment identity.
- Missing provenance or hidden/public classification fails closed.

## PVRL-102 — Define oracle/reference provenance model

- [ ] Implement oracle-independence categories:
  - existing human/source tests;
  - deterministic formal/programmatic contract;
  - repository-native tests;
  - independently generated contract/reference;
  - same-family generated contract/reference with explicit correlation caveat.
- [ ] Keep environment-author identity distinct from grader/oracle identity.
- [ ] Keep reference implementation identity distinct from student trajectory identity.
- [ ] Prevent same-family generated references from being labeled fully independent.
- [ ] Add tests for provenance downgrade/validation rules.

Acceptance criteria:

- Every admitted task has explicit oracle provenance and independence grade.

## PVRL-103 — Implement reference-first validation

- [ ] Execute clean initial workspace against hidden contract first.
- [ ] Require intended fail/incomplete condition before reference application.
- [ ] Apply the reference implementation/patch in a controlled phase.
- [ ] Require hidden tests to pass after reference application.
- [ ] Require preservation/public tests to remain valid.
- [ ] Repeat validation in a fresh sandbox.
- [ ] Freeze stdout/stderr/exit-code/result hashes needed for provenance.
- [ ] Separate candidate failure from harness/environment failure.
- [ ] Fail closed on non-reproducible reference validation.

Acceptance criteria:

- No environment is admitted merely because a generated reference looks plausible.
- A harness failure cannot be converted into a normal candidate failure/reward.

## PVRL-104 — Harden execution sandbox

- [ ] Disable outbound network by default.
- [ ] Keep hidden tests outside solver-writable workspace.
- [ ] Prevent candidate access to host secrets, SSH material, cloud credentials, Docker/Podman sockets, or mounted personal storage.
- [ ] Restore/remount pristine grader-controlled files before scoring.
- [ ] Bound wall clock, CPU, memory, process count, output size, file size, and disk usage.
- [ ] Capture bounded stdout/stderr and exact exit status.
- [ ] Ensure each attempt uses a disposable workspace.
- [ ] Add adversarial tests for modifying tests/grader files.
- [ ] Add adversarial tests for reading hidden files via path traversal/symlink tricks.
- [ ] Add tests for network and process-limit enforcement.

Acceptance criteria:

- Candidate edits to public/protected test files cannot change the reward source of truth.
- Hidden tests are not readable from the solver context/workspace.

## PVRL-105 — Add contamination controls

- [ ] Reuse/extend protected benchmark registry.
- [ ] Reject exact protected task IDs/content.
- [ ] Add normalized/textual overlap checks appropriate to synthetic task prompts/tests.
- [ ] Check environment-author few-shot examples for protected material.
- [ ] Check calibration/development/qualification partitions for overlap.
- [ ] Freeze contamination evidence in each admitted environment manifest.

Acceptance criteria:

- HumanEval, MBPP, repository holdout, and future qualification material cannot enter the training curriculum.

### Gate PVRL-G1 — Environment integrity

- [ ] Immutable environment schema proven.
- [ ] Reference-first validation proven.
- [ ] Sandbox/grader isolation proven.
- [ ] Contamination checks proven.

Stop rule:

- Do not collect policy calibration rollouts for training authorization if environment integrity is unresolved.

---

# PVRL-200 — Agent harness and exact trajectory identity

## PVRL-201 — Define Leaf-lite tool contract

- [ ] Freeze minimal tool vocabulary for repository-style phases: `read`, `write`, `edit`, `glob`, `bash`.
- [ ] Define a simpler subset for self-contained v0 tasks if sufficient.
- [ ] Remove duplicate/ambiguous ways to perform the same action.
- [ ] Define deterministic tool-result serialization.
- [ ] Define tool timeouts/output truncation semantics.
- [ ] Keep grader-only operations out of the policy tool namespace.
- [ ] Add parser/schema tests for every tool call and result.

Acceptance criteria:

- The harness presents a small stable interface and cannot expose hidden grader state.

## PVRL-202 — Implement exact Qwen3.5 token ledger

- [ ] Maintain exact per-session `prompt_ids` and `output_ids`.
- [ ] Never reconstruct historical assistant tokens from normalized strings.
- [ ] Append only newly rendered user/tool observations and generation prompt.
- [ ] Enforce exact uncompacted-history invariant:

```text
next_prompt_ids[:len(previous_prompt_ids + previous_output_ids)]
    == previous_prompt_ids + previous_output_ids
```

- [ ] Preserve exact assistant reasoning/action tokens as emitted by rollout engine.
- [ ] Treat non-append-only continuation as fatal outside an explicit compaction transition.
- [ ] Add regression fixtures reproducing history re-templating drift.
- [ ] Add tests across assistant text, tool calls, whitespace, structured arguments, and reasoning blocks.

Acceptance criteria:

- Ordinary multi-turn continuation never silently forks because of re-templating.

## PVRL-203 — Prove rollout/log-prob/loss-mask alignment

- [ ] Bind each action segment to exact token IDs and turn boundaries.
- [ ] Bind rollout log probabilities to the exact generating policy/checkpoint when used.
- [ ] Bind reward attribution to exact trajectory ID.
- [ ] Prove assistant/action mask aligns 1:1 with trainable action tokens.
- [ ] Reject zero-action-token trajectories.
- [ ] Reject length/count mismatches instead of repairing heuristically.
- [ ] Persist compact token/mask audit evidence for representative trajectories.

Acceptance criteria:

- Training cannot consume a trajectory whose action tokens differ from those actually sampled.

## PVRL-204 — Run harness A/B control before training

Dependency: FTR-G1 must be satisfied before interpreting results.

- [ ] Freeze a bounded non-protected development/control set.
- [ ] Evaluate exact unchanged base under existing/canonical interaction path.
- [ ] Evaluate exact unchanged base under Leaf-lite path.
- [ ] Hold model, decoding, task set, resource budget, and grader fixed.
- [ ] Compare task-by-task outcomes and operational metrics.
- [ ] Freeze whether harness alone materially improves/degrades behavior.
- [ ] Select one harness for all subsequent weight-training comparisons.

Acceptance criteria:

- Harness effects are separated from weight-update effects.

### Gate PVRL-G2 — Agent/trajectory integrity

- [ ] Leaf-lite contract frozen.
- [ ] Exact token continuation proven.
- [ ] Loss/log-prob/reward alignment proven.
- [ ] Harness A/B effect classified.

Stop rule:

- No multi-turn RL updates while token-history alignment is uncertain.

---

# PVRL-300 — Candidate task synthesis and learnability frontier

## PVRL-301 — Build candidate environment authoring pipeline

- [ ] Author self-contained Python tasks outside protected benchmarks.
- [ ] Allow Qwen3.8-27B to propose specification, tests, and reference implementation/patch.
- [ ] Preserve exact author model revision/config/seed/output identity.
- [ ] Separate author output from admitted immutable environment.
- [ ] Run all candidates through PVRL-G1 reference-first validation.
- [ ] Reject or quarantine invalid/unverifiable candidates rather than repairing silently.
- [ ] Make authoring resumable/idempotent for interruptible cloud runs.
- [ ] Never execute untrusted generated code in a notebook process with Google Drive/credentials mounted.

Acceptance criteria:

- Author output is only a candidate environment until independent executable validation succeeds.

## PVRL-302 — Freeze calibration policy

- [ ] Freeze current-policy checkpoint identity.
- [ ] Freeze stochastic rollout count: initial v0 target `4` attempts/task.
- [ ] Freeze temperature/top-p/top-k/max-token policy.
- [ ] Freeze seed schedule.
- [ ] Freeze tool/harness contract.
- [ ] Freeze grader/runtime identity.
- [ ] Record calibration outputs separately from training trajectories.

Acceptance criteria:

- Calibration semantics cannot change after observing solve rates.

## PVRL-303 — Implement frontier classifier

For four calibration attempts with `k` successes:

- [ ] classify `k=4` as mastered/easy;
- [ ] classify `k=0` as currently too hard/no group-relative signal;
- [ ] classify `1<=k<=3` as frontier candidate;
- [ ] prefer `k=2` when sufficient candidates exist;
- [ ] record parse/compile/timeout/harness failure separately from behavioral failure;
- [ ] exclude harness-invalid calibration groups.
- [ ] report frontier density by task category.

Acceptance criteria:

- Curriculum admission is based on current-policy learnability rather than manual judgment after inspecting answers.

## PVRL-304 — Freeze first curriculum

- [ ] Generate approximately 50-100 candidate environments under a precommitted generation count.
- [ ] Apply deterministic validation/filtering.
- [ ] Calibrate every eligible candidate.
- [ ] Select approximately 20-50 frontier tasks according to precommitted ranking/tie rules.
- [ ] Freeze curriculum manifest and exact environment IDs/hashes.
- [ ] Freeze category composition and oracle-independence composition.
- [ ] Freeze calibration evidence SHA set.
- [ ] Prevent post-selection task edits.

Acceptance criteria:

- First RL curriculum is immutable before the first optimizer update.

### Gate PVRL-G3 — Curriculum quality

- [ ] Candidate tasks reproducibly valid.
- [ ] Sufficient frontier density exists.
- [ ] Curriculum is disjoint from development/protected/qualification sets.
- [ ] Curriculum manifest is immutable.

Stop rule:

- If frontier density is too low, revise task generation in a new protocol generation before training; do not loosen admission post hoc.

---

# PVRL-400 — Synchronous QLoRA group-relative trainer

## PVRL-401 — Freeze v0 hardware/training preparation

- [ ] Use pinned Qwen3.5-4B policy revision.
- [ ] Use QLoRA 4-bit NF4 base loading.
- [ ] Use BF16 compute where supported.
- [ ] Start with LoRA rank `8` and alpha `32` unless a mechanical preflight proves invalid.
- [ ] Freeze exact target modules after architecture audit.
- [ ] Freeze microbatch/gradient-accumulation settings from measured memory preflight.
- [ ] Freeze gradient-checkpointing policy.
- [ ] Capture CUDA/PyTorch/Transformers/PEFT/TRL versions.
- [ ] Add zero-update/no-training inference equivalence control.

Acceptance criteria:

- Quantization/PEFT preparation cannot silently alter model behavior before updates.

## PVRL-402 — Implement binary executable reward

- [ ] Reward `1.0` only when all required hidden behavioral checks pass.
- [ ] Reward `0.0` for normal candidate behavioral failure.
- [ ] Exclude harness/environment failures from valid reward samples.
- [ ] Do not use answer/reference similarity.
- [ ] Do not add compile/style/length reward in v0.
- [ ] Persist reward evidence bound to exact trajectory and grader result SHA.

Acceptance criteria:

- Reward is derived from grader-controlled behavioral correctness only.

## PVRL-403 — Implement group-relative advantages

- [ ] Freeze exact normalization formula and epsilon.
- [ ] Compute mean/std within the exact task rollout group.
- [ ] Define deterministic treatment of zero-variance groups.
- [ ] Do not convert zero-variance groups to SFT examples.
- [ ] Add analytical unit tests for all-pass/all-fail/mixed groups.
- [ ] Add numerical stability tests.

Acceptance criteria:

- Advantage values are reproducible and mathematically bound to exact reward groups.

## PVRL-404 — Implement assistant/action-token-only policy loss

- [ ] Mask system tokens.
- [ ] Mask user/task tokens.
- [ ] Mask tool definitions.
- [ ] Mask tool observations.
- [ ] Unmask only policy-generated assistant/action tokens.
- [ ] Preserve reasoning/action token semantics without re-templating historical output.
- [ ] Detect zero-target examples.
- [ ] Persist representative token/mask audits.

Acceptance criteria:

- RL objective cannot accidentally optimize prompt/tool-observation tokens.

## PVRL-405 — Enforce synchronous on-policy update ordering

- [ ] Collect a rollout group from one exact current checkpoint.
- [ ] Score that group.
- [ ] Compute advantages.
- [ ] Update policy.
- [ ] Make new checkpoint authoritative before collecting the next group/update batch.
- [ ] Prevent stale rollout mixing in v0.
- [ ] Bind every rollout batch to parent checkpoint SHA/adapter identity.

Acceptance criteria:

- v0 has zero intentional policy lag and needs no asynchronous importance correction.

## PVRL-406 — Checkpoint/resume/provenance hardening

Bind every authoritative checkpoint to:

- [ ] base revision;
- [ ] tokenizer revision;
- [ ] chat-template SHA;
- [ ] source git SHA;
- [ ] PVRL protocol/config SHA;
- [ ] curriculum manifest SHA;
- [ ] environment-manifest SHA set;
- [ ] rollout/update index;
- [ ] optimizer/scheduler state;
- [ ] quantization/LoRA config;
- [ ] parent checkpoint identity;
- [ ] RNG/seed state sufficient for safe resume;
- [ ] runtime/library identities.

Also:

- [ ] Fail closed on mismatched resume state.
- [ ] Preserve checkpoint atomically before marking an update complete.
- [ ] Test interruption before/after optimizer step and before/after checkpoint commit.

Acceptance criteria:

- A resumed run cannot silently continue from a different curriculum/model/config state.

### Gate PVRL-G4 — Training mechanics

Dependencies: FTR-G1, PVRL-G1, PVRL-G2, PVRL-G3.

- [ ] QLoRA zero-update equivalence proven.
- [ ] Reward semantics proven.
- [ ] Advantage math proven.
- [ ] Action-only loss proven.
- [ ] On-policy ordering proven.
- [ ] Checkpoint/resume identity proven.

Stop rule:

- No optimizer update until all PVRL-G4 evidence is green.

---

# PVRL-500 — First bounded verifiable-RL experiment

## PVRL-501 — Freeze development evaluation before training

- [ ] Create a development set disjoint from curriculum/calibration/protected/future qualification data.
- [ ] Freeze task IDs/specifications/tests/hashes.
- [ ] Freeze unchanged-base task-level results under selected harness.
- [ ] Freeze paired minimum effect-size threshold.
- [ ] Freeze maximum acceptable regression count/rate.
- [ ] Freeze per-category non-catastrophic-regression constraints.
- [ ] Freeze checkpoint-selection rule before training.
- [ ] Freeze second-seed trigger for marginal results.

Acceptance criteria:

- Development success cannot be defined after seeing trained checkpoints.

## PVRL-502 — Execute tiny synchronous RL trajectory

- [ ] Use only the frozen PVRL-304 curriculum.
- [ ] Use initial target of four RL trajectories/task group.
- [ ] Keep run short/bounded with precommitted update/checkpoint count.
- [ ] Preserve exact rollout/reward/advantage/checkpoint artifacts.
- [ ] Record training reward/loss as operational evidence only.
- [ ] Do not inspect fresh qualification data.

## PVRL-503 — Evaluate every authorized checkpoint development-only

For each checkpoint report:

- [ ] base-correct/policy-correct;
- [ ] base-correct/policy-wrong regressions;
- [ ] base-wrong/policy-correct improvements;
- [ ] both wrong;
- [ ] net paired change;
- [ ] per-category behavior;
- [ ] parse/compile/timeout/harness breakdown;
- [ ] exact checkpoint and task-result hashes.

- [ ] Apply the precommitted checkpoint-selection rule.
- [ ] Do not cherry-pick qualitative examples to override the gate.

## PVRL-504 — Replicate marginal/unstable result when required

- [ ] If the effect is below strong-evidence threshold but not clearly harmful, run the precommitted second seed.
- [ ] Preserve the same curriculum/development protocol.
- [ ] Classify effect as reproducible improvement, preservation, harm, or instability.

### Gate PVRL-G5 — First behavioral-improvement gate

- [ ] First bounded RL behavior classified.

Interpretation:

- clear reproducible improvement -> authorize PVRL-600;
- preservation/no meaningful effect -> revise only with a new bounded hypothesis;
- harm -> stop scaling and perform causal analysis;
- instability -> characterize variance/update magnitude before continuing.

---

# PVRL-600 — Adaptive curriculum iteration

## PVRL-601 — Freeze improved checkpoint as new current policy

- [ ] Bind exact selected parent checkpoint.
- [ ] Preserve original unchanged base as ultimate comparison reference.
- [ ] Freeze all environment-authoring/calibration settings for iteration 2 or explicitly version changes.

## PVRL-602 — Generate and calibrate new candidate tasks

- [ ] Generate/mutate new non-protected candidates.
- [ ] Run reference-first validation.
- [ ] Calibrate against improved current policy, not original base.
- [ ] Recompute frontier density.
- [ ] Freeze second curriculum before updates.

## PVRL-603 — Execute second bounded RL trajectory

- [ ] Train only on second frozen curriculum.
- [ ] Preserve same training-mechanics controls.
- [ ] Compare against parent checkpoint and original base.

## PVRL-604 — Test adaptive-curriculum hypothesis

- [ ] Determine whether second frontier regeneration yields additional paired development gain.
- [ ] Report whether prior iteration's mastered tasks move out of the frontier as expected.
- [ ] Report whether new frontier shifts in difficulty/category composition.
- [ ] Separate curriculum effect from any protocol/harness change.

### Gate PVRL-G6 — Iterative learning gate

- [ ] Adaptive curriculum provides meaningful additional value or is explicitly falsified/bounded.

Stop rule:

- Do not scale task count merely because training reward increased.

---

# PVRL-700 — Repository/multi-file expansion

Run only after PVRL-G5; preferably after PVRL-G6.

## PVRL-701 — Add immutable repository snapshot environments

- [ ] Freeze repository/source snapshot identity.
- [ ] Define fail-to-pass and pass-to-pass tests.
- [ ] Preserve dependency/cache identities.
- [ ] Keep hidden tests outside writable checkout.
- [ ] Record patch/file mutation inventory.

## PVRL-702 — Extend Leaf-lite multi-file harness

- [ ] Validate `read/write/edit/glob/bash` under repository tasks.
- [ ] Bound command execution and output.
- [ ] Prevent nested uncontrolled agents/subprocess farms.
- [ ] Preserve exact tool-event token ledger.

## PVRL-703 — Increase context only from evidence

- [ ] Measure context requirement distribution.
- [ ] Increase physical context in bounded steps.
- [ ] Measure VRAM/throughput impact.
- [ ] Fail closed on required-state truncation.

## PVRL-704 — Add explicit self-summary compaction if needed

- [ ] Define compaction trigger.
- [ ] Preserve system/task state.
- [ ] Remove only documented complete turn groups.
- [ ] Record summary prompt/model/checkpoint and exact summary token IDs.
- [ ] Mark token-ledger discontinuity as an explicit compaction boundary.
- [ ] Test compacted vs uncompacted behavior on bounded tasks.

## PVRL-705 — Harden reward-hacking analysis for repository tasks

- [ ] Detect protected-test/grader mutation attempts.
- [ ] Detect environment introspection attempts.
- [ ] Detect network/process escape attempts.
- [ ] Preserve attempt categories without rewarding them.

### Gate PVRL-G7 — Repository-scale readiness

- [ ] Multi-file task integrity proven.
- [ ] Long-context/compaction policy proven if used.
- [ ] Grader isolation remains intact.

---

# PVRL-800 — Fresh qualification and promotion decision

## PVRL-801 — Freeze untouched qualification set

Only after development success:

- [ ] Create/freeze new qualification set not used in design, calibration, task authoring, or checkpoint selection.
- [ ] Freeze exact role as qualification.
- [ ] Freeze model/checkpoint to be evaluated before revealing results.
- [ ] Freeze effect-size and suite-level constraints.

## PVRL-802 — Run one-shot qualification

- [ ] Evaluate selected policy once under frozen protocol.
- [ ] Compare task-by-task to unchanged base.
- [ ] Preserve all exact artifacts.
- [ ] Do not tune after viewing qualification while retaining a fresh-qualification claim.

## PVRL-803 — Promotion decision

- [ ] Apply frozen promotion rules.
- [ ] Distinguish operational workflow success from model qualification success.
- [ ] Record promote/reject conclusion durably.

### Gate PVRL-G8 — Qualification

- [ ] Fresh qualification completed and classified.

---

# PVRL-900 — Causal and scaling closure

## PVRL-901 — Produce causal evidence matrix

Classify each proposition as supported, weakened, falsified, or unresolved:

- [ ] executable reward can improve Qwen3.5-4B on disjoint Python development tasks;
- [ ] learnability-frontier selection is useful relative to uncalibrated tasks;
- [ ] Leaf-lite harness materially changes small-model capability;
- [ ] exact token-ledger enforcement is necessary for multi-turn trajectory integrity;
- [ ] synchronous QLoRA group-relative training is stable on 16 GiB hardware;
- [ ] adaptive curriculum regeneration adds value after a first improvement;
- [ ] repository-scale expansion is justified.

## PVRL-902 — Compare against old SFT/distillation evidence

- [ ] Compare verified-RL gains/regressions with P9/FTR SFT outcomes.
- [ ] Do not claim PVRL success by itself proves every old failure was caused by data.
- [ ] Integrate FTR causal findings where available.

## PVRL-903 — Freeze scaling recommendation

Choose exactly one evidence-backed recommendation:

- [ ] scale environment count;
- [ ] scale curriculum iterations;
- [ ] move to repository tasks;
- [ ] revise reward/objective in a new protocol;
- [ ] stop PVRL because evidence does not support further investment.

## PVRL-904 — Close engineering/scientific phase

- [ ] Freeze final report and artifact index.
- [ ] Freeze unresolved questions.
- [ ] Preserve exact model/checkpoint/protocol identities.
- [ ] Mark development/diagnostic/qualification freshness status permanently.

---

# Cross-cutting rules

These rules apply to every PVRL task:

- Never silently mutate/reuse failed, provisional, or contaminated artifacts.
- Never call workflow completion model/scientific success.
- Never train on protected benchmark data.
- Never treat environment-author model output as correct without executable verification.
- Never count harness failures as ordinary candidate failures.
- Never re-template historical Qwen3.5 assistant actions in a way that changes their token IDs.
- Never mix stale-policy trajectories into synchronous v0 updates.
- Never start a large experiment when a smaller causal test can answer the question.
- Never weaken a precommitted gate after seeing the result without creating a new protocol generation and documenting why.
- Keep exact source/model/tokenizer/template/environment/curriculum/checkpoint identities in every authoritative artifact.
