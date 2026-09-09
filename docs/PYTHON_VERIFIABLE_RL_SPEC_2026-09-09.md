# Python Verifiable Reinforcement Learning Specification

Date: 2026-09-09

Companion TODO:

`docs/PYTHON_VERIFIABLE_RL_TODO_2026-09-09.md`

Status: successor architecture specification; implementation is dependency-gated by the Python fine-tuning root-cause remediation track.

## 1. Purpose

This specification defines a successor training architecture for improving the pinned `Qwen/Qwen3.5-4B` Python coding model using executable correctness as the primary learning signal.

The architecture is intentionally different from the existing teacher-answer distillation path. A larger model may help author tasks, tests, and reference implementations, but it is not treated as an oracle whose generated answer is automatically suitable as a student target.

The central design rule is:

> Correctness is established by an independently executed behavioral contract, not by model identity, prose quality, syntactic plausibility, or teacher confidence.

The first implementation phase is deliberately small and synchronous so that we can establish causality before scaling to repository-level software-engineering reinforcement learning.

## 2. Relationship to the FTR remediation track

The Python Fine-Tuning Root-Cause Remediation track remains authoritative for diagnosing why prior SFT/distillation experiments failed. PVRL does not replace or bypass that work.

PVRL has two classes of work:

1. **Infrastructure/specification work that may proceed before FTR-G1**, including task schemas, sandbox contracts, provenance structures, token-ledger tests, and non-training harness code.
2. **Model-behavior/training experiments that are blocked until the required FTR gates are satisfied.**

At minimum, no PVRL model-training conclusion may be drawn until FTR-G1 establishes evaluation integrity. If FTR later identifies additional defects that affect PVRL assumptions, those defects become PVRL prerequisites before the affected experiment proceeds.

The starting repository boundary for this specification is master commit:

`6fb599f3528387d77f8293a2a8a2260326aa20ae`

That commit contains the merged FTR-101 exact unchanged-base reproduction implementation. FTR-102 was still in progress when this PVRL track was created.

## 3. Scientific objective

The primary research question is:

> Can a small curriculum of independently verifiable Python tasks, selected near the current Qwen3.5-4B policy's learnability frontier and optimized with group-relative executable reward, produce a reproducible behavioral improvement over the exact unchanged base?

The first experiment must be capable of falsifying this proposition cheaply.

### 3.1 Primary null hypothesis

`H0`: A bounded executable-reward RL intervention does not produce a meaningful paired improvement over the exact unchanged Qwen3.5-4B base on a frozen, disjoint development set.

### 3.2 Primary alternative hypothesis

`H1`: A bounded executable-reward RL intervention on frontier-selected, independently verified tasks produces a meaningful paired improvement over the exact unchanged base without unacceptable regressions.

### 3.3 What does not count as success

The following are operational evidence only and do not establish model improvement:

- a workflow exits successfully;
- training loss decreases;
- reward rises on the training curriculum;
- the model solves tasks that were used for RL;
- a checkpoint is produced;
- a larger model generated the task or reference implementation;
- the output compiles;
- a validator returns `not applicable`;
- a single marginal development gain after checkpoint selection.

Success requires precommitted, disjoint behavioral evaluation.

## 4. Reference architecture

PVRL is inspired by the central ideas in FrogNano / TaskPilot: online task synthesis, executable environments, learnability-frontier calibration, and reinforcement learning driven by behavioral outcomes rather than traditional trajectory distillation.

Primary reference:

- Minseon Kim et al., **FrogNano: Training a 4B Coding Agent via Online Task Synthesis**, arXiv:2609.07925, 2026.
- https://arxiv.org/abs/2609.07925

PVRL is not intended to reproduce FrogNano exactly. The initial implementation is adapted for a single RTX 4070 Ti SUPER-class 16 GiB GPU and the existing repository's QLoRA/evaluation infrastructure.

### 4.1 Deliberate deviations from FrogNano in v0

PVRL v0 will initially use:

- self-contained Python environments rather than full repository SWE tasks;
- short contexts rather than 65k/131k contexts;
- synchronous rollouts and updates rather than asynchronous distributed rollout infrastructure;
- QLoRA/NF4/BF16 rather than assuming full-parameter BF16 training;
- 4 policy attempts per calibration task initially rather than large distributed rollout groups;
- binary executable correctness before any shaped length/reward terms;
- no reference-KL term initially unless evidence requires it;
- no policy-staleness correction while rollouts are strictly on-policy and synchronous.

Any later move toward asynchronous rollouts, long-context agent trajectories, or distributed training must be introduced as a separately gated protocol revision.

## 5. Fixed model identities

Unless a later protocol revision explicitly changes them, PVRL starts with the same frozen identities used by the FTR track.

### 5.1 Student/policy

- Repository: `Qwen/Qwen3.5-4B`
- Revision: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`

### 5.2 Environment-author model

- Repository: `Qwen/Qwen3.8-27B`
- Revision: `72a217afab8029b39e4af1c7273a829995a3dbaf`

The 27B model may author candidate environment material. It is not automatically a correctness oracle and its output is never admitted merely because it came from the larger model.

### 5.3 Frozen unchanged-base benchmark reference

The existing frozen base scores remain the canonical historical reference unless FTR establishes a justified replacement protocol:

- HumanEval: `128/164`
- MBPP: `290/500`
- repository holdout: `6/11`
- combined: `424/675`

Protected benchmark tasks must never become PVRL training tasks, hidden environment templates, calibration tasks, prompt exemplars, or teacher-authored derivatives.

## 6. Core trust model

PVRL separates four roles that must not be conflated:

1. **Environment author** — proposes a specification, reference implementation, tests, fixtures, or task mutation.
2. **Reference implementation** — demonstrates that the proposed task is solvable under the proposed contract.
3. **Grader/oracle** — executes the behavioral contract independently of the policy answer.
4. **Policy/student** — attempts the task and receives reward only from the grader outcome.

The author and reference implementation may come from the same model family, but that lowers oracle-independence confidence. The grader must still execute independently and the evidence must preserve this correlation caveat.

## 7. Environment model

PVRL v0 tasks are executable Python environments with an immutable manifest.

Each environment must contain or reference:

- immutable environment ID and schema version;
- task-category label;
- user-visible problem specification;
- initial workspace tree and file hashes;
- public files visible to the solver;
- hidden grading contract/tests inaccessible to the solver;
- reference implementation or reference patch when required;
- author provenance;
- oracle provenance and independence grade;
- source/repository provenance when applicable;
- dependency lock/image identity;
- resource limits;
- network policy;
- reference-first validation evidence;
- contamination-check evidence;
- policy-calibration evidence once generated;
- exact manifest SHA-256.

Task material is immutable after admission. A modified task receives a new environment ID/generation and cannot silently reuse the old evidence.

## 8. Reference-first validation

An environment is not valid because its tests pass against a generated reference. It must satisfy a stronger sequence.

For self-contained v0 tasks, the minimum contract is:

1. construct the clean initial workspace;
2. execute hidden tests against the initial/baseline implementation and verify the intended fail condition;
3. apply the reference implementation/patch;
4. execute hidden tests and require the intended pass condition;
5. execute all public/preservation tests and require no prohibited regression;
6. repeat in a fresh sandbox to prove reproducibility;
7. freeze stdout/stderr/exit status/timing class and artifact hashes needed for provenance.

For tasks where the initial state is intentionally incomplete rather than incorrect, the manifest must define an equivalent fail-to-pass contract explicitly rather than pretending a baseline implementation exists.

### 8.1 Harness failures

Infrastructure failures are not candidate failures.

The grader must distinguish at least:

- candidate behavioral failure;
- candidate parse/compile failure;
- candidate timeout/resource failure;
- grader/harness failure;
- environment construction failure;
- unavailable dependency/image;
- corrupted/identity-mismatched environment.

Harness failures invalidate the attempt for reward/training purposes and must propagate as infrastructure errors.

## 9. Oracle independence grading

Every environment must receive an oracle-independence category. At minimum:

1. existing human/source tests;
2. deterministic formal/programmatic contract;
3. repository-native tests;
4. independently generated contract/reference from a separate provenance path;
5. same-family generated contract/reference with explicit correlation caveat.

A same-family generated reference must never be described as fully independent.

The initial PVRL curriculum should prefer stronger oracle categories where practical.

## 10. Sandbox and reward-hacking defenses

The policy must not be able to modify or inspect the source of truth used for reward.

PVRL v0 sandbox requirements:

- outbound network disabled;
- hidden tests outside the solver-writable workspace;
- grader launched separately from the policy process;
- candidate workspace disposable per attempt;
- protected grader files remounted or restored before scoring;
- bounded wall-clock timeout;
- bounded CPU, memory, file size, process count, and disk usage;
- no host Docker/Podman socket exposure;
- no host secrets, tokens, SSH keys, cloud credentials, or mounted personal storage;
- deterministic dependency/image identity;
- captured exit code/stdout/stderr with size bounds;
- explicit classification of reward-hacking attempts when detectable.

A candidate that modifies public tests may still be graded only against pristine grader-controlled tests. Candidate changes to protected tests or grader code must never affect reward.

## 11. Reward semantics

### 11.1 Initial reward

PVRL v0 uses binary behavioral reward:

- `1.0`: all required hidden behavioral checks pass under the grader contract;
- `0.0`: the candidate executes normally but fails the contract.

Harness/infrastructure failures produce no valid reward sample.

### 11.2 No shaped reward initially

Do not initially reward:

- compilation by itself;
- style;
- answer similarity to a reference;
- shorter text;
- teacher/model confidence;
- partial public-test success;
- hidden reasoning format.

A shaped reward may be proposed later only after a protocol revision identifies a concrete failure mode and adds tests proving that the shaping term cannot dominate correctness.

## 12. Learnability-frontier calibration

Each candidate task must be calibrated against the exact current policy checkpoint before curriculum admission.

PVRL v0 starts with four independent stochastic policy attempts per candidate task under a frozen sampling configuration.

Let `k` be the number of successful attempts out of four.

Initial categories:

- `k = 4`: mastered/easy;
- `k = 0`: currently too hard or invalid for useful group-relative learning;
- `1 <= k <= 3`: frontier candidate;
- `k = 2`: preferred initial target when enough tasks are available.

The calibration policy, sampling temperature, top-p/top-k, maximum tokens, seed schedule, and tool/harness configuration must be frozen before observing task outcomes.

Calibration attempts are evidence, not training trajectories unless a later protocol explicitly admits them.

### 12.1 Task admission

The initial curriculum must be frozen before RL begins. Selection rules must be deterministic from precommitted metadata and calibration outcomes.

Do not manually cherry-pick tasks after inspecting model behavior.

## 13. Leaf-lite agent interface

Repository-scale PVRL will eventually use a deliberately small tool surface inspired by lightweight coding-agent harnesses.

The target tool vocabulary is:

- `read`
- `write`
- `edit`
- `glob`
- `bash`

PVRL v0 self-contained function tasks may start with an even smaller equivalent file-edit/run interface.

The harness must avoid unnecessary planner/task-manager abstractions, nested agents, permission chatter, or duplicate ways to perform the same operation.

### 13.1 Harness A/B requirement

Before attributing any behavioral gain to RL weights, the unchanged base must be evaluated on a bounded development set under:

- the existing/canonical interaction path; and
- the PVRL Leaf-lite path.

If the harness alone materially changes performance, that effect is frozen and reported separately. Weight-training claims use one fixed harness thereafter.

## 14. Exact multi-turn token ledger

Multi-turn Qwen3.5 agent RL has a known class of alignment hazards when prior assistant messages are reconstructed from strings and re-templated. Re-tokenization can change historical assistant tokens, which breaks the relationship between stored rollout actions/log probabilities and the continuation actually consumed by the model.

Reference issue:

- THUDM/slime issue #2288, `Qwen3.5 history re-templating causes false trajectory forks`.
- https://github.com/THUDM/slime/issues/2288

PVRL therefore requires an exact per-session token ledger.

For uncompacted history, every turn must satisfy:

```text
next_prompt_ids[:len(previous_prompt_ids + previous_output_ids)]
    == previous_prompt_ids + previous_output_ids
```

Historical assistant tokens must come from the exact canonical token IDs emitted by the rollout engine, not from parsing, string normalization, or re-applying the chat template to old assistant text.

Only newly appended user/tool observations and the next-generation prompt may be newly rendered/tokenized.

A non-append-only continuation is a fatal trajectory-integrity error unless a separately specified context-compaction transition is being performed.

### 14.1 Loss/log-prob alignment

For every training trajectory, the following arrays/segments must remain aligned:

- action token IDs;
- rollout log probabilities when used;
- assistant/action loss mask;
- turn boundaries;
- reward attribution;
- prompt/context prefix identity.

A mismatch invalidates the trajectory. Do not repair it heuristically after collection.

## 15. Context policy

PVRL v0 deliberately avoids long-context requirements.

Initial target:

- physical context: approximately 2k-4k tokens, increased only if task evidence requires it;
- bounded completion: approximately 1k-2k tokens for self-contained Python tasks;
- no silent left/right truncation of required task state;
- fail closed when the required solution or contract cannot fit.

Long-context repository tasks are a later phase.

### 15.1 Context compaction

Self-summary/context compaction is not part of v0. If introduced later it must:

- be an explicit trajectory transition;
- preserve task/system state;
- record exactly which turn groups were removed;
- record the summary prompt/model/checkpoint and produced token IDs;
- reset token-prefix continuity only at the documented compaction boundary;
- never be confused with ordinary append-only continuation;
- be tested for reward/evaluation impact before use in training.

## 16. Synchronous on-policy v0

The first PVRL trainer is synchronous:

```text
collect rollout group -> execute reward -> compute group advantages -> update -> collect next group
```

No rollout may be collected from an older policy after a new policy checkpoint becomes authoritative for the next update.

This deliberately avoids distributed policy-staleness correction in v0.

If asynchronous rollout is introduced later, the protocol must add explicit rollout-policy checkpoint identity, importance-ratio/staleness handling, and tests proving that training tokens/log probabilities correspond to the policy that generated them.

## 17. Group-relative objective

For a task group `i` with rewards `R_ij`, compute a group-relative normalized advantage using a frozen formula. The default starting form is:

```text
A_ij = (R_ij - mean(R_i)) / (std(R_i) + epsilon)
```

with a precommitted epsilon.

Groups with zero reward variance provide no meaningful group-relative signal and should not silently become ordinary supervised examples.

The exact optimizer objective, clipping behavior, per-token/per-sequence normalization, and treatment of zero-variance groups must be frozen in a protocol config and covered by unit tests before the first RL update.

## 18. Assistant/action-token-only optimization

Policy-gradient loss must apply only to policy-generated assistant/action tokens.

The following are context, not targets:

- system prompt;
- user task specification;
- tool definitions;
- tool observations;
- grader messages;
- environment metadata.

Token-level audit evidence must prove the mask boundaries on representative multi-turn trajectories.

No trajectory with zero trainable action tokens may be admitted.

## 19. Hardware-adapted training path

The initial local target is an RTX 4070 Ti SUPER-class 16 GiB GPU.

The expected first implementation uses:

- QLoRA;
- 4-bit NF4 base quantization;
- BF16 compute where supported;
- LoRA rank 8 initially;
- alpha 32 initially;
- conservative microbatching;
- gradient accumulation sized by measured memory;
- gradient checkpointing if required;
- exact CUDA/PyTorch/Transformers/PEFT/TRL identity capture.

These are PVRL experimental choices, not claims about FrogNano's exact optimizer implementation.

No training run begins until a memory preflight and a zero-update/no-op control prove the selected inference/training preparation path is mechanically sound.

## 20. Checkpoint and resume contract

Every authoritative training checkpoint must bind to:

- exact base revision;
- tokenizer revision;
- chat-template SHA;
- source git SHA;
- PVRL protocol/config SHA;
- environment-manifest SHA;
- curriculum-selection manifest SHA;
- rollout batch/update index;
- optimizer/scheduler state identity;
- LoRA/quantization config;
- RNG/seed state sufficient for deterministic resume where practical;
- parent checkpoint identity;
- training-library/runtime versions.

Resume must fail closed on identity mismatch.

A checkpoint produced after partial or corrupted optimizer state restoration is not authoritative.

## 21. Data partitioning and freshness

Every task set must be labeled as exactly one of:

- training curriculum;
- calibration/diagnostic;
- development evaluation;
- qualification evaluation;
- protected benchmark.

A task cannot occupy multiple roles unless the protocol explicitly declares the later role scientifically contaminated.

Once results from a set influence architecture, task generation, hyperparameters, selection rules, or checkpoint choice, that set is no longer fresh qualification data.

Protected HumanEval, MBPP, and repository holdout material must never flow into training or curriculum generation.

## 22. Development evaluation

The first bounded RL experiment uses a frozen development set disjoint from:

- training environments;
- calibration environments;
- environment-author few-shot examples;
- protected benchmarks;
- future qualification data.

Evaluation is paired task-by-task against the exact unchanged base under the same frozen harness and decoding settings.

Reports must include at least:

- base correct / policy correct;
- base correct / policy wrong regressions;
- base wrong / policy correct improvements;
- both wrong;
- net paired change;
- per-category changes;
- parse/compile/harness/timeout breakdown;
- exact task IDs and artifact hashes.

A development success threshold must be committed before observing the trained result.

## 23. First bounded experiment

The first training experiment is intentionally diagnostic.

Target scale:

- generate 50-100 candidate self-contained Python environments;
- calibrate each with four current-policy attempts;
- admit approximately 20-50 frontier tasks according to precommitted rules;
- use four RL trajectories per task group initially;
- train one short, bounded trajectory with precommitted checkpoints;
- evaluate development-only;
- replicate with another seed if the effect is too small to distinguish from instability.

The exact counts become protocol constants before the run. These ranges describe design intent, not post-hoc flexibility.

## 24. Adaptive curriculum iteration

Only after the first trained checkpoint passes its development gate may PVRL test the online-curriculum hypothesis.

The next iteration must:

1. freeze the improved checkpoint as current policy;
2. generate or mutate new candidate tasks without using fresh qualification data;
3. recalibrate candidates against that checkpoint;
4. freeze the new frontier-selected curriculum;
5. perform another bounded RL intervention;
6. compare the second checkpoint against both its parent and the original unchanged base.

The key causal question is whether regenerating the curriculum near the new policy frontier produces another reproducible improvement.

Do not scale merely because training reward increases.

## 25. Repository-scale expansion

Repository/multi-file SWE environments are a later milestone, not a prerequisite for validating PVRL.

Before expansion, the track must already demonstrate:

- environment integrity;
- reward isolation;
- exact multi-turn token continuity;
- deterministic checkpoint provenance;
- positive bounded development evidence;
- adaptive curriculum benefit or a justified reason to scale without it.

Repository-scale work must then add:

- immutable repository snapshots;
- fail-to-pass and pass-to-pass test semantics;
- dependency/cache isolation;
- multi-file patch accounting;
- longer context controls;
- optional compaction;
- stronger anti-cheating analysis;
- explicit public/private test separation.

## 26. Stop rules

PVRL uses fail-closed gates.

Stop model experiments if any of the following occurs:

- FTR evaluation integrity is unresolved;
- unchanged-base identity cannot be reproduced;
- no-op adapter/QLoRA path changes model behavior unexpectedly;
- hidden tests can be read or modified by the solver;
- reference-first validation is non-reproducible;
- harness errors are being converted into candidate failures;
- token-prefix continuity is violated;
- rollout/log-prob/loss-mask alignment is not exact;
- protected benchmark contamination is detected;
- checkpoint identity/resume is ambiguous;
- the first bounded RL experiment harms development behavior beyond the precommitted stop threshold.

Stop curriculum scaling if:

- frontier tasks are too sparse to support the objective;
- reward variance is dominated by harness instability;
- improvement exists only on training/calibration tasks;
- development gain is marginal and fails replication;
- regressions exceed the precommitted non-regression constraint.

## 27. Evidence policy

Every PVRL gate must emit machine-readable evidence plus a short human-readable closure note.

Evidence files must record exact source commit and relevant artifact hashes. Workflow success is not sufficient evidence.

Never mutate prior failed/provisional evidence in place to make a later run look successful. New protocol generations or reruns receive new identities and explicitly supersede earlier evidence.

## 28. Security and operational constraints

Generation/training workflows must be safe for interruption and restart.

For cloud task-authoring jobs:

- checkpoint authoring output periodically;
- persist immutable completed records before advancing cursors;
- use idempotent resume semantics;
- never execute untrusted generated code in a notebook process with Google Drive or credentials mounted;
- transfer generated material to the controlled execution environment for validation.

For the local/self-hosted runner:

- execute untrusted candidate code only in the constrained sandbox;
- do not mount repository credentials into candidate containers;
- preserve artifact evidence before cleanup.

## 29. Reproducibility requirements

Before any result can support a scientific claim, preserve:

- exact source SHA;
- exact model/tokenizer revisions;
- chat-template identity;
- runtime/dependency identity;
- task/environment manifest identities;
- seeds and sampling policy;
- rollout grouping and reward records;
- trainer config;
- checkpoint identities;
- evaluation config/harness identity;
- task-level outputs and outcomes.

Repeated execution should distinguish deterministic identity drift from expected stochastic rollout variance.

## 30. Promotion philosophy

PVRL does not lower the existing promotion bar merely because RL is a new training method.

A candidate model must first show meaningful development improvement. Only then may a new untouched qualification set be frozen and evaluated according to a separately committed promotion protocol.

Existing protected benchmark results that already influenced design remain diagnostic/historical evidence and cannot be relabeled as fresh qualification evidence.

## 31. Initial task categories

The candidate generator should intentionally cover a balanced subset of verifiable Python capabilities, for example:

- exact-spec algorithms and functions;
- parsing/serialization;
- data transformation;
- classes/data structures;
- numerical edge cases;
- stateful logic;
- small debugging/repair tasks;
- bounded file-processing tasks;
- simple CLI behavior where deterministic;
- test-driven bug fixes in tiny synthetic packages.

Framework/API tasks with unstable external dependencies are deferred until the deterministic environment story is strong.

## 32. Non-goals for v0

PVRL v0 does not attempt to:

- reproduce FrogNano's exact distributed system;
- train at 65k/131k context;
- solve SWE-bench directly;
- use unrestricted networking;
- perform broad hyperparameter sweeps;
- treat Qwen3.8-27B outputs as gold student answers;
- use protected benchmark tasks for reward training;
- claim that one successful RL run proves the old SFT method was wrong;
- add asynchronous rollout complexity before synchronous causality is established.

## 33. Required final outputs of the PVRL track

A successful PVRL engineering phase ultimately produces:

1. immutable executable-environment tooling;
2. hardened grader/sandbox isolation;
3. a Leaf-lite coding-agent harness;
4. exact Qwen3.5 multi-turn token-ledger machinery;
5. learnability-frontier calibration tooling;
6. a synchronous QLoRA group-relative RL trainer;
7. checkpoint/resume/provenance hardening;
8. one bounded first-curriculum experiment;
9. one adaptive-curriculum successor experiment if authorized;
10. a fresh qualification decision only after development success;
11. a final causal/scaling report stating what is supported, falsified, or unresolved.

The companion TODO is the authoritative dependency order for implementing this specification.
