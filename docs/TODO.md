# Tiny Qwen Coder — TODO

This TODO operationalizes `docs/SPEC.md` into phased, testable work items for a shared `Qwen/Qwen3.5-4B` base model with interchangeable programming-language LoRA adapters.

Python is the first adapter. TypeScript and Rust follow after the Python pipeline proves the architecture.

Status convention:

- `[ ]` not started
- `[~]` in progress
- `[x]` complete
- `[!]` blocked

Task IDs are stable once implementation begins and SHOULD be referenced in commits and experiment reports.

---

# Phase 0 — Repository Bootstrap

Goal: establish a reproducible, language-neutral Python project with quality gates before adding model-training logic.

## P0-001 — Initialize Python project

- [x] Create `pyproject.toml`.
- [x] Use `uv` for environment/dependency management.
- [x] Define supported Python version.
- [x] Create `src/tiny_qwen_coder/`.
- [x] Create `tests/`.
- [x] Generate and commit `uv.lock`.

Acceptance criteria:

- `uv sync --frozen` succeeds from a clean checkout.
- `python -c "import tiny_qwen_coder"` succeeds inside the project environment.

## P0-002 — Add core dependencies

- [x] Add PyTorch.
- [x] Add Transformers.
- [x] Add Datasets.
- [x] Add PEFT.
- [x] Add TRL.
- [x] Add Accelerate.
- [x] Add test/lint/type-check dependencies.
- [x] Pin through `uv.lock`.

Acceptance criteria:

- Environment resolves from scratch.
- A diagnostic command can report dependency versions.

## P0-003 — Establish language-neutral repository layout

- [x] Add `configs/base/`.
- [x] Add `configs/languages/`.
- [x] Add `configs/data/`.
- [x] Add `configs/train/`.
- [x] Add `configs/eval/`.
- [x] Add `configs/runtime/`.
- [x] Add package modules for model, adapters, languages, data, training, evaluation, runtime, and reporting.
- [x] Add thin scripts for model inspection, data preparation, training, evaluation, inference, and serving.
- [x] Add `tests/unit/`, `tests/integration/`, and `tests/fixtures/`.

Acceptance criteria:

- Layout matches `docs/SPEC.md`.
- No Python-only script architecture such as `train_python.py` is introduced.
- Generic scripts delegate substantive logic to package modules.

## P0-004 — Add `.gitignore`

- [x] Ignore `.venv/` and Python caches.
- [x] Ignore local Hugging Face caches.
- [x] Ignore generated `data/`.
- [x] Ignore generated `artifacts/`.
- [x] Ignore `outputs/` adapters/checkpoints.
- [x] Ignore benchmark sandboxes/logs.

Acceptance criteria:

- Representative training/evaluation artifacts are not staged by default.

## P0-005 — Add quality gates

- [x] Configure Ruff formatting.
- [x] Configure Ruff linting.
- [x] Select/configure mypy or pyright.
- [x] Configure pytest.
- [x] Add package smoke test.

Acceptance criteria:

- Format check passes.
- Lint passes.
- Type checking passes.
- Pytest passes.
- `git diff --check` is clean.

## P0-006 — Add README bootstrap documentation

- [x] Explain the one-base/many-adapters architecture.
- [x] Explain Python-first scope.
- [x] Explain planned TypeScript/Rust adapters.
- [x] Document environment setup.
- [x] Explain CPU development versus GPU training.
- [x] Link `docs/SPEC.md` and `docs/TODO.md`.

Acceptance criteria:

- A new developer can set up the repo and run quality gates from README alone.

## P0-007 — Add bounded GitHub Actions CI

- [x] `uv sync --frozen`.
- [x] Format check.
- [x] Lint.
- [x] Type check.
- [x] CPU-safe tests.
- [x] Avoid required full-model download/training.

Acceptance criteria:

- CI passes on a clean CPU runner.
- Required CI assumes no GPU.

---

# Phase 1 — Canonical Base Model and Configuration Contract

Goal: make the shared base-model identity immutable and make all experiments configuration-driven.

## P1-001 — Define canonical base-model config

- [x] Add `configs/base/qwen35-4b.yaml`.
- [x] Record repository `Qwen/Qwen3.5-4B`.
- [x] Resolve/pin exact Hugging Face revision before canonical training.
- [x] Record tokenizer revision policy.
- [x] Record precision policy.

Acceptance criteria:

- Model identity is represented by repository + exact revision, not model name alone.

## P1-002 — Define language config schema

- [x] Stable language ID.
- [x] aliases/extensions.
- [x] repository-detection signals.
- [x] system prompt/version.
- [x] data-source config references.
- [x] evaluation config references.
- [x] language-specific validator/executor hooks.

Acceptance criteria:

- Python, TypeScript, and Rust configs can be represented without changing the schema.

## P1-003 — Define training/evaluation/run config schemas

- [x] Data-preparation schema.
- [x] LoRA training schema.
- [x] Evaluation schema.
- [x] Runtime/adapter-selection schema.
- [x] Strict unknown-field handling.

Acceptance criteria:

- Invalid config fails before expensive work starts.
- Equivalent config parses deterministically.

## P1-004 — Implement run identity and manifests

- [x] Stable/generated run ID.
- [x] Git commit SHA.
- [x] timestamp.
- [x] base model/revision.
- [x] language.
- [x] adapter family/ID.
- [x] seed.
- [x] dependency versions.
- [x] host/GPU metadata.

Acceptance criteria:

- Every training/evaluation run emits a machine-readable manifest.

## P1-005 — Implement deterministic seeding

- [x] Python random.
- [x] NumPy if used.
- [x] PyTorch CPU/GPU.
- [x] document unavoidable nondeterminism.

Acceptance criteria:

- Dataset shuffle/split is repeatable with the same inputs/seed.

## P1-006 — Implement environment report

- [x] Python version.
- [x] PyTorch/CUDA.
- [x] Transformers/TRL/PEFT/Datasets/Accelerate.
- [x] GPU and VRAM.

Acceptance criteria:

- Report can be generated without training.

---

# Phase 2 — Model Inspection and Adapter Compatibility Foundation

Goal: understand the actual Qwen3.5 architecture and establish a robust adapter contract before training.

## P2-001 — Implement model inspection utility

- [x] Add `scripts/inspect_model.py`.
- [x] Report model class and parameter count.
- [x] Distinguish text backbone, vision encoder, and multimodal/projector modules.
- [x] Enumerate module hierarchy relevant to LoRA.
- [x] Report tokenizer/chat-template metadata.
- [x] Record exact upstream revision.

Acceptance criteria:

- Utility produces human- and machine-readable reports for the canonical base.

## P2-002 — Discover PEFT target modules

- [x] Enumerate linear/projection modules.
- [x] Identify full-attention projections.
- [x] Identify MLP projections.
- [x] Identify hybrid/DeltaNet-specific modules.
- [x] Exclude vision-encoder/projector modules from language LoRA targets by default.
- [x] Define selective-target candidate from observed names.

Acceptance criteria:

- No canonical target list is copied blindly from another architecture.

## P2-003 — Validate `all-linear` strategy

- [x] Instantiate PEFT with `target_modules="all-linear"` or supported equivalent.
- [x] Record matched modules.
- [x] Record trainable parameter count.

Acceptance criteria:

- Selective and all-linear strategies can be compared later.

## P2-004 — Define adapter-manifest schema

- [x] adapter ID.
- [x] family.
- [x] language.
- [x] base repository/revision.
- [x] tokenizer/chat-template identity.
- [x] training Git SHA/config hash.
- [x] dataset manifest IDs.
- [x] LoRA hyperparameters/targets.
- [x] trainable parameters.
- [x] evaluation artifact references.

Acceptance criteria:

- Manifest can represent Python, TypeScript, Rust, and future language adapters.

## P2-005 — Implement compatibility validator

- [x] Compare adapter base repository.
- [x] Compare exact base revision.
- [x] Compare tokenizer/template identity where required.
- [x] Validate expected LoRA architecture metadata.
- [x] Fail closed on base revision mismatch by default.

Acceptance criteria:

- Synthetic incompatible adapter manifests are rejected deterministically.

## P2-006 — Validate chat-template loss masking

- [x] Test pinned Qwen chat template.
- [x] Verify TRL assistant-only mask behavior.
- [x] Add system/user/assistant token test.
- [x] Implement tested completion-only fallback if necessary.

Acceptance criteria:

- Project can prove which tokens receive loss.

## P2-007 — Add canonical model-load smoke test

- [x] Load Qwen3.5-4B in BF16 on compatible GPU.
- [x] Generate deterministic/simple completion.
- [x] Record base-model memory footprint.

Acceptance criteria:

- Canonical base loads/generates successfully with pinned stack.

## P2-008 — Select canonical 4B training memory strategy

- [x] Run BF16 LoRA forward/backward preflight at sequence length 2,048 and micro-batch 1.
- [x] Record peak allocated/reserved VRAM and practical safety headroom.
- [x] If BF16 LoRA is not comfortably memory-safe, validate 4-bit QLoRA with BF16 compute.
- [x] Record quantization details when used, including 4-bit type, double-quantization policy, and compute dtype.
- [x] Freeze the canonical P0 training mode before Phase 7.

Acceptance criteria:

- The 16 GB reference GPU has a measured, reproducible training configuration before the full Python run.
- BF16 versus QLoRA is selected from evidence rather than assumption.

---

# Phase 3 — Generic Language and Dataset Framework

Goal: build reusable infrastructure before embedding Python-specific assumptions.

## P3-001 — Define `LanguageSpec`/language plugin interface

- [ ] Stable ID and aliases.
- [ ] extensions/signals.
- [ ] data adapters.
- [ ] validators.
- [ ] protected benchmarks.
- [ ] execution/evaluation hooks.

Acceptance criteria:

- Dummy Python/TypeScript/Rust specs can be registered through the same interface.

## P3-002 — Implement language registry

- [ ] Register language by ID.
- [ ] Resolve aliases.
- [ ] reject unknown languages clearly.
- [ ] support listing registered languages.

Acceptance criteria:

- Generic commands select language through registry/config rather than `if python` logic spread across modules.

## P3-003 — Define normalized training-record schema

- [ ] messages/system-user-assistant representation.
- [ ] language.
- [ ] source ID/revision.
- [ ] provenance/license metadata.
- [ ] optional validation metadata.

Acceptance criteria:

- Multiple upstream source formats normalize to one internal representation.

## P3-004 — Implement generic required-content filtering

- [ ] Empty prompt rejection.
- [ ] Empty response rejection.
- [ ] malformed record rejection.
- [ ] safe line-ending/encoding normalization.
- [ ] rejection reason accounting.

Acceptance criteria:

- Filters operate independently of programming language.

## P3-005 — Implement tokenizer-aware length filtering

- [ ] Use canonical Qwen tokenizer/template.
- [ ] Record token-length distribution.
- [ ] configurable min/max.
- [ ] explicit truncation policy only.

Acceptance criteria:

- No example is silently truncated.

## P3-006 — Implement exact deduplication

- [ ] normalized prompt/response hashes.
- [ ] source-ID duplicate handling.
- [ ] duplicate statistics.

Acceptance criteria:

- Exact duplicates cannot cross train/validation splits.

## P3-007 — Implement deterministic splitting

- [ ] shuffle with configured seed.
- [ ] split after deduplication.
- [ ] prevent linked duplicates crossing split.

Acceptance criteria:

- Same inputs/config/seed produce same membership.

## P3-008 — Implement dataset manifest

- [ ] language.
- [ ] source IDs/revisions/licenses.
- [ ] code/config/seed identity.
- [ ] counts and rejection reasons.
- [ ] dedup stats.
- [ ] token stats.
- [ ] fingerprints/checksums.
- [ ] contamination findings.

Acceptance criteria:

- Prepared corpus can be audited without reading all examples.

## P3-009 — Add dataset-pipeline tests

- [ ] normalization.
- [ ] length filtering.
- [ ] dedup.
- [ ] deterministic split.
- [ ] manifest generation.
- [ ] plugin-specific validation hook.

Acceptance criteria:

- CPU-only fixtures validate the generic pipeline.

---

# Phase 4 — Generic Evaluation and Execution Framework

Goal: build evaluation infrastructure reusable across languages before Python fine-tuning.

## P4-001 — Create protected benchmark registry

- [ ] Register protected datasets per language.
- [ ] Prevent accidental use by normal training configs.

Acceptance criteria:

- Synthetic attempts to select protected benchmarks for SFT fail clearly.

## P4-002 — Add contamination checks

- [ ] exact normalized prompt matching.
- [ ] exact normalized solution/code matching where possible.
- [ ] suspicious high-overlap reporting.

Acceptance criteria:

- Injected benchmark copies in fixtures are detected.

## P4-003 — Implement constrained execution harness

- [ ] Disposable work directory/container.
- [ ] timeout.
- [ ] CPU/memory bounds where practical.
- [ ] network off by default.
- [ ] no host credentials.
- [ ] stdout/stderr/status capture.

Acceptance criteria:

- Deliberately looping/malicious synthetic candidates cannot trivially compromise or indefinitely hang the host.

## P4-004 — Define common evaluation result schema

- [ ] problem ID.
- [ ] language.
- [ ] generated text/code.
- [ ] parse/compile status.
- [ ] tests passed/total.
- [ ] timeout/error category.
- [ ] generation stats.
- [ ] adapter/base identity.

Acceptance criteria:

- Python, TypeScript, and Rust evaluators can emit the same high-level schema.

## P4-005 — Build general/tool regression suite

- [ ] instruction following.
- [ ] JSON structured output.
- [ ] simple reasoning.
- [ ] shell reasoning.
- [ ] Git reasoning.
- [ ] tool-call formatting/selection.

Acceptance criteria:

- Suite is frozen before first canonical Python LoRA evaluation.

## P4-006 — Freeze generation/evaluation settings

- [ ] temperature.
- [ ] top-p/top-k as applicable.
- [ ] seed.
- [ ] max new tokens.
- [ ] stopping behavior.
- [ ] prompt/template version.

Acceptance criteria:

- Base and adapters can be evaluated identically.

---

# Phase 5 — Python Dataset Pipeline

Goal: implement the first language plugin and prepare the canonical Python P0 corpus.

## P5-001 — Add Python language config/plugin

- [ ] `python` ID and `.py` extension.
- [ ] repository detection signals (`pyproject.toml`, `uv.lock`, etc.).
- [ ] versioned Python system prompt.
- [ ] Python validator hooks.

Acceptance criteria:

- Generic pipeline can select Python entirely through language config/registry.

## P5-002 — Add OLMo Python-instruct loader

- [ ] Load `OLMo-Coding/starcoder-python-instruct`.
- [ ] Pin/record upstream revision.
- [ ] preserve provenance.
- [ ] retain Python 3 according to source metadata.

Acceptance criteria:

- Small source sample normalizes through generic schema.

## P5-003 — Add Magicoder Python loader

- [ ] Load `ise-uiuc/Magicoder-OSS-Instruct-75K`.
- [ ] pin/record revision.
- [ ] select Python rows.
- [ ] preserve provenance.

Acceptance criteria:

- Small sample normalizes through generic schema.

## P5-004 — Add Python-specific quality checks

- [ ] Python 2 exclusion.
- [ ] conservative `ast.parse()` validation for standalone code.
- [ ] preserve legitimate snippets/diffs/fragments.
- [ ] record validation metadata/rejection reason.

Acceptance criteria:

- Complete invalid Python examples can be detected without blindly deleting valid fragments.

## P5-005 — Create P0 Python corpus

Target:

- ~30k accepted OLMo Python 3 examples.
- ~10k accepted Magicoder Python examples when available.
- ~40k total before 95/5 split.

Tasks:

- [ ] Apply filters before final counts.
- [ ] Fill shortfall only according to explicit config.
- [ ] Emit actual composition.

Acceptance criteria:

- Final counts are measured, not assumed.

## P5-006 — Emit/freeze Python P0 dataset manifest

- [ ] revisions/licenses.
- [ ] filtering stats.
- [ ] dedup stats.
- [ ] train/validation membership fingerprints.
- [ ] token stats.
- [ ] contamination results.

Acceptance criteria:

- P0 dataset is reproducibly defined before training.

---

# Phase 6 — Python Baseline Evaluation

Goal: freeze the unchanged-base Python and regression baseline before any LoRA training.

## P6-001 — Register Python protected benchmarks

- [ ] HumanEval.
- [ ] MBPP.
- [ ] repository-owned holdout suite.

Acceptance criteria:

- All are inaccessible to normal SFT data configs.

## P6-002 — Implement HumanEval evaluator

- [ ] Select/pin maintained implementation.
- [ ] normalize prompting.
- [ ] constrained execution.
- [ ] per-problem and aggregate output.

Acceptance criteria:

- Repeated deterministic runs are reproducible within framework limits.

## P6-003 — Implement MBPP evaluator

- [ ] Select/pin maintained implementation.
- [ ] constrained test execution.
- [ ] common result schema.

Acceptance criteria:

- Produces comparable per-problem/aggregate artifacts.

## P6-004 — Build custom Python suite

Include representative tasks covering:

- [ ] standard library/data transforms.
- [ ] pathlib.
- [ ] JSON.
- [ ] regex.
- [ ] dataclasses/typing.
- [ ] generators/decorators/context managers.
- [ ] exceptions.
- [ ] async/await.
- [ ] subprocess logic.
- [ ] SQLite.
- [ ] pytest-oriented tasks.

Acceptance criteria:

- Tests are deterministic, executable, repo-owned, and excluded from training.

## P6-005 — Run unchanged-base Python baseline

- [ ] HumanEval.
- [ ] MBPP.
- [ ] custom suite.
- [ ] general/tool regression suite.
- [ ] memory/performance metadata.

Acceptance criteria:

- Baseline artifacts are complete and frozen before P0 adapter training.

---

# Phase 7 — Python P0 LoRA

Goal: produce the first language adapter using the training mode frozen by the 4B GPU memory preflight.

## P7-001 — Finalize selective LoRA targets

- [ ] Use P2 inspection results.
- [ ] record exact modules.
- [ ] record trainable parameter count.

Acceptance criteria:

- Target set is architecture-verified.

## P7-002 — Implement generic adapter trainer

- [ ] Base revision from canonical config.
- [ ] language from registry/config.
- [ ] LoRA parameters from config.
- [ ] assistant/completion-only loss.
- [ ] checkpoint/log output.
- [ ] run + adapter manifests.

Acceptance criteria:

- Trainer contains no hard-coded Python-only training logic.

## P7-003 — Add P0 Python training config

Initial candidate:

```text
training mode           frozen P2-008 selection
compute dtype           BF16
sequence length         2048
LoRA rank               16
LoRA alpha              32
LoRA dropout            0.05
bias                    none
learning rate           2e-4
scheduler               cosine
warmup ratio            0.03
epochs                  1
gradient checkpointing  enabled when compatible
loss                    assistant/completion only
```

- [ ] Micro-batch selected empirically.
- [ ] Gradient accumulation selected empirically.

Acceptance criteria:

- Config is fully machine-readable and captured in run manifest.

## P7-004 — Add preflight validation

- [ ] Canonical base revision matches config.
- [ ] Dataset manifest is frozen/compatible.
- [ ] assistant-only masking verified.
- [ ] LoRA targets match modules.
- [ ] output path safe.
- [ ] GPU/training-mode compatibility reported.

Acceptance criteria:

- Invalid setup fails before expensive training.

## P7-005 — Run short GPU smoke training

- [ ] Bounded sample/steps.
- [ ] forward/backward succeeds.
- [ ] loss finite.
- [ ] checkpoint/adapter saves.
- [ ] peak VRAM recorded.

Acceptance criteria:

- Training path works end-to-end before full P0 run.

## P7-006 — Run full Python P0 training

- [ ] Train on frozen P0 corpus.
- [ ] record loss/throughput.
- [ ] record peak allocated/reserved VRAM.
- [ ] save LoRA adapter and manifests.

Acceptance criteria:

- Run completes without OOM/NaN.
- Output is a LoRA adapter, not a merged full model.

## P7-007 — Validate adapter load/inference

- [ ] Load canonical base.
- [ ] attach Python adapter.
- [ ] generate fixed smoke prompts.
- [ ] disable adapter and recover base behavior.

Acceptance criteria:

- Adapter can be enabled/disabled without rebuilding the full model.

---

# Phase 8 — Python P0 Evaluation and Promotion Decision

Goal: determine whether the first adapter genuinely improved Python without unacceptable regressions.

## P8-001 — Evaluate Python P0 adapter

- [ ] HumanEval.
- [ ] MBPP.
- [ ] custom Python suite.
- [ ] same frozen generation config as baseline.

Acceptance criteria:

- Direct base-vs-adapter comparison artifact generated.

## P8-002 — Run general/tool regression suite

- [ ] instruction following.
- [ ] JSON.
- [ ] reasoning.
- [ ] shell/Git.
- [ ] tool-call formatting/selection.

Acceptance criteria:

- Regressions quantified, not hand-waved.

## P8-003 — Add preliminary cross-language smoke tests

- [ ] small TypeScript prompts.
- [ ] small Rust prompts.
- [ ] compare base vs Python adapter.

Acceptance criteria:

- Catastrophic non-Python collapse would be detected before promotion.

## P8-004 — Write P0 experiment report

Include:

- [ ] dataset identity.
- [ ] training config.
- [ ] VRAM/throughput.
- [ ] baseline metrics.
- [ ] adapter metrics.
- [ ] regressions.
- [ ] qualitative examples only as supplemental evidence.

Acceptance criteria:

- Report supports a clear promote/reject/iterate decision.

## P8-005 — Promote or reject P0 adapter

- [ ] Establish quantitative promotion thresholds using actual baseline evidence.
- [ ] Mark recommended Python adapter ID if accepted.
- [ ] Preserve rejected experiment metadata.

Acceptance criteria:

- "Recommended" has an explicit evidentiary meaning.

---

# Phase 9 — Python LoRA Experiment Matrix

Goal: improve the Python adapter through controlled experiments.

## P9-001 — Rank sweep

- [ ] r=8.
- [ ] r=16 baseline.
- [ ] r=32.
- [ ] r=64.

Acceptance criteria:

- All other major variables held fixed.
- Comparative report produced.

## P9-002 — Target-module sweep

- [ ] selective baseline.
- [ ] Define a genuinely distinct language-only target candidate, if justified by P2 evidence.
- [ ] Treat literal PEFT `all-linear` as an explicit multimodal-scope experiment only; otherwise record it as not applicable.

Acceptance criteria:

- Trainable parameters, VRAM, speed, and benchmark deltas are compared for genuinely distinct target sets.
- Set-equivalent target configurations discovered by P2-003 are not misrepresented as separate experiments.

## P9-003 — Dataset-size sweep

Candidates:

- [ ] 10k.
- [ ] 25k.
- [ ] 50k.
- [ ] 100k+ if justified.

Acceptance criteria:

- Learning curve/diminishing-return evidence produced.

## P9-004 — Training-length/learning-rate study

- [ ] additional epoch/step options.
- [ ] lower/higher LR candidates.

Acceptance criteria:

- Overfitting/general regression explicitly measured.

## P9-005 — BF16 LoRA vs QLoRA comparison

- [ ] If P0 used BF16 LoRA, add a controlled 4-bit QLoRA comparison.
- [ ] If P0 used QLoRA, add a BF16 comparison only when measured memory headroom makes it safe.
- [ ] compare VRAM/speed/quality.

Acceptance criteria:

- Quantization effect is isolated from unrelated changes.

## P9-006 — Select recommended Python adapter

Acceptance criteria:

- Best adapter selected from target-language gain, general/tool preservation, cross-language behavior, VRAM, and speed—not training loss alone.

---

# Phase 10 — Runtime Adapter Manager and Hot Switching

Goal: prove the core product concept: one resident base model with interchangeable LoRA adapters.

## P10-001 — Implement adapter registry

- [ ] discover/register adapter manifests.
- [ ] lookup by family/language/ID.
- [ ] identify recommended adapter per language.

Acceptance criteria:

- Registry works without hard-coded Python paths.

## P10-002 — Implement adapter load/activate/disable API

- [ ] load adapter.
- [ ] activate adapter.
- [ ] disable adapters/base-only mode.
- [ ] query active adapter.

Acceptance criteria:

- Base model remains resident during supported adapter switches.

## P10-003 — Enforce compatibility during load

- [ ] exact base revision.
- [ ] tokenizer/template constraints.
- [ ] adapter schema version.

Acceptance criteria:

- Incompatible synthetic adapter is rejected before activation.

## P10-004 — Test repeated switching

Scenario:

```text
base → python → base → python → base
```

- [ ] outputs reproducible under deterministic settings.
- [ ] no cumulative adapter leakage.
- [ ] bounded VRAM.

Acceptance criteria:

- Switching behavior is stable over repeated cycles.

## P10-005 — Add CLI/runtime selection

Examples conceptually:

```text
--adapter python
--adapter none
```

Acceptance criteria:

- User can explicitly choose active language adapter.

---

# Phase 11 — TypeScript Adapter

Goal: prove that the framework generalizes to a second programming language without architectural rewrites.

## P11-001 — Add TypeScript language plugin/config

- [ ] `.ts`/`.tsx` extensions.
- [ ] `package.json`/`tsconfig.json` signals.
- [ ] versioned system prompt.
- [ ] TypeScript validation/execution hooks.

Acceptance criteria:

- Generic commands support `--language typescript` or equivalent.

## P11-002 — Survey/select TypeScript training datasets

- [ ] identify suitable open datasets.
- [ ] inspect licensing/provenance.
- [ ] distinguish TypeScript from JavaScript when possible.
- [ ] decide `.tsx` scope.
- [ ] pin revisions.

Acceptance criteria:

- Dataset decision documented before canonical training.

## P11-003 — Implement TypeScript quality filters

- [ ] parsing/syntax checks where safe.
- [ ] optional type-check validation for complete units.
- [ ] preserve valid snippets requiring project context.

Acceptance criteria:

- Quality checks do not assume every example is a standalone project.

## P11-004 — Define TypeScript protected benchmarks/eval suite

Cover representative areas:

- [ ] type system/generics.
- [ ] async/Promise.
- [ ] Node APIs.
- [ ] modules.
- [ ] transformations/error handling.
- [ ] `.tsx` if in scope.
- [ ] compiler/tests.

Acceptance criteria:

- Evaluation frozen before canonical TypeScript training.

## P11-005 — Run unchanged-base TypeScript baseline

Acceptance criteria:

- Complete baseline artifact exists.

## P11-006 — Train TypeScript P0 adapter

- [ ] Use same canonical base revision.
- [ ] Start from shared LoRA architecture/hyperparameter baseline unless evidence requires change.
- [ ] emit adapter manifest.

Acceptance criteria:

- Adapter trains without language-specific trainer fork.

## P11-007 — Evaluate/promote TypeScript adapter

- [ ] target-language suite.
- [ ] general/tool suite.
- [ ] Python/Rust cross-language tests.

Acceptance criteria:

- Promotion follows same evidence contract as Python.

## P11-008 — Validate Python↔TypeScript switching

Scenario:

```text
base → python → typescript → python → base
```

Acceptance criteria:

- Switches succeed without base-model reload and without state leakage.

---

# Phase 12 — Rust Adapter

Goal: prove reuse on a compiler-heavy systems language and exploit compiler feedback for evaluation.

## P12-001 — Add Rust language plugin/config

- [ ] `.rs` extension.
- [ ] `Cargo.toml`/`Cargo.lock` signals.
- [ ] versioned system prompt.
- [ ] Rust validation/execution hooks.

Acceptance criteria:

- Generic commands support Rust without trainer/evaluator architecture fork.

## P12-002 — Survey/select Rust training datasets

- [ ] identify suitable open datasets.
- [ ] inspect licensing/provenance.
- [ ] pin revisions.
- [ ] decide snippet/project balance.

Acceptance criteria:

- Dataset decision documented before canonical training.

## P12-003 — Implement Rust quality filters

- [ ] parse/compile checks where safe.
- [ ] preserve snippets requiring context.
- [ ] record compiler-validation metadata where used.

Acceptance criteria:

- Complete invalid examples can be detected without rejecting valid contextual snippets wholesale.

## P12-004 — Define Rust protected benchmarks/eval suite

Cover:

- [ ] ownership/borrowing.
- [ ] lifetimes where appropriate.
- [ ] traits/generics.
- [ ] iterators.
- [ ] `Result`/error handling.
- [ ] concurrency.
- [ ] async if in scope.
- [ ] Cargo project edits.
- [ ] compiler/tests.
- [ ] Clippy where appropriate.

Acceptance criteria:

- Evaluation frozen before canonical Rust training.

## P12-005 — Run unchanged-base Rust baseline

Acceptance criteria:

- Complete baseline artifact exists.

## P12-006 — Train Rust P0 adapter

- [ ] same canonical base revision.
- [ ] shared LoRA architecture baseline.
- [ ] generic trainer.
- [ ] adapter manifest.

Acceptance criteria:

- Rust adapter is another artifact family member, not another full model.

## P12-007 — Evaluate/promote Rust adapter

- [ ] target-language suite.
- [ ] general/tool suite.
- [ ] Python/TypeScript cross-language tests.

Acceptance criteria:

- Promotion follows common evidence contract.

## P12-008 — Validate three-way switching

Scenario:

```text
base → python → typescript → rust → python → base
```

Acceptance criteria:

- Same base remains loaded.
- Active adapter identity is correct after every transition.
- VRAM remains bounded.

---

# Phase 13 — Full Cross-Language Study

Goal: quantify specialization, interference, and storage/runtime benefits across all canonical adapters.

## P13-001 — Freeze cross-language matrix

Rows:

- base/no adapter.
- Python adapter.
- TypeScript adapter.
- Rust adapter.

Columns:

- Python suite.
- TypeScript suite.
- Rust suite.
- general/tool suite.

Acceptance criteria:

- Every cell is evaluated under controlled generation settings.

## P13-002 — Produce specialization/interference report

- [ ] target-language gain.
- [ ] unrelated-language regression.
- [ ] general/tool regression.
- [ ] adapter size.
- [ ] switching latency.
- [ ] VRAM impact.

Acceptance criteria:

- Tradeoffs are quantitatively visible.

## P13-003 — Compare storage footprint

Compare:

```text
one base + 3 adapters
```

versus hypothetical:

```text
3 independent full model copies
```

Acceptance criteria:

- Actual artifact sizes reported.

## P13-004 — Establish canonical recommended adapters

- [ ] Python.
- [ ] TypeScript.
- [ ] Rust.

Acceptance criteria:

- Each recommendation points to exact manifest/config/evaluation artifacts.

---

# Phase 14 — Automatic Selection and Polyglot Repositories

Goal: make adapter selection useful in real development workflows while preserving explicit control.

## P14-001 — Implement repository language detection

Signals include:

- Python: `pyproject.toml`, `.py`, `uv.lock`.
- TypeScript: `package.json`, `tsconfig.json`, `.ts`, `.tsx`.
- Rust: `Cargo.toml`, `.rs`.

Acceptance criteria:

- Deterministic scoring/result for synthetic repositories.

## P14-002 — Add explicit override precedence

Acceptance criteria:

- User-selected adapter always wins over automatic detection.

## P14-003 — Handle ambiguous/polyglot repositories

- [ ] report multiple detected languages.
- [ ] avoid pretending one language owns the whole repo.
- [ ] support explicit task-level selection.

Acceptance criteria:

- Tauri-style Rust+TypeScript fixture is recognized as polyglot.

## P14-004 — Explore file-sensitive selection

Acceptance criteria:

- Switching based on current target file is experimentally evaluated, not enabled blindly.

## P14-005 — Explore dynamic switching during agent loops

Acceptance criteria:

- Conversation context remains coherent.
- Adapter state does not leak/accumulate.
- Benefit over fixed-adapter behavior is measured.

---

# Phase 15 — Tool/Agent Specialization

Goal: evolve language-specialized coding assistants into verified repo-level coding agents.

## P15-001 — Define coding-agent tool contract

Candidate constrained tools:

- [ ] list files.
- [ ] read file.
- [ ] search.
- [ ] apply patch/write file.
- [ ] run bounded command.
- [ ] inspect diff.

Acceptance criteria:

- Tool schemas are stable and testable.

## P15-002 — Build agent evaluation tasks

Tasks SHOULD require:

- repository inspection.
- edit generation.
- formatter/compiler/test execution.
- recovery after failure.

Acceptance criteria:

- Success is executable and deterministic where practical.

## P15-003 — Record agent baseline per language

- [ ] base/no adapter.
- [ ] language adapter.

Acceptance criteria:

- Existing language adapters' agent behavior is known before additional agent training.

## P15-004 — Build verified trajectory format

Include:

- task.
- tool definitions.
- tool calls.
- tool outputs.
- edits.
- validation failures.
- repairs.
- final verified success.

Acceptance criteria:

- Trajectories can be validated and normalized for SFT.

## P15-005 — Investigate behavior specialization strategy

Compare possibilities such as:

- agent training folded into each language adapter.
- separate behavior adapter.
- sequential training.
- future composition/fusion.

Acceptance criteria:

- Strategy chosen from measured interference/compatibility, not convenience alone.

## P15-006 — Train/evaluate agent experiment

Metrics:

- tool-call validity.
- correct tool selection.
- patch success.
- compiler/test success.
- recovery rate.
- turns/tokens.
- loop/premature-done rate.

Acceptance criteria:

- Experiment beats its frozen baseline on repo-level tasks to be considered successful.

---

# Phase 16 — OpenCode-Compatible Serving

Goal: serve the shared base and active adapter through a client-compatible local endpoint.

## P16-001 — Select local serving backend

Evaluate then-current support for:

- Qwen3.5 architecture.
- LoRA adapters.
- adapter switching.
- tool-call parsing.
- OpenAI-compatible APIs.

Acceptance criteria:

- Backend selection documented from current verified capability.

## P16-002 — Implement serving configuration

- [ ] canonical base model.
- [ ] adapter registry.
- [ ] active adapter selection.
- [ ] OpenAI-compatible endpoint.

Acceptance criteria:

- Inference works through HTTP/client path, not only direct Python.

## P16-003 — Validate tool calling through serving stack

Acceptance criteria:

- Tool-call structure survives model → server → client round trip.

## P16-004 — Connect OpenCode

- [ ] document provider config.
- [ ] explicit language-adapter workflow.
- [ ] run disposable repo tasks.

Acceptance criteria:

- OpenCode can interact with local Tiny Qwen Coder endpoint.

## P16-005 — Evaluate real repository tasks

Measure:

- task success.
- test/build success.
- tool validity.
- turns.
- tokens.
- latency.
- loops/premature completion.

Acceptance criteria:

- OpenCode readiness is based on executable repo-level evidence.

---

# Phase 17 — Continued Pretraining Research

Goal: determine whether raw language-specific source-code adaptation improves later LoRA SFT.

## P17-001 — Define raw-code license/provenance policy

Acceptance criteria:

- Allowed source/license policy documented before ingesting large corpora.

## P17-002 — Build language-specific raw-code pipeline

- [ ] Python first.
- [ ] TypeScript later.
- [ ] Rust later.

Acceptance criteria:

- Provenance/dedup/contamination controls remain auditable.

## P17-003 — Run bounded Python CPT experiment

Acceptance criteria:

- Compare base→SFT versus base→CPT→SFT under frozen eval.

## P17-004 — Extend only if evidence supports it

Acceptance criteria:

- TypeScript/Rust CPT work is justified by Python evidence or language-specific rationale.

---

# Phase 18 — Adapter Composition/Fusion Research

Goal: investigate advanced composition only after independent language adapters are stable.

## P18-001 — Define composition hypotheses

Candidates:

- language + behavior adapter.
- two language adapters in polyglot task.
- weighted/fused adapters.

Acceptance criteria:

- Each experiment has an explicit expected benefit and failure mode.

## P18-002 — Build composition regression matrix

Acceptance criteria:

- Single adapters remain controls.

## P18-003 — Test language + behavior composition

Acceptance criteria:

- Composition must outperform or provide a meaningful tradeoff versus retraining a combined adapter.

## P18-004 — Test multi-language composition

Acceptance criteria:

- No claim of successful fusion without executable polyglot evidence.

---

# Phase 19 — Packaging, Publication, and Reproducibility Closure

Goal: make successful adapters and results reusable without duplicating the full base model.

## P19-001 — Define adapter release format

Include:

- binary LoRA weights.
- adapter manifest.
- base-model exact revision.
- training config.
- evaluation summary.
- license/provenance notes.

Acceptance criteria:

- A user can determine exactly which base checkpoint is required.

## P19-002 — Add reproducibility command/documentation

Acceptance criteria:

- From repo + upstream sources + pinned config, a user can reconstruct a selected adapter experiment.

## P19-003 — Publish canonical comparison report

Include:

- Python results.
- TypeScript results.
- Rust results.
- cross-language matrix.
- general/tool regressions.
- adapter sizes.
- VRAM/training performance.
- switching behavior.

Acceptance criteria:

- Claims are traceable to machine-readable artifacts.

## P19-004 — Verify central project invariant

The final system MUST demonstrate:

```text
one pinned Qwen3.5-4B base model
+
small interchangeable Python/TypeScript/Rust LoRA adapters
```

without requiring three independent full model copies.

Acceptance criteria:

- Base is loaded once in the runtime test.
- Python, TypeScript, and Rust adapters can each be activated.
- Base-only mode can be restored.
- Exact adapter/base compatibility is enforced.
- Cross-language evaluation results are published.

---

# Immediate execution order

The next implementation sequence is:

1. `P0-001` through `P0-007` — bootstrap repository and CI.
2. `P1-*` — pin the canonical base/configuration contract.
3. `P2-*` — inspect Qwen3.5 and define adapter compatibility/LoRA targeting.
4. `P3-*` and `P4-*` — build generic language/data/evaluation infrastructure.
5. `P5-*` and `P6-*` — prepare Python data and freeze the baseline.
6. `P7-*` and `P8-*` — train and evaluate the first Python LoRA.
7. `P9-*` — improve/select the recommended Python adapter.
8. `P10-*` — prove adapter hot switching on one resident base.
9. `P11-*` — TypeScript adapter.
10. `P12-*` — Rust adapter.
11. `P13-*` onward — cross-language, polyglot, agent, OpenCode, and advanced research.

The project MUST resist introducing TypeScript/Rust-specific forks before the shared framework is proven. Python is the first specialization, not the architecture of the system.
