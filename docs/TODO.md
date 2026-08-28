# Tiny Qwen Coder — TODO

This TODO operationalizes `docs/SPEC.md` into phased, testable work items.

Status convention:

- `[ ]` not started
- `[~]` in progress
- `[x]` complete
- `[!]` blocked

Task IDs are stable and SHOULD be referenced in commits and experiment reports.

---

# Phase 0 — Repository Bootstrap

Goal: establish a reproducible Python project with quality gates before adding model-training logic.

## P0-001 — Initialize Python project

- [ ] Create `pyproject.toml`.
- [ ] Use `uv` for environment and dependency management.
- [ ] Define the supported Python version.
- [ ] Create the `src/tiny_qwen_coder/` package.
- [ ] Create `tests/`.
- [ ] Generate and commit `uv.lock`.

Acceptance criteria:

- `uv sync --frozen` succeeds from a clean checkout.
- `python -c "import tiny_qwen_coder"` succeeds inside the project environment.

## P0-002 — Add project dependencies

- [ ] Add PyTorch.
- [ ] Add Transformers.
- [ ] Add Datasets.
- [ ] Add PEFT.
- [ ] Add TRL.
- [ ] Add Accelerate.
- [ ] Add test/lint/type-check dependencies.
- [ ] Record dependency versions through `uv.lock`.

Acceptance criteria:

- The environment resolves from scratch.
- A dependency-version diagnostic command prints the versions used by an experiment.

## P0-003 — Establish repository layout

- [ ] Add `configs/data/`.
- [ ] Add `configs/train/`.
- [ ] Add `configs/eval/`.
- [ ] Add `scripts/`.
- [ ] Add package modules for configuration, data, training, evaluation, and reporting.
- [ ] Add `tests/unit/`, `tests/integration/`, and `tests/fixtures/`.

Acceptance criteria:

- Layout matches the intent of `docs/SPEC.md`.
- Scripts contain thin entry points and delegate substantive logic to package modules.

## P0-004 — Add `.gitignore`

- [ ] Ignore `.venv/`.
- [ ] Ignore Python caches.
- [ ] Ignore local Hugging Face caches.
- [ ] Ignore `data/` generated artifacts.
- [ ] Ignore `artifacts/` generated artifacts.
- [ ] Ignore `outputs/` checkpoints/adapters.
- [ ] Ignore local benchmark sandboxes and logs.

Acceptance criteria:

- A representative local training/evaluation run does not leave large/generated files staged by default.

## P0-005 — Add developer quality gates

- [ ] Configure Ruff formatting.
- [ ] Configure Ruff linting.
- [ ] Select and configure a static type checker.
- [ ] Configure pytest.
- [ ] Add a minimal smoke test.

Acceptance criteria:

- Formatting check passes.
- Lint passes.
- Type checking passes.
- Pytest passes.
- `git diff --check` is clean.

## P0-006 — Add README bootstrap documentation

- [ ] Explain the project objective.
- [ ] Explain that P0 starts from `Qwen/Qwen3.5-0.8B`.
- [ ] Explain that the Base checkpoint is a later comparison.
- [ ] Document environment setup.
- [ ] Document CPU-only development versus GPU training expectations.
- [ ] Link `docs/SPEC.md` and `docs/TODO.md`.

Acceptance criteria:

- A new developer can create the environment and run the quality gates using only the README.

## P0-007 — Add bounded GitHub Actions CI

- [ ] Run `uv sync --frozen`.
- [ ] Run format check.
- [ ] Run lint.
- [ ] Run type checking.
- [ ] Run CPU-safe tests.
- [ ] Avoid downloading/training Qwen in required CI.

Acceptance criteria:

- CI passes on a clean CPU runner.
- No required CI job assumes a GPU.

---

# Phase 1 — Configuration and Reproducibility Foundation

Goal: make experiments configuration-driven and auditable.

## P1-001 — Define configuration schema

- [ ] Define data-preparation configuration.
- [ ] Define training configuration.
- [ ] Define evaluation configuration.
- [ ] Define run/output configuration.
- [ ] Validate unknown/invalid fields instead of silently ignoring them.

Acceptance criteria:

- Invalid configuration fails with a clear error before expensive work starts.
- Equivalent config files parse deterministically.

## P1-002 — Implement run identity

- [ ] Generate or accept a stable run ID.
- [ ] Record Git commit SHA when available.
- [ ] Record timestamp.
- [ ] Record host/GPU metadata.
- [ ] Record dependency versions.
- [ ] Record seed.

Acceptance criteria:

- Every training/evaluation run emits a machine-readable manifest with a unique identity.

## P1-003 — Implement deterministic seeding

- [ ] Seed Python random.
- [ ] Seed NumPy if used.
- [ ] Seed PyTorch CPU/GPU.
- [ ] Configure deterministic behavior where practical.
- [ ] Document unavoidable nondeterminism.

Acceptance criteria:

- Dataset shuffling/splitting is repeatable with the same seed.

## P1-004 — Implement environment report

- [ ] Record Python version.
- [ ] Record PyTorch version.
- [ ] Record CUDA version.
- [ ] Record Transformers/TRL/PEFT versions.
- [ ] Record GPU name and total VRAM.

Acceptance criteria:

- The report can be generated without starting a training run.

---

# Phase 2 — Model Inspection and Loading

Goal: inspect the actual Qwen3.5 architecture before committing to LoRA targets.

## P2-001 — Implement model inspection utility

- [ ] Add `scripts/inspect_model.py`.
- [ ] Load tokenizer/model metadata.
- [ ] Report model class.
- [ ] Report total parameter count.
- [ ] Report module hierarchy relevant to LoRA.
- [ ] Report tokenizer/chat-template information.
- [ ] Record model revision/commit where available.

Acceptance criteria:

- Running the utility on `Qwen/Qwen3.5-0.8B` produces a stable machine-readable and human-readable report.

## P2-002 — Discover PEFT target modules

- [ ] Enumerate linear/projection modules.
- [ ] Identify conventional attention projection names that actually exist.
- [ ] Identify MLP projections that actually exist.
- [ ] Identify hybrid/DeltaNet-specific linear modules.
- [ ] Define a selective-target candidate from observed module names.

Acceptance criteria:

- No LoRA target in P0 is chosen solely by copying a Llama/Qwen recipe without validating it against the loaded model.

## P2-003 — Validate `all-linear` compatibility

- [ ] Build a PEFT configuration using `target_modules="all-linear"` or supported equivalent.
- [ ] Verify which modules are adapted.
- [ ] Record trainable parameter count.
- [ ] Add a unit/integration test around target discovery that does not require full GPU training.

Acceptance criteria:

- Selective and all-linear strategies can be instantiated and compared later.

## P2-004 — Validate chat-template loss masking

- [ ] Test the pinned tokenizer chat template.
- [ ] Verify whether TRL `assistant_only_loss=True` yields a valid assistant mask.
- [ ] Add a test with system/user/assistant tokens.
- [ ] If unsupported, implement and test the completion-only fallback defined in `SPEC.md`.

Acceptance criteria:

- The project can prove which tokens receive loss.
- It does not silently train user/system tokens while claiming assistant-only SFT.

## P2-005 — Add model load smoke test

- [ ] Load Qwen3.5-0.8B in BF16 on a compatible GPU.
- [ ] Generate a deterministic/simple completion.
- [ ] Record actual base-model memory footprint.

Acceptance criteria:

- Model loads and generates successfully with the pinned stack.

---

# Phase 3 — Dataset Pipeline

Goal: construct a deterministic, auditable Python instruction SFT corpus.

## P3-001 — Add upstream dataset adapters

- [ ] Add loader for `OLMo-Coding/starcoder-python-instruct`.
- [ ] Add loader for `ise-uiuc/Magicoder-OSS-Instruct-75K`.
- [ ] Normalize source fields behind a common internal schema.
- [ ] Preserve source IDs and provenance fields.

Acceptance criteria:

- Small streamed/sampled fixtures from each source normalize into the same internal representation.

## P3-002 — Pin/record upstream revisions

- [ ] Record Hugging Face dataset revision or commit information when available.
- [ ] Fail or warn when a requested pinned revision cannot be resolved.
- [ ] Store revision information in the dataset manifest.

Acceptance criteria:

- A prepared dataset identifies the exact upstream versions used as closely as the upstream API permits.

## P3-003 — Filter Python 2

- [ ] Use OLMo source metadata to retain Python 3 rows.
- [ ] Apply equivalent source-specific Python-language/version filters to Magicoder where possible.
- [ ] Record rejection counts.

Acceptance criteria:

- Known Python 2-tagged examples cannot enter the P0 training set.

## P3-004 — Validate required content

- [ ] Reject empty instructions.
- [ ] Reject empty responses.
- [ ] Reject malformed records.
- [ ] Normalize line endings and obvious encoding artifacts where safe.
- [ ] Record each rejection reason.

Acceptance criteria:

- Every accepted SFT example has a non-empty user instruction and assistant response.

## P3-005 — Implement length filtering

- [ ] Tokenize with the actual Qwen tokenizer/template.
- [ ] Record token-length distributions.
- [ ] Enforce configurable min/max thresholds.
- [ ] Ensure P0 examples fit the configured 2,048-token training sequence or have an explicit truncation policy.

Acceptance criteria:

- No example is silently truncated without the configured policy and reporting.

## P3-006 — Implement exact deduplication

- [ ] Deduplicate exact normalized prompt/response pairs.
- [ ] Detect duplicate source IDs when applicable.
- [ ] Record duplicate counts.

Acceptance criteria:

- Exact duplicate records cannot appear in both train and validation splits.

## P3-007 — Implement conservative syntax-quality checks

- [ ] Detect standalone Python responses that can be safely passed to `ast.parse()`.
- [ ] Record syntax-validity metadata.
- [ ] Reject clearly broken standalone-code examples under a configurable policy.
- [ ] Preserve legitimate fragments/diffs/REPL snippets instead of blindly parsing everything as a module.

Acceptance criteria:

- Syntax filtering improves quality without systematically eliminating valid non-module examples.

## P3-008 — Normalize to conversational SFT format

- [ ] Add versioned system prompt.
- [ ] Map instruction to `user`.
- [ ] Map solution to `assistant`.
- [ ] Preserve provenance separately from training messages.

Acceptance criteria:

- Records can be rendered through the exact Qwen chat template used during training.

## P3-009 — Build deterministic split logic

- [ ] Shuffle deterministically with configured seed.
- [ ] Split after deduplication.
- [ ] Default P0 split to 95/5 unless config changes it.
- [ ] Ensure source-linked duplicates do not cross split boundaries.

Acceptance criteria:

- Re-running the same revision/config/seed yields the same split membership.

## P3-010 — Create P0 dataset composition

Target:

- approximately 30k accepted OLMo Python3 rows
- approximately 10k accepted Magicoder Python rows
- approximately 40k total before 95/5 split

Tasks:

- [ ] Build the source-selection policy.
- [ ] Apply filtering before final count.
- [ ] If Magicoder yields fewer accepted Python examples than the target, document the shortfall and fill only according to configured policy.

Acceptance criteria:

- Final counts are reported from actual accepted data, not assumed targets.

## P3-011 — Emit dataset manifest

- [ ] Source IDs and revisions.
- [ ] Licenses as reported upstream.
- [ ] seed/config hash.
- [ ] source row counts.
- [ ] rejection counts by reason.
- [ ] accepted counts.
- [ ] train/validation counts.
- [ ] dedup stats.
- [ ] token-length stats.
- [ ] dataset fingerprints/checksums where available.

Acceptance criteria:

- Every prepared dataset can be audited without examining the entire generated corpus.

## P3-012 — Add dataset pipeline tests

- [ ] Unit tests for source normalization.
- [ ] Unit tests for Python 2 exclusion.
- [ ] Unit tests for length filtering.
- [ ] Unit tests for deduplication.
- [ ] Unit tests for split determinism.
- [ ] Unit tests for syntax-policy edge cases.
- [ ] End-to-end small-fixture preparation test.

Acceptance criteria:

- CPU-only tests validate the pipeline without downloading the full dataset.

---

# Phase 4 — Benchmark Integrity and Baseline Evaluator

Goal: freeze meaningful evaluation before fine-tuning.

## P4-001 — Create evaluation-only dataset registry

- [ ] Register HumanEval as evaluation-only.
- [ ] Register MBPP as evaluation-only.
- [ ] Add mechanism for future protected datasets.

Acceptance criteria:

- Training configuration cannot intentionally select protected datasets without an explicit hard failure or override that is impossible to mistake for a normal run.

## P4-002 — Add contamination checks

- [ ] Exact normalized prompt matching.
- [ ] Exact normalized solution/code matching where permitted.
- [ ] High-overlap/suspicious-match reporting.
- [ ] Record contamination findings in the dataset manifest.

Acceptance criteria:

- Known exact benchmark copies inserted into a synthetic fixture are detected.

## P4-003 — Implement HumanEval evaluator

- [ ] Select a maintained benchmark implementation.
- [ ] Pin version/revision.
- [ ] Normalize prompt generation.
- [ ] Execute generated solutions in a constrained environment.
- [ ] Record per-problem and aggregate results.

Acceptance criteria:

- Repeated runs with deterministic decoding/configuration produce reproducible results within expected framework behavior.

## P4-004 — Implement MBPP evaluator

- [ ] Select a maintained benchmark implementation.
- [ ] Pin version/revision.
- [ ] Execute generated solutions against tests in a constrained environment.
- [ ] Record per-problem and aggregate results.

Acceptance criteria:

- Same reporting schema as HumanEval where practical.

## P4-005 — Build custom Python evaluation suite

Cover representative tasks involving:

- [ ] standard-library data transformations
- [ ] `pathlib`
- [ ] JSON
- [ ] regular expressions
- [ ] dataclasses
- [ ] typing
- [ ] iterators/generators
- [ ] decorators
- [ ] context managers
- [ ] exceptions
- [ ] async/await
- [ ] subprocess logic
- [ ] SQLite
- [ ] pytest-oriented code

Acceptance criteria:

- Tests are repository-owned, deterministic, executable, and not included in SFT data.

## P4-006 — Implement safe execution harness

- [ ] Per-task disposable working directory/container.
- [ ] Wall-clock timeout.
- [ ] CPU/memory limits where practical.
- [ ] Network disabled by default.
- [ ] No host credentials/secrets.
- [ ] Capture stdout/stderr/exit status.

Acceptance criteria:

- A deliberately malicious/looping synthetic candidate cannot indefinitely hang or trivially access sensitive host state.

## P4-007 — Add Python result metrics

- [ ] syntax validity
- [ ] tests passed/total
- [ ] pass@1
- [ ] timeout rate
- [ ] exception category
- [ ] generated token count
- [ ] latency/tokens per second where practical

Acceptance criteria:

- Metrics are available per problem and in aggregate.

## P4-008 — Build general-regression suite

Include small controlled tests for:

- [ ] instruction following
- [ ] JSON structured output
- [ ] simple general reasoning
- [ ] shell reasoning
- [ ] Git reasoning
- [ ] tool/function-call formatting or selection

Acceptance criteria:

- The suite is frozen before the first LoRA evaluation.

## P4-009 — Freeze decoding/evaluation configuration

- [ ] temperature
- [ ] top-p/top-k as applicable
- [ ] seed
- [ ] max new tokens
- [ ] stop behavior
- [ ] prompt/template version

Acceptance criteria:

- Baseline and adapter evaluations can use identical settings.

## P4-010 — Run and record untouched-model baseline

- [ ] Evaluate `Qwen/Qwen3.5-0.8B` on HumanEval.
- [ ] Evaluate on MBPP.
- [ ] Evaluate custom Python suite.
- [ ] Evaluate general/tool regression suite.
- [ ] Record performance/memory metadata.

Acceptance criteria:

- Baseline result artifact is complete and immutable enough to support subsequent comparisons.

---

# Phase 5 — P0 BF16 LoRA Training

Goal: produce the first measured Python specialization adapter.

## P5-001 — Implement LoRA training entry point

- [ ] Add `scripts/train_lora.py`.
- [ ] Load training configuration.
- [ ] Load Qwen3.5-0.8B in BF16.
- [ ] Apply PEFT LoRA.
- [ ] Load prepared train/validation splits.
- [ ] Use verified assistant/completion-only loss.
- [ ] Save adapter checkpoints and metrics.

Acceptance criteria:

- A tiny synthetic/sampled dataset can complete a smoke-training run.

## P5-002 — Add trainable-parameter audit

- [ ] Log total parameters.
- [ ] Log trainable parameters.
- [ ] Log trainable percentage.
- [ ] Log complete matched LoRA target modules.
- [ ] Assert base parameters are frozen except intentionally trainable PEFT state.

Acceptance criteria:

- The run aborts if unexpected large portions of the base model are trainable.

## P5-003 — Add VRAM instrumentation

- [ ] Reset peak memory before training.
- [ ] Record peak allocated VRAM.
- [ ] Record peak reserved VRAM.
- [ ] Record GPU total VRAM.

Acceptance criteria:

- Run report contains actual peak VRAM rather than an estimate.

## P5-004 — Add throughput instrumentation

- [ ] examples/sec
- [ ] tokens/sec where practical
- [ ] total wall-clock training time
- [ ] step time summaries

Acceptance criteria:

- P0 efficiency can be compared with later experiments.

## P5-005 — Validate conservative batch configuration

Initial candidate:

```text
sequence length        2048
micro-batch            4
grad accumulation      4
effective batch        16
```

- [ ] Run memory smoke test.
- [ ] Reduce/increase micro-batch based on measured headroom.
- [ ] Keep final values in config, not hard-coded code.

Acceptance criteria:

- Training runs stably on the target 16 GB GPU with comfortable memory headroom.

## P5-006 — Freeze P0 LoRA configuration

Initial intended values:

```text
rank                   16
alpha                  32
dropout                0.05
bias                    none
learning rate          2e-4
scheduler              cosine
warmup ratio           0.03
epochs                  1
precision              BF16
sequence length        2048
```

- [ ] Finalize target modules from Phase 2 evidence.
- [ ] Save config under `configs/train/`.
- [ ] Give it a stable config/version name.

Acceptance criteria:

- P0 can be launched from a single committed config.

## P5-007 — Run tiny end-to-end training smoke test

- [ ] Use a very small dataset subset.
- [ ] Complete forward/backward/update.
- [ ] Save adapter.
- [ ] Reload adapter.
- [ ] Generate a response.

Acceptance criteria:

- The complete training/save/load path works before the 40k run.

## P5-008 — Run P0 40k training experiment

- [ ] Train one epoch.
- [ ] Preserve logs/checkpoints according to artifact policy.
- [ ] Record validation loss.
- [ ] Record hardware/throughput/VRAM.
- [ ] Record final adapter hash/path.

Acceptance criteria:

- Training completes without OOM or silent truncation/masking errors.

## P5-009 — Validate adapter reload independently

- [ ] Start a fresh process.
- [ ] Load unchanged base checkpoint.
- [ ] Load saved LoRA adapter.
- [ ] Generate known smoke prompts.

Acceptance criteria:

- Adapter does not depend on hidden in-memory training state.

---

# Phase 6 — P0 Evaluation and Report

Goal: determine whether the first LoRA actually improved Python capability and what it damaged.

## P6-001 — Run frozen Python evaluation against P0 adapter

- [ ] HumanEval.
- [ ] MBPP.
- [ ] Custom Python suite.

Acceptance criteria:

- Exact same evaluation configuration as baseline unless an explicitly documented compatibility fix is required.

## P6-002 — Run frozen regression suite against P0 adapter

- [ ] instruction following
- [ ] JSON output
- [ ] reasoning
- [ ] shell/Git reasoning
- [ ] tool/function-call behavior

Acceptance criteria:

- Regressions are visible in the final comparison.

## P6-003 — Produce per-problem comparison

- [ ] baseline fail -> adapter pass
- [ ] baseline pass -> adapter fail
- [ ] both pass
- [ ] both fail
- [ ] syntax and runtime failure deltas

Acceptance criteria:

- Aggregate gains can be traced back to individual tasks.

## P6-004 — Produce P0 report

Report at least:

- [ ] model/revision
- [ ] dataset composition
- [ ] LoRA config
- [ ] trainable parameter count
- [ ] training loss/validation loss
- [ ] peak VRAM
- [ ] training throughput/time
- [ ] baseline scores
- [ ] adapter scores
- [ ] regressions
- [ ] contamination status
- [ ] observed failure modes

Acceptance criteria:

- A reviewer can determine whether P0 was beneficial without reading raw logs.

## P6-005 — Decide P0 outcome

Classify the run as one of:

- [ ] useful improvement
- [ ] neutral/inconclusive
- [ ] Python improvement with unacceptable regression
- [ ] failed specialization
- [ ] invalid experiment

Acceptance criteria:

- Decision cites measured evidence and defines the next experimental variable.

---

# Phase 7 — LoRA Target-Module Experiments

Goal: understand how Qwen3.5's hybrid architecture responds to PEFT targeting.

## P7-001 — Run selective-target baseline

- [ ] Preserve P0 as the canonical selective-target result.

## P7-002 — Run `all-linear` experiment

- [ ] Same dataset.
- [ ] Same seed.
- [ ] Same rank/alpha/dropout.
- [ ] Same training length.
- [ ] Same evaluation.

Acceptance criteria:

- Target strategy is the primary meaningful variable changed.

## P7-003 — Compare architecture coverage

- [ ] trainable parameter count
- [ ] target-module distribution
- [ ] DeltaNet/hybrid coverage
- [ ] VRAM
- [ ] throughput
- [ ] Python performance
- [ ] regression performance

Acceptance criteria:

- Choose a preferred target strategy based on measured quality/cost tradeoff.

---

# Phase 8 — LoRA Rank Sweep

Goal: find the smallest useful adapter capacity and characterize diminishing returns.

## P8-001 — Rank 8

- [ ] Train/evaluate with otherwise frozen best-known config.

## P8-002 — Rank 16

- [ ] Preserve/refresh canonical comparison as required.

## P8-003 — Rank 32

- [ ] Train/evaluate.

## P8-004 — Rank 64

- [ ] Train/evaluate.

## P8-005 — Rank sweep report

Compare:

- [ ] adapter size
- [ ] trainable parameters
- [ ] VRAM
- [ ] throughput
- [ ] Python benchmarks
- [ ] regression suite

Acceptance criteria:

- Select preferred rank based on quality/size/compute tradeoff rather than assumption.

---

# Phase 9 — Dataset-Size Sweep

Goal: measure how much instruction data the tiny model actually benefits from.

Use the best target strategy/rank established so far.

## P9-001 — ~10k examples

- [ ] Train/evaluate.

## P9-002 — ~25k examples

- [ ] Train/evaluate.

## P9-003 — ~50k examples

- [ ] Train/evaluate.

## P9-004 — ~100k examples

- [ ] Train/evaluate.

## P9-005 — Larger-data decision

- [ ] Determine whether 250k+ or the full available corpus is justified by the trend.

## P9-006 — Dataset scaling report

- [ ] Plot/report Python score versus accepted training examples.
- [ ] Plot/report regression score versus accepted training examples.
- [ ] Report compute cost.

Acceptance criteria:

- Identify the point of diminishing returns or justify additional scale.

---

# Phase 10 — Sequence-Length and Batch Experiments

Goal: use the available 16 GB VRAM intelligently without conflating context length with model quality.

## P10-001 — Measure 2k canonical run

- [ ] Preserve peak VRAM and throughput.

## P10-002 — 4k sequence experiment

- [ ] Adjust micro-batch only as required for memory.
- [ ] Keep effective batch documented.

## P10-003 — 8k sequence experiment

- [ ] Run only if dataset/task distribution justifies it.

## P10-004 — Sequence-length report

Compare:

- [ ] truncation rate
- [ ] VRAM
- [ ] throughput
- [ ] benchmark quality
- [ ] long-example quality

Acceptance criteria:

- Choose sequence length on measured utility, not maximum available context.

---

# Phase 11 — BF16 LoRA vs QLoRA

Goal: determine whether quantized training is useful for this model rather than assuming it is necessary.

## P11-001 — Preserve canonical BF16 result

- [ ] Record exact reference config/result.

## P11-002 — Implement QLoRA config

- [ ] Add supported 4-bit quantization stack.
- [ ] Validate adapter training/save/load.
- [ ] Record quantization details.

## P11-003 — Controlled QLoRA experiment

- [ ] Match dataset/rank/training length/eval to BF16 run.

## P11-004 — Compare

- [ ] peak VRAM
- [ ] throughput
- [ ] adapter quality
- [ ] benchmark quality
- [ ] regression behavior

Acceptance criteria:

- QLoRA becomes recommended only if its measured tradeoff is useful.

---

# Phase 12 — Base Checkpoint Comparison

Goal: isolate the value of Qwen's post-training for this specialization task.

## P12-001 — Validate Base loading/template strategy

- [ ] Load `Qwen/Qwen3.5-0.8B-Base`.
- [ ] Validate official chat/control-token handling.
- [ ] Validate SFT masking.

## P12-002 — Train equivalent Base LoRA

- [ ] Same Python dataset.
- [ ] Same seed.
- [ ] Same selected LoRA strategy.
- [ ] Same training duration.

## P12-003 — Evaluate equivalent Base LoRA

- [ ] Same Python suite.
- [ ] Same regression suite where meaningful.

## P12-004 — Compare post-trained vs Base specialization

Acceptance criteria:

- Report distinguishes Python gains from instruction/tool behavior that originated in post-training.

---

# Phase 13 — Raw Python Continued Pretraining Investigation

Goal: determine whether Python distributional pretraining adds value beyond instruction SFT.

## P13-001 — Survey candidate raw Python corpora

- [ ] Investigate The Stack/BigCode-derived Python sources.
- [ ] Prefer permissive/open-license subsets.
- [ ] Document source licenses/provenance.
- [ ] Evaluate contamination risk.

Acceptance criteria:

- No raw-code source enters training without a documented provenance/license review.

## P13-002 — Build raw-Python cleaning pipeline

- [ ] language/version checks
- [ ] file-size limits
- [ ] duplicate/near-duplicate handling
- [ ] generated/vendor code filtering where practical
- [ ] benchmark contamination checks
- [ ] token packing/chunking policy

## P13-003 — Implement causal-LM CPT training path

- [ ] Raw text/code objective.
- [ ] Separate configuration namespace from SFT.
- [ ] Separate run identity/artifacts.

## P13-004 — Run bounded CPT experiment

- [ ] Start with a small controlled token budget.
- [ ] Evaluate code completion and Python benchmark effects.

## P13-005 — Follow CPT with Python SFT

Compare:

```text
post-trained -> Python SFT
```

and/or

```text
Base -> Python CPT -> Python SFT
```

Acceptance criteria:

- CPT is retained only if it improves quality enough to justify additional complexity/compute.

---

# Phase 14 — Tool-Calling Dataset and Agent SFT

Goal: evolve the Python specialist into a constrained coding agent.

## P14-001 — Define versioned tool schemas

Initial candidates:

- [ ] `list_files`
- [ ] `read_file`
- [ ] `search`
- [ ] `apply_patch`
- [ ] `run_command`
- [ ] `git_diff`

Acceptance criteria:

- Schemas are JSON-schema-valid, minimal, stable, and versioned.

## P14-002 — Build tool-call evaluator before tool fine-tuning

Measure:

- [ ] parse validity
- [ ] correct function selection
- [ ] argument validity
- [ ] unnecessary calls
- [ ] continuation after tool result

Acceptance criteria:

- Starting Python adapter receives a frozen agent/tool baseline before agent SFT.

## P14-003 — Define agent trajectory format

Records MUST support:

- [ ] user task
- [ ] available tools
- [ ] assistant tool call
- [ ] tool result
- [ ] repeated calls/results
- [ ] final assistant response
- [ ] verification metadata

## P14-004 — Build synthetic verified coding tasks

Start with disposable Python repositories containing:

- [ ] failing unit test
- [ ] lint failure
- [ ] type-check failure
- [ ] straightforward bug
- [ ] missing implementation
- [ ] refactor task

Acceptance criteria:

- Each task has an objectively verifiable expected terminal state.

## P14-005 — Collect successful trajectories

- [ ] Run a teacher/stronger coding agent or curated solution process.
- [ ] Capture tool calls and results.
- [ ] Verify final tests/lint/type checks.
- [ ] Keep only trajectories meeting verification policy.

## P14-006 — Collect failure-and-repair trajectories

Prioritize:

- [ ] pytest failure -> repair
- [ ] Ruff failure -> repair
- [ ] type-check failure -> repair
- [ ] import/runtime failure -> repair
- [ ] bad patch -> corrected patch

Acceptance criteria:

- Training examples include verified recovery behavior rather than only perfect one-shot trajectories.

## P14-007 — Train agent/tool SFT adapter

- [ ] Use TRL-supported tool-call representation or equivalent validated implementation.
- [ ] Preserve tools column/schemas.
- [ ] Train with controlled dataset size.
- [ ] Evaluate against frozen tool suite.

## P14-008 — Measure Python regression after agent SFT

- [ ] HumanEval/MBPP/custom suite.
- [ ] general regression suite.

Acceptance criteria:

- Agent gains and Python-quality tradeoffs are both reported.

---

# Phase 15 — Repository-Level Coding Agent Evaluation

Goal: measure actual software-engineering behavior instead of only function calling.

## P15-001 — Build disposable repository benchmark

Task types:

- [ ] locate and fix a bug
- [ ] add a small feature
- [ ] repair a failing test
- [ ] repair lint/type errors
- [ ] refactor with behavior preserved

## P15-002 — Add agent metrics

- [ ] task success
- [ ] tool-call parse validity
- [ ] correct tool selection
- [ ] unnecessary call count
- [ ] file-read relevance
- [ ] patch validity
- [ ] test pass after edit
- [ ] recovery after failure
- [ ] turns to completion
- [ ] tokens consumed
- [ ] wall-clock time
- [ ] loop rate
- [ ] premature-done rate

## P15-003 — Add safety boundaries

- [ ] disposable filesystem
- [ ] command timeout
- [ ] network policy
- [ ] no secrets
- [ ] no host Docker socket
- [ ] resource limits

Acceptance criteria:

- Agent evaluation cannot trivially mutate the developer's real repository or host environment.

## P15-004 — Compare model stages

Compare at least:

- [ ] untouched Qwen3.5-0.8B
- [ ] best Python LoRA
- [ ] best Python + agent/tool SFT

Acceptance criteria:

- Agent training is shown to improve repo-level task success, not merely tool-call syntax.

---

# Phase 16 — OpenCode Integration

Goal: determine whether the final tiny model is practically usable behind OpenCode.

## P16-001 — Select local inference server

Evaluate compatible OpenAI-style serving through one or more of:

- [ ] vLLM
- [ ] SGLang
- [ ] llama.cpp-compatible path if/when adapter export supports it cleanly
- [ ] another well-supported OpenAI-compatible local server

Acceptance criteria:

- Selection is based on correct Qwen3.5 + adapter + tool behavior, not familiarity alone.

## P16-002 — Serve base + adapter

- [ ] Load model.
- [ ] Load adapter or supported merged artifact.
- [ ] Expose OpenAI-compatible endpoint.
- [ ] Verify deterministic chat request.

## P16-003 — Validate tool-call protocol through server

- [ ] Send tool definitions.
- [ ] Verify emitted tool calls survive server parsing/serialization.
- [ ] Verify tool results can be fed back.

Acceptance criteria:

- Tool behavior works end-to-end over the same protocol OpenCode will use.

## P16-004 — Configure OpenCode

- [ ] Add documented local provider configuration.
- [ ] Keep credentials/endpoints environment-driven.
- [ ] Document context/token limits.

## P16-005 — Run OpenCode task suite

- [ ] Execute disposable Python repository tasks.
- [ ] Record success/failure and agent telemetry.

## P16-006 — Produce OpenCode readiness report

Classify the model as:

- [ ] not viable
- [ ] viable for constrained/simple tasks
- [ ] useful Python micro-agent
- [ ] broadly useful local coding agent

Acceptance criteria:

- Classification is evidence-based and tied to repository-task metrics.

---

# Phase 17 — Packaging and Publication

Goal: make successful results reusable without losing provenance.

## P17-001 — Define adapter naming/versioning

Include:

- [ ] base model lineage
- [ ] experiment/config version
- [ ] dataset-manifest reference
- [ ] adapter rank/target strategy

## P17-002 — Add adapter model card template

Document:

- [ ] intended use
- [ ] base model
- [ ] datasets
- [ ] training configuration
- [ ] evaluation results
- [ ] regressions/limitations
- [ ] licensing/provenance
- [ ] hardware

## P17-003 — Add reproducible publish command

- [ ] Publish adapter to an appropriate registry only when explicitly authorized.
- [ ] Never publish secrets/local paths.

## P17-004 — Optional merge/export path

- [ ] Merge adapter into base weights only as a deployment artifact.
- [ ] Preserve adapter-level lineage.
- [ ] Evaluate merged result against adapter-loaded result.

Acceptance criteria:

- Deployment convenience does not obscure experiment provenance.

---

# Phase 18 — Final Comparative Study

Goal: summarize what actually matters for a 0.8B Python coding model.

## P18-001 — Assemble canonical experiment table

Include:

- [ ] starting checkpoint
- [ ] training stage
- [ ] data count/token count
- [ ] LoRA rank
- [ ] target strategy
- [ ] sequence length
- [ ] trainable parameters
- [ ] adapter size
- [ ] peak VRAM
- [ ] training time/throughput
- [ ] HumanEval
- [ ] MBPP
- [ ] custom Python suite
- [ ] regression suite
- [ ] tool metrics
- [ ] repo-agent success

## P18-002 — Identify Pareto-optimal configurations

Evaluate quality versus:

- [ ] adapter size
- [ ] VRAM
- [ ] training cost
- [ ] inference cost
- [ ] regression cost

## P18-003 — Document conclusions

Answer at least:

1. How much can LoRA improve Qwen3.5-0.8B at Python?
2. What dataset size gives diminishing returns?
3. What LoRA rank is sufficient?
4. Does Qwen3.5's hybrid architecture benefit from all-linear targeting?
5. Does raw Python CPT materially help?
6. How much tool/agent training can be added without sacrificing Python quality?
7. Is the resulting model actually useful in OpenCode?
8. What tasks remain beyond the capability ceiling of a 0.8B model?

Acceptance criteria:

- Conclusions are traceable to committed configurations and recorded experiment artifacts.

---

# Immediate execution order

The first implementation sequence SHOULD be:

1. `P0-001` through `P0-007` — bootstrap repository and CI.
2. `P1-001` through `P1-004` — reproducibility/config foundation.
3. `P2-001` through `P2-005` — inspect Qwen3.5 and validate LoRA/template mechanics.
4. `P3-001` through `P3-012` — deterministic Python dataset pipeline.
5. `P4-001` through `P4-010` — benchmark integrity and untouched baseline.
6. `P5-001` through `P5-009` — first BF16 LoRA training.
7. `P6-001` through `P6-005` — evaluate and decide whether P0 worked.

Do NOT start hyperparameter sweeps before the baseline, dataset manifest, P0 training path, and frozen evaluation path are trustworthy.
