# Tiny Qwen Coder — Specification

## 1. Purpose

`tiny-qwen-coder` is an experimental project for specializing Qwen3.5-0.8B for Python software engineering using parameter-efficient fine-tuning (PEFT), beginning with BF16 LoRA supervised fine-tuning (SFT) and progressing toward a small tool-using Python coding agent suitable for local coding workflows such as OpenCode.

The project is intended to answer a concrete research question:

> How far can a sub-1B general/agentic model be pushed toward reliable Python software engineering while preserving its existing instruction-following and tool-use behavior?

The project MUST emphasize reproducibility, executable evaluation, controlled experiments, and explicit regression measurement. Improvements MUST be demonstrated quantitatively rather than inferred from training loss or subjective prompt examples.

---

## 2. Goals

### 2.1 Primary goals

1. Fine-tune `Qwen/Qwen3.5-0.8B` with LoRA to improve Python code generation and Python software-engineering capability.
2. Preserve, as much as practical, the original model's general instruction-following, structured-output, and tool-use behavior.
3. Establish a reproducible baseline and evaluation harness before training.
4. Build a deterministic, auditable Python SFT dataset pipeline with provenance and contamination controls.
5. Keep the initial experiment small enough to iterate rapidly on a single 16 GB NVIDIA GPU.
6. Compare LoRA configurations scientifically, changing one meaningful variable at a time.
7. Evolve the best Python-specialized model toward repo-level, tool-using coding-agent behavior.
8. Make all training and evaluation commands reproducible from repository configuration rather than ad hoc shell history.

### 2.2 Secondary goals

1. Measure the effect of LoRA rank, target modules, sequence length, and dataset size.
2. Compare BF16 LoRA with optional QLoRA only after the BF16 baseline is established.
3. Compare fine-tuning the post-trained model with fine-tuning `Qwen/Qwen3.5-0.8B-Base` using otherwise equivalent data and evaluation.
4. Evaluate whether Python specialization causes catastrophic forgetting or agent/tool-use regression.
5. Investigate continued pretraining on permissively licensed raw Python code as a later stage.
6. Investigate SFT on verified coding-agent trajectories involving repository inspection, editing, test execution, and repair.
7. Produce adapters that can be served through an OpenAI-compatible local inference stack and evaluated with OpenCode.

---

## 3. Non-goals

The initial project is NOT intended to:

1. Perform full-parameter fine-tuning of Qwen3.5-0.8B.
2. Train a foundation model from scratch.
3. Maximize benchmark scores at the expense of benchmark contamination or reproducibility.
4. Train directly on HumanEval, MBPP, or other designated evaluation sets.
5. Use GitHub Actions for GPU training.
6. Commit model weights, generated datasets, Hugging Face caches, checkpoints, or other large generated artifacts to Git.
7. Optimize for every programming language in the initial phases.
8. Claim OpenCode readiness solely because a model can produce Python snippets; coding-agent behavior must be evaluated separately.
9. Merge LoRA adapters into the base model until adapter-level evaluation is complete.
10. Introduce quantization into the first experiment unless required by an unforeseen compatibility constraint.

---

## 4. Model strategy

### 4.1 Initial training target

The P0/P3 first fine-tuning experiment SHALL use:

- Model: `Qwen/Qwen3.5-0.8B`
- Role: post-trained starting checkpoint
- Precision for training: BF16 where supported
- Fine-tuning method: LoRA through Hugging Face PEFT/TRL

The post-trained checkpoint is chosen first because the project ultimately cares about a coding assistant and coding agent, not only raw code completion. Existing instruction-following and tool-use behavior therefore form part of the baseline that the Python specialization should preserve.

### 4.2 Base-model comparison

`Qwen/Qwen3.5-0.8B-Base` SHALL be treated as a separate later experiment, not silently substituted for the initial model.

The Base checkpoint is especially relevant because Qwen documents it as a pre-trained-only checkpoint intended for fine-tuning and research, and its chat control tokens were included during pretraining to support efficient LoRA-style PEFT with the official chat template without requiring embedding fine-tuning.

A controlled comparison SHOULD eventually evaluate:

- post-trained Qwen3.5-0.8B + Python LoRA
- Qwen3.5-0.8B-Base + equivalent Python LoRA

The comparison MUST keep dataset, seed, LoRA configuration, evaluation prompts, decoding parameters, and benchmark implementation as equivalent as practical.

### 4.3 Architecture awareness

Qwen3.5 uses a hybrid architecture rather than only conventional full-attention Transformer blocks. LoRA target-module selection MUST therefore be discovered from the loaded model rather than assumed from an older Llama/Qwen recipe.

The repository MUST include a model-inspection utility that records:

- model class
- parameter count
- trainable/frozen parameter counts
- module names and types relevant to PEFT targeting
- tokenizer/chat-template metadata
- dtype
- device placement
- model revision/commit identifier when available

The first selective-target LoRA experiment MAY target conventional attention and MLP projection modules when those exact modules exist. A later controlled experiment SHALL compare selective targeting against PEFT `target_modules="all-linear"` or an equivalent complete linear-layer strategy.

---

## 5. Hardware target

### 5.1 Primary development target

The primary training environment is a single NVIDIA GPU with approximately 16 GB VRAM.

The project SHOULD prioritize a comfortable BF16 LoRA workflow instead of minimizing VRAM at all costs.

Initial expected configuration:

- BF16 base-model loading
- gradient checkpointing enabled where compatible
- sequence length: 2,048 tokens
- micro-batch selected empirically from a conservative starting point
- gradient accumulation used to increase effective batch size without forcing larger activation memory

### 5.2 Memory measurement

Every training run MUST record:

- GPU model
- CUDA/runtime versions where practical
- total VRAM
- peak allocated VRAM
- peak reserved VRAM when available
- batch size
- gradient accumulation
- sequence length
- LoRA rank
- LoRA target strategy
- tokens/examples processed per second when available

No hard-coded VRAM estimate SHALL be treated as an acceptance criterion; the repository MUST measure actual peak memory for each run.

---

## 6. Software stack

The initial implementation SHALL use:

- Python
- `uv` for dependency/environment management
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Hugging Face PEFT
- Hugging Face TRL
- Accelerate

Additional dependencies MAY be introduced for evaluation, static analysis, testing, or optional performance kernels, but they MUST be pinned through the project dependency mechanism.

The project SHOULD avoid requiring proprietary training services.

### 6.1 Development quality gates

The repository SHOULD include:

- Ruff formatting and linting
- static type checking (preferably mypy or pyright, selected once during bootstrap)
- pytest
- deterministic unit tests for dataset transforms and configuration validation
- `git diff --check`-clean changes

GPU-dependent tests MUST NOT be required for normal CI unless a future dedicated GPU runner is explicitly configured.

---

## 7. Repository layout

The target layout is:

```text
tiny-qwen-coder/
├── docs/
│   ├── SPEC.md
│   └── TODO.md
├── pyproject.toml
├── uv.lock
├── README.md
├── .gitignore
├── configs/
│   ├── data/
│   ├── train/
│   └── eval/
├── src/
│   └── tiny_qwen_coder/
│       ├── __init__.py
│       ├── config.py
│       ├── data/
│       ├── training/
│       ├── evaluation/
│       └── reporting/
├── scripts/
│   ├── inspect_model.py
│   ├── prepare_data.py
│   ├── train_lora.py
│   ├── evaluate.py
│   └── infer.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── data/                 # generated/local, ignored
├── artifacts/            # generated/local, ignored
└── outputs/              # generated/local, ignored
```

The exact module names MAY evolve, but command-line entry points MUST call shared library code rather than duplicating substantive logic in scripts.

---

## 8. Artifact and Git policy

The following MUST be excluded from Git:

- Hugging Face model caches
- downloaded model weights
- generated train/validation datasets
- checkpoints
- LoRA adapter binary weights unless deliberately published through an appropriate model registry/release mechanism
- benchmark-generated code sandboxes
- TensorBoard/W&B-style local run directories unless specifically curated
- Python caches and local virtual environments

Git SHOULD contain:

- source code
- configuration
- dataset manifests and hashes where legally/practically appropriate
- compact evaluation summaries
- experiment metadata
- documentation
- small synthetic test fixtures

Raw or generated data MUST NOT be committed merely to make a run reproducible. Reproducibility SHOULD come from pinned upstream identifiers, deterministic preparation logic, seeds, manifests, and checksums.

---

## 9. Dataset strategy

### 9.1 Initial SFT sources

The initial experiment SHALL build a Python-only instruction-tuning corpus from high-quality sources. The first intended sources are:

1. `OLMo-Coding/starcoder-python-instruct`
   - approximately 1.26M instruction/code rows at the time this specification was written
   - Apache-2.0 dataset metadata
   - Python code paired with synthetic natural-language instructions
   - source metadata distinguishes Python 2 and Python 3
   - examples were prefiltered for bounded token length by the dataset authors

2. `ise-uiuc/Magicoder-OSS-Instruct-75K`
   - use only rows identified as Python
   - retain source/provenance fields required for later audit

The exact dataset revisions MUST be pinned or recorded in every run manifest.

### 9.2 P0 dataset target

The first full SFT experiment SHOULD target approximately 40,000 examples, initially budgeted as:

- 30,000 accepted Python 3 examples from `OLMo-Coding/starcoder-python-instruct`
- 10,000 accepted Python examples from Magicoder when enough rows survive filtering; otherwise fill from the primary source and record the final composition

The final sample count MUST be determined after filtering, not fabricated to match a target.

A deterministic 95/5 train/validation split MAY be used for P0 unless later evidence justifies another ratio.

### 9.3 Dataset normalization

Training records SHOULD be normalized into a conversational representation such as:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert Python software engineer. Produce correct, idiomatic, maintainable Python. Follow the user's requirements precisely."
    },
    {
      "role": "user",
      "content": "..."
    },
    {
      "role": "assistant",
      "content": "..."
    }
  ]
}
```

The exact system prompt MUST be versioned in configuration and MUST NOT be changed between baseline/fine-tuned comparisons without recording the change.

### 9.4 Filtering

The preparation pipeline MUST implement and report deterministic filtering for at least:

- Python 2 exclusion where source metadata provides the distinction
- empty/malformed prompt or response removal
- maximum/minimum length constraints
- exact duplicate removal
- normalization-aware duplicate detection where practical
- obviously invalid records
- Python syntax validation where the response is intended to be a complete parseable program/module

Syntax filtering MUST NOT incorrectly discard valid examples that are intentionally fragments, REPL snippets, diffs, or otherwise not parseable as standalone modules. The dataset pipeline MUST distinguish or conservatively retain such examples rather than applying `ast.parse()` blindly to every response.

Later phases SHOULD investigate:

- near-duplicate detection
- generated boilerplate detection
- benchmark-contamination scanning
- dependency/library age and quality heuristics
- license-aware source filtering

### 9.5 Provenance manifest

Every prepared dataset MUST generate a manifest containing at least:

- dataset identifiers
- dataset revisions when available
- source license metadata as reported upstream
- preparation code version/commit
- random seed
- source row counts
- accepted row counts
- rejection counts by reason
- train/validation counts
- deduplication statistics
- token-length summary
- stable fingerprints/checksums available from the tooling

---

## 10. Benchmark contamination policy

HumanEval, MBPP, and any additional designated holdout benchmark MUST NOT be used as SFT training data.

The project MUST maintain an explicit registry of evaluation-only datasets.

The data-preparation pipeline SHOULD implement contamination checks where source text permits practical matching. At minimum it SHOULD support:

- exact prompt matching
- normalized code/text matching
- suspicious high-overlap reporting

A benchmark score MUST NOT be reported as evidence of improvement if known training contamination makes the comparison invalid.

---

## 11. Evaluation strategy

### 11.1 Baseline first

No P0 fine-tuning result is considered valid unless the unchanged starting model has first been evaluated using the same frozen evaluation configuration intended for the fine-tuned adapter.

The evaluation harness MUST persist enough information to reproduce comparisons:

- model/checkpoint identifier
- adapter identifier, if any
- prompt/template version
- benchmark version
- random seed
- decoding parameters
- max generation length
- execution/test timeout
- environment metadata
- per-problem result
- aggregate result

### 11.2 Python capability evaluation

Initial Python evaluation SHOULD include:

- HumanEval or an actively maintained compatible implementation
- MBPP or an actively maintained compatible implementation
- a repository-owned custom Python evaluation suite

The custom suite SHOULD cover a range such as:

- standard library usage
- `pathlib`
- JSON
- regular expressions
- dataclasses
- typing
- iterators/generators
- decorators
- context managers
- exceptions
- async/await and `asyncio`
- subprocess handling
- SQLite
- unit testing with pytest
- common data transformations

Optional later suites MAY cover NumPy, Pydantic, FastAPI, pandas, or other libraries, but library-specific evaluation MUST record package versions.

### 11.3 Execution-based scoring

Where a task has executable tests, correctness MUST be scored through isolated execution rather than only text similarity.

The harness SHOULD record:

- parse/syntax validity
- test pass/fail counts
- pass@1
- timeout
- runtime exception category
- generated token count
- latency/tokens per second where practical

Generated code MUST be executed in a constrained disposable environment. The evaluator MUST NOT run arbitrary generated code with unrestricted host access.

### 11.4 Regression suite

Because the initial checkpoint is post-trained and agent-capable, evaluation MUST include non-Python regression checks.

At minimum the project SHOULD maintain tests for:

- general instruction following
- structured JSON output
- simple reasoning
- basic shell/Git reasoning without unsafe execution
- tool-call formatting or function-selection behavior

The objective is not necessarily zero regression. The objective is to make tradeoffs visible.

A successful adapter SHOULD demonstrate:

- measurable Python improvement
- no severe collapse in instruction following
- no severe collapse in structured/tool behavior

Quantitative thresholds SHALL be established after the reproducible baseline exists, rather than invented before baseline measurement.

---

## 12. Initial LoRA experiment (P0 training configuration)

The first training run SHALL be intentionally conservative and easy to reproduce.

Initial target configuration:

```text
model                  Qwen/Qwen3.5-0.8B
precision              BF16
fine-tuning            LoRA SFT
sequence length        2048
LoRA rank              16
LoRA alpha             32
LoRA dropout           0.05
bias                    none
learning rate          2e-4 initial candidate
scheduler              cosine initial candidate
warmup ratio           0.03 initial candidate
epochs                  1
gradient checkpointing enabled when compatible
loss                    assistant/completion tokens only
```

Micro-batch size and gradient accumulation MUST be selected from actual hardware measurements. A reasonable initial trial is micro-batch 4 with gradient accumulation 4 if it fits comfortably; the training program MUST make these values configuration-driven.

### 12.1 Assistant-only loss

The training pipeline SHOULD use TRL assistant-only loss when the active Qwen chat template produces a valid assistant token mask. The implementation MUST verify this capability rather than assuming it from another Qwen model family.

If the upstream template does not support the required mask behavior in the pinned dependency version, the project MUST either:

1. use a verified compatible template/masking implementation, or
2. use a prompt-completion representation with completion-only loss.

It MUST NOT silently train loss over user/system tokens while claiming assistant-only SFT.

### 12.2 LoRA target modules

P0 SHALL inspect the actual model before fixing the target list.

A selective target set MAY include discovered conventional projection modules such as attention q/k/v/o and MLP gate/up/down projections where those exact names exist.

The run manifest MUST record the complete matched target-module set and number of trainable parameters.

A later experiment SHALL compare selective targets with `all-linear` or an equivalent comprehensive linear-module configuration.

---

## 13. Experiment methodology

### 13.1 One major variable at a time

After P0, experiments SHOULD change one major variable at a time whenever practical.

Candidate sweeps include:

- LoRA rank: 8, 16, 32, 64
- dataset size: approximately 10k, 25k, 50k, 100k, then larger if justified
- selective vs all-linear targets
- sequence length: 2k, 4k, 8k when useful
- one epoch vs additional training only when validation/evaluation justify it

### 13.2 Run identity

Every training/evaluation run MUST have a stable run ID and machine-readable manifest.

The manifest SHOULD include:

- Git commit SHA
- model ID/revision
- dataset manifest/fingerprint
- complete training config
- dependency versions
- hardware
- seed
- start/end timestamps
- training metrics
- peak VRAM
- output adapter location/hash
- baseline/evaluation configuration references

### 13.3 Reproducibility

A run SHALL be considered reproducible when another compatible environment can reconstruct the dataset and training configuration from the repository plus documented upstream artifacts without depending on undocumented manual state.

Bit-identical GPU training is not required where upstream kernels are nondeterministic, but all known randomness MUST be seeded and nondeterminism MUST be documented.

---

## 14. Adapter lifecycle

The project SHALL preserve the LoRA adapter separately during evaluation.

The normal sequence is:

```text
base model
    +
LoRA adapter
    -> evaluation
```

Merging the adapter into full model weights MAY be implemented later for deployment convenience, but merged weights MUST NOT replace adapter-level experiment artifacts or make provenance ambiguous.

Adapters SHOULD be named with enough metadata to distinguish experiment lineage.

---

## 15. Continued pretraining phase

Raw Python continued pretraining (CPT) is a later experiment and MUST be separated conceptually from instruction SFT.

The goal of CPT would be to improve distributional familiarity with:

- Python syntax and idioms
- library/API patterns
- code completion
- project/module structure

Raw-code CPT SHOULD use permissively/openly licensed Python data with explicit provenance and filtering. Potential BigCode/The Stack-derived sources MAY be investigated, but source licensing and contamination MUST be reviewed before adoption.

A CPT experiment SHOULD be followed by Python instruction SFT before comparison as a coding assistant.

The project SHOULD compare at least:

```text
post-trained model -> Python SFT
```

against a controlled pipeline such as:

```text
base model -> Python CPT -> Python SFT
```

when resources justify it.

---

## 16. Tool-use and coding-agent phase

The long-term target is not merely a code generator. It is a compact Python coding agent capable of participating in a constrained software-engineering loop.

### 16.1 Agent task model

Target behavior includes:

```text
inspect repository
-> identify relevant files
-> read/search
-> choose edit
-> apply edit
-> run formatter/linter/tests
-> interpret failure
-> repair
-> verify
-> summarize
```

### 16.2 Minimal tool vocabulary

Agent training SHOULD initially constrain the tool vocabulary to a small, stable surface such as:

- `list_files`
- `read_file`
- `search`
- `write_file` or `apply_patch`
- `run_command`
- `git_diff`

The exact tool schemas MUST be versioned.

### 16.3 Agent SFT data

Tool SFT records SHOULD preserve:

- user request
- available tool JSON schemas
- assistant tool calls
- tool results
- intermediate repair attempts
- verified final state

TRL supports tool-calling SFT dataset representations containing conversation messages, tool calls/tool responses, and available tools. The implementation MUST validate actual behavior in the pinned TRL version.

High-value agent data SHOULD include failed attempts and verified recovery, particularly:

- pytest failure -> diagnosis -> repair
- Ruff/type-check failure -> repair
- import/runtime error -> repair
- incorrect tool choice -> corrected sequence

Synthetic trajectories SHOULD be verified by execution before inclusion whenever feasible.

---

## 17. OpenCode integration

A later deployment/evaluation phase SHALL expose the selected adapter through an OpenAI-compatible local inference endpoint supported by OpenCode.

OpenCode evaluation MUST distinguish:

1. model can be connected to OpenCode
2. model emits syntactically valid tool calls
3. model chooses the correct tools
4. model completes realistic repository tasks

Candidate agent metrics include:

- tool-call parse validity
- correct tool selection
- unnecessary tool-call count
- repo navigation accuracy
- edit/patch validity
- test pass rate after edits
- recovery rate after failed tests
- turns to completion
- generated tokens
- wall-clock time
- infinite-loop rate
- premature-completion rate

The project SHOULD use disposable repositories/tasks so generated commands and edits cannot damage the development host.

---

## 18. Security requirements

Model-generated code and commands are untrusted input.

Evaluation and agent testing MUST:

- use disposable/sandboxed execution environments
- apply timeouts
- constrain filesystem access
- avoid exposing secrets or credentials
- prevent uncontrolled network access when it is not required
- avoid granting generated code access to the host Docker socket or equivalent privileged control plane

Dataset ingestion MUST also treat source content as data rather than executable instructions.

---

## 19. CI policy

Normal GitHub Actions CI SHOULD run only bounded CPU-friendly checks such as:

- dependency/bootstrap validation
- formatting
- linting
- type checking
- unit tests
- small synthetic dataset pipeline tests
- configuration schema validation
- tiny mocked model/evaluation tests where practical

CI MUST NOT download/train the full Qwen model as a required pull-request or push check unless explicitly introduced later with bounded caching/resources.

Long-running benchmarks and GPU fine-tuning SHALL be local/manual experiment jobs with committed configuration and summarized results.

---

## 20. Reporting

Each completed experiment SHOULD produce both machine-readable and human-readable output.

Suggested machine-readable files:

```text
artifacts/<run-id>/manifest.json
artifacts/<run-id>/train_metrics.json
artifacts/<run-id>/eval_results.json
```

A compact Markdown summary MAY be generated for review.

Comparative reports SHOULD make it easy to answer:

- Did Python capability improve?
- By how much?
- At what VRAM/time/token cost?
- Did general capability regress?
- Did tool use regress or improve?
- Which experimental variable changed?

---

## 21. P0 success definition

The first end-to-end milestone is complete when all of the following are true:

1. The repository bootstraps reproducibly with `uv`.
2. Model inspection works for Qwen3.5-0.8B and records relevant PEFT target information.
3. A deterministic Python SFT preparation pipeline builds a documented train/validation dataset from pinned upstream sources.
4. Evaluation-only datasets are protected from intentional training inclusion.
5. The untouched Qwen3.5-0.8B baseline has frozen Python and regression evaluation results.
6. A BF16 rank-16 LoRA trains successfully for the configured P0 run on the target GPU.
7. Peak VRAM and training throughput are measured.
8. The adapter reloads independently from the base model.
9. The same frozen evaluation suite is run against the adapter.
10. A report compares baseline and fine-tuned results, including regressions.
11. The experiment is reproducible from committed code/configuration and recorded upstream revisions.

P0 is considered informative even if Python scores fail to improve. A correctly measured negative result is preferable to an unmeasured apparent success.

---

## 22. Longer-term success definition

The project succeeds at its broader objective if it can demonstrate a Python-specialized sub-1B model that:

- materially outperforms its starting checkpoint on held-out executable Python tasks
- retains useful general instruction-following capability
- retains or improves reliable structured/tool-call behavior
- can operate through a constrained coding-agent tool loop
- completes a measurable subset of realistic disposable Python repository tasks
- runs locally with modest inference resources
- is reproducible and auditable from published code/configuration and lawful dataset provenance

A model need not match large frontier coding agents to be useful. The research target is to determine what a deliberately specialized, tightly tooled 0.8B-class model can do efficiently and reliably.

---

## 23. Reference sources

These sources informed the initial design and SHOULD be re-checked when dependencies/models are upgraded:

- Qwen3.5-0.8B: https://huggingface.co/Qwen/Qwen3.5-0.8B
- Qwen3.5-0.8B-Base: https://huggingface.co/Qwen/Qwen3.5-0.8B-Base
- OLMo Coding StarCoder Python Instruct: https://huggingface.co/datasets/OLMo-Coding/starcoder-python-instruct
- Magicoder OSS Instruct: https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K
- TRL SFTTrainer: https://huggingface.co/docs/trl/main/sft_trainer
- PEFT LoRA: https://huggingface.co/docs/peft/main/en/package_reference/lora

---

## 24. Specification change policy

Material changes to the experimental contract SHOULD update this specification before or with implementation.

Examples of material changes include:

- changing the primary model checkpoint
- changing designated evaluation-only datasets
- replacing LoRA with another primary training method
- changing the fundamental success criteria
- introducing benchmark data into training
- changing agent tool schemas
- changing artifact provenance requirements

Routine implementation details that preserve the contract do not require a specification revision.
