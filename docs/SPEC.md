# Tiny Qwen Coder — Specification

## 1. Purpose

`tiny-qwen-coder` is an experimental framework for turning a single pinned `Qwen/Qwen3.5-4B` checkpoint into a family of programming-language specialists by attaching interchangeable LoRA adapters.

The core architecture is:

```text
                         Qwen3.5-4B
                      pinned base model
                             │
             ┌───────────────┼───────────────┐
             │               │               │
             ▼               ▼               ▼
        Python LoRA     TypeScript LoRA     Rust LoRA
             │               │               │
             ▼               ▼               ▼
       Python coder      TS coder         Rust coder
```

The base model is intended to remain loaded while small language-specific LoRA weights are activated or swapped according to the language being worked on. The project MUST NOT require a separate full model copy for each programming language.

Python is the first implementation and proving ground. TypeScript and Rust are planned follow-on adapters. The training, evaluation, artifact, and serving infrastructure MUST therefore be language-neutral from the beginning even when early tasks contain Python-specific logic.

The primary research question is:

> How effectively can one 4B coding-capable base model support multiple specialized programming-language personalities through interchangeable LoRA adapters while preserving general instruction-following, tool-use, and cross-language capability?

The project MUST emphasize reproducibility, executable evaluation, controlled experiments, adapter compatibility, and explicit regression measurement. Improvements MUST be demonstrated quantitatively rather than inferred from loss curves or subjective examples.

---

## 2. Architectural principles

### 2.1 One canonical base model

All production language adapters MUST target the same immutable base-model identity:

- repository: `Qwen/Qwen3.5-4B`
- exact Hugging Face revision/commit: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- tokenizer revision: identical to the base-model revision unless explicitly documented otherwise
- chat template: versioned and kept compatible across adapters

The model name alone is insufficient. Every adapter manifest MUST record the exact base revision it was trained against.

An adapter trained against a different base revision MUST be treated as incompatible unless an explicit compatibility experiment proves otherwise.

### 2.2 Interchangeable language specialization

The first planned adapter family is:

```text
language/python
language/typescript
language/rust
```

The runtime SHOULD be able to:

1. load the base model once;
2. load one or more compatible LoRA adapters;
3. activate a selected adapter without reloading the base model;
4. return to the unadapted base model;
5. report which adapter is active;
6. reject adapters whose manifest is incompatible with the loaded base.

Hot-switch behavior MUST be tested rather than assumed.

### 2.3 Language adapters are distinct from behavior adapters

The project distinguishes two concepts:

**Language specialization**

- Python
- TypeScript
- Rust
- future languages such as Go or C++

**Behavior specialization**

- tool use
- repository navigation
- debugging
- code review
- test generation
- compiler/test-failure recovery

The initial project SHALL train language adapters independently. Arbitrary LoRA stacking, fusion, or composition MUST NOT be assumed to work correctly simply because the PEFT runtime can load multiple adapters.

Behavior-adapter composition is a later research topic and requires its own evaluation.

### 2.4 Language-neutral infrastructure

Shared training code MUST NOT be implemented as separate scripts such as:

```text
train_python.py
train_typescript.py
train_rust.py
```

Instead, the preferred interface is conceptually:

```text
prepare-data --language python
train-adapter --language python
evaluate --language python
```

or an equivalent configuration-driven API.

Language-specific behavior belongs in declarative configuration and well-defined language plugins/adapters, while generic data preparation, LoRA training, run tracking, evaluation orchestration, artifact generation, and serving logic remain shared.

---

## 3. Goals

### 3.1 Primary goals

1. Establish one pinned `Qwen/Qwen3.5-4B` checkpoint as the canonical base model.
2. Fine-tune a Python LoRA that measurably improves Python code generation and software-engineering performance.
3. Preserve, as much as practical, the base model's general instruction-following, structured-output, and tool-use behavior.
4. Build language-neutral dataset, training, evaluation, reporting, and adapter-management infrastructure.
5. Make language adapters interchangeable without reloading the full base model.
6. Add TypeScript and Rust LoRAs using the same infrastructure after Python validates the approach.
7. Measure cross-language effects so specialization gains and regressions are visible.
8. Keep initial experiments comfortable on a single NVIDIA GPU with approximately 16 GB VRAM.
9. Compare LoRA configurations scientifically, changing one meaningful variable at a time.
10. Evolve the best language-specialized models toward repo-level, tool-using coding-agent behavior suitable for OpenCode or similar clients.

### 3.2 Secondary goals

1. Measure the effect of LoRA rank, target modules, sequence length, dataset size, and training mix.
2. Measure BF16 LoRA versus 4-bit QLoRA feasibility on the reference 16 GB GPU and freeze one canonical training mode before full adapter training.
3. Compare fine-tuning the post-trained checkpoint with `Qwen/Qwen3.5-4B-Base` as a separate controlled research branch.
4. Investigate continued pretraining on permissively licensed source code before SFT.
5. Investigate verified coding-agent trajectories involving repository inspection, editing, execution, and repair.
6. Evaluate automatic language detection and adapter selection.
7. Evaluate adapter switching in polyglot repositories.
8. Explore adapter fusion/composition only after single-adapter behavior is understood.

---

## 4. Non-goals

The initial project is NOT intended to:

1. Perform full-parameter fine-tuning of Qwen3.5-4B.
2. Train a foundation model from scratch.
3. Maintain a separate full Qwen model copy per programming language.
4. Merge every LoRA into a standalone full model for normal use.
5. Train Python, TypeScript, and Rust simultaneously before the Python pipeline is validated.
6. Assume independently trained LoRAs compose cleanly.
7. Maximize benchmark scores through contamination or benchmark memorization.
8. Train directly on designated evaluation sets such as HumanEval or MBPP.
9. Use GitHub Actions for required GPU training.
10. Commit model weights, generated datasets, Hugging Face caches, or checkpoints to Git.
11. Claim OpenCode readiness solely because the model can generate code snippets.
12. Use quantization without a measured memory/compatibility justification or change quantization policy mid-experiment.

---

## 5. Canonical base-model contract

### 5.1 Initial base checkpoint

The first canonical language-adapter experiments SHALL use:

- model repository: `Qwen/Qwen3.5-4B`
- role: post-trained shared base checkpoint
- preferred fine-tuning mode: BF16 LoRA through Hugging Face PEFT/TRL when it fits with adequate VRAM headroom
- fallback fine-tuning mode: 4-bit QLoRA with BF16 compute when BF16 LoRA is not comfortably memory-safe
- canonical training mode: selected from measured GPU preflight and frozen before the first full adapter training run

The post-trained checkpoint is chosen because the project ultimately targets coding-assistant and coding-agent behavior, not only raw code completion. Existing instruction-following and tool-use behavior is therefore part of the capability baseline that specialization SHOULD preserve.

### 5.2 Exact revision pinning

The canonical Qwen3.5-4B revision is pinned to `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Changing that SHA is a base-family migration and MUST be explicit because existing adapters are revision-bound.

The canonical base descriptor SHOULD be represented in configuration, for example:

```yaml
base_model:
  repository: Qwen/Qwen3.5-4B
  revision: 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
```

Every adapter manifest MUST include at least:

- base repository
- base revision
- tokenizer revision
- chat-template identifier/hash where practical
- Transformers version
- PEFT version
- LoRA target strategy
- LoRA rank/alpha/dropout

### 5.3 Base-model comparison branch

`Qwen/Qwen3.5-4B-Base` SHALL be treated as a separate experiment, not silently substituted for the canonical post-trained base.

A controlled comparison MAY evaluate:

- post-trained Qwen3.5-4B + language LoRA
- Qwen3.5-4B-Base + equivalent language LoRA

The comparison MUST keep dataset, seed, LoRA configuration, evaluation prompts, decoding parameters, and benchmark implementation equivalent as practical.

Adapters trained on `Qwen3.5-4B-Base` belong to a separate compatibility family and MUST NOT be advertised as interchangeable with adapters trained on `Qwen3.5-4B`.

### 5.4 Text-only specialization on a multimodal checkpoint

Qwen3.5-4B is released as a causal language model with a vision encoder. The initial Tiny Qwen Coder adapters are text/code specialists, not vision adapters. Vision-encoder and multimodal/projector components MUST remain frozen and MUST NOT be selected as language-LoRA targets unless a separate experiment explicitly changes that scope. Model inspection MUST distinguish the language backbone from vision/multimodal modules.

### 5.5 Architecture awareness

Qwen3.5-4B uses a hybrid Gated DeltaNet/full-attention architecture rather than only conventional full-attention Transformer blocks. LoRA target-module selection MUST therefore be discovered from the actual loaded model rather than copied blindly from an older Llama/Qwen recipe, and language-adapter target discovery MUST exclude vision/multimodal components by default.

The repository MUST include a model-inspection utility that records:

- model class
- total parameter count
- module hierarchy
- linear/projection module names
- trainable/frozen parameter counts after PEFT attachment
- tokenizer/chat-template metadata
- dtype
- device placement
- exact model revision

The first selective-target LoRA MAY use discovered attention and MLP projections. A controlled experiment SHALL compare selective targeting against PEFT `target_modules="all-linear"` or an equivalent supported strategy.

---

## 6. Adapter contract

### 6.1 Adapter identity

Every trained adapter MUST have a stable logical identity composed of at least:

```text
family/language/experiment
```

Example:

```text
language/python/p0-r16-40k
language/typescript/p0-r16-40k
language/rust/p0-r16-40k
```

Exact naming MAY evolve, but identity MUST be machine-readable and unambiguous.

### 6.2 Adapter manifest

Each adapter artifact MUST be accompanied by a machine-readable manifest containing at least:

- adapter ID
- language
- adapter family (`language` initially)
- base-model repository and exact revision
- tokenizer/chat-template identity
- training Git commit SHA
- dataset manifest IDs/hashes
- training config hash
- seed
- LoRA target modules
- rank
- alpha
- dropout
- trainable parameter count
- precision
- sequence length
- optimizer/scheduler settings
- training steps/epochs
- peak VRAM
- validation metrics
- evaluation artifact IDs
- creation timestamp

### 6.3 Compatibility checks

A runtime MUST refuse or loudly warn before activating an adapter when any required compatibility field conflicts with the loaded base.

At minimum, exact base revision mismatch MUST be treated as incompatible by default.

### 6.4 Adapter storage

Adapters are expected to be much smaller than the full model. Normal runtime storage SHOULD therefore resemble:

```text
base/
└── qwen3.5-4b/

adapters/
└── language/
    ├── python/
    ├── typescript/
    └── rust/
```

Git MUST NOT contain the binary adapter weights unless explicitly published through an appropriate release/model-registry mechanism. Git SHOULD contain compact manifests, configuration, benchmark summaries, and provenance metadata.

---

## 7. Runtime adapter switching

### 7.1 Required behavior

The serving/runtime layer SHOULD support the conceptual operations:

```text
load_base()
load_adapter("python")
load_adapter("typescript")
load_adapter("rust")
set_adapter("python")
disable_adapter()
active_adapter()
```

The exact API depends on PEFT and the serving stack.

The base model SHOULD remain resident in VRAM while adapters are switched.

### 7.2 Correctness requirements

Adapter switching tests MUST verify that:

1. activating Python produces Python-adapter behavior;
2. switching to Rust produces Rust-adapter behavior without a base reload;
3. switching back to Python restores the same Python adapter state;
4. disabling adapters restores base-model behavior;
5. incompatible adapters are rejected;
6. repeated switching does not accumulate adapter effects accidentally;
7. memory usage remains bounded.

### 7.3 Automatic selection

A later runtime MAY infer the adapter from repository evidence such as:

**Python**

- `pyproject.toml`
- `uv.lock`
- `.py` files

**TypeScript**

- `package.json`
- `tsconfig.json`
- `.ts`/`.tsx` files

**Rust**

- `Cargo.toml`
- `Cargo.lock`
- `.rs` files

Automatic detection MUST have an explicit user override.

### 7.4 Polyglot repositories

Polyglot repositories MUST NOT be forced into a single permanent language classification.

Potential later strategies include:

1. explicit adapter selection per task;
2. selection based on the currently edited file;
3. dynamic switching during an agent loop;
4. future adapter composition/fusion.

The initial runtime SHOULD prefer explicit selection and deterministic behavior over clever automatic switching.

---

## 8. Hardware target

### 8.1 Primary training environment

The primary development target is a single NVIDIA GPU with approximately 16 GB VRAM.

The project SHOULD test BF16 LoRA first on the reference 16 GB GPU, but MUST NOT assume that a 4B checkpoint leaves enough training headroom. If BF16 LoRA is not comfortably memory-safe at the canonical 2,048-token preflight, P0 SHALL use 4-bit QLoRA with BF16 compute. The choice MUST be based on measured peak VRAM and then frozen for the experiment.

Initial expected configuration:

- BF16 base-model load/generation smoke test
- BF16 LoRA forward/backward preflight at sequence length 2,048 and micro-batch 1
- 4-bit QLoRA fallback validation if BF16 LoRA lacks safe headroom
- gradient checkpointing enabled where compatible
- sequence length: 2,048 tokens
- conservative micro-batch selected empirically
- gradient accumulation used to increase effective batch size

### 8.2 Memory measurement

Every training run MUST record:

- GPU model
- CUDA/runtime versions
- total VRAM
- peak allocated VRAM
- peak reserved VRAM when available
- micro-batch
- gradient accumulation
- effective batch
- sequence length
- LoRA rank
- LoRA target strategy
- trainable parameter count
- tokens/examples per second where available

No estimated VRAM figure SHALL be treated as a formal acceptance criterion. Actual peak memory MUST be measured.

---

## 9. Software stack

The initial implementation SHALL use:

- Python
- `uv`
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Hugging Face PEFT
- Hugging Face TRL
- Accelerate

Additional dependencies MAY be introduced for evaluation, static analysis, sandboxing, or optional performance kernels, but MUST be pinned through the project dependency mechanism.

### 9.1 Development quality gates

The repository SHOULD include:

- Ruff formatting
- Ruff linting
- one static type checker selected during bootstrap
- pytest
- deterministic unit tests for configuration and data transforms
- `git diff --check`-clean changes

GPU-dependent tests MUST NOT be required for ordinary CI unless a dedicated GPU runner is deliberately configured later.

---

## 10. Repository layout

The target layout is language-neutral:

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
│   ├── base/
│   │   └── qwen35-4b.yaml
│   ├── languages/
│   │   ├── python.yaml
│   │   ├── typescript.yaml
│   │   └── rust.yaml
│   ├── data/
│   │   ├── python/
│   │   ├── typescript/
│   │   └── rust/
│   ├── train/
│   ├── eval/
│   │   ├── python/
│   │   ├── typescript/
│   │   └── rust/
│   └── runtime/
├── src/
│   └── tiny_qwen_coder/
│       ├── __init__.py
│       ├── config.py
│       ├── model.py
│       ├── adapters/
│       ├── languages/
│       ├── data/
│       ├── training/
│       ├── evaluation/
│       ├── runtime/
│       └── reporting/
├── scripts/
│   ├── inspect_model.py
│   ├── prepare_data.py
│   ├── train_adapter.py
│   ├── evaluate.py
│   ├── infer.py
│   └── serve.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── data/                 # generated/local, ignored
├── artifacts/            # generated/local, ignored
└── outputs/              # generated/local, ignored
```

The exact module names MAY evolve. CLI scripts MUST remain thin entry points around shared package code.

---

## 11. Language plugin contract

Language-specific behavior SHOULD be represented behind a common interface or configuration contract.

A language definition SHOULD provide, as applicable:

- stable language ID
- aliases/file extensions
- repository-detection signals
- source dataset adapters
- filtering/validation hooks
- system-prompt specialization
- protected evaluation datasets
- execution/compiler/test harness
- syntax checker/parser
- benchmark definitions
- result metrics

Conceptually:

```python
LanguageSpec(
    id="python",
    extensions=[".py"],
    ...
)
```

The generic training pipeline MUST NOT need language-specific branching scattered throughout unrelated modules.

---

## 12. Dataset strategy

### 12.1 Generic requirements

Every language dataset pipeline MUST provide:

- deterministic source selection
- upstream revision recording
- provenance retention
- license metadata
- malformed/empty record filtering
- token-length filtering
- exact deduplication
- deterministic train/validation splitting
- contamination checks against protected evaluation data
- language-specific quality validation where safe
- a machine-readable dataset manifest

### 12.2 Initial Python sources

The first Python SFT corpus SHALL initially draw from:

1. `OLMo-Coding/starcoder-python-instruct`
   - use Python 3 examples
   - retain source/provenance metadata

2. `ise-uiuc/Magicoder-OSS-Instruct-75K`
   - use rows identified as Python
   - retain source/provenance metadata

Exact upstream revisions MUST be pinned or recorded in every prepared dataset manifest.

### 12.3 Initial Python P0 target

The first full Python SFT experiment SHOULD target approximately 40,000 accepted examples, initially budgeted as:

- approximately 30,000 accepted Python 3 OLMo-Coding examples
- approximately 10,000 accepted Magicoder Python examples when available after filtering

If the second source yields fewer accepted Python examples, the pipeline MAY fill from the primary source according to explicit configuration. Final composition MUST be reported from actual accepted data.

A deterministic 95/5 train/validation split MAY be used initially.

### 12.4 Conversational normalization

Training records SHOULD normalize to the common chat representation:

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

The system prompt MUST be versioned per language. Language-specific prompts SHOULD use the same overall behavioral contract so adapter comparisons are meaningful.

### 12.5 Python filtering

The Python plugin MUST initially support:

- Python 2 exclusion where metadata permits
- empty/malformed prompt/response removal
- length limits using the canonical Qwen tokenizer/template
- exact duplicate removal
- conservative Python syntax validation where the response is intended to be a standalone module/program

Syntax validation MUST NOT blindly reject legitimate snippets, REPL fragments, diffs, or partial code.

### 12.6 Planned TypeScript data

The TypeScript phase SHALL identify and document suitable open/code datasets before training. The pipeline SHOULD distinguish TypeScript from JavaScript where source metadata permits and SHOULD include both `.ts` and `.tsx` when appropriate.

Quality validation MAY use TypeScript parsing/type-checking on examples that form complete compilable units, but MUST avoid rejecting valid snippets solely because they lack project context.

### 12.7 Planned Rust data

The Rust phase SHALL identify and document suitable open/code datasets before training. Quality validation SHOULD use Rust parsing/compilation for complete units where feasible and preserve snippets that legitimately require surrounding project context.

Rust evaluation SHOULD make strong use of compiler and test feedback because the Rust toolchain provides high-quality executable correctness signals.

### 12.8 Dataset manifest

Every prepared dataset MUST emit a manifest containing at least:

- language
- dataset identifiers/revisions
- source license metadata as reported upstream
- preparation Git commit
- preparation config hash
- seed
- source row counts
- accepted counts
- rejection counts by reason
- train/validation counts
- deduplication statistics
- token-length statistics
- contamination findings
- available dataset fingerprints/checksums

---

## 13. Benchmark contamination policy

Each language plugin MUST register designated evaluation-only datasets.

Evaluation-only data MUST NOT be intentionally used as SFT training data.

The preparation pipeline SHOULD support at least:

- exact normalized prompt matching
- exact normalized solution/code matching where legally/practically available
- suspicious high-overlap reporting

A benchmark score MUST NOT be presented as evidence of generalization when known contamination invalidates the comparison.

For Python, HumanEval and MBPP are protected evaluation sets from the beginning.

---

## 14. Evaluation architecture

### 14.1 Baseline first

No language adapter is considered valid unless the unchanged canonical base model has first been evaluated using the same frozen evaluation configuration intended for the adapter comparison.

Evaluation artifacts MUST record:

- base repository/revision
- adapter identity or `none`
- language
- prompt/template version
- benchmark revision
- seed
- decoding parameters
- max generation length
- execution timeout
- environment/toolchain versions
- per-problem outcomes
- aggregate metrics

### 14.2 Python evaluation

Initial Python evaluation SHOULD include:

- HumanEval or a maintained compatible implementation
- MBPP or a maintained compatible implementation
- a repository-owned executable Python suite

The custom suite SHOULD cover representative tasks involving:

- standard library
- `pathlib`
- JSON
- regular expressions
- dataclasses
- typing
- iterators/generators
- decorators
- context managers
- exceptions
- async/await
- subprocess logic
- SQLite
- pytest-oriented code

### 14.3 TypeScript evaluation

The TypeScript phase MUST define an executable benchmark suite before training the canonical TypeScript adapter.

It SHOULD measure areas such as:

- TypeScript type-system use
- generics
- async/Promise behavior
- Node.js APIs
- module systems
- data transformations
- error handling
- `.tsx`/React tasks when included in scope
- tests and compiler/type-check outcomes

Exact public benchmarks SHALL be selected and pinned during the TypeScript phase based on then-current quality and licensing.

### 14.4 Rust evaluation

The Rust phase MUST define an executable benchmark suite before training the canonical Rust adapter.

It SHOULD measure:

- ownership/borrowing
- lifetimes where appropriate
- traits/generics
- iterators
- error handling
- concurrency
- async where included
- Cargo project changes
- compiler correctness
- unit/integration tests
- Clippy or equivalent static checks where appropriate

### 14.5 Safe execution

Generated code MUST run in a constrained disposable environment with:

- wall-clock timeout
- bounded CPU/memory where practical
- network disabled by default
- no host credentials/secrets
- captured stdout/stderr/exit status

### 14.6 Common metrics

Where applicable, all language adapters SHOULD report:

- syntax/parse validity
- compile/type-check validity
- tests passed/total
- pass@1
- timeout rate
- runtime/compiler error category
- generated token count
- latency/tokens per second

---

## 15. Cross-language evaluation matrix

Every mature language adapter MUST be evaluated not only on its target language but also against at least the other canonical language suites.

The project SHOULD maintain a matrix such as:

| Active adapter | Python | TypeScript | Rust | General/tool |
| --- | ---: | ---: | ---: | ---: |
| none/base | baseline | baseline | baseline | baseline |
| Python | expected ↑ | measure | measure | measure |
| TypeScript | measure | expected ↑ | measure | measure |
| Rust | measure | measure | expected ↑ | measure |

The objective is to quantify specialization and catastrophic interference.

A language adapter MAY reduce unrelated-language performance slightly. Severe collapse MUST be visible and SHOULD block promotion to a recommended adapter unless the tradeoff is explicitly accepted.

Cross-language tests MUST use identical base revision and compatible generation settings.

---

## 16. General/tool regression suite

Because the canonical checkpoint is post-trained, every language adapter MUST be checked against a frozen non-language regression suite.

The suite SHOULD include:

- general instruction following
- structured JSON output
- simple reasoning
- shell/Git reasoning
- tool/function-call formatting
- tool selection on small deterministic examples

The objective is:

```text
target-language coding       ↑↑
general behavior             approximately preserved
tool behavior                approximately preserved
other languages              measured, not ignored
```

Quantitative promotion thresholds SHALL be established after the reproducible base baseline exists.

---

## 17. Initial Python LoRA experiment

The first canonical adapter SHALL be Python.

Initial candidate configuration:

```text
base model              Qwen/Qwen3.5-4B @ pinned revision
adapter family          language
language                python
training mode           frozen P2 memory decision (BF16 LoRA preferred; 4-bit QLoRA fallback)
fine-tuning             LoRA SFT
sequence length         2048
LoRA rank               16
LoRA alpha              32
LoRA dropout            0.05
bias                    none
learning rate           2e-4 initial candidate
scheduler               cosine initial candidate
warmup ratio            0.03 initial candidate
epochs                  1
gradient checkpointing  enabled when compatible
loss                    assistant/completion tokens only
```

Micro-batch and gradient accumulation MUST be selected from actual hardware measurements. For the 4B checkpoint, the initial BF16 training preflight SHALL start at micro-batch 1 and increase only after measured headroom is established.

### 17.1 Assistant-only loss

The training pipeline SHOULD use TRL assistant-only loss when the pinned Qwen chat template produces a valid assistant token mask.

The implementation MUST verify the mask. If unsupported, it MUST use a tested equivalent masking strategy or completion-only representation. It MUST NOT silently train user/system tokens while claiming assistant-only SFT.

### 17.2 LoRA target discovery

The P0 Python adapter MUST use target modules discovered from the actual loaded Qwen3.5 architecture.

The training manifest MUST record the exact matched modules and trainable parameter count.

A later controlled experiment SHALL compare selective targeting with `all-linear` or another complete supported linear-target strategy.

### 17.3 Adapter promotion

The first Python run is an experiment, not automatically the canonical release.

Promotion requires:

- complete dataset manifest
- complete training manifest
- baseline comparison
- target-language improvement
- cross-language/regression report
- adapter load/switch validation
- no unresolved benchmark contamination finding

---

## 18. Hyperparameter experiment policy

After P0, experiments MAY vary:

- LoRA rank: e.g. 8, 16, 32, 64
- alpha
- target strategy
- dataset size: e.g. 10k, 25k, 50k, 100k+
- sequence length
- epochs/steps
- learning rate
- source mixture
- BF16 LoRA versus QLoRA

Experiments SHOULD change one major variable at a time unless explicitly designed as a factorial study.

All comparisons MUST use frozen evaluation configurations.

---

## 19. Continued pretraining

After SFT baselines exist, the project MAY investigate continued causal-language-model pretraining on permissively licensed source code.

A possible language pipeline becomes:

```text
canonical base
     ↓
language-specific continued pretraining
     ↓
language instruction SFT
     ↓
language adapter/model experiment
```

Because continued pretraining through LoRA and SFT LoRA behavior may interact, this is a separate experiment family and MUST NOT silently replace the P0 recipe.

Source-code licensing and provenance require additional scrutiny before large raw-code training.

---

## 20. Coding-agent trajectory training

After language SFT is understood, the project SHALL investigate verified coding-agent trajectories such as:

```text
user task
  ↓
read/list/search repository
  ↓
edit code
  ↓
run formatter/compiler/tests
  ↓
observe failure
  ↓
repair
  ↓
verification succeeds
```

Useful metrics include:

- tool-call JSON validity
- correct tool selection
- unnecessary tool calls
- patch applicability
- compile/test success
- recovery after failure
- turns to success
- token consumption
- premature completion
- loop frequency

Language-agent training MUST be distinguishable from pure language SFT in manifests and evaluation.

---

## 21. OpenCode/runtime integration

The long-term serving objective is:

```text
OpenCode or compatible client
            │
            ▼
OpenAI-compatible local endpoint
            │
            ▼
Tiny Qwen Coder runtime
            │
      ┌─────┴─────┐
      │           │
      ▼           ▼
Qwen3.5-4B   adapter manager
      │           │
      └─────┬─────┘
            ▼
 active language LoRA
```

The runtime SHOULD expose or otherwise support explicit adapter selection.

A future client integration MAY automatically select adapters based on project/file context, but explicit override MUST remain available.

OpenCode readiness MUST be demonstrated with repository-level executable tasks, not inferred from HumanEval/MBPP alone.

---

## 22. Polyglot and dynamic-switch research

After Python, TypeScript, and Rust adapters all exist, the project SHOULD evaluate realistic polyglot repositories.

Important cases include:

- Python backend + TypeScript frontend
- Rust/Tauri backend + TypeScript frontend
- build scripts/configuration crossing language boundaries

Experiments MAY compare:

1. one adapter fixed for the entire task;
2. explicit adapter switch between subtasks;
3. file-sensitive automatic adapter selection;
4. dynamic switching during an agent loop;
5. adapter composition/fusion.

Dynamic switching MUST verify that conversation state remains coherent and that adapter state does not leak or accumulate unexpectedly.

---

## 23. Reproducibility

Every training/evaluation run MUST record enough metadata to reconstruct the experiment, including:

- Git commit SHA
- base model repository/revision
- tokenizer/chat-template identity
- language
- adapter family/ID
- dataset identifiers/revisions/manifests
- exact training/evaluation config
- random seeds
- dependency versions
- hardware metadata
- CUDA/runtime information
- LoRA target modules
- trainable parameter count
- batch/sequence settings
- peak VRAM
- output artifact hashes where practical

A run SHALL NOT be considered reproducible merely because a shell command appears in chat history.

### 23.1 Deterministic seeding contract

The project-wide seed MUST be an integer in the inclusive range `0..2^32-1`, which is
compatible with Python, NumPy's legacy global RNG, PyTorch, and common dataset tooling.
Before stochastic data preparation, training, or evaluation begins, the shared seeding
utility MUST:

- seed Python `random`;
- seed NumPy;
- seed PyTorch CPU RNGs;
- seed all CUDA-visible PyTorch devices when CUDA is available;
- enable PyTorch deterministic algorithms in fail-closed mode;
- disable cuDNN benchmarking and enable deterministic cuDNN behavior;
- set a deterministic cuBLAS workspace configuration; and
- set `PYTHONHASHSEED` for child processes.

Dataset shuffle/split helpers MUST use an isolated RNG initialized from the configured seed
so their output is unaffected by unrelated consumption of the process-global RNG. PyTorch
DataLoader/sampler code SHOULD use the project's seeded `torch.Generator` and worker seeding
hook rather than relying on implicit worker RNG state.

Deterministic seeding does **not** promise bitwise-identical results across arbitrary
environments. Remaining sources of nondeterminism or numerical variation include different
GPU architectures, drivers, CUDA/cuDNN/PyTorch versions, distributed reduction order,
third-party kernels, and operations for which PyTorch has no deterministic implementation.
The project therefore records environment and dependency metadata in run manifests. When
PyTorch identifies a requested operation as nondeterministic while deterministic algorithms
are enabled, execution SHOULD fail rather than silently accept nondeterminism.

`PYTHONHASHSEED` is fixed when a Python interpreter starts. Setting it inside a running
process only controls child processes; callers that require deterministic hash iteration in
the parent interpreter MUST launch Python with the desired `PYTHONHASHSEED` already set.

### 23.2 Standalone environment report

The repository MUST provide a standalone machine-readable environment report that can run
without loading the Qwen model, preparing a dataset, or starting training. The report MUST
include the Python interpreter version, PyTorch version, CUDA availability/runtime, cuDNN
version when available, the core ML dependency versions, host platform metadata, and one
record per CUDA-visible GPU with its name, compute capability, total VRAM, and current free
VRAM. The report SHOULD be suitable for preflight diagnostics and for attaching to experiment
artifacts. `tiny-qwen-coder-env` is the canonical command-line entry point and MUST emit JSON.

---

## 24. Experiment artifact layout

A run SHOULD create a structure such as:

```text
artifacts/
└── language/
    └── python/
        └── <run-id>/
            ├── run.json
            ├── dataset-manifest.json
            ├── training-config.json
            ├── training-metrics.jsonl
            ├── evaluation-summary.json
            ├── evaluation-details.jsonl
            └── adapter-manifest.json

outputs/
└── language/
    └── python/
        └── <run-id>/
            ├── adapter_config.json
            └── adapter_model.safetensors
```

The same layout MUST work for TypeScript, Rust, and future languages without code duplication.

---

## 25. Git and artifact policy

Git MUST exclude:

- downloaded model weights
- Hugging Face caches
- generated training datasets
- checkpoints
- binary LoRA adapter weights unless explicitly published
- benchmark execution sandboxes
- local run logs unsuitable for source control
- virtual environments and Python caches

Git SHOULD contain:

- source code
- pinned configuration
- dataset manifests/hashes where appropriate
- compact evaluation summaries
- experiment metadata
- documentation
- small synthetic test fixtures

Reproducibility SHOULD come from pinned identifiers, deterministic preparation, configs, seeds, manifests, and hashes—not from committing huge generated artifacts.

---

## 26. CI policy

Required GitHub Actions CI SHOULD remain CPU-safe and bounded.

CI SHOULD run:

- dependency resolution from lockfile
- format check
- lint
- type check
- unit tests
- small synthetic dataset-pipeline tests
- configuration validation
- adapter-manifest/compatibility tests using mocks or tiny fixtures

Required CI MUST NOT:

- download the full Qwen model solely to validate ordinary changes
- perform real GPU LoRA training
- run large public benchmarks

GPU validation MAY be added later through a dedicated optional/self-hosted workflow.

---

## 27. Security requirements

Generated code and untrusted dataset content MUST be treated as untrusted input.

Evaluation execution MUST use constrained disposable environments and MUST NOT expose:

- host SSH keys
- cloud credentials
- GitHub tokens
- Hugging Face write tokens
- unrelated host files

Training scripts MUST NOT execute arbitrary code found in datasets as part of preprocessing.

Model/dataset downloads SHOULD be pinned and preferably use safetensors where available.

---

## 28. Success criteria

### 28.1 Python milestone

The Python milestone succeeds when:

1. the canonical base revision is pinned;
2. Python data preparation is deterministic and auditable;
3. base Python/general/tool baselines are recorded;
4. the frozen canonical LoRA mode (BF16 LoRA or the measured QLoRA fallback) completes reproducibly on the target hardware;
5. the adapter materially improves executable Python performance;
6. general/tool behavior does not catastrophically collapse;
7. cross-language impact is measured;
8. the adapter can be loaded, disabled, and reactivated against the same resident base model;
9. all manifests and artifacts are sufficient to reproduce the run.

### 28.2 Multi-language milestone

The multi-language milestone succeeds when:

1. Python, TypeScript, and Rust adapters all target the same pinned canonical base revision;
2. all three use the same generic training/evaluation infrastructure;
3. each materially improves its target language over the unchanged base;
4. adapter switching works without reloading the base model;
5. incompatibility checks prevent accidental cross-base adapter use;
6. a cross-language evaluation matrix is published;
7. storage/runtime clearly demonstrates one base model plus small interchangeable adapters.

### 28.3 Coding-agent milestone

The coding-agent milestone succeeds when at least one language-specialized configuration can complete a meaningful set of repository-level tasks using tools, make edits, execute validation, recover from failures, and outperform the unchanged base on the same frozen task suite.

---

## 29. Planned language order

Unless evidence justifies a change, implementation order is:

1. **Python** — prove the architecture and training methodology.
2. **TypeScript** — validate reuse of the generic pipeline on a different dynamic/typed ecosystem and `.tsx` possibilities.
3. **Rust** — validate reuse on a compiler-heavy systems language with strong executable feedback.
4. cross-language and polyglot experiments.
5. behavior/tool specialization and adapter-composition research.

Python-specific implementation decisions MUST NOT unnecessarily constrain TypeScript or Rust.

---

## 30. Guiding rule

The central invariant of `tiny-qwen-coder` is:

> **One pinned Qwen3.5-4B base model; many small, independently trained, interchangeable programming-language LoRA adapters.**

Every major design choice SHOULD be evaluated against that invariant.
