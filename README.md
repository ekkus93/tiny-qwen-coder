# Tiny Qwen Coder

Tiny Qwen Coder is a research project for turning a single shared `Qwen/Qwen3.5-4B` base model into a family of programming-language specialists by attaching interchangeable LoRA adapters.

The base model is intended to remain fixed. Language expertise is packaged separately:

```text
Qwen3.5-4B
    |
    +-- Python LoRA
    +-- TypeScript LoRA
    `-- Rust LoRA
```

The long-term runtime goal is to load the base model once and switch only the active LoRA adapter when the working language changes. Python is the first implementation target. TypeScript and Rust are planned follow-on adapters built on the same language-neutral data, training, evaluation, and runtime infrastructure.

## Project status

The repository is being implemented incrementally from [`docs/TODO.md`](docs/TODO.md). The architecture, experimental methodology, reproducibility requirements, dataset policy, evaluation strategy, and adapter compatibility rules are defined in [`docs/SPEC.md`](docs/SPEC.md).

Current scope:

- one pinned `Qwen/Qwen3.5-4B` base model;
- BF16 LoRA preferred on the 16 GB reference GPU, with 4-bit QLoRA selected if measured training headroom requires it;
- text/code-only specialization for the multimodal 4B checkpoint, with vision components kept frozen;
- Python as the first language specialization;
- TypeScript and Rust adapters after the Python pipeline proves the architecture;
- explicit regression testing so specialization does not silently destroy general instruction-following or tool-use behavior;
- later OpenCode/tool-agent integration after language-adapter training and switching are reliable.

## Requirements

For normal development and CPU-safe tests:

- Linux, macOS, or another environment supported by the pinned Python dependencies;
- Python 3.11 or newer;
- [`uv`](https://docs.astral.sh/uv/) for dependency and environment management;
- Git.

A GPU is **not** required for repository development, linting, type checking, unit tests, dataset-pipeline tests, or other CPU-safe validation.

GPU-backed work such as loading the canonical model in BF16, measuring VRAM, LoRA training, and model evaluation will require a compatible NVIDIA GPU and CUDA/PyTorch environment. GPU work is intentionally separated from the required CPU-safe development/CI path.

## Setup

Clone the repository and enter it:

```bash
git clone https://github.com/ekkus93/tiny-qwen-coder.git
cd tiny-qwen-coder
```

Create the project environment exactly from the committed lockfile:

```bash
uv sync --frozen
```

Verify the package is importable:

```bash
uv run --frozen python -c "import tiny_qwen_coder"
```

Print the resolved Python and project dependency versions:

```bash
uv run --frozen tiny-qwen-coder-versions
```

The project declares Python `>=3.11`; `.python-version` selects Python 3.11 when the requested interpreter is available.

## Quality gates

Run these commands from the repository root before publishing a change.

### Formatting

Check formatting without modifying files:

```bash
uv run --frozen ruff format --check .
```

To apply Ruff formatting locally:

```bash
uv run --frozen ruff format .
```

### Linting

```bash
uv run --frozen ruff check .
```

To apply Ruff's safe automatic fixes where available:

```bash
uv run --frozen ruff check --fix .
```

### Type checking

```bash
uv run --frozen mypy src scripts tests
```

The repository currently runs mypy in strict mode.

### Tests

```bash
uv run --frozen pytest
```

### Git whitespace check

```bash
git diff --check
```

A normal pre-publish validation sequence is therefore:

```bash
uv sync --frozen
uv run --frozen ruff format --check .
uv run --frozen ruff check .
uv run --frozen mypy src scripts tests
uv run --frozen pytest
git diff --check
```

## CPU development versus GPU work

The project deliberately distinguishes inexpensive repository validation from model workloads.

### CPU-safe development

The following should remain runnable without downloading the full model or requiring CUDA:

- formatting and linting;
- static type checking;
- unit tests;
- configuration parsing and validation;
- dataset-pipeline tests using bounded fixtures;
- adapter-manifest and compatibility tests using synthetic metadata;
- other small deterministic infrastructure tests.

Required GitHub Actions CI is intended to remain CPU-safe as well.

### GPU-backed work

GPU execution is reserved for work that genuinely needs the model, including:

- canonical `Qwen3.5-4B` model-load smoke tests;
- BF16 LoRA and QLoRA feasibility/VRAM measurements;
- LoRA smoke training and canonical training runs;
- base-versus-adapter generation benchmarks;
- adapter switching/runtime performance measurements;
- later agent/OpenCode experiments.

Model checkpoints, generated adapters, datasets, benchmark sandboxes, logs, and experiment artifacts are intentionally excluded from normal Git tracking.

## Repository layout

```text
configs/
    base/       shared base-model identity/config
    languages/  programming-language definitions
    data/       dataset preparation configs
    train/      training configs
    eval/       evaluation configs
    runtime/    adapter/runtime configs

src/tiny_qwen_coder/
    model/      shared model loading/inspection
    adapters/   LoRA metadata and compatibility
    languages/  language registry/plugins
    data/       language-neutral data pipeline
    training/   generic adapter training
    evaluation/ common evaluation framework
    runtime/    inference/adapter switching/serving
    reporting/  manifests and reports

scripts/
    inspect_model.py
    prepare_data.py
    train_adapter.py
    evaluate.py
    infer.py
    serve.py

tests/
    unit/
    integration/
    fixtures/
```

The generic scripts are intentionally language-neutral. The project should add a new language through configuration/plugins rather than by creating separate entry points such as `train_python.py`, `train_typescript.py`, and `train_rust.py`.

## Adapter model

All canonical language adapters must target the same pinned base-model revision, tokenizer/chat-template identity, and compatible LoRA architecture. An adapter manifest will record those identities so an incompatible adapter can be rejected instead of silently attached to the wrong base model.

The initial target family is:

| Adapter | Status | Purpose |
| --- | --- | --- |
| Python | First target | Python generation, reasoning, repair, and later tool-driven development |
| TypeScript | Planned | TypeScript/JavaScript ecosystem specialization |
| Rust | Planned | Rust generation, compiler-guided repair, and Cargo-oriented development |

Polyglot repositories and automatic/dynamic adapter selection are later milestones. The initial system will prove that one base model can be reused while adapters are selected explicitly and independently.

## Documentation

- [`docs/SPEC.md`](docs/SPEC.md) — project specification and design contract.
- [`docs/TODO.md`](docs/TODO.md) — phased implementation plan and acceptance criteria.

The specification is authoritative for architecture and experimental rules; the TODO is the execution checklist used to implement it.

## License

See [`LICENSE`](LICENSE).
