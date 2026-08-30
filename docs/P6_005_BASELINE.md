# P6-005 unchanged-base Python baseline

P6-005 freezes the canonical evaluation evidence for the unchanged
`Qwen/Qwen3.5-4B` base before any P0 Python adapter training. The run covers,
in this fixed order:

1. HumanEval;
2. MBPP;
3. the repository-owned Python holdout suite;
4. the frozen general/tool regression suite.

The baseline also records CUDA memory, model-load time, generation latency and
throughput, host/dependency provenance, and SHA-256 digests for every required
artifact.

## Canonical contract

The runner reads `configs/eval/python/base_baseline_v1.yaml` and fails closed if
its base model, suite order, language, generation settings, execution settings,
or Python system prompt drift from the frozen contract. The base is loaded
without an adapter in BF16. Generation is greedy and Qwen3.5 thinking mode is
disabled explicitly so code and structured-output scoring receive only the
answer channel.

Generated code is never executed on the host. HumanEval, MBPP, and the custom
holdout run through the constrained OCI harness with networking disabled and the
digest-pinned image:

```text
python:3.11.14-slim@sha256:c8271b1f627d0068857dce5b53e14a9558603b527e46f1f901722f935b786a39
```

The harness uses `--pull never`, so that image must already exist in the OCI
runtime selected on the evaluation machine. Podman is preferred when both
Podman and Docker are installed.

## Prerequisites

Run from a clean checkout of the intended `master` commit. The machine must
have:

- an NVIDIA CUDA GPU large enough for the BF16 base model and generation cache;
- the project environment installed from the frozen lockfile;
- Podman or Docker available locally;
- the pinned Python execution image already present in that runtime;
- access to the pinned Qwen model/tokenizer and protected HumanEval/MBPP data,
  either from the local Hugging Face cache or from the network during loading.

A dirty Git tree is rejected because the frozen manifest binds the results to
one exact source commit.

## Run

From the repository root:

```bash
uv sync --frozen
uv run --frozen tiny-qwen-coder-python-baseline
```

To select another visible CUDA device:

```bash
uv run --frozen tiny-qwen-coder-python-baseline --device-index 1
```

The canonical output directory is:

```text
artifacts/eval/python/base-baseline-v1/
```

Generation checkpoints are written beneath `.baseline-work/` while a run is in
progress. If generation is interrupted, rerunning the same command reuses only
checkpoint records whose item ID, exact prompt hash, and generation-contract
hash still match. The work directory is removed after a successful freeze.

## Frozen artifacts

A successful run requires all of the following before
`baseline-manifest.json` is written:

```text
provenance.json
runtime-metadata.json
humaneval/humaneval-results.jsonl
humaneval/humaneval-aggregate.json
mbpp/mbpp-results.jsonl
mbpp/mbpp-aggregate.json
repository-holdout/repository-holdout-results.jsonl
repository-holdout/repository-holdout-aggregate.json
general-tool-regression/general-tool-regression-results.jsonl
general-tool-regression/general-tool-regression-aggregate.json
```

The manifest binds the source Git SHA, exact base/tokenizer identity, frozen
evaluation settings, Python system prompt, generation contract, and SHA-256 of
every artifact. Runs from a dirty tree or without recorded CUDA provenance are
not eligible for freezing.

## Verify

After the run, validate every artifact digest without loading the model:

```bash
uv run --frozen tiny-qwen-coder-python-baseline --verify-only
```

Verification fails if a required artifact is missing, altered, renamed, or no
longer matches the frozen artifact-set digest.

The P0 adapter training phase must not begin until this command succeeds for the
canonical GPU-produced baseline.
