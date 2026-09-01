# P8-003 — Preliminary cross-language smoke tests

P8-003 asks a narrow diagnostic question: did the accepted Python P0 LoRA cause a catastrophic loss of basic non-Python coding behavior before the adapter is considered for promotion?

This is **not** a TypeScript or Rust benchmark and must not be used as evidence that either language is production-ready. The full TypeScript and Rust language plugins, protected benchmarks, compilers/execution hooks, and canonical adapters belong to later phases.

## Frozen probe

The probe is frozen before the canonical GPU run and contains six self-contained code-only prompts:

- TypeScript: typed addition, a generic `firstOrUndefined`, and an async typed function.
- Rust: typed integer addition, a generic `Option<T>` first-element helper, and checked division returning `Option<i32>`.

Each response is scored structurally without executing generated code. The scorer requires the requested language-specific signature and key semantic fragments, rejects Markdown fences and obvious wrong-language syntax, and rejects responses that consume the entire bounded token budget.

The frozen suite SHA-256 is:

`7f6a6329cabbdb62c1823b0f16cad5bf8067300164c5a3b6ade84bcaaf802f52`

The system prompt is deliberately neutral rather than the Python training prompt:

> You are a coding assistant. Follow the user's requested programming language exactly. When the user requests only code, return only code with no Markdown fences or explanation.

Both the unchanged base and Python P0 adapter receive exactly the same prompt/template and deterministic generation settings.

## Exact model and adapter identity

The base is the canonical pinned `Qwen/Qwen3.5-4B` revision from `configs/base/qwen35-4b.yaml`.

The only accepted adapter is the P7-006 Python P0 artifact:

- adapter ID: `language/python/p0`
- adapter weights SHA-256: `c94606250e112f72362eb883a55f7b2c8af854d445f6bb6194352c2806a8f276`
- adapter weight size: `65004840` bytes
- training run ID: `training-python-20260831T180916446466Z-02df92a9-eafc119d`
- training source Git SHA: `02df92a9c2d347b9fb013dc25714fe066c6bcafe`

The workflow downloads the canonical P7-006 GitHub Actions artifact and refuses a missing, ambiguous, or hash-mismatched adapter. The runtime then performs the stronger P7-007 artifact/provenance validation before loading the base.

## Catastrophic-collapse rule

The detection rule is frozen before seeing P8-003 GPU output.

The probe is conclusive only when the unchanged base passes at least two of three TypeScript cases **and** at least two of three Rust cases. If either base-language score is below `2/3`, the run is recorded as `inconclusive_base`; it must not be interpreted as evidence that Python P0 avoided cross-language collapse, and P8-003 remains open.

Once the base signal is adequate, a catastrophic non-Python collapse is flagged if either condition holds:

1. For TypeScript or Rust individually, the base passes at least two of the three cases and Python P0 passes zero.
2. Across all six cases, the base passes at least four and Python P0 passes half or fewer of the number passed by the base.

A conclusive run records one of `catastrophic_regression` or `no_catastrophic_regression`. The measurement remains valid even when a catastrophic regression is detected. A negative model result is evidence, not an evaluation-harness failure.

## Runtime contract

The canonical workflow `.github/workflows/python-p0-cross-language-smoke.yml` is manual-only. It:

1. requires CUDA and BF16 support;
2. uses a clean source checkout;
3. downloads and hash-validates the exact accepted P7-006 adapter;
4. loads the exact base once in BF16;
5. generates all six base responses;
6. attaches the LoRA to that same resident base object and verifies it remains unmerged/inference-only;
7. generates the same six prompts with Python P0 enabled;
8. scores every response and records per-case transitions plus per-language/overall totals;
9. independently rereads and recomputes the persisted report, including the source Git identity;
10. uploads compact evidence for seven days, or partial evidence for three days on failure.

There is no CPU/full-precision/model-substitution fallback. A missing GPU, base revision drift, tokenizer/template drift, adapter provenance mismatch, malformed evidence, unexpected PEFT state, or source-identity mismatch fails closed.

## Interpretation

P8-003 is supplemental diagnostic evidence for P8-004. The decisive Python result remains P8-001, where P0 materially regressed HumanEval, MBPP, and the repository holdout. A conclusive clean P8-003 result would only narrow the failure mode: it would suggest the Python adapter's largest observed damage is concentrated in Python/coding-task behavior rather than a wholesale inability to emit basic TypeScript/Rust-shaped code.
