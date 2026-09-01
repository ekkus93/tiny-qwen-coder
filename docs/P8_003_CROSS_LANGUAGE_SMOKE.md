# P8-003 — Preliminary cross-language smoke tests

P8-003 asks a narrow diagnostic question: did the accepted Python P0 LoRA cause a catastrophic loss of basic non-Python coding behavior before the adapter is considered for promotion?

This is **not** a TypeScript or Rust benchmark and must not be used as evidence that either language is production-ready. The full TypeScript and Rust language plugins, protected benchmarks, compiler/execution hooks, and canonical adapters belong to later phases.

## Frozen prompt suite

The prompt suite contains six self-contained code-only prompts:

- TypeScript: typed addition, a generic `firstOrUndefined`, and an async typed function.
- Rust: typed integer addition, a generic `Option<T>` first-element helper, and checked division returning `Option<i32>`.

The prompt-suite SHA-256 is frozen as:

`7f6a6329cabbdb62c1823b0f16cad5bf8067300164c5a3b6ade84bcaaf802f52`

The system prompt is deliberately neutral rather than the Python training prompt:

> You are a coding assistant. Follow the user's requested programming language exactly. When the user requests only code, return only code with no Markdown fences or explanation.

The system prompt remains versioned as `cross-language-smoke-v1` because v2 changes only measurement/scoring, not generation. Both the unchanged base and Python P0 adapter receive exactly the same prompt/template and deterministic generation settings.

## V1 — preserved inconclusive measurement

P8-003 v1 used a single strict structural score. It rejected Markdown fences because the prompts requested code only. That behavior remains frozen and is **not** retroactively weakened.

The canonical v1 GPU measurement was GitHub Actions run `33557769986` at source Git SHA `27550522181fc8cf3a490a03c983df55f6022430`.

- artifact ID: `9820048413`
- artifact ZIP SHA-256: `cf43d9d381647030a18f6be1df52c664614446fbf3aca1238a85c4a3f1bbdd10`
- TypeScript: base `0/3`, Python P0 `3/3`
- Rust: base `0/3`, Python P0 `3/3`
- overall: base `0/6`, Python P0 `6/6`
- conclusion: `inconclusive_base`
- `baseline_adequate_for_collapse_detection`: `false`

Every unchanged-base response contained the requested language-shaped code but wrapped it in one Markdown fence (`typescript` for TypeScript and `rust` for Rust). Python P0 emitted corresponding unfenced code. Therefore v1 measured a format-adherence difference rather than an adequate baseline for catastrophic semantic-collapse detection.

The exact independently verified v1 JSON report, including raw generations and token IDs, is committed permanently at `docs/evidence/P8_003_V1_VERIFIED_REPORT_33557769986.json`. The expiring Actions artifact is not the sole copy of the evidence.

## V2 — frozen two-dimensional scoring contract

V2 preserves the v1 generation inputs and adds a separately versioned scoring contract. It measures two independent dimensions from the same freshly generated responses.

Scoring contract version:

`cross-language-smoke-scoring-v2`

Scoring contract SHA-256:

`33c2459c64631ee7cd8903c36a6fe6ecb81df6ce6e1848bad096b5803cc77dd2`

### 1. Format adherence

The `format_adherence` dimension retains the v1 strict behavior:

- the response must be non-empty;
- generation must stop before the bounded token budget is exhausted;
- Markdown fences fail;
- obvious wrong-language patterns fail;
- all frozen language-specific structural patterns must match.

This dimension remains useful for diagnosing whether Python P0 changed instruction/format following, but it does **not** decide catastrophic cross-language collapse in v2.

### 2. Semantic shape

The `semantic_shape` dimension asks whether the response still has the requested TypeScript/Rust structural semantics while allowing the specific formatting variation exposed by v1.

A response is eligible for structural scoring only when it is either:

1. plain code with no Markdown fence; or
2. exactly one whole-response Markdown fenced block, with no text outside the fence, whose language tag is exactly `typescript` for TypeScript or `rust` for Rust.

For an accepted fence, v2 unwraps only that fence and applies the **same frozen required/forbidden language patterns** used by the strict score. It does not execute generated code.

V2 fails closed on:

- an untagged fence;
- a wrong-language fence tag;
- prose before or after a fenced block;
- malformed, nested, or multiple fences;
- obvious wrong-language syntax;
- missing requested structural/semantic fragments;
- empty output; or
- generation that reaches the full token budget.

## Catastrophic-collapse rule

The catastrophic-collapse decision uses **only `semantic_shape`**. `format_adherence` is supplemental diagnostic evidence.

The semantic measurement is conclusive only when the unchanged base passes at least two of three TypeScript cases **and** at least two of three Rust cases. If either base-language semantic score is below `2/3`, the run is `inconclusive_base`; it must not be interpreted as evidence that Python P0 avoided cross-language collapse.

Once the semantic base signal is adequate, catastrophic non-Python collapse is flagged if either condition holds:

1. For TypeScript or Rust individually, the base passes at least two of the three cases and Python P0 passes zero.
2. Across all six cases, the base passes at least four and Python P0 passes half or fewer of the number passed by the base.

A conclusive semantic run records one of `catastrophic_regression` or `no_catastrophic_regression`. A negative model result is evidence, not an evaluation-harness failure.

## Exact model and adapter identity

The base is the canonical pinned `Qwen/Qwen3.5-4B` revision from `configs/base/qwen35-4b.yaml`.

The only accepted adapter is the P7-006 Python P0 artifact:

- adapter ID: `language/python/p0`
- adapter weights SHA-256: `c94606250e112f72362eb883a55f7b2c8af854d445f6bb6194352c2806a8f276`
- adapter weight size: `65004840` bytes
- training run ID: `training-python-20260831T180916446466Z-02df92a9-eafc119d`
- training source Git SHA: `02df92a9c2d347b9fb013dc25714fe066c6bcafe`

The workflow downloads the canonical P7-006 GitHub Actions artifact and refuses a missing, ambiguous, or hash-mismatched adapter. The runtime then performs the stronger P7-007 artifact/provenance validation before loading the base.

## V2 runtime contract

The canonical workflow `.github/workflows/python-p0-cross-language-smoke.yml` is manual-only. It:

1. requires CUDA and BF16 support;
2. uses a clean source checkout;
3. downloads and hash-validates the exact accepted P7-006 adapter;
4. validates both the frozen prompt-suite hash and v2 scoring-contract hash before generation;
5. loads the exact base once in BF16;
6. generates all six base responses;
7. attaches the LoRA to that same resident base object and verifies it remains unmerged/inference-only;
8. generates the same six prompts with Python P0 enabled;
9. records both `format_adherence` and `semantic_shape` for every case plus per-language/overall aggregates;
10. makes the catastrophic-collapse decision from `semantic_shape` only;
11. independently rereads and recomputes the persisted v2 report, including source Git identity and both frozen fingerprints;
12. uploads compact v2 evidence for seven days, or partial evidence for three days on failure.

There is no CPU/full-precision/model-substitution fallback. A missing GPU, base revision drift, tokenizer/template drift, adapter provenance mismatch, malformed evidence, unexpected PEFT state, source-identity mismatch, prompt-suite drift, or scoring-contract drift fails closed.

## Accepted V2 measurement

The canonical v2 GPU measurement is GitHub Actions run `33570451764` at source Git SHA `56039856392b5a4a3eecad147518c3657ccd683f`.

The run completed every required step successfully: CUDA/BF16 validation, clean-source verification, exact P7-006 artifact download and hash validation, frozen v2 generation, independent report verification, and evidence upload.

Evidence identity:

- artifact ID: `9824776245`
- artifact name: `p8-003-v2-python-p0-cross-language-56039856392b5a4a3eecad147518c3657ccd683f`
- artifact ZIP SHA-256: `ae57a061ddd02ff76f5dde4d454165f4d4421efc51778acade8d9537d8787b1a`
- persisted/independently verified report SHA-256: `de386ec1eca9fcfed4f797db223d686d7ef131292c9b9624b5d33d87d42357ee`
- measurement version: `cross-language-smoke-v2`
- scoring contract: `cross-language-smoke-scoring-v2`
- scoring contract SHA-256: `33c2459c64631ee7cd8903c36a6fe6ecb81df6ce6e1848bad096b5803cc77dd2`
- exact resolved Qwen revision: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- same resident base object after adapter attachment: `true`
- adapter enabled, unmerged, inference-only: `true`
- GPU: `NVIDIA GeForce RTX 4070 Ti SUPER`
- peak allocated VRAM: `9330175488` bytes
- peak reserved VRAM: `9403629568` bytes
- `measurement_complete`: `true`

The report embedded in the artifact and the independently verified report are byte-identical.

### Decision-bearing semantic result

| Language | Base | Python P0 | Regressions |
| --- | ---: | ---: | ---: |
| TypeScript | `3/3` | `3/3` | `0` |
| Rust | `3/3` | `3/3` | `0` |
| **Overall** | **`6/6`** | **`6/6`** | **`0`** |

The unchanged base is adequate for collapse detection in both languages. The v2 semantic conclusion is therefore:

`no_catastrophic_regression`

and:

`catastrophic_non_python_collapse_detected: false`

### Supplemental format result

The strict `format_adherence` result remains:

| Language | Base | Python P0 |
| --- | ---: | ---: |
| TypeScript | `0/3` | `3/3` |
| Rust | `0/3` | `3/3` |
| **Overall** | **`0/6`** | **`6/6`** |

All six base responses were semantically acceptable but enclosed in one correctly tagged whole-response Markdown fence. Python P0 produced corresponding unfenced code. V2 therefore confirms that the v1 `0/6 → 6/6` observation was a formatting/instruction-adherence difference, not evidence of a six-case TypeScript/Rust semantic gain.

The accepted report, including all raw generations and token IDs, is preserved at `docs/evidence/P8_003_V2_VERIFIED_REPORT_33570451764.json`. That repository copy is JSON-normalized; the exact byte identity of the canonical artifact report is the report SHA-256 recorded above.

## Interpretation

P8-003 is complete. Under the frozen v2 decision contract, the Python P0 adapter did **not** cause catastrophic loss of the basic TypeScript/Rust semantic shapes covered by this six-case diagnostic.

That result narrows the failure mode; it does not rehabilitate P0. P8-001 remains the decisive Python-coding result: HumanEval, MBPP, the repository holdout, and the combined Python coding score all materially regressed after P0 training. The P8-003 result should therefore be carried into P8-004 as evidence that the observed damage is not a blanket collapse of these basic non-Python coding behaviors.
