# P8-005 — Python P0 promotion decision

## Decision

`language/python/p0` is **rejected** as the recommended Python adapter.

The adapter remains a valid, preserved experiment artifact and becomes the Phase 9 negative control. Rejection is a quality/promotion decision, not an assertion that the P7 training artifact is corrupt: P7-006 completed successfully, P7-007 validated load/disable/re-enable behavior, and the exact adapter remains reproducible from its frozen training identity.

No Python adapter is marked recommended by P8-005. The machine-readable decision therefore records:

```text
recommended_adapter_id: null
```

P8-005 does not introduce the runtime adapter registry planned for Phase 10. It establishes the evidentiary policy that a later registry must use when it assigns the word `recommended`.

---

## Frozen promotion policy

The machine-readable source of truth is:

`configs/eval/python/promotion_v1.yaml`

Policy identity:

- policy ID: `python-promotion-v1`
- policy SHA-256: `2c884ca66b3b09071971e89777c9877eddd730c9d4cc59e7475b1cbce963b22e`
- language: `python`
- unchanged-base source Git SHA: `da537443ab80b1380bee0fc3c7d9d01ca0574f35`
- unchanged-base artifact-set SHA-256: `4bf616c3e84bdd74f8cf6467fc2d9d760f04d3b1b44660a81e365ff6f99a72fc`

`tiny_qwen_coder.evaluation.promotion.load_frozen_python_promotion_policy()` verifies the exact file SHA-256 before parsing and rejects unknown/missing fields. There is no fallback to a looser policy when the file is malformed or changed.

### Important timing limitation

This v1 policy was established in P8-005 after P0 measurements existed; it was not preregistered before P0 training. It must therefore not be described as a pre-P0 decision rule.

To limit hindsight bias, the quantitative target-language thresholds are anchored to the accepted unchanged-base P6-005 scores rather than to values chosen to narrowly exclude P0. The policy is frozen prospectively for subsequent Python adapter experiments unless a separately versioned policy is reviewed before those candidate results are used for promotion.

P0's rejection is not sensitive to the 2-point improvement margin: P0 is already far below the unchanged base on every protected Python suite.

---

## Quantitative meaning of `recommended`

A Python adapter may be marked recommended under `python-promotion-v1` only if **every** gate below passes.

### 1. Target-language improvement

The accepted unchanged-base combined Python result is:

`424/675 = 0.6281481481481481`

A candidate must improve the combined protected Python pass rate by at least `0.02` absolute. Because the suite contains 675 fixed problems, the smallest integer pass count satisfying that requirement is:

`438/675 = 0.6488888888888888`

That is 14 more passing problems than the unchanged base and about 2.074 percentage points of realizable improvement on the discrete suite.

### 2. No protected Python suite may fall below base

A candidate must also preserve at least the unchanged-base count on every component suite:

| Suite | Minimum candidate result |
| --- | ---: |
| HumanEval | `128/164` |
| MBPP | `290/500` |
| repository holdout | `6/11` |

This prevents a large gain on one suite from concealing a target-language regression on another.

### 3. General/tool preservation

Using the frozen P8-002 suite, a candidate must:

- pass at least `2/12`, matching the accepted base result; and
- introduce **zero** base-pass → adapter-fail regressions.

This is a preservation gate, not a claim that `2/12` represents strong general/tool capability. The P8-002 report already documents the suite's weak baseline headroom and exact-format sensitivity.

### 4. Cross-language preservation

Using the P8-003 v2 semantic contract, a candidate must:

- conclude `no_catastrophic_regression`;
- pass at least `2/3` TypeScript semantic-shape cases; and
- pass at least `2/3` Rust semantic-shape cases.

This remains a catastrophic-collapse guard, not a TypeScript/Rust quality benchmark.

### 5. Dataset contamination evidence

The training dataset must have contamination status `clean` under the project's intended protected-benchmark contamination checks. `not_run`, missing, malformed, or contaminated evidence is not promotion-eligible.

### 6. Adapter load/integrity validation

The exact candidate must pass the adapter load/inference validation contract. Missing validation or an identity/provenance mismatch is not promotion-eligible.

All gates are conjunctive. There is no weighted score, exception path, or quiet override that can compensate for a failed gate.

---

## P0 evidence evaluated against the policy

The permanent machine-readable decision is:

`docs/evidence/P8_005_P0_PROMOTION_DECISION.json`

Decision-record SHA-256:

`1169a3b07f22e672bfc3c9f5624222ccb5c49af05b143a5d5f25d62ed690791e`

The record embeds exact adapter/training provenance plus the canonical P6/P7/P8 evidence references and the deterministic output of `evaluate_python_adapter_promotion()`.

| Gate | Requirement | P0 | Result |
| --- | --- | --- | --- |
| combined Python gain | `>=438/675` | `187/675` | **FAIL** |
| HumanEval preservation | `>=128/164` | `88/164` | **FAIL** |
| MBPP preservation | `>=290/500` | `97/500` | **FAIL** |
| holdout preservation | `>=6/11` | `2/11` | **FAIL** |
| general/tool preservation | `>=2/12`, zero regressions | `2/12`, zero regressions | PASS |
| cross-language preservation | no catastrophic regression, TS/Rust `>=2/3` | TS `3/3`, Rust `3/3` | PASS |
| contamination evidence | `clean` | `not_run` | **FAIL** |
| adapter load validation | validated | validated | PASS |

P0 therefore fails five independent promotion gates:

1. `python_combined_gain`;
2. `humaneval_preservation`;
3. `mbpp_preservation`;
4. `repository_holdout_preservation`; and
5. `contamination_evidence`.

The deterministic disposition is:

```text
disposition: reject
recommended_adapter_id: null
```

---

## Why P0 is rejected

The decisive evidence is the protected Python regression, not the contamination omission by itself.

P8-001 measured:

| Suite | Base | P0 | Delta passes |
| --- | ---: | ---: | ---: |
| HumanEval | `128/164` | `88/164` | `-40` |
| MBPP | `290/500` | `97/500` | `-193` |
| repository holdout | `6/11` | `2/11` | `-4` |
| **Combined** | **`424/675`** | **`187/675`** | **`-237`** |

The combined pass rate fell from `0.6281481481481481` to `0.277037037037037`, an absolute loss of `0.3511111111111111` and a relative reduction of about 55.90% from base.

P8-002 and P8-003 narrow the failure mode: they do not show additional pass/fail collapse on the frozen general/tool suite or catastrophic loss of the six TypeScript/Rust semantic-shape cases. Those results are useful diagnostics, but they cannot override a large regression in the target capability the adapter was intended to improve.

The dataset contamination status `not_run` is an additional eligibility failure and must remain visible. It is not used as a convenient substitute for the direct Python evidence.

---

## Rejected-experiment preservation

P0 must not be deleted or rewritten merely because it failed promotion.

The preserved rejected identity is:

- adapter ID: `language/python/p0`
- adapter SHA-256: `c94606250e112f72362eb883a55f7b2c8af854d445f6bb6194352c2806a8f276`
- adapter size: `65,004,840` bytes
- training source Git SHA: `02df92a9c2d347b9fb013dc25714fe066c6bcafe`
- training run ID: `training-python-20260831T180916446466Z-02df92a9-eafc119d`
- P7-006 workflow run: `33422910444`
- P7-006 artifact: `9789946698`

Canonical evidence retained by reference in the decision record includes:

- P6-005 unchanged-base run `33301242379` / artifact `9729636096`;
- P7-007 adapter-load validation run `33509937071`;
- P8-001 Python evaluation run `33538724658` / artifact `9814936298`;
- P8-002 general/tool run `33554096916` / artifact `9818707785`;
- P8-003 v2 run `33570451764` / artifact `9824776245`; and
- `docs/P8_004_P0_EXPERIMENT_REPORT.md`.

The decision record explicitly marks the experiment `rejected_experiment`, assigns it the Phase 9 role `negative_control`, and records that neither the adapter nor the observed results are to be deleted/re-written.

---

## Phase 9 handoff

Future Python experiments should be evaluated against the same `python-promotion-v1` policy while it remains current. P0 provides the controlled negative reference for rank, target-module, dataset-size, learning-rate/training-length, and BF16-vs-QLoRA studies.

A later candidate earns the `recommended` label only by satisfying every frozen promotion gate. Training loss, anecdotal examples, or a single benchmark win are insufficient substitutes.

P8-005 is complete when the policy, evaluator tests, permanent rejected-decision record, documentation, and TODO status are merged with passing CPU quality gates.
