# P8-001 — Evaluate Python P0 adapter

P8-001 measures the accepted Python P0 LoRA adapter against the frozen P6-005 unchanged-base baseline. It does not make the P8-005 promotion decision and it does not run the P8-002 general/tool regression suite.

## Canonical identities

The evaluation is pinned to:

- base model: `Qwen/Qwen3.5-4B`
- base/tokenizer revision: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- adapter ID: `language/python/p0`
- P7-006 training source: `02df92a9c2d347b9fb013dc25714fe066c6bcafe`
- P7-006 training run ID: `training-python-20260831T180916446466Z-02df92a9-eafc119d`
- adapter weight SHA-256: `c94606250e112f72362eb883a55f7b2c8af854d445f6bb6194352c2806a8f276`
- adapter weight size: `65,004,840` bytes
- accepted P6-005 baseline source: `da537443ab80b1380bee0fc3c7d9d01ca0574f35`
- accepted P6-005 artifact-set SHA-256: `4bf616c3e84bdd74f8cf6467fc2d9d760f04d3b1b44660a81e365ff6f99a72fc`

The P7-006 artifact is revalidated with the P7-007 fail-closed adapter validator before the expensive base model is loaded. The P6-005 baseline artifact is independently rehashed and validated before its metrics can participate in the comparison.

## Protected benchmark contract

P8-001 reuses the P6 frozen generation and execution contract:

| Suite | Problems |
| --- | ---: |
| HumanEval | 164 |
| MBPP | 500 |
| repository holdout (custom Python suite) | 11 |
| **Total** | **675** |

Generation remains deterministic and identical to the baseline configuration: seed `1729`, greedy decoding, temperature `0`, top-p `1`, top-k `0`, maximum `512` new tokens, the canonical evaluation prompt version, and thinking disabled by the pinned Qwen chat-template path.

The adapter-specific generation-contract fingerprint includes the P6 base generation contract plus the exact adapter ID, family, weight hash, byte size, training run ID, and training Git SHA. Consequently, an unchanged-base P6 checkpoint or a checkpoint from a different adapter cannot be silently reused as P8-001 evidence.

## Two-stage execution boundary

The workflow deliberately separates model generation from untrusted-code execution:

1. **GPU generation** runs on the self-hosted CUDA runner. It downloads and validates the canonical P7-006 training artifact, loads the canonical BF16 base once, attaches the unmerged inference-only LoRA adapter, and generates exactly 675 responses. It does not execute generated code.
2. **Hosted scoring** downloads only the compact generation evidence plus the accepted P6 baseline. Generated code is scored through the existing constrained OCI execution harness using the same digest-pinned Python execution image as P6-005. The scoring stage has no P7 training artifact and cannot regenerate missing responses.

A generation-stage manifest hashes all transported checkpoints and runtime metadata. Hosted scoring recomputes the adapter-specific generation contract and refuses missing, reordered, altered, stale, or identity-mismatched checkpoints.

## Acceptance and comparison evidence

An accepted P8-001 evaluation requires all three suites to have:

- the exact protected problem count;
- `harness_errors == 0`;
- internally consistent passed/failed counts; and
- a numeric `pass_at_1` in `[0, 1]`.

The comparison artifact reports, for each suite, base and adapter passed/failed counts, base and adapter pass@1, and absolute deltas. It also reports the micro-averaged result over all 675 coding problems. P8-001 intentionally imposes no minimum improvement threshold; later P8 tasks combine these measurements with regression evidence to decide promotion, rejection, or iteration.

The accepted P6-005 reference values are rechecked before comparison:

- HumanEval: `128 / 164` (`0.7804878048780488`)
- MBPP: `290 / 500` (`0.58`)
- repository holdout: `6 / 11` (`0.5454545454545454`)
- combined: `424 / 675`

## Persisted artifacts

The final compact evidence directory is `artifacts/eval/python/p0-v1/` and contains:

- `generation-stage.json`
- `runtime-metadata.json`
- three adapter-generation JSONL checkpoints under `.baseline-work/`
- HumanEval results and aggregate
- MBPP results and aggregate
- repository-holdout results and aggregate
- `comparison.json`
- `evaluation-manifest.json`

`evaluation-manifest.json` hashes the complete final evidence inventory. The verifier rehashes the inventory, revalidates the generation-stage identity, revalidates P6-005, rereads the three persisted adapter aggregates, and recomputes `comparison.json`. Any mismatch fails closed.

## Workflow trigger lifecycle

While P8-001 is being introduced, `.github/workflows/python-p0-evaluation.yml` has a one-shot `master` push trigger limited to that workflow path, plus manual dispatch. This allows the merge that introduces the workflow to launch the canonical GPU evaluation without making unrelated commits expensive. After canonical acceptance evidence is captured, the push trigger should be removed so future P8-001 reruns are explicit/manual only.

P8-001 must not be marked complete merely because the implementation or CPU CI passes. Completion requires a successful canonical GPU generation run, constrained scoring, independent evidence verification, and a persisted direct base-vs-adapter comparison.
