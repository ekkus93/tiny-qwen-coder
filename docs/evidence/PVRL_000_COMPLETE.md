# PVRL-000 — Program Boundary Freeze

Status: COMPLETE

Date: 2026-09-09

Authoritative machine-readable evidence:

`docs/evidence/PVRL_000_PROGRAM_BOUNDARY.json`

## Frozen boundary

PVRL starts from master SHA:

`6fb599f3528387d77f8293a2a8a2260326aa20ae`

The initial policy is pinned `Qwen/Qwen3.5-4B` revision:

`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`

The larger environment-author model is pinned `Qwen/Qwen3.8-27B` revision:

`72a217afab8029b39e4af1c7273a829995a3dbaf`

The 27B model is explicitly an environment author. Its generated answers, references, or tests are candidates that require executable validation; model size does not authorize a student target.

The frozen historical unchanged-base reference remains HumanEval `128/164`, MBPP `290/500`, repository holdout `6/11`, combined `424/675`.

## FTR dependency

At PVRL track creation, FTR-101 is complete and FTR-102 is still open/in progress. The exact FTR-102 head at this record is `0bc81414bcefaec4c69e5daadd399b7a1ea0fc70`; CPU CI run `34379857723` has passed while GPU/control run `34379852161` remains in progress.

Therefore FTR-G1 is **not yet satisfied** and PVRL model training is **not authorized** by this boundary freeze.

Infrastructure/specification work may continue while FTR completes.

## Protected-data rule

HumanEval, MBPP, repository holdout, and future qualification material are prohibited from PVRL training, calibration, environment-author examples, task-generation derivatives, and reward shaping.

## Scientific scope

PVRL v0 tests whether a small curriculum of independently verifiable, frontier-selected Python tasks can improve the exact Qwen3.5-4B base through synchronous group-relative executable-reward RL.

The initial scope is self-contained executable Python, short context, binary behavioral reward, and single-16-GiB-GPU QLoRA. Repository-scale SWE, long context, asynchronous rollout systems, and unrestricted networking are explicitly deferred.

## Closure

PVRL-000 is complete because the model identities, baseline reference, FTR dependency, protected-data rule, hypotheses, v0 scope, non-goals, and stop rules are durably frozen.

PVRL-101 is the next implementation task, but model-training work remains blocked by FTR-G1.
