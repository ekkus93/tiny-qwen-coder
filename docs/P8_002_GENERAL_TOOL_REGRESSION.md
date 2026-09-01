# P8-002 — General/tool regression for Python P0

P8-002 quantifies whether the accepted P7-006 `language/python/p0` adapter damages non-coding behavior. It uses the general/tool suite frozen before P0 training and compares the adapter directly with the accepted P6-005 unchanged-base evidence.

## Frozen contract

The suite is `general_tool_regression` version 1 with semantic SHA-256 `9de462c05a05455b2cc5af8c0246d897fe7991510d470e837d205540922239f9`. It contains exactly 12 cases, two in each category:

- instruction following;
- JSON structured output;
- simple reasoning;
- shell reasoning;
- Git reasoning; and
- tool-call formatting/selection.

P8-002 does not alter prompts or expectations after observing P0 results. Generation uses the same pinned Qwen3.5-4B base, Python `python-v1` system prompt, tokenizer/chat template, frozen decoding settings, and exact P7-006 adapter identity used by P8-001.

## Evidence boundaries

The permanent workflow is manual-only. GPU generation downloads and revalidates the canonical P7-006 artifact before loading the model, generates exactly 12 responses, and persists an adapter-specific generation checkpoint plus a stage manifest that rehashes it.

Hosted scoring downloads only that compact generation evidence and the accepted P6-005 baseline artifact. The full P6 artifact is revalidated before any baseline result participates in the comparison. Missing P8 checkpoints cause a hard failure; hosted scoring cannot silently regenerate responses.

## Comparison

The final comparison records:

- overall base and adapter pass counts;
- per-category base/adapter pass counts and deltas;
- every individual base and adapter response;
- base and adapter deterministic scoring details; and
- per-case transitions: `regression`, `improvement`, `preserved_pass`, or `preserved_fail`.

The independent verifier rehashes the stage/checkpoint/comparison artifacts and recomputes the comparison from the frozen suite and validated P6 evidence.

P8-002 is complete only after one canonical GPU run passes generation, scoring, independent verification, and evidence upload. No minimum score is imposed by this measurement task; P8-004/P8-005 use the quantified regression evidence in the promotion/rejection decision.
