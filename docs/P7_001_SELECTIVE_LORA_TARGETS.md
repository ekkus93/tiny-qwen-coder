# P7-001 — Frozen Selective LoRA Targets

P7-001 promotes the architecture candidate discovered in P2 into the canonical
language-LoRA target contract used by Phase 7 training.

The machine-readable source of truth is
`configs/base/qwen35-4b-selective-lora-v1.yaml`. Its loader pins the exact file
SHA-256 so an unreviewed target change fails closed.

## Canonical base identity

- repository: `Qwen/Qwen3.5-4B`
- revision: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- target strategy: `selective`

The target profile is base-architecture metadata, not Python-specific training
metadata. Python, TypeScript, Rust, and future language adapters using this
canonical base start from the same architecture contract unless a later
controlled experiment explicitly freezes a different profile.

## Exact PEFT target modules

P2-002 inspected the pinned checkpoint and classified the language-model linear
projections. P7-001 freezes the following leaf names:

### Full attention

- `k_proj`
- `o_proj`
- `q_proj`
- `v_proj`

### MLP

- `down_proj`
- `gate_proj`
- `up_proj`

### Gated DeltaNet / linear attention

- `in_proj_a`
- `in_proj_b`
- `in_proj_qkv`
- `in_proj_z`
- `out_proj`

The resulting canonical `target_modules` tuple is:

```text
down_proj
gate_proj
in_proj_a
in_proj_b
in_proj_qkv
in_proj_z
k_proj
o_proj
out_proj
q_proj
up_proj
v_proj
```

The language output head (`lm_head`), vision encoder, and multimodal projector
remain excluded from the language-LoRA target set.

## Measured trainable parameter count

P2-008 attached rank-16 LoRA adapters to this exact selective set while running
the canonical QLoRA memory preflight. The measured trainable parameter count was:

```text
32,464,896
```

P7-001 records both the measurement rank (`16`) and parameter count
(`32,464,896`). Validation fails if a rank-16 attachment reports a different
count.

This count is evidence for the frozen rank-16 architecture. Future rank sweeps
are expected to produce different trainable counts and must record their own
measurements rather than changing this P7-001 evidence.

## Architecture verification

`require_profile_matches_discovery()` compares a P2
`PeftTargetDiscoveryReport` with the frozen profile and rejects:

- base repository drift;
- base revision drift;
- any unclassified text-backbone linear module;
- a changed selective module set;
- category-level target drift.

`load_frozen_selective_lora_target_profile()` additionally verifies the exact
profile-file SHA-256 before parsing it.

Together, these checks make the P7-001 target set a revision-bound,
architecture-verified input for the generic trainer implemented by P7-002.
