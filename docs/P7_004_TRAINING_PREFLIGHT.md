# P7-004 — Adapter Training Preflight

P7-004 adds a mandatory fail-fast validation stage immediately before adapter training. The preflight is intentionally lighter than P7-005: it may load the pinned tokenizer to prove loss masking, but it does not load the 4B model or run forward/backward work.

The standalone command is:

```bash
uv run --frozen tiny-qwen-coder-train-preflight \
  --config configs/train/python/p0.yaml
```

The normal `tiny-qwen-coder-train-adapter` path invokes the same preflight automatically before it creates the configured output directory or loads model weights.

## Frozen base and LoRA architecture

The resolved base repository/revision must exactly match the immutable P7-001 selective LoRA profile. Canonical training currently requires the frozen selective target strategy, exact 12 target leaves, and measured rank 16. The report records the profile ID/SHA-256 and the P2-008 measured trainable-parameter count so the architecture proof is auditable without loading model weights.

Because P7-001 target discovery was performed against the exact pinned base revision, any base-revision or target-list drift invalidates the architecture proof and fails closed.

## Frozen dataset integrity

Preflight verifies all of the following before training:

- the configured dataset manifest bytes are unchanged from plan resolution;
- `dataset-manifest.sha256` exactly authenticates those bytes;
- the manifest seed matches the training seed;
- train/validation JSONL files parse through the strict normalized-record loader;
- split counts match the frozen manifest;
- ordered normalized-content checksums match the frozen manifest; and
- no exact normalized record occurs in both train and validation splits.

This catches stale, replaced, truncated, or cross-contaminated record files even when the manifest path itself is unchanged.

## Loss-mask proof

For `assistant_only` training, the pinned tokenizer's current checkpoint chat-template SHA-256 must match the dataset manifest and P2-006 must reproduce a true TRL assistant-only generation mask. Falling back to completion-only masking is not accepted for an `assistant_only` configuration.

The generic completion-only mode uses the existing P2-006 completion-boundary proof.

## Output safety

The configured run directory must be a new child beneath `<repo>/artifacts/train/`. Preflight rejects the root itself, destinations outside the allowed tree, an already-existing destination, and any existing symlink component in the path. Validation creates no output directory.

## Hardware and training-mode compatibility

The production hardware probe records CUDA visibility, GPU count/name/VRAM, BF16 support, and bitsandbytes availability. BF16 configurations require BF16-capable CUDA hardware. QLoRA additionally requires bitsandbytes. These checks validate capability only; P7-005 remains responsible for the bounded real-GPU forward/backward smoke run and measured memory behavior.

## Report artifact

A successful preflight serializes deterministic JSON. The training runtime writes the accepted report to:

```text
<output>/training-preflight.json
```

before loading model weights. The report contains the training-config hash and evidence for dataset integrity, base/LoRA architecture, loss masking, output safety, and hardware compatibility.
