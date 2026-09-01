# P7-007 — Validate adapter load/inference

P7-007 proves that the canonical Python P0 LoRA produced by P7-006 can be
attached to the exact pinned `Qwen/Qwen3.5-4B` base, used for deterministic
inference, disabled to recover unchanged-base behavior, and re-enabled without
reloading or rebuilding the full base model.

## Canonical source adapter

This validation is intentionally bound to the accepted P7-006 training run:

- workflow run: `33422910444`
- training source Git SHA: `02df92a9c2d347b9fb013dc25714fe066c6bcafe`
- artifact: `python-p0-full-training-02df92a9c2d347b9fb013dc25714fe066c6bcafe`
- adapter: `language/python/p0`
- adapter file: `adapter/adapter_model.safetensors`
- adapter size: `65,004,840` bytes
- adapter SHA-256: `c94606250e112f72362eb883a55f7b2c8af854d445f6bb6194352c2806a8f276`

The validator does not accept an arbitrary local `artifacts/train/python/p0`
directory as canonical evidence. The workflow downloads the exact P7-006
artifact and fails unless it finds exactly one expected training-output root.

## Fail-closed artifact preflight

Before the 4B base model is loaded, the validator requires and cross-checks:

- the strict adapter manifest;
- the P7-006 training report, resolved training config, and run manifest;
- exact base model and tokenizer repository/revision identity;
- canonical `language/python/p0` adapter identity;
- 4,750 completed optimizer steps and the canonical P0 training-config hash;
- PEFT LoRA type, rank, alpha, dropout, bias, target modules, and inference mode;
- the recorded SHA-256 and byte size of `adapter_model.safetensors`;
- the recorded inference chat-template hash; and
- absence of merged/full-model weights in the adapter directory.

No base-revision fallback, adapter substitution, missing-file fallback, or
best-effort continuation is allowed.

## GPU inference contract

The CUDA validator then:

1. loads the exact pinned base checkpoint once in BF16;
2. verifies the resolved upstream model revision;
3. generates two fixed, non-benchmark Python smoke prompts greedily with the
   unchanged base;
4. attaches the P0 LoRA with `is_trainable=False`;
5. verifies the PEFT adapter is active, unmerged, and has zero trainable
   parameters;
6. generates the same fixed prompts with the adapter enabled;
7. enters PEFT's `disable_adapter()` context on the same loaded model and
   requires exact base token IDs and decoded text;
8. exits the context, verifies the adapter is enabled again, and requires exact
   reproduction of the earlier adapted token IDs and decoded text; and
9. records GPU memory, load times, identities, status snapshots, and all four
   deterministic generations in the acceptance report.

The adapter is *not* required to change the output of every smoke prompt. That
would be an invalid acceptance criterion because a correct specialized model may
produce the same answer as the base model for an easy deterministic prompt.
Adapter attachment is instead proven from PEFT state plus successful
enable/disable/re-enable behavior.

## Local invocation

After extracting the canonical P7-006 artifact, run on a BF16-capable CUDA host:

```bash
uv sync --frozen
uv run --frozen python scripts/validate_adapter_inference.py \
  --training-output /path/to/extracted/artifacts/train/python/p0 \
  --output artifacts/eval/python/p7-007/adapter-inference-validation.json
```

The command emits the JSON report only after all acceptance conditions pass.

## Acceptance evidence

GPU acceptance is pending on branch `ralph/p7-007-adapter-inference-validation`.
The workflow remains manual-only while CPU quality gates are established. A temporary
branch-only one-shot trigger will be armed only after those gates pass, then removed
after GPU acceptance so the final merged workflow remains manual-only.
