# P7-006 — Full Python P0 Training

P7-006 runs the first complete Python language-adapter training job against the
frozen P0 corpus. It uses the exact P7-003 configuration at
`configs/train/python/p0.yaml` and the mandatory P7-004 preflight.

The bounded P7-005 smoke run proved the production generic trainer end to end on
the reference RTX 4070 Ti SUPER before this full run was allowed to launch. Its
accepted run used the same QLoRA configuration and completed one optimizer step
with finite loss and substantial VRAM headroom.

## Canonical execution

The live GPU workflow is `.github/workflows/python-p0-full-training.yml`. It can
be dispatched manually and also runs once when that workflow is first added to
`master`. Subsequent ordinary code changes therefore do not silently trigger an
expensive full retrain.

The workflow uses the self-hosted Linux x64 GPU runner, installs the frozen
`qlora` dependency set, verifies CUDA/BF16/bitsandbytes, materializes the frozen
P0 corpus, runs the full preflight, and executes:

```bash
uv run --frozen python -m tiny_qwen_coder.training.full_training \
  --config configs/train/python/p0.yaml \
  --repo-root .
```

`tiny_qwen_coder.training.full_training` is deliberately a thin evidence wrapper
around the generic P7-002 `run_adapter_training` implementation. It does not
replace the training mechanics that P7-005 already proved on the GPU.

The canonical output remains `artifacts/train/python/p0`.

## Frozen workload

The accepted P0 corpus contains:

- 38,000 training records;
- 2,000 validation records;
- one epoch;
- micro-batch size 1;
- gradient accumulation 8; and
- 4,750 expected optimizer steps.

P7-005 measured 0.972 training samples/second for its bounded eight-example
step. A straight-line estimate puts the full training portion near 10.9 hours,
plus validation and setup overhead. The canonical workflow therefore retains a
24-hour timeout rather than relying on an unbounded job.

## Full-run evidence contract

A successful full run emits `training-report.json` in addition to the resolved
configuration, frozen dataset-manifest copy, run manifest, adapter manifest,
training metrics, checkpoints, and adapter directory. The report records:

- exact source training config path and resolved canonical config SHA-256;
- exact base/tokenizer identity and frozen dataset-manifest identity;
- frozen train and validation record counts;
- micro-batch, gradient accumulation, effective batch, sequence length, epochs,
  and completed optimizer steps;
- final training loss, final validation loss, and every logged training loss;
- trainer-reported training runtime, samples/second, and optimizer steps/second;
- end-to-end wrapper runtime;
- peak CUDA allocated and reserved bytes;
- final checkpoint and LoRA adapter paths; and
- SHA-256 inventory/fingerprint for persisted adapter, preflight, config,
  dataset, run-manifest, adapter-manifest, and metric evidence.

The wrapper fails closed if required metrics are missing or non-finite,
throughput is missing/non-positive, CUDA peak memory is invalid, the final
checkpoint is missing, the generic trainer and persisted evidence disagree, or
the output is not a non-empty PEFT LoRA adapter.

## No merged-model fallback

P7-006 must produce a portable LoRA adapter, not a copied or merged 4B model.
Both the wrapper and workflow require `adapter_config.json` to declare
`peft_type: LORA`, require non-empty `adapter_model.safetensors` or
`adapter_model.bin`, and reject full-model filenames such as
`model.safetensors`, sharded `model-*.safetensors`, or `pytorch_model*.bin` in
both the final adapter directory and final training checkpoint.

No compatibility probing or quiet fallback is used. The frozen TRL/Transformers
stack is expected to satisfy the reviewed trainer contract exactly; incompatible
API or artifact behavior fails the job.

## Acceptance

P7-006 is complete only after the canonical self-hosted GPU workflow:

1. trains all 38,000 frozen P0 training records for one epoch;
2. completes exactly 4,750 optimizer steps without OOM or non-finite loss;
3. evaluates all 2,000 frozen validation records;
4. saves the final LoRA checkpoint and adapter without merged/full-model weights;
5. passes the independent workflow verifier; and
6. uploads the canonical evidence artifact.

The resulting adapter is a training candidate only. P7-007 must still prove
load/enable/disable inference behavior, and Phase 8 determines whether the
adapter is promotable based on benchmark and regression results.
