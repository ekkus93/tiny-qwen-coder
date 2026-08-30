# P7-005 — Python P0 GPU Smoke Training

P7-005 proves the real QLoRA training path on the reference 16 GiB GPU before the full P0 run. The smoke run uses the exact P7-003 training configuration and P7-004 preflight gates, but adds a separate bounded execution contract at `configs/train/python/p0_smoke.yaml`.

## Bounded contract

The canonical smoke run is intentionally small and non-promotable:

| Setting | Value |
| --- | ---: |
| source training config | `configs/train/python/p0.yaml` |
| output | `artifacts/train/python/p0-smoke` |
| optimizer steps | 1 |
| train records | 8 |
| validation records | 4 |
| micro-batch | 1 |
| gradient accumulation | 8 |
| sequence-length cap | 2,048 |

One optimizer step therefore requires eight real micro-batch forward/backward passes without recycling training records. The smoke runner refuses bounds above four optimizer steps, 64 train records, or 16 validation records.

The smoke output is separate from `artifacts/train/python/p0` and the report always records `smoke_only: true` and `promotable: false`. It is acceptance evidence only and must never be treated as the P0 adapter.

## Fail-fast preflight

The smoke runner reuses the full P7-004 contract before creating its output directory or loading the 4B model:

- exact canonical base revision and frozen selective LoRA targets;
- frozen dataset manifest, split counts, checksums, and no train/validation overlap;
- exact assistant-only loss-mask proof;
- CUDA, BF16, and bitsandbytes/QLoRA compatibility; and
- the same no-existing-path/no-symlink output safety policy, applied to the smoke output directory.

## Live GPU workflow

`.github/workflows/python-p0-smoke-training.yml` runs training on the self-hosted Linux x64 runner. It installs the locked `qlora` extra, verifies CUDA/BF16/bitsandbytes, materializes the deterministic frozen P0 corpus, and then executes:

```bash
uv run --frozen tiny-qwen-coder-train-smoke \
  --config configs/train/python/p0_smoke.yaml \
  --repo-root .
```

The workflow uploads the smoke evidence even on failure so GPU/runtime problems can be diagnosed without weakening the acceptance checks.

## Acceptance evidence

A successful `smoke-training-report.json` is required to prove all of the following:

- exactly one optimizer step completed;
- training emitted at least one numeric finite loss;
- final training loss is numeric and finite;
- validation loss is numeric and finite;
- `checkpoints/checkpoint-1` exists and is non-empty;
- the separately saved adapter contains `adapter_config.json` and non-empty adapter weights;
- peak allocated CUDA memory is positive;
- peak reserved CUDA memory is positive and at least peak allocated memory; and
- checkpoint/adapter files are SHA-256 inventoried in the report.

P7-005 is complete only after the live self-hosted workflow satisfies these checks. The measured losses, VRAM peaks, runtime, artifact fingerprint, source Git SHA, and uploaded Actions artifact form the evidence for proceeding to P7-006.
