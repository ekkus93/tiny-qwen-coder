# P9-007 local GPU runbook

This runbook executes the repaired V4-2000 student experiment on the RTX 4070 Ti SUPER without
placing generated teacher data in Git.

## Frozen inputs

Use only the qualified Drive export for:

```text
MyDrive/tiny-qwen-coder/distillation/qwen38-27b-v4-2000-salvage-v1/
```

The canonical local training configuration is:

```text
configs/train/python/p9_distilled_v4_2000_r8_lr1e5.yaml
```

The trajectory protocol is:

```text
configs/train/python/p9_distilled_v4_2000_trajectory_v1.yaml
```

It freezes one common 185-optimizer-step schedule with adapter-only snapshots at steps 25, 50,
100, and 185. Do not train four separate horizons: doing so would change the cosine schedule seen
by each candidate and invalidate checkpoint comparison.

## 1. Install the QLoRA environment

From a clean repository checkout:

```bash
uv sync --frozen --extra qlora
```

Confirm CUDA and BF16:

```bash
uv run --frozen python - <<'PY'
import bitsandbytes
import torch

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("CUDA device 0 does not support BF16")
props = torch.cuda.get_device_properties(0)
print(f"torch={torch.__version__}")
print(f"bitsandbytes={bitsandbytes.__version__}")
print(f"cuda_runtime={torch.version.cuda}")
print(f"gpu={props.name}")
print(f"total_memory_bytes={props.total_memory}")
PY
```

## 2. Import the exact repaired corpus

Copy/download the completed salvage ZIP onto the GPU machine, then run:

```bash
uv run --frozen python scripts/import_v4_2000_salvage.py \
  --archive /absolute/path/to/qwen38-27b-v4-2000-salvage-v1-20260908T103445Z-1-001.zip
```

The importer refuses to overwrite an existing destination and writes the verified corpus to:

```text
data/python/qwen38-27b-v4-2000-salvage-v1/
```

Do not substitute the old contaminated `qwen38-27b-v4-2000/final/` data.

## 3. Validate the experiment contract

```bash
uv run --frozen python -m tiny_qwen_coder.training.distilled_trajectory --repo-root .
```

The validation must report:

```text
train_records           1479
effective_batch_size    8
derived_one_pass_steps  185
trajectory_max_steps    185
checkpoint_steps        [25, 50, 100, 185]
```

It also revalidates the frozen P9-004 HumanEval/MBPP development membership and keeps the repository
holdout qualification-only.

## 4. Run mandatory preflight

```bash
uv run --frozen tiny-qwen-coder-train-preflight \
  --config configs/train/python/p9_distilled_v4_2000_r8_lr1e5.yaml \
  --repo-root . \
  --output /tmp/p9-007c-preflight.json
```

Do not continue after any preflight failure.

## 5. Run the one-step GPU smoke

```bash
uv run --frozen tiny-qwen-coder-train-smoke \
  --config configs/train/python/p9_distilled_v4_2000_r8_lr1e5_smoke.yaml \
  --repo-root .
```

The smoke is non-promotable. It exists only to prove that the real QLoRA path can load the frozen
corpus, allocate safely, complete one optimizer step, evaluate, and save adapter-only evidence on
the 4070 Ti.

Do not launch the trajectory unless the smoke succeeds.

## 6. Train one checkpointed trajectory

The four development candidates must come from one common optimizer/LR schedule:

```bash
uv run --frozen python -m tiny_qwen_coder.training.distilled_trajectory_training \
  --repo-root .
```

Expected output root:

```text
artifacts/train/python/p9-distilled-v4-2000-r8-lr1e5/
```

Required adapter-only snapshots:

```text
snapshots/step-0025/
snapshots/step-0050/
snapshots/step-0100/
snapshots/step-0185/
```

The training report itself is not promotable. Development evaluation decides whether any snapshot
may consume the one-shot qualification set.

## 7. Generate development responses

```bash
uv run --frozen python -m tiny_qwen_coder.evaluation.python_distilled_trajectory \
  generate-trajectory \
  --training-output artifacts/train/python/p9-distilled-v4-2000-r8-lr1e5 \
  --repo-root . \
  --device-index 0
```

Generation is restricted to the frozen 45 HumanEval + 130 MBPP development tasks. It records zero
repository-holdout requests.

## 8. Score all four snapshots

```bash
for step in 25 50 100 185; do
  uv run --frozen python -m tiny_qwen_coder.evaluation.python_distilled_trajectory \
    score \
    --training-output artifacts/train/python/p9-distilled-v4-2000-r8-lr1e5 \
    --step "${step}" \
    --repo-root .
done
```

A checkpoint is eligible only when all three gates pass:

```text
combined >= 104/175
HumanEval >= 33/45
MBPP >= 70/130
```

Training or validation loss cannot authorize qualification.

## 9. Apply the mechanical development selector

```bash
uv run --frozen python -m tiny_qwen_coder.evaluation.python_distilled_trajectory \
  select \
  --scores-root artifacts/eval/python/p9-distilled-v4-2000-development-v1 \
  --repo-root .
```

The selector chooses the highest combined development score and uses fewer optimizer steps as the
only tie breaker. If no checkpoint is eligible, it sets `qualification_authorized` to `false` and
the experiment stops before the qualification set.

If a checkpoint is selected, freeze that exact adapter snapshot and its development-selection
evidence before implementing or running the winner-only one-shot qualification step.
