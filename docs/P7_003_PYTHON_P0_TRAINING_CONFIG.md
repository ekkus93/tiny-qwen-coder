# P7-003 — Python P0 Training Configuration

P7-003 freezes the first concrete language-adapter training configuration at
`configs/train/python/p0.yaml`. The generic P7-002 trainer remains unchanged;
this file supplies Python P0's language identity, frozen dataset locations,
training hyperparameters, LoRA target set, and QLoRA settings.

## Canonical configuration

The P0 config is revision-bound through `configs/base/qwen35-4b.yaml` to
`Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` and uses:

| Setting | Frozen P0 value |
| --- | --- |
| training mode | `qlora_4bit` |
| 4-bit type | NF4 |
| double quantization | enabled |
| compute dtype | BF16 |
| sequence length | 2,048 |
| micro-batch | 1 |
| gradient accumulation | 8 |
| effective batch | 8 examples |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| LoRA bias | `none` |
| target strategy | selective P7-001 profile |
| learning rate | 2e-4 |
| scheduler | cosine |
| warmup ratio | 0.03 |
| epochs | 1 |
| gradient checkpointing | enabled |
| loss | assistant-only |
| seed | 1729 |

The config points at the P5 frozen-corpus layout:

- `data/python/p0/dataset-manifest.json`
- `data/python/p0/train.jsonl`
- `data/python/p0/validation.jsonl`

and writes training artifacts beneath `artifacts/train/python/p0`.

## Batch selection evidence

P2-008 is the authoritative memory experiment. On the 16 GiB RTX 4070 Ti
SUPER reference GPU, BF16 LoRA completed the forward pass but OOMed during
backward. The 4-bit NF4 fallback completed forward, backward, and an AdamW
optimizer step at sequence length 2,048 and micro-batch 1, with peak reserved
VRAM of 14,661,189,632 bytes and 2,027,028,480 bytes (1.888 GiB) of reserved
headroom. That is above the predeclared 1.554 GiB safety requirement.

Therefore P0 does not attempt a larger micro-batch before the bounded P7-005
smoke run: `micro_batch_size: 1` is the empirically measured safe value.

Gradient accumulation does not enlarge the per-forward/backward micro-batch, so
P0 freezes `gradient_accumulation_steps: 8`, giving an effective batch of eight
examples while preserving the measured micro-batch memory envelope. This is the
conservative first-run optimizer cadence used by the generic trainer contract;
P7-005 must still verify finite loss, throughput, and actual peak VRAM before the
full P7-006 training run. Any later change requires a new reviewed config rather
than silently mutating P0.

## Loss policy

P0 selects `assistant_only`. P2-006 proved that the pinned TRL stack can supply
the Qwen training chat template with generation markers and train the assistant
span, including its stop token. P7-002 maps this config value directly to TRL's
assistant-only loss mode. The completion-only path remains the generic fallback,
but it is not the canonical P0 selection.

## Selective target set

The config copies the exact frozen P7-001 target leaves:

- `down_proj`
- `gate_proj`
- `in_proj_a`
- `in_proj_b`
- `in_proj_qkv`
- `in_proj_z`
- `k_proj`
- `o_proj`
- `out_proj`
- `q_proj`
- `up_proj`
- `v_proj`

At rank 16, P2-008 measured 32,464,896 trainable parameters for this target
profile. Unit tests require the P0 config's targets and rank to match the frozen
P7-001 profile, so target drift fails before GPU training.

## Machine-readable provenance

P7-002 resolves this YAML into an `AdapterTrainingPlan`, fingerprints the
canonical parsed config, and writes the full resolved configuration to
`training-config.json` beside the run. The completed adapter manifest stores the
same config SHA-256 together with the exact base/tokenizer revisions, dataset
manifest reference, LoRA metadata, optimizer/scheduler settings, sequence
length, steps, and measured peak VRAM. The run manifest records the common run,
Git, host, dependency, base, language, adapter, and seed provenance.

This satisfies P7-003 without embedding Python-specific hyperparameters in the
generic trainer.
