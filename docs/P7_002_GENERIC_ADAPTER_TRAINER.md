# P7-002 — Generic Adapter Trainer

P7-002 implements the shared supervised-fine-tuning runner used by every language adapter. The trainer is deliberately language-neutral: programming-language identity is resolved through the language registry, while the base checkpoint, prepared dataset, LoRA/QLoRA settings, loss policy, and output locations are supplied by machine-readable configuration.

The CLI entry point is:

```bash
uv run --frozen tiny-qwen-coder-train-adapter --config <training-config.yaml>
```

P7-003 owns the first concrete Python P0 training config. P7-002 does not embed Python source names, validators, prompts, benchmarks, or hyperparameters.

## Inputs

A training config supplies the canonical base-config path, canonical language ID, adapter identity, frozen dataset manifest, normalized train/validation JSONL paths, output directory, deterministic seed, LoRA or QLoRA mode, sequence/batch/optimizer schedule, assistant-only or completion-only loss, LoRA hyperparameters/target modules, and explicit QLoRA quantization settings.

Before model loading, `resolve_adapter_training_plan()` resolves the language through `LanguageRegistry`, reads the exact model/tokenizer revisions from the canonical base config, fingerprints the canonicalized training config, fingerprints the frozen dataset manifest, and verifies that the dataset language and tokenizer identity match the selected language/base.

`train_records` and `validation_records` are explicit config fields rather than filename conventions. That keeps the runner reusable for future language pipelines whose prepared-corpus layouts differ.

## Normalized records and loss modes

`tiny_qwen_coder.data.loading` strictly deserializes the language-neutral `NormalizedTrainingRecord` JSONL schema. Source-specific loading and language-specific validation remain data-pipeline responsibilities.

Assistant-only loss passes records as conversational `messages` and enables TRL assistant-only masking. Completion-only loss splits the final assistant turn into conversational `completion` and all preceding messages into `prompt`; missing prompts or final assistant turns fail closed.

No source-language branch exists in the plan, artifact, or runtime modules.

## LoRA and QLoRA

LoRA configuration comes directly from the training config. Selective targeting uses the configured exact target leaves; `all_linear` maps to PEFT's literal `all-linear` strategy.

BF16 LoRA loads the canonical shared base in BF16. QLoRA constructs an explicit 4-bit `BitsAndBytesConfig` and runs PEFT's `prepare_model_for_kbit_training()` before TRL attaches the adapter. This preserves the P2-008 preflight path rather than introducing a second QLoRA preparation sequence.

Actual training requires a CUDA-visible GPU. CPU-only plan/config/record/manifest behavior remains unit-testable.

## Output contract

Each output directory uses stable paths:

```text
<output>/
  checkpoints/
  adapter/
  dataset-manifest.json
  training-config.json
  training-metrics.jsonl
  run-manifest.json
  adapter-manifest.json
```

The portable completed-adapter manifest records exact base/tokenizer revisions, effective training chat-template identity, run/config/dataset provenance, LoRA hyperparameters, measured full LoRA module paths, measured trainable parameter count, precision/sequence/optimizer/scheduler settings, completed steps/epochs, peak reserved VRAM, and validation loss when available.

## Acceptance evidence

CPU regression tests use a synthetic Rust language plugin and Rust dataset manifest to resolve a complete training plan and construct a completed adapter manifest. That proves another language can use the trainer without adding Python-specific trainer logic. Additional tests cover strict normalized-record loading, assistant/completion conversion, tokenizer-revision drift, and adapter-ID/language consistency.
