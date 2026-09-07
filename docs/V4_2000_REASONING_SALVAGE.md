# V4-2000 reasoning-leak salvage

The first completed `qwen38-27b-v4-2000` run is **frozen evidence, not valid student training data**.

The Qwen3.8 chat template prefilled the opening `<think>` token. vLLM therefore returned many thinking-mode completions as:

```text
private reasoning text
</think>

final answer
```

rather than as a self-contained:

```text
<think>private reasoning text</think>

final answer
```

The original parser stripped reasoning only when the returned string itself began with `<think>`. As a result, the first v4-2000 corpus treated the reasoning body plus the closing `</think>` marker as part of the trainable assistant response.

## Frozen affected evidence

The downloaded Google Drive archive inspected after the run has SHA-256:

```text
f21c7deacd5f5cfb2a8598d2512da58aaccf1d22893247c9236a2c42f9664ef7
```

Its first-pass generation run identity records:

```text
implementation_sha256 = ba028085bc8385fc65c9bcfe4df33dcc02f938835f478e6a3d51a323e214212b
input_file_sha256      = e4960c198beee58dadaa9ee808c2dd84a59d5ffd82c676be9c1459c673a9b8cb
total_records          = 2000
```

Mechanical inspection found:

- 1,983 `finish_reason=stop` records, all containing a closing-only `</think>` boundary;
- 17 `finish_reason=length` records;
- 4 of those length-truncated records contained a closing `</think>` boundary; and
- 13 length-truncated records ended while still inside reasoning and therefore contained no final-answer boundary.

The old merged/final corpus must not be used for student training.

## Code repair

The teacher runtime now normalizes vLLM closing-only thinking output at the inference boundary before the existing generation/compression parsers see it. Finalization also fails closed if any assistant training message still contains `<think>` or `</think>`.

This changes the generation/compression implementation identity. Do not edit old checkpoint identities or attempt to force the old generation checkpoint through the new ordinary loader.

## Salvage strategy

The expensive 2,000 first-pass generations do **not** need to be repeated.

`salvage_teacher_reasoning.py` independently validates the frozen legacy generation evidence:

1. validates the source input SHA-256 and run identity;
2. validates every generation shard and `.sha256` sidecar;
3. validates config, implementation, input, record, prompt, seed, teacher, and stored-response identities;
4. splits closing-only `reasoning</think>answer` rows using the corrected reasoning parser;
5. converts length-truncated rows that never reached `</think>` into an empty assistant final answer while retaining only a reasoning digest/character count in provenance;
6. writes a new deterministic sanitized source JSONL plus checksum, summary, and independent salvage run identity; and
7. never mutates the original generation checkpoint.

The sanitized source deliberately retains the active v3 teacher-only budget metadata and prompt policy because the selective v4 compressor needs those frozen per-record budgets. The v4 merge step strips the active teacher-only policy before diagnostics or finalization, exactly as before.

## Fresh compression only

The old v4 compression checkpoint must not be reused. Its target selection and rewrite prompts were based on contaminated draft answers that still included hidden reasoning.

The salvage workflow therefore uses a new compression checkpoint namespace and recomputes targets from the sanitized first-pass answers. Only records that remain all of the following are rewritten:

1. `finish_reason=stop`;
2. over the Qwen3.5-4B 2,048-token student envelope; and
3. over their frozen v3 final-answer budget.

The compression pass remains Qwen3.8-27B BF16 with `reasoning_effort=low` and an 8,192-token generation cap.

## New durable namespace

Use:

```text
MyDrive/tiny-qwen-coder/distillation/qwen38-27b-v4-2000-salvage-v1/
```

The original remains untouched at:

```text
MyDrive/tiny-qwen-coder/distillation/qwen38-27b-v4-2000/
```

## Executable salvage runbook

The canonical executable repair workflow is:

[`scripts/teacher_distillation/qwen38_teacher_distillation_v4_2000_salvage_colab.ipynb`](../scripts/teacher_distillation/qwen38_teacher_distillation_v4_2000_salvage_colab.ipynb)

Direct Colab link:

https://colab.research.google.com/github/ekkus93/tiny-qwen-coder/blob/master/scripts/teacher_distillation/qwen38_teacher_distillation_v4_2000_salvage_colab.ipynb

Do not run the preserved ordinary v4 notebook against the affected 2,000-record evidence. For this run, the salvage notebook is authoritative. It validates the frozen legacy generation checkpoint, writes the sanitized first-pass corpus into the fresh salvage namespace, recomputes the selective-compression target set, generates only missing fresh compression shards, diagnoses and finalizes the repaired corpus, and runs the final qualification and reasoning-marker audit.

Use a fresh A100 80 GB Colab runtime only after the repository revision containing the salvage notebook has passed the ordinary CPU quality gates. Freeze that exact repository ZIP under the filename expected by the notebook and keep its adjacent SHA-256 sidecar unchanged for all resumes of the salvage namespace.

## Acceptance criteria

The repaired merged corpus must again pass all of the following before training:

- normal-stop rate >= 0.90;
- student-length acceptance among normal stops >= 0.85;
- protected-benchmark contamination status `clean`;
- no `<think>` marker in any accepted/train/validation assistant message; and
- no `</think>` marker in any accepted/train/validation assistant message.

Do not train from the original v4-2000 `final/` directory even though its previous qualification file says `qualified: true`; that qualification predated discovery of the reasoning-leak parser defect.
