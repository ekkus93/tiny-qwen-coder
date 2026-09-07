# Bounded v4 selective-compression study

This note freezes the scientific intent of the first v4 teacher-distillation study.

## Motivation

The bounded v3 study preserved high-reasoning Qwen3.8-27B generation and used a per-record final-answer budget derived from the Qwen3.5-4B 2,048-token student envelope. The live 200-record result produced 199 normal stops and 159 student-length-accepted normal-stop records. Follow-up diagnostics showed that every one of the 40 student-overlength normal-stop records exceeded its frozen answer budget. The remaining failure is therefore budget-instruction compliance, not the per-record budget arithmetic.

## Intervention

V4 keeps the sealed v3 first-pass answers immutable and applies a second Qwen3.8-27B pass only to records that are all of:

1. normal-stop v3 generations;
2. over the 2,048-token student record boundary; and
3. over their frozen v3 final-answer budget.

For the observed v3-200 evidence this should select 40 records. The compression pass uses the same pinned BF16 teacher/runtime and generation cap but `low` reasoning because the programming problem has already been solved. It receives the original student conversation, the v3 final answer, and the frozen answer budget. It never receives v3 hidden reasoning.

Compressed outputs are written to a new durable checkpoint namespace. V3 shards are never edited. The merged v4 corpus preserves already-fitting v3 answers unchanged and substitutes only sealed compression outputs for selected targets.

## Scale gate

The bounded v4 study may scale only if all of the following are true:

- normal-stop rate is at least 0.90;
- student-length acceptance among normal stops is at least 0.85; and
- protected-benchmark contamination status is `clean`.

The gate must not be lowered after observing the v4 result.

## Execution

Use `scripts/teacher_distillation/qwen38_teacher_distillation_v4_colab.ipynb` with an A100 80 GB runtime and a frozen exact-commit repository ZIP named `tiny-qwen-coder-v4.zip` on Google Drive. The notebook performs preflight selection, resumable compression, merged-corpus diagnostics, fail-closed finalization, qualification, and an explicit scaling guard.
