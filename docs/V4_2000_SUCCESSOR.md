# V4-2000 teacher-data successor

The bounded v4-200 selective-compression study qualified for scaling with the frozen mechanical gates:

- normal-stop rate >= 0.90;
- student-length acceptance among normal stops >= 0.85; and
- protected-benchmark contamination status `clean`.

The 2,000-record successor is a **fresh experiment namespace**. It must not reuse or mutate the v4-200 checkpoint directories.

## Canonical notebook

Use:

`scripts/teacher_distillation/qwen38_teacher_distillation_v4_2000_colab.ipynb`

The notebook requires a fresh A100 80 GB Colab runtime and the exact repository ZIP for the commit being executed, renamed on Drive to:

`MyDrive/tiny-qwen-coder/code/tiny-qwen-coder-v4-2000.zip`

The durable experiment namespace is:

`MyDrive/tiny-qwen-coder/distillation/qwen38-27b-v4-2000/`

## Frozen successor sequence

1. Verify that the preserved v4-200 `qualification.json` still authorizes scaling.
2. Rebuild a deterministic source-stratified 2,000-record subset from the canonical prompt-only input.
3. Apply the frozen v3 per-record answer-budget policy.
4. Generate/resume 2,000 first-pass Qwen3.8-27B answers with the unchanged v3 high-reasoning contract and durable 16-record shards.
5. Diagnose the first-pass generation.
6. Mechanically select only normal-stop records that are both over the Qwen3.5-4B 2,048-token student envelope and over their frozen v3 answer budget.
7. Run/resume the qualified v4 low-reasoning selective-compression pass only for that target set.
8. Merge compressed replacements with untouched first-pass answers.
9. Diagnose and finalize the merged corpus with Python-quality filtering, 2,048-token rejection, exact deduplication, deterministic splitting, and fail-closed protected-benchmark contamination checks.
10. Apply the same final readiness thresholds before student training. The notebook blocks training readiness when that gate fails.

The notebook does not start LoRA training automatically. Preserve the final v4-2000 evidence before starting the student experiment.
