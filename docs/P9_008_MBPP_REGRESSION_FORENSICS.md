# P9-008 — MBPP qualification regression forensics

## Purpose

P9-008 is a read-only diagnostic study following the failed P9-007E winner-only qualification. It
does **not** train a model, regenerate protected responses, repartition benchmarks, or evaluate any
P9-007 runner-up. It compares the already-frozen unchanged-base MBPP result artifact with the
already-consumed step-185 P9-007E MBPP qualification artifact.

The question is narrow:

> Why did the repaired teacher-distilled adapter preserve HumanEval and repository holdout while
> losing 24 MBPP passes relative to the unchanged base?

## Frozen P9-007E outcome

The one-shot qualification completed successfully in GitHub Actions run `34275104625`, job
`102226070875`, at source SHA `69fa3ee934cb59e081bee8356322e569ce42c3a9`. The workflow itself
was green, but the model failed the frozen target-language promotion boundary.

```text
                             step 185       unchanged base
HumanEval                    128/164        128/164
MBPP                         266/500        290/500
repository holdout             6/11           6/11
combined                     400/675        424/675
promotion minimum            438/675
```

P9-007 is therefore closed as a failed qualification. No runner-up may be tested on qualification,
and teacher generation must not be scaled beyond the repaired 2,000-example study under the same
recipe.

Durable qualification evidence is frozen in:

```text
docs/evidence/P9_007_QUALIFICATION_RUN_34275104625.json
```

## Exact forensic inputs

P9-008 consumes two immutable result artifacts only.

### Unchanged base

```text
workflow run       33301242379
source SHA         da537443ab80b1380bee0fc3c7d9d01ca0574f35
artifact ID        9729636096
artifact name      python-base-baseline-da537443ab80b1380bee0fc3c7d9d01ca0574f35
MBPP results SHA   c1598681b431906ba497b71ba4f1d1ce7c1cbba3d1a98db356f310fa6a65ab2e
full MBPP          290/500
```

### P9-007E selected adapter

```text
workflow run       34275104625
source SHA         69fa3ee934cb59e081bee8356322e569ce42c3a9
artifact ID        10077334268
artifact name      p9-007e-qualification-69fa3ee934cb59e081bee8356322e569ce42c3a9
MBPP results SHA   a9e550976a1809ea14b31f1587d9f6e655677f3185d913344dea33ba422c4df2
qualification MBPP 196/370
selected step      185
```

The qualification membership is exactly 370 MBPP tasks. On that same membership, the frozen base
passes 220 tasks, so the qualification-only MBPP delta is also exactly `196 - 220 = -24`.

## Exact flip matrix

Comparing pass/fail status task by task gives:

| Base | Step 185 | Count | Meaning |
| --- | --- | ---: | --- |
| pass | pass | 179 | retained capability |
| fail | fail | 133 | unchanged failure |
| pass | fail | **41** | regression |
| fail | pass | **17** | improvement |

The 17 improvements partially offset 41 newly broken tasks, yielding the observed net `-24`.

This is more informative than the aggregate score: the adapter is not uniformly worse. It changes
which MBPP tasks the model can solve, but the destructive side of that exchange is much larger.

## Failure-mode evidence

The 41 pass-to-fail regressions break down as:

```text
test failures   40 / 41   97.56%
parse failures   1 / 41    2.44%
harness errors   0 / 41
```

The single parse regression is a 512-token truncation. The other 40 regressions successfully reach
the executable test stage. This rules out harness instability, OCI failures, and widespread parser
breakage as explanations for the MBPP loss.

Among the 40 test-stage regressions, the observed error messages include 34 assertion failures,
four `NameError` failures, and two `TypeError` failures. Manual inspection of the frozen generated
code shows recurring classes of semantic mistakes: omitted imports, wrong required function names,
off-by-one formulas, inverted specification semantics, wrong return shapes, index/value confusion,
and plausible but incorrect algorithm templates.

There are also two pairs of regressions whose step-185 generated code is byte-for-byte identical
after normalization:

```text
MBPP/199 and MBPP/388
MBPP/76  and MBPP/347
```

P9-008 records only their generated-code SHA-256 and task IDs in durable output, not protected
prompts or tests. Repeated identical wrong solutions on different tasks are evidence that the
adapter strengthened particular incorrect solution templates rather than merely adding random
sampling noise.

## Length and truncation

Generation length correlates with failure, but it does not explain the main regression.

```text
step-185 passing tasks      mean 140.54 tokens, median 113, max-token hits  4/196
step-185 failing tasks      mean 235.70 tokens, median 200, max-token hits 18/174
pass->fail regressions base mean 228.44 tokens, median 200, max-token hits  0/41
pass->fail regressions FT   mean 234.00 tokens, median 214, max-token hits  3/41
```

Only one of the 41 regressions is actually classified as a parse/truncation failure. Therefore
raising `max_new_tokens` would not address the dominant problem.

## Interpretation

The strongest supported conclusion is:

> The repaired corpus fixed the catastrophic behavior of the old P0 fine-tunes, but the current
> teacher-distillation objective still teaches semantic habits that trade away MBPP correctness.

The positive HumanEval preservation matters: the adapter can absorb useful teacher signal without
globally destroying coding ability. The negative MBPP flip matrix matters more for the next design:
we need to improve **semantic verification and exact-spec behavior**, not search another rank or
learning rate around the same data objective.

## Next experiment boundary

The next student study should keep the conservative optimizer/adapter structure fixed initially and
change the **training-data acceptance objective**. A candidate example should not become trainable
merely because the teacher output looks like good Python or passes static filters.

The next corpus generation/selection design should require independently executable evidence where
possible:

1. exact requested function name/signature and dependency/import validation;
2. Python parse/compile success;
3. execution against tests or generated behavioral contracts that were not produced by the same
   teacher response being graded;
4. edge-case coverage for short specification-following tasks;
5. rejection of outputs whose semantics cannot be independently checked;
6. compact answers favored when equivalent, without treating length alone as a correctness signal.

Do not use MBPP or the protected evaluation tasks as training data. Fresh synthetic tasks, public
repository tests, or separately generated executable contracts can provide MBPP-like semantic
pressure without contaminating the benchmark.

Keep the existing 2,000-example scale ceiling until this redesigned objective demonstrates a real
improvement on a development-only gate. Do not begin another broad rank/LR/epoch sweep first.

## Reproduction

The deterministic analyzer is:

```text
src/tiny_qwen_coder/evaluation/python_distilled_mbpp_forensics.py
```

It writes only aggregate diagnostics, task IDs, error categories, token counts, and generated-code
hashes. It does not write benchmark prompts/tests into the diagnostic artifact.

The GitHub workflow:

```text
.github/workflows/python-p9-mbpp-forensics.yml
```

downloads the exact two frozen Actions artifacts, verifies the exact MBPP result-file SHA-256
values, reproduces the flip matrix, and fails closed if the observed signature changes.
