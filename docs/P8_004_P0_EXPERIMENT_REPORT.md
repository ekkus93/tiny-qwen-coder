# P8-004 — Python P0 experiment report

## Executive conclusion

Python P0 is a technically valid LoRA training artifact but a failed quality experiment.

The adapter trained to completion, remained loadable/unmerged, and did not show catastrophic loss of the small TypeScript/Rust behaviors covered by P8-003. However, the decision-relevant Python coding evidence is strongly negative across every protected suite:

- HumanEval: `128/164` → `88/164`;
- MBPP: `290/500` → `97/500`;
- repository holdout: `6/11` → `2/11`; and
- combined Python coding: `424/675` (`0.6281481481481481`) → `187/675` (`0.277037037037037`).

That combined result is a loss of 237 passing problems, an absolute pass-rate delta of `-0.3511111111111111` (35.11 percentage points), and a relative reduction of approximately 55.90% from the unchanged-base pass rate.

**P8-004 recommendation:** do **not** promote `language/python/p0` as the recommended Python adapter. Preserve it as negative experimental evidence and iterate in Phase 9. The formal promote/reject state transition remains P8-005.

The clean P8-002 and P8-003 results narrow the failure mode but do not outweigh the direct Python regression: the adapter did not trigger a measurable pass/fail collapse in the frozen general/tool suite or the small cross-language semantic smoke suite, yet it materially damaged the exact target capability it was trained to improve.

---

## 1. Experiment identity

### Base model

- repository: `Qwen/Qwen3.5-4B`
- model revision: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- tokenizer repository: `Qwen/Qwen3.5-4B`
- tokenizer revision: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`

### Adapter

- adapter ID: `language/python/p0`
- family: `language`
- language: `python`
- P7-006 training source Git SHA: `02df92a9c2d347b9fb013dc25714fe066c6bcafe`
- training run ID: `training-python-20260831T180916446466Z-02df92a9-eafc119d`
- adapter weight SHA-256: `c94606250e112f72362eb883a55f7b2c8af854d445f6bb6194352c2806a8f276`
- adapter weight size: `65,004,840` bytes
- canonical P7-006 workflow run: `33422910444`
- canonical P7-006 Actions artifact: `9789946698`
- P7-006 artifact ZIP SHA-256: `1aba44c566699dc69ee20e7b03481680629dbfa8b94aaab65450b56174dc55f3`

The canonical P7-006 artifact contains the resolved training configuration, frozen dataset manifest, run manifest, adapter manifest, training metrics, final training report, checkpoint, and final unmerged PEFT LoRA adapter.

---

## 2. Dataset identity

The frozen training dataset is:

- manifest ID: `dataset/python/p0`
- manifest SHA-256: `900d62a32466a63faa585a53ebb0a74cf0f8ae188ea5c689caddcd3e4f1fece4`
- corpus ID: `python-p0`
- dataset config SHA-256: `4f9663e72b22d81ce8975e6f6ed87ee7457d3bef0a08fe211e700dd5ea12fbff`
- seed: `1729`
- maximum token length: `2048`
- accepted records: `40,000`
- train records: `38,000`
- validation records: `2,000`
- validation fraction: `0.05`

### Source composition

| Source | Frozen revision | License recorded by manifest | Accepted |
| --- | --- | --- | ---: |
| `OLMo-Coding/starcoder-python-instruct` | `5bcafbc00100ec7cf1e6e5a9e353dc2f4eaad9fc` | Apache-2.0 | 30,000 |
| `ise-uiuc/Magicoder-OSS-Instruct-75K` | `5f839b1f368a76b161028bb9edff055db34022b2` | MIT | 10,000 |
| **Total** | — | — | **40,000** |

The OLMo loader selected rows whose source metadata identified `python3`; the Magicoder loader selected rows whose language field identified `python`.

### Filtering and token distribution

The materialized pipeline scanned `44,108` records to accept `40,000`:

- `3,619` records were rejected for exceeding the token-length limit;
- `489` records were rejected by Python validation/quality checks;
- `0` content-stage rejections were recorded;
- `0` exact-duplicate rejections were recorded.

For accepted examples, the tokenizer-aware distribution had:

- mean: `651.343275` tokens;
- median (`p50`): `546` tokens;
- `p95`: `1,592` tokens; and
- `p99`: `1,938` tokens.

The train/validation membership SHA-256 is `78559765eac305528e5ba96ae3dae04feffda39fc4727d450d602f6e68697428`.

### Dataset evidence limitation

The frozen dataset manifest records contamination status as `not_run`, with no contamination check IDs or findings. This must not be silently presented as a clean contamination result. It is a limitation of the P0 evidence and should be corrected for future promotion-eligible experiments.

Because P0 substantially *underperformed* the unchanged base on all three Python coding suites, this missing contamination check does not rehabilitate P0. It does, however, reduce the strength of any positive benchmark claim that might otherwise have been made from this corpus.

---

## 3. Frozen training configuration

The canonical source is `configs/train/python/p0.yaml`, whose resolved configuration SHA-256 is:

`4b0c742ad3a55f4eaffd4f2283be7291d6434eb89b07c13dc90c2166238a5f46`

| Parameter | P0 value |
| --- | --- |
| training mode | 4-bit QLoRA |
| model load dtype | BF16 |
| QLoRA quantization | NF4, double quantization, BF16 compute |
| sequence length | 2,048 |
| epochs | 1 |
| micro-batch size | 1 |
| gradient accumulation | 8 |
| effective batch size | 8 |
| optimizer steps | 4,750 |
| learning rate | `2e-4` |
| scheduler | cosine |
| warmup ratio | `0.03` |
| gradient checkpointing | enabled |
| loss | assistant-only |
| seed | `1729` |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | `0.05` |
| LoRA bias | none |
| target strategy | selective language-backbone projections |

The selective target contract covers full-attention projections, MLP projections, and Gated DeltaNet/linear-attention projections while excluding the language output head, vision encoder, and multimodal projector. At rank 16 it exposes `32,464,896` trainable parameters across the frozen target architecture.

---

## 4. Full-training result

The canonical P7-006 run completed the full frozen workload without OOM, non-finite loss, or merged/full-model fallback.

| Metric | Measured result |
| --- | ---: |
| training records | 38,000 |
| validation records | 2,000 |
| completed optimizer steps | 4,750 |
| training loss | `0.6934559548340345` |
| validation loss | `0.7679226398468018` |
| trainer runtime | `45,979.9559 s` (~12.77 h) |
| end-to-end runtime | `47,050.625130466186 s` (~13.07 h) |
| training throughput | `0.826 samples/s` |
| optimizer throughput | `0.103 steps/s` |
| peak CUDA allocated | `9,119,821,824` bytes (~8.49 GiB) |
| peak CUDA reserved | `13,413,384,192` bytes (~12.49 GiB) |

The training host recorded one `NVIDIA GeForce RTX 4070 Ti SUPER` with `16,688,218,112` bytes of total CUDA memory. The measured peak reservation used about 80.38% of that reported memory, leaving practical headroom during the accepted run.

Training therefore succeeded as an engineering exercise: the configuration was reproducible, memory-safe on the reference GPU, finite, and produced a valid portable LoRA. The later benchmark regression is a model-quality failure, not a failed training harness.

---

## 5. Frozen unchanged-base reference

P8-001 compares P0 against the accepted P6-005 unchanged-base baseline rather than against a newly regenerated convenience baseline.

The accepted baseline is:

- workflow run: `33301242379`
- source Git SHA: `da537443ab80b1380bee0fc3c7d9d01ca0574f35`
- Actions artifact: `9729636096`
- artifact name: `python-base-baseline-da537443ab80b1380bee0fc3c7d9d01ca0574f35`
- artifact ZIP SHA-256: `bcc08b94e0204e19d38fe28d0771a597687cbe44ad08af4759233a3c824e2e21`
- frozen artifact-set SHA-256: `4bf616c3e84bdd74f8cf6467fc2d9d760f04d3b1b44660a81e365ff6f99a72fc`

Both base and adapter evaluation use the pinned model/tokenizer revision and frozen deterministic generation settings. P8-001 generation uses seed `1729`, greedy decoding, temperature `0`, top-p `1`, top-k `0`, a `512` new-token maximum, the canonical evaluation prompt version, and thinking disabled.

---

## 6. Python coding evaluation — P8-001

Canonical P8-001 run `33538724658` on source `6aaea1bef3b6df97b2bf8d61103a89f6ee7fa43c` completed all 675 adapter generations, constrained execution/scoring, independent verification, and evidence upload.

Evidence artifact:

- artifact ID: `9814936298`
- artifact name: `p8-001-python-p0-evaluation-6aaea1bef3b6df97b2bf8d61103a89f6ee7fa43c`
- artifact ZIP SHA-256: `b02bea72d0c730e20f8e147c1464fcd17fd109c10d7b8a6a8db1015729282ad4`

| Suite | Unchanged base | Python P0 | Pass delta | Pass-rate delta | Relative change from base |
| --- | ---: | ---: | ---: | ---: | ---: |
| HumanEval | `128/164` (`0.7804878048780488`) | `88/164` (`0.5365853658536586`) | `-40` | `-0.24390243902439024` | `-31.25%` |
| MBPP | `290/500` (`0.58`) | `97/500` (`0.194`) | `-193` | `-0.38599999999999995` | `-66.55%` |
| repository holdout | `6/11` (`0.5454545454545454`) | `2/11` (`0.18181818181818182`) | `-4` | `-0.3636363636363636` | `-66.67%` |
| **Combined** | **`424/675` (`0.6281481481481481`)** | **`187/675` (`0.277037037037037`)** | **`-237`** | **`-0.3511111111111111`** | **`-55.90%`** |

All three suites move in the same negative direction. This matters more than any isolated anecdotal output: the regression spans two standard protected coding suites and the repository-owned holdout, with zero harness errors in the accepted measurement.

The MBPP and repository-holdout drops are especially severe, but no single suite is carrying the conclusion. Even HumanEval, where the adapter retained the strongest absolute performance, lost 40 passes and 24.39 percentage points.

---

## 7. General/tool regression — P8-002

Canonical P8-002 run `33554096916` on source `0485c375bcc384d0a0dfdfb423db740a4b78b109` passed generation, baseline validation, scoring, independent verification, and evidence upload.

Evidence artifact:

- artifact ID: `9818707785`
- artifact ZIP SHA-256: `d023e9d1563518fcba4ff1528d0290996004df943f834798d057ff7b22acc74b`
- frozen suite version: `general_tool_regression` v1
- suite semantic SHA-256: `9de462c05a05455b2cc5af8c0246d897fe7991510d470e837d205540922239f9`

| Category | Base | Python P0 | Delta |
| --- | ---: | ---: | ---: |
| instruction following | `0/2` | `0/2` | `0` |
| JSON structured output | `2/2` | `2/2` | `0` |
| simple reasoning | `0/2` | `0/2` | `0` |
| shell reasoning | `0/2` | `0/2` | `0` |
| Git reasoning | `0/2` | `0/2` | `0` |
| tool-call formatting/selection | `0/2` | `0/2` | `0` |
| **Overall** | **`2/12`** | **`2/12`** | **`0`** |

There were zero pass→fail regressions and zero fail→pass improvements under the frozen scorer.

This is reassuring only in a narrow sense. The unchanged base itself passes only 2 of 12 cases, so equality at `2/12` is evidence that P0 did not make this particular low-scoring frozen suite detectably worse—not evidence of strong general/tool capability.

The suite also has known exact-format sensitivity. That behavior is preserved because the suite was frozen before P0 evaluation; P8-004 does not retroactively normalize it.

---

## 8. Cross-language smoke — P8-003

P8-003 is a six-case catastrophic-collapse detector, not a TypeScript/Rust benchmark.

The original v1 measurement correctly remained `inconclusive_base`: the strict code-only scorer rejected all six base responses because they were wrapped in correctly tagged Markdown fences. V2 was then separately versioned and frozen before fresh GPU generation so semantic shape and strict format adherence could be measured independently.

Canonical v2 run `33570451764` at source `56039856392b5a4a3eecad147518c3657ccd683f` passed all provenance, generation, independent verification, and upload gates.

Evidence identity:

- artifact ID: `9824776245`
- artifact ZIP SHA-256: `ae57a061ddd02ff76f5dde4d454165f4d4421efc51778acade8d9537d8787b1a`
- verified report SHA-256: `de386ec1eca9fcfed4f797db223d686d7ef131292c9b9624b5d33d87d42357ee`
- scoring-contract SHA-256: `33c2459c64631ee7cd8903c36a6fe6ecb81df6ce6e1848bad096b5803cc77dd2`

### Decision-bearing semantic shape

| Language | Base | Python P0 | Regressions |
| --- | ---: | ---: | ---: |
| TypeScript | `3/3` | `3/3` | `0` |
| Rust | `3/3` | `3/3` | `0` |
| **Overall** | **`6/6`** | **`6/6`** | **`0`** |

Conclusion: `no_catastrophic_regression`.

### Supplemental strict format adherence

| Language | Base | Python P0 |
| --- | ---: | ---: |
| TypeScript | `0/3` | `3/3` |
| Rust | `0/3` | `3/3` |
| **Overall** | **`0/6`** | **`6/6`** |

P0 therefore changed output-format adherence on these six prompts without causing the semantic collapse the test was designed to detect.

---

## 9. Supplemental qualitative observations

Quantitative protected evaluation remains primary. The following examples help explain the measured behavior but must not override aggregate results.

### General/tool behavior changed despite equal pass/fail totals

P8-002 preserved the same `2/12` score but did not preserve every output:

- the arithmetic case changed from base `14` to adapter `17 - 5 + 8 = 20`;
- the Git staging case changed from base `git add -A` to adapter `git add .`.

Both remained failures under the frozen suite, illustrating why equal aggregate pass counts do not imply identical behavior.

### Cross-language formatting improved on the smoke prompts

In P8-003, all six unchanged-base responses had the requested TypeScript/Rust semantic shape but used one correctly tagged Markdown fence. Python P0 emitted corresponding unfenced code. The v2 semantic dimension therefore remained `6/6 → 6/6`, while strict format adherence moved `0/6 → 6/6`.

That is a real formatting/instruction-adherence difference on the frozen six prompts, but it is not evidence that P0 improved TypeScript or Rust programming ability.

---

## 10. Interpretation

### What succeeded

1. **Reproducible training machinery:** P0 completed the frozen 38k/2k workload on the reference GPU and emitted complete fail-closed evidence.
2. **Portable adapter architecture:** P7-007 proved the LoRA can be attached, disabled, re-enabled, and returned to base-only behavior without rebuilding or merging the 4B base.
3. **Memory strategy:** 4-bit QLoRA fit safely enough on the 16 GB reference GPU for the full run.
4. **No detected blanket non-Python collapse:** the P8-003 semantic smoke remained `6/6` for both base and P0.
5. **No additional pass/fail damage in the frozen general/tool suite:** P8-002 remained `2/12` for both models.

### What failed

The central P0 hypothesis was that Python-specific LoRA SFT would improve Python coding performance while retaining acceptable unrelated behavior. The first half failed decisively: Python coding performance deteriorated across every protected suite.

Training loss and validation loss show that the SFT objective was optimized to a finite solution, but they do not establish task improvement. In this experiment, lower/finite supervised loss did not translate into better protected coding performance.

### What the evidence does not establish

The existing evidence is not sufficient to assign one definitive root cause. Plausible hypotheses for Phase 9 include:

- the `2e-4` learning rate and/or one-epoch update magnitude being too aggressive for this base;
- the 40k-example instruction distribution being mismatched to the protected coding tasks;
- quality/composition issues in the selected P0 corpus;
- the breadth of the selective target set still permitting harmful specialization;
- QLoRA-specific effects compared with BF16 LoRA; and
- interaction among rank, dataset size, learning rate, and training length.

These are experiment hypotheses, not conclusions. Phase 9 should isolate them with controlled sweeps rather than changing several variables simultaneously.

---

## 11. Evidence limitations and follow-up requirements

1. **Dataset contamination was not run.** Future promotion-eligible datasets should have the intended contamination checks complete before training/evaluation claims are finalized.
2. **The general/tool suite has weak baseline headroom.** Base and P0 both score `2/12`, and the suite has known exact-format sensitivity. A future version may improve diagnostic power, but this frozen result must not be rewritten.
3. **P8-003 is intentionally tiny.** Six structural prompts can detect catastrophic collapse; they cannot establish broad TypeScript/Rust quality.
4. **No causal attribution from one P0 run.** The experiment establishes that this exact frozen configuration is poor, not that LoRA, QLoRA, the datasets, or the target architecture are inherently unsuitable.
5. **No promotion threshold was frozen before P0 evaluation.** P8-005 must establish explicit quantitative evidentiary meaning for `recommended` using the now-available baseline/P0 measurements; it must not choose a threshold designed to rescue this observed P0 result.

---

## 12. P8-004 decision summary

| Dimension | Result | Interpretation |
| --- | --- | --- |
| training completion | PASS | reproducible, finite, memory-safe full run |
| adapter portability | PASS | valid unmerged LoRA; enable/disable behavior verified |
| HumanEval | FAIL | `0.78049 → 0.53659` |
| MBPP | FAIL | `0.58 → 0.194` |
| repository holdout | FAIL | `0.54545 → 0.18182` |
| combined Python coding | FAIL | `0.62815 → 0.27704`, `-237` passes |
| general/tool preservation | NEUTRAL | `2/12 → 2/12`, weak baseline signal |
| catastrophic TS/Rust collapse | PASS | semantic `6/6 → 6/6` on small smoke |
| dataset contamination evidence | INCOMPLETE | manifest records `not_run` |

### Recommendation handed to P8-005

**Reject Python P0 as the recommended Python adapter.**

Preserve the exact adapter, configuration, dataset manifest, and evaluation evidence as the Phase 9 control. Do not delete or rewrite the failed result. The next experiments should seek a configuration that recovers or exceeds the unchanged-base Python benchmark scores without introducing unacceptable general/tool or cross-language regressions.

P8-005 should formalize that rejection and define explicit promotion thresholds for subsequent adapters. Phase 9 should then vary one major factor at a time so P0 remains a useful negative control rather than a dead end.

---

## 13. Evidence index

| Evidence | Canonical identity |
| --- | --- |
| unchanged-base baseline | run `33301242379`, source `da537443ab80b1380bee0fc3c7d9d01ca0574f35`, artifact `9729636096` |
| P0 full training | run `33422910444`, source `02df92a9c2d347b9fb013dc25714fe066c6bcafe`, artifact `9789946698` |
| P0 adapter load/inference validation | run `33509937071` |
| P8-001 Python coding evaluation | run `33538724658`, source `6aaea1bef3b6df97b2bf8d61103a89f6ee7fa43c`, artifact `9814936298` |
| P8-002 general/tool regression | run `33554096916`, source `0485c375bcc384d0a0dfdfb423db740a4b78b109`, artifact `9818707785` |
| P8-003 strict v1 | run `33557769986`, source `27550522181fc8cf3a490a03c983df55f6022430`, artifact `9820048413` |
| P8-003 semantic v2 | run `33570451764`, source `56039856392b5a4a3eecad147518c3657ccd683f`, artifact `9824776245` |

Primary repository documentation:

- `docs/P6_005_BASELINE.md`
- `docs/P7_001_SELECTIVE_LORA_TARGETS.md`
- `docs/P7_003_PYTHON_P0_TRAINING_CONFIG.md`
- `docs/P7_006_FULL_PYTHON_P0_TRAINING.md`
- `docs/P7_007_ADAPTER_LOAD_INFERENCE.md`
- `docs/P8_001_PYTHON_P0_EVALUATION.md`
- `docs/P8_002_GENERAL_TOOL_REGRESSION.md`
- `docs/P8_003_CROSS_LANGUAGE_SMOKE.md`
- `docs/evidence/P8_003_V1_VERIFIED_REPORT_33557769986.json`
- `docs/evidence/P8_003_V2_VERIFIED_REPORT_33570451764.json`
