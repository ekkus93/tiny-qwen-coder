# FTR-202 A100 execution handoff — 2026-09-10

## Purpose

This handoff freezes the exact execution source and operator procedure for the empirical FTR-202 teacher benchmark. The protocol, FTR-203 gate, transport-integrity checks, and historical Docker scoring runtime are already qualified. The remaining scientific work is the real Qwen3.8-27B generation run, isolated scoring, pairwise comparison, and automatic FTR-G2 decision.

## Immutable execution source

Use this exact repository commit for **both generation and scoring**:

```text
90d42cca541cc7a96493d5d80d95718e5930d551
```

Authoritative post-merge CI:

```text
https://github.com/ekkus93/tiny-qwen-coder/actions/runs/34544502972
```

That CI passed after the FTR-202 scorer was pinned to the historical Docker runtime and exact frozen execution image.

Do not substitute a later `master` SHA for this experiment. The transport verifier intentionally rejects scoring from a source SHA different from the generation SHA.

## Frozen identities and gate

Teacher:

```text
Qwen/Qwen3.8-27B@72a217afab8029b39e4af1c7273a829995a3dbaf
```

Frozen base public-coding score:

```text
HumanEval 128/164
MBPP      290/500
Combined  418/664
```

FTR-203 authorizes this teacher/configuration only if all precommitted checks pass:

- HumanEval + MBPP teacher score is at least `472/664`.
- Net public-coding improvement is at least `+54` tasks.
- Absolute public-coding gain is at least `+0.08`.
- Paired one-sided exact discordant-task p-value is at most `0.05`.
- HumanEval does not regress.
- MBPP does not regress.
- Repository holdout loses no more than one passed task.
- General/tool regression loses no more than one passed case.

A gate failure is a valid experimental outcome. It means teacher distillation from this exact teacher/configuration must stop.

## Phase A — Colab A100 generation

Use:

```text
scripts/evaluation/ftr_202_teacher_benchmark_colab.ipynb
```

The notebook checks out the immutable source SHA above and runs only:

```text
scripts/evaluation/run_ftr_202_colab.py
```

Requirements:

- Google Colab A100 80 GB runtime.
- Google Drive mounted for durable checkpoints.
- Enough Drive capacity for response/checkpoint evidence.
- Network access is allowed for model download during generation.
- Generated benchmark code is never executed in Colab.

The default durable root in the notebook is:

```text
/content/drive/MyDrive/tiny-qwen-coder/ftr-202
```

The generated evidence directory will be:

```text
/content/drive/MyDrive/tiny-qwen-coder/ftr-202/90d42cca541cc7a96493d5d80d95718e5930d551/ftr-201-teacher-direct-v1
```

If Colab disconnects, reopen the notebook and rerun it with the same Drive path. The generation stage resumes from exact persisted task checkpoints.

Successful completion creates this file inside the generation directory:

```text
FTR_202_GENERATION_HANDOFF.json
```

The handoff records the exact source SHA, GPU evidence, generation-stage digest, and SHA-256 digest of every transported generation artifact.

## Phase B — Transport

Copy the **entire** `ftr-201-teacher-direct-v1` directory from Drive to the isolated scoring machine.

Do not modify, rename, regenerate, or selectively copy files inside it. The scorer verifies the handoff manifest, `generation-stage.json`, every listed artifact digest, task inventory, and path containment before executing any generated code.

## Phase C — Prepare isolated scoring machine

Use a machine with Docker. Google Drive must not be mounted.

Check out the exact execution source:

```bash
git clone https://github.com/ekkus93/tiny-qwen-coder.git
cd tiny-qwen-coder
git checkout --detach 90d42cca541cc7a96493d5d80d95718e5930d551
test "$(git rev-parse HEAD)" = "90d42cca541cc7a96493d5d80d95718e5930d551"
test -z "$(git status --porcelain)"
```

Install the frozen environment:

```bash
python -m pip install "uv==0.12.13"
uv sync --frozen
```

Remove model/cloud credentials from the scoring environment:

```bash
unset GOOGLE_APPLICATION_CREDENTIALS
unset HF_TOKEN
unset HUGGING_FACE_HUB_TOKEN
```

Prepare the exact historical execution image:

```bash
docker pull "python:3.11.14-slim@sha256:c8271b1f627d0068857dce5b53e14a9558603b527e46f1f901722f935b786a39"
docker image inspect "python:3.11.14-slim@sha256:c8271b1f627d0068857dce5b53e14a9558603b527e46f1f901722f935b786a39" >/dev/null
```

The scorer refuses Podman fallback and refuses an unpinned or missing image.

## Phase D — Restore the frozen base evidence

The base reference is frozen to:

```text
workflow run: 33301242379
artifact: python-base-baseline-da537443ab80b1380bee0fc3c7d9d01ca0574f35
source SHA: da537443ab80b1380bee0fc3c7d9d01ca0574f35
```

With an authenticated GitHub CLI, one convenient download path is:

```bash
mkdir -p /tmp/ftr202-base-download
gh run download 33301242379 \
  -R ekkus93/tiny-qwen-coder \
  -n python-base-baseline-da537443ab80b1380bee0fc3c7d9d01ca0574f35 \
  -D /tmp/ftr202-base-download

BASE_DIR="$(find /tmp/ftr202-base-download -type d -path '*/artifacts/eval/python/base-baseline-v1' -print -quit)"
test -n "$BASE_DIR"
printf 'BASE_DIR=%s\n' "$BASE_DIR"
```

Downloading the artifact is preparation only. Do not leave unrelated cloud/model credentials exported when running candidate scoring.

## Phase E — Isolated scoring and automatic FTR-G2 decision

Assume the transported teacher generation directory is:

```text
/tmp/ftr202-generation/ftr-201-teacher-direct-v1
```

Run:

```bash
uv run --frozen python scripts/evaluation/run_ftr_202_isolated_score.py \
  --repo-root . \
  --generation-dir /tmp/ftr202-generation/ftr-201-teacher-direct-v1 \
  --base-dir "$BASE_DIR" \
  --report /tmp/ftr-202-teacher-superiority.json
```

The scorer:

1. requires a clean checkout at `90d42cca541cc7a96493d5d80d95718e5930d551`;
2. rejects Drive/model/cloud credential exposure;
3. verifies the complete transported handoff before execution;
4. requires Docker and the exact pinned execution image;
5. scores HumanEval, MBPP, repository holdout, and general/tool regression with the canonical scorer;
6. compares teacher and frozen base task-by-task;
7. applies the already-frozen FTR-203 gate.

Exit status:

- `0`: teacher superiority demonstrated; the report decision authorizes the next stage.
- `3`: benchmark completed but FTR-G2 failed; stop teacher distillation from this teacher/configuration.
- any other nonzero status: infrastructure/protocol/scoring failure; do not interpret it as a capability result.

## Evidence to preserve

Preserve at minimum:

- the complete transported generation directory;
- `FTR_202_GENERATION_HANDOFF.json`;
- scored teacher artifacts produced under `artifacts/eval/python/ftr-201-teacher-direct-v1`;
- `/tmp/ftr-202-teacher-superiority.json`;
- terminal output showing the exact Docker image and source SHA;
- the final decision before starting any FTR-300 or new distillation work.

## Current status

At handoff freeze time:

- protocol: complete;
- FTR-203 gate: precommitted;
- generation runner: complete;
- transport integrity: complete;
- historical Docker scoring parity: complete;
- exact execution source: CI-qualified;
- real A100 generation: **not yet run**;
- isolated scoring: **not yet run**;
- pairwise teacher/base result: **unknown**;
- FTR-G2: **undecided**.
