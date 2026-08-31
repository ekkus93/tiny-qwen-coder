# Hugging Face adapter promotion and archival

`tiny-qwen-coder-promote-hf` is the fail-closed promotion path for a completed full-training
output. It is intentionally separate from training so an experiment is not published merely
because training exited successfully.

## Safety contract

The command refuses promotion unless all of the following are true:

- `training-report.json` uses the supported full-training schema and contains finite losses,
  positive record/step counts, and a final checkpoint under `checkpoints/`;
- every file in the training report's persisted-artifact inventory still matches its recorded
  size and SHA-256, and the aggregate artifact-set SHA-256 also matches;
- `adapter/adapter_config.json` declares PEFT `LORA`;
- `adapter/adapter_model.safetensors` exists and is non-empty;
- merged/full-model weights and legacy `adapter_model.bin` are absent;
- adapter-manifest identity and step count agree with the training report;
- run-manifest Git provenance is present;
- the adapter directory contains no unreviewed file that would otherwise be silently omitted.

The bounded durable archive contains:

- `adapter_model.safetensors` and `adapter_config.json`;
- the Qwen chat template/tokenizer files emitted with the adapter, when present;
- a generated model card;
- adapter, run, dataset, training-config, preflight, metrics, and training reports;
- `archive-manifest.json`, which fingerprints the promoted file set.

`training_args.bin` and the complete `checkpoints/` tree are deliberately excluded from the
Hub archive. The former is unnecessary Python-serialized training state; the latter contains
large optimizer/resume state and is not needed to use the finished LoRA adapter.

## Authentication

Create a Hugging Face user access token with write permission and expose it through an
environment variable. The token is never accepted as a command-line value, which avoids
putting it in shell history or the process list.

```bash
export HF_TOKEN='...'
```

The default environment-variable name is `HF_TOKEN`. Use `--token-env OTHER_NAME` if needed.

## First promotion

Choose a durable archive directory **outside** the GitHub Actions `_work` checkout. For
example:

```bash
uv run --frozen tiny-qwen-coder-promote-hf \
  --output-dir artifacts/train/python/p0 \
  --archive-dir "$HOME/models/tiny-qwen-coder/python-p0/<run-id>" \
  --repo-id '<hf-user>/tiny-qwen-coder-python-p0' \
  --create-private-repo
```

`--create-private-repo` creates the target model repository if necessary and requests private
visibility. Without it, the target repository must already exist and be accessible with the
supplied token.

The command creates one Hub commit for the complete managed file set. It then re-downloads
every managed file from the exact returned commit OID with `force_download=True` and compares
both file size and SHA-256 with the local archive.

A successful verification writes:

```text
<archive-dir>/huggingface-promotion.json
```

That report records the exact Hub commit, source Git commit, archive-manifest hash, and every
verified file digest.

## Optional checkpoint cleanup

Checkpoint deletion is explicit and occurs **only after** the exact-commit Hub verification
succeeds:

```bash
uv run --frozen tiny-qwen-coder-promote-hf \
  --output-dir artifacts/train/python/p0 \
  --archive-dir "$HOME/models/tiny-qwen-coder/python-p0/<run-id>" \
  --repo-id '<hf-user>/tiny-qwen-coder-python-p0' \
  --delete-checkpoints-after-verify
```

If upload, download, size verification, or SHA-256 verification fails, `checkpoints/` is left
untouched and no `huggingface-promotion.json` success record is written. Cleanup never deletes
the local durable adapter archive or the original `adapter/` directory.

## Existing repositories

The command treats the Hub repository as managed state. An existing repository may be reused
only if it contains no unmanaged files and, when an `archive-manifest.json` already exists,
that manifest identifies the same adapter ID. The Hub commit uses the inspected current commit
as `parent_commit`, so a concurrent repository update causes promotion to fail rather than
silently overwriting it.

One Hugging Face model repository per adapter identity is recommended. Later promoted runs of
the same adapter can replace the root files while Hugging Face Git history preserves the older
commits.
