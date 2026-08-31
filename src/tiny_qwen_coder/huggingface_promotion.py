"""Fail-closed archival and Hugging Face promotion for completed LoRA adapters."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn, Protocol

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download


class HuggingFacePromotionError(RuntimeError):
    """Raised when promotion or archival evidence is incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class ArchiveFileDigest:
    """One file in the bounded, promotable adapter archive."""

    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class PromotionIdentity:
    """Validated identity and provenance for a completed training output."""

    adapter_id: str
    run_id: str
    language: str
    source_git_sha: str
    source_training_config: str
    source_training_config_sha256: str
    dataset_manifest_id: str
    dataset_manifest_sha256: str
    base_model_repository: str
    base_model_revision: str
    global_steps: int
    training_loss: float
    validation_loss: float
    final_checkpoint_dir: str


@dataclass(frozen=True, slots=True)
class HuggingFacePromotionReport:
    """Local durable evidence that an exact Hub commit was re-downloaded and verified."""

    schema_version: int
    repo_id: str
    revision: str
    adapter_id: str
    run_id: str
    source_git_sha: str
    archive_dir: str
    archive_manifest_sha256: str
    verified_files: tuple[ArchiveFileDigest, ...]
    checkpoints_deleted: bool
    verified_at_utc: str


class PromotionHub(Protocol):
    """Small Hub boundary used to keep promotion logic deterministic and testable."""

    def prepare_repo(
        self,
        repo_id: str,
        *,
        adapter_id: str,
        archive_manifest: Path,
        create_private_repo: bool,
    ) -> str | None: ...

    def commit_files(
        self,
        repo_id: str,
        *,
        archive_dir: Path,
        files: tuple[ArchiveFileDigest, ...],
        parent_commit: str | None,
        commit_message: str,
    ) -> str: ...

    def download_file(
        self,
        repo_id: str,
        *,
        revision: str,
        filename: str,
        destination_dir: Path,
    ) -> Path: ...


class HuggingFaceHub:
    """Production Hub implementation using one commit and exact-revision downloads."""

    def __init__(self, *, token: str) -> None:
        if not token.strip():
            raise HuggingFacePromotionError("Hugging Face token must not be empty")
        self._token = token
        self._api = HfApi()

    def prepare_repo(
        self,
        repo_id: str,
        *,
        adapter_id: str,
        archive_manifest: Path,
        create_private_repo: bool,
    ) -> str | None:
        if create_private_repo:
            try:
                self._api.create_repo(
                    repo_id=repo_id,
                    repo_type="model",
                    private=True,
                    exist_ok=True,
                    token=self._token,
                )
            except Exception as exc:  # pragma: no cover - exercised through the Hub
                raise HuggingFacePromotionError(
                    f"could not create or access private Hugging Face repo {repo_id}: {exc}"
                ) from exc

        try:
            remote_files = set(
                self._api.list_repo_files(repo_id, repo_type="model", token=self._token)
            )
            model_info = self._api.model_info(repo_id, token=self._token)
        except Exception as exc:  # pragma: no cover - exercised through the Hub
            raise HuggingFacePromotionError(
                f"could not inspect Hugging Face repo {repo_id}: {exc}"
            ) from exc

        archive_payload = _load_json_object(archive_manifest, context="archive manifest")
        raw_managed = archive_payload.get("files")
        if not isinstance(raw_managed, list):
            raise HuggingFacePromotionError("archive manifest files must be a list")
        managed_files = {
            item["path"]
            for item in raw_managed
            if isinstance(item, dict) and isinstance(item.get("path"), str)
        }
        managed_files.add("archive-manifest.json")
        allowed_remote = managed_files | {".gitattributes"}
        unexpected = sorted(remote_files - allowed_remote)
        if unexpected:
            rendered = ", ".join(unexpected)
            raise HuggingFacePromotionError(
                "Hugging Face repo contains unmanaged files; refusing to overwrite: " + rendered
            )

        if remote_files - {".gitattributes"}:
            if "archive-manifest.json" not in remote_files:
                raise HuggingFacePromotionError(
                    "existing Hugging Face repo is not managed by Tiny Qwen Coder"
                )
            parent_sha = getattr(model_info, "sha", None)
            if not isinstance(parent_sha, str) or not parent_sha:
                raise HuggingFacePromotionError(
                    "existing Hugging Face repo has no parent commit SHA"
                )
            try:
                with tempfile.TemporaryDirectory(prefix="tqc-hf-existing-") as tmp:
                    downloaded = hf_hub_download(
                        repo_id=repo_id,
                        filename="archive-manifest.json",
                        repo_type="model",
                        revision=parent_sha,
                        token=self._token,
                        local_dir=tmp,
                        force_download=True,
                    )
                    existing = _load_json_object(
                        Path(downloaded), context="existing remote archive manifest"
                    )
            except Exception as exc:  # pragma: no cover - exercised through the Hub
                raise HuggingFacePromotionError(
                    f"could not verify existing Hugging Face archive identity: {exc}"
                ) from exc
            if existing.get("adapter_id") != adapter_id:
                raise HuggingFacePromotionError(
                    "existing Hugging Face repo belongs to a different adapter identity"
                )

        parent = getattr(model_info, "sha", None)
        if parent is None:
            return None
        if not isinstance(parent, str) or not parent:
            raise HuggingFacePromotionError(
                "Hugging Face repo returned an invalid parent commit SHA"
            )
        return parent

    def commit_files(
        self,
        repo_id: str,
        *,
        archive_dir: Path,
        files: tuple[ArchiveFileDigest, ...],
        parent_commit: str | None,
        commit_message: str,
    ) -> str:
        operations = [
            CommitOperationAdd(
                path_in_repo=item.path,
                path_or_fileobj=archive_dir / item.path,
            )
            for item in files
        ]
        try:
            result = self._api.create_commit(
                repo_id=repo_id,
                repo_type="model",
                operations=operations,
                commit_message=commit_message,
                parent_commit=parent_commit,
                token=self._token,
            )
        except Exception as exc:  # pragma: no cover - exercised through the Hub
            raise HuggingFacePromotionError(
                f"could not commit adapter archive to Hugging Face repo {repo_id}: {exc}"
            ) from exc
        revision = getattr(result, "oid", None)
        if not isinstance(revision, str) or len(revision) < 20:
            raise HuggingFacePromotionError("Hugging Face commit did not return a valid commit OID")
        return revision

    def download_file(
        self,
        repo_id: str,
        *,
        revision: str,
        filename: str,
        destination_dir: Path,
    ) -> Path:
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="model",
                revision=revision,
                token=self._token,
                local_dir=destination_dir,
                force_download=True,
            )
        except Exception as exc:  # pragma: no cover - exercised through the Hub
            raise HuggingFacePromotionError(
                f"could not download {filename} from Hugging Face commit {revision}: {exc}"
            ) from exc
        return Path(downloaded)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path, *, context: str) -> dict[str, object]:
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise HuggingFacePromotionError(f"could not read {context} {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise HuggingFacePromotionError(f"{context} is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise HuggingFacePromotionError(f"{context} must be a JSON object: {path}")
    return payload


def _require_str(payload: dict[str, object], key: str, *, context: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise HuggingFacePromotionError(f"{context}.{key} must be a non-empty string")
    return value


def _require_positive_int(payload: dict[str, object], key: str, *, context: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HuggingFacePromotionError(f"{context}.{key} must be a positive integer")
    return value


def _require_finite_float(payload: dict[str, object], key: str, *, context: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HuggingFacePromotionError(f"{context}.{key} must be numeric")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise HuggingFacePromotionError(f"{context}.{key} must be finite")
    return resolved


def _safe_output_path(root: Path, relative: str, *, context: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise HuggingFacePromotionError(f"{context} must be relative: {relative}")
    resolved_root = root.resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise HuggingFacePromotionError(f"{context} escapes output root: {relative}") from exc
    return resolved


def _digest_file(root: Path, path: Path) -> ArchiveFileDigest:
    if path.is_symlink():
        raise HuggingFacePromotionError(f"archive source must not be a symlink: {path}")
    if not path.is_file():
        raise HuggingFacePromotionError(f"archive source file is missing: {path}")
    return ArchiveFileDigest(
        path=path.relative_to(root).as_posix(),
        size_bytes=path.stat().st_size,
        sha256=_sha256_file(path),
    )


def _verify_persisted_artifacts(output_dir: Path, report: dict[str, object]) -> None:
    raw = report.get("persisted_artifacts")
    if not isinstance(raw, list) or not raw:
        raise HuggingFacePromotionError("training report has no persisted_artifacts inventory")
    canonical: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise HuggingFacePromotionError(
                f"training report persisted_artifacts[{index}] must be an object"
            )
        relative = item.get("path")
        size = item.get("size_bytes")
        expected_sha = item.get("sha256")
        if not isinstance(relative, str) or not relative or relative in seen:
            raise HuggingFacePromotionError("persisted artifact path is missing or duplicated")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise HuggingFacePromotionError(f"persisted artifact size is invalid: {relative}")
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise HuggingFacePromotionError(f"persisted artifact SHA-256 is invalid: {relative}")
        path = _safe_output_path(output_dir, relative, context="persisted artifact path")
        if path.is_symlink() or not path.is_file():
            raise HuggingFacePromotionError(f"persisted artifact is missing or unsafe: {relative}")
        if path.stat().st_size != size:
            raise HuggingFacePromotionError(f"persisted artifact size mismatch: {relative}")
        if _sha256_file(path) != expected_sha:
            raise HuggingFacePromotionError(f"persisted artifact SHA-256 mismatch: {relative}")
        seen.add(relative)
        canonical.append({"path": relative, "size_bytes": size, "sha256": expected_sha})

    canonical.sort(key=lambda item: str(item["path"]))
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    actual_set_sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    expected_set_sha = report.get("artifact_set_sha256")
    if actual_set_sha != expected_set_sha:
        raise HuggingFacePromotionError("training report artifact_set_sha256 does not match files")


def _validate_training_output(output_dir: Path) -> PromotionIdentity:
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise HuggingFacePromotionError(
            f"training output directory is missing or unsafe: {output_dir}"
        )
    report = _load_json_object(output_dir / "training-report.json", context="training report")
    if report.get("schema_version") != 1:
        raise HuggingFacePromotionError("unsupported training-report schema_version")
    _verify_persisted_artifacts(output_dir, report)

    adapter_id = _require_str(report, "adapter_id", context="training report")
    run_id = _require_str(report, "run_id", context="training report")
    language = _require_str(report, "language", context="training report")
    global_steps = _require_positive_int(report, "global_steps", context="training report")
    _require_positive_int(report, "train_records", context="training report")
    _require_positive_int(report, "validation_records", context="training report")
    training_loss = _require_finite_float(report, "training_loss", context="training report")
    validation_loss = _require_finite_float(report, "validation_loss", context="training report")
    source_training_config = _require_str(
        report, "source_training_config", context="training report"
    )
    source_training_config_sha256 = _require_str(
        report, "source_training_config_sha256", context="training report"
    )
    dataset_manifest_id = _require_str(report, "dataset_manifest_id", context="training report")
    dataset_manifest_sha256 = _require_str(
        report, "dataset_manifest_sha256", context="training report"
    )
    final_checkpoint_dir = _require_str(report, "final_checkpoint_dir", context="training report")
    checkpoint = _safe_output_path(
        output_dir, final_checkpoint_dir, context="final checkpoint directory"
    )
    if checkpoint.is_symlink() or not checkpoint.is_dir() or not any(checkpoint.iterdir()):
        raise HuggingFacePromotionError("final training checkpoint is missing or empty")
    if Path(final_checkpoint_dir).parts[0] != "checkpoints":
        raise HuggingFacePromotionError("final checkpoint is not inside checkpoints/")

    adapter_dir_name = _require_str(report, "adapter_dir", context="training report")
    adapter_dir = _safe_output_path(output_dir, adapter_dir_name, context="adapter directory")
    if adapter_dir.is_symlink() or not adapter_dir.is_dir():
        raise HuggingFacePromotionError("adapter directory is missing or unsafe")
    adapter_config = _load_json_object(
        adapter_dir / "adapter_config.json", context="adapter config"
    )
    if str(adapter_config.get("peft_type", "")).upper() != "LORA":
        raise HuggingFacePromotionError("saved adapter is not a PEFT LoRA adapter")
    safetensors = adapter_dir / "adapter_model.safetensors"
    if safetensors.is_symlink() or not safetensors.is_file() or safetensors.stat().st_size <= 0:
        raise HuggingFacePromotionError(
            "canonical promotion requires non-empty adapter_model.safetensors"
        )
    for forbidden_name in (
        "adapter_model.bin",
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    ):
        if (adapter_dir / forbidden_name).exists():
            raise HuggingFacePromotionError(
                f"adapter directory contains forbidden promotion file: {forbidden_name}"
            )

    adapter_manifest = _load_json_object(
        output_dir / "adapter-manifest.json", context="adapter manifest"
    )
    if adapter_manifest.get("adapter_id") != adapter_id:
        raise HuggingFacePromotionError("adapter manifest identity disagrees with training report")
    summary = adapter_manifest.get("training_summary")
    if not isinstance(summary, dict) or summary.get("steps") != global_steps:
        raise HuggingFacePromotionError(
            "adapter manifest step count disagrees with training report"
        )

    run_manifest = _load_json_object(output_dir / "run-manifest.json", context="run manifest")
    git = run_manifest.get("git")
    if not isinstance(git, dict):
        raise HuggingFacePromotionError("run manifest git provenance must be an object")
    source_git_sha = _require_str(git, "sha", context="run manifest git provenance")

    base_model = report.get("base_model")
    if not isinstance(base_model, dict):
        raise HuggingFacePromotionError("training report base_model must be an object")
    base_model_repository = _require_str(base_model, "repository", context="base model")
    base_model_revision = _require_str(base_model, "revision", context="base model")

    return PromotionIdentity(
        adapter_id=adapter_id,
        run_id=run_id,
        language=language,
        source_git_sha=source_git_sha,
        source_training_config=source_training_config,
        source_training_config_sha256=source_training_config_sha256,
        dataset_manifest_id=dataset_manifest_id,
        dataset_manifest_sha256=dataset_manifest_sha256,
        base_model_repository=base_model_repository,
        base_model_revision=base_model_revision,
        global_steps=global_steps,
        training_loss=training_loss,
        validation_loss=validation_loss,
        final_checkpoint_dir=final_checkpoint_dir,
    )


def _model_card(identity: PromotionIdentity) -> str:
    return f"""---
base_model: {identity.base_model_repository}
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- sft
- code
- tiny-qwen-coder
---

# Tiny Qwen Coder — {identity.adapter_id}

This repository contains a **LoRA adapter only**. It does not contain merged or full
base-model weights. Load it on top of `{identity.base_model_repository}` at revision
`{identity.base_model_revision}`.

## Provenance

- Adapter ID: `{identity.adapter_id}`
- Training run: `{identity.run_id}`
- Source Git commit: `{identity.source_git_sha}`
- Training config: `{identity.source_training_config}`
- Training config SHA-256: `{identity.source_training_config_sha256}`
- Dataset manifest: `{identity.dataset_manifest_id}`
- Dataset manifest SHA-256: `{identity.dataset_manifest_sha256}`
- Optimizer steps: {identity.global_steps}
- Training loss: {identity.training_loss:.12g}
- Validation loss: {identity.validation_loss:.12g}

The repository includes `archive-manifest.json` plus the training and dataset evidence
needed to audit the adapter. Tiny Qwen Coder's promotion command re-downloaded every
managed file from the exact Hugging Face commit and verified its SHA-256 before any
optional checkpoint cleanup was allowed.
"""


def _copy_archive_source(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise HuggingFacePromotionError(f"archive source is missing or unsafe: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _archive_source_files(output_dir: Path) -> tuple[tuple[Path, str], ...]:
    adapter_dir = output_dir / "adapter"
    allowed_adapter = (
        "adapter_config.json",
        "adapter_model.safetensors",
        "chat_template.jinja",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    explicitly_omitted = {"README.md", "training_args.bin"}
    actual_names = {path.name for path in adapter_dir.iterdir() if path.is_file()}
    unknown = sorted(actual_names - set(allowed_adapter) - explicitly_omitted)
    if unknown:
        raise HuggingFacePromotionError(
            "adapter directory contains unreviewed files; refusing silent omission: "
            + ", ".join(unknown)
        )

    files: list[tuple[Path, str]] = []
    for name in allowed_adapter:
        source = adapter_dir / name
        if source.exists():
            files.append((source, name))
    required_adapter = {"adapter_config.json", "adapter_model.safetensors"}
    if not required_adapter.issubset({destination for _, destination in files}):
        raise HuggingFacePromotionError("adapter archive is missing required PEFT files")

    evidence_names = (
        "adapter-manifest.json",
        "dataset-manifest.json",
        "run-manifest.json",
        "training-config.json",
        "training-metrics.jsonl",
        "training-preflight.json",
        "training-report.json",
    )
    for name in evidence_names:
        files.append((output_dir / name, name))
    return tuple(files)


def _archive_manifest_payload(
    identity: PromotionIdentity,
    files: tuple[ArchiveFileDigest, ...],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "adapter_id": identity.adapter_id,
        "run_id": identity.run_id,
        "language": identity.language,
        "source_git_sha": identity.source_git_sha,
        "source_training_config": identity.source_training_config,
        "source_training_config_sha256": identity.source_training_config_sha256,
        "dataset_manifest_id": identity.dataset_manifest_id,
        "dataset_manifest_sha256": identity.dataset_manifest_sha256,
        "base_model": {
            "repository": identity.base_model_repository,
            "revision": identity.base_model_revision,
        },
        "global_steps": identity.global_steps,
        "training_loss": identity.training_loss,
        "validation_loss": identity.validation_loss,
        "files": [asdict(item) for item in files],
    }


def _verify_archive_exact(archive_dir: Path, expected: tuple[ArchiveFileDigest, ...]) -> None:
    if archive_dir.is_symlink() or not archive_dir.is_dir():
        raise HuggingFacePromotionError(f"archive directory is missing or unsafe: {archive_dir}")
    actual_paths = {
        path.relative_to(archive_dir).as_posix()
        for path in archive_dir.rglob("*")
        if path.is_file() and path.name != "huggingface-promotion.json"
    }
    expected_paths = {item.path for item in expected}
    if actual_paths != expected_paths:
        raise HuggingFacePromotionError("existing local archive file set does not match promotion")
    for item in expected:
        path = archive_dir / item.path
        if path.is_symlink() or path.stat().st_size != item.size_bytes:
            raise HuggingFacePromotionError(f"local archive size mismatch: {item.path}")
        if _sha256_file(path) != item.sha256:
            raise HuggingFacePromotionError(f"local archive SHA-256 mismatch: {item.path}")


def _build_local_archive(
    output_dir: Path,
    archive_dir: Path,
    identity: PromotionIdentity,
) -> tuple[ArchiveFileDigest, ...]:
    output_resolved = output_dir.resolve()
    archive_resolved = archive_dir.resolve()
    try:
        archive_resolved.relative_to(output_resolved)
    except ValueError:
        pass
    else:
        raise HuggingFacePromotionError(
            "local archive must be outside the disposable training output directory"
        )

    archive_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{archive_dir.name}.staging-", dir=archive_dir.parent))
    try:
        for source, destination in _archive_source_files(output_dir):
            _copy_archive_source(source, staging / destination)
        (staging / "README.md").write_text(_model_card(identity), encoding="utf-8")

        files_without_manifest = tuple(
            _digest_file(staging, path) for path in sorted(staging.rglob("*")) if path.is_file()
        )
        manifest_payload = _archive_manifest_payload(identity, files_without_manifest)
        (staging / "archive-manifest.json").write_text(
            json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        expected = tuple(
            _digest_file(staging, path) for path in sorted(staging.rglob("*")) if path.is_file()
        )

        if archive_dir.exists():
            _verify_archive_exact(archive_dir, expected)
            shutil.rmtree(staging)
        else:
            os.replace(staging, archive_dir)
            _verify_archive_exact(archive_dir, expected)
        return expected
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


def _verify_remote_commit(
    hub: PromotionHub,
    *,
    repo_id: str,
    revision: str,
    expected: tuple[ArchiveFileDigest, ...],
) -> None:
    with tempfile.TemporaryDirectory(prefix="tqc-hf-verify-") as tmp:
        destination = Path(tmp)
        for item in expected:
            downloaded = hub.download_file(
                repo_id,
                revision=revision,
                filename=item.path,
                destination_dir=destination,
            )
            if downloaded.is_symlink() or not downloaded.is_file():
                raise HuggingFacePromotionError(
                    f"verified Hub download is missing or unsafe: {item.path}"
                )
            if downloaded.stat().st_size != item.size_bytes:
                raise HuggingFacePromotionError(f"remote size mismatch: {item.path}")
            if _sha256_file(downloaded) != item.sha256:
                raise HuggingFacePromotionError(f"remote SHA-256 mismatch: {item.path}")


def _write_promotion_report(
    archive_dir: Path,
    *,
    repo_id: str,
    revision: str,
    identity: PromotionIdentity,
    expected: tuple[ArchiveFileDigest, ...],
    checkpoints_deleted: bool,
) -> HuggingFacePromotionReport:
    manifest_digest = next(
        (item.sha256 for item in expected if item.path == "archive-manifest.json"), None
    )
    if manifest_digest is None:
        raise HuggingFacePromotionError("archive manifest is missing from verified file set")
    report = HuggingFacePromotionReport(
        schema_version=1,
        repo_id=repo_id,
        revision=revision,
        adapter_id=identity.adapter_id,
        run_id=identity.run_id,
        source_git_sha=identity.source_git_sha,
        archive_dir=str(archive_dir.resolve()),
        archive_manifest_sha256=manifest_digest,
        verified_files=expected,
        checkpoints_deleted=checkpoints_deleted,
        verified_at_utc=datetime.now(UTC).isoformat(),
    )
    (archive_dir / "huggingface-promotion.json").write_text(
        json.dumps(asdict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def promote_completed_adapter(
    output_dir: Path,
    *,
    archive_dir: Path,
    repo_id: str,
    hub: PromotionHub,
    create_private_repo: bool = False,
    delete_checkpoints_after_verify: bool = False,
) -> HuggingFacePromotionReport:
    """Archive, upload, re-download, verify, and optionally clean checkpoints."""

    if not repo_id.strip() or "/" not in repo_id:
        raise HuggingFacePromotionError("repo_id must be a namespaced Hugging Face model repo")
    identity = _validate_training_output(output_dir)
    expected = _build_local_archive(output_dir, archive_dir, identity)
    manifest = archive_dir / "archive-manifest.json"
    parent_commit = hub.prepare_repo(
        repo_id,
        adapter_id=identity.adapter_id,
        archive_manifest=manifest,
        create_private_repo=create_private_repo,
    )
    revision = hub.commit_files(
        repo_id,
        archive_dir=archive_dir,
        files=expected,
        parent_commit=parent_commit,
        commit_message=f"Archive {identity.adapter_id} run {identity.run_id}",
    )
    _verify_remote_commit(
        hub,
        repo_id=repo_id,
        revision=revision,
        expected=expected,
    )

    checkpoints_deleted = False
    if delete_checkpoints_after_verify:
        checkpoint_root = output_dir / Path(identity.final_checkpoint_dir).parts[0]
        checkpoint_root = checkpoint_root.resolve()
        output_root = output_dir.resolve()
        try:
            checkpoint_root.relative_to(output_root)
        except ValueError as exc:
            raise HuggingFacePromotionError(
                "checkpoint cleanup target escapes output root"
            ) from exc
        if checkpoint_root.name != "checkpoints" or not checkpoint_root.is_dir():
            raise HuggingFacePromotionError("checkpoint cleanup target is invalid")
        shutil.rmtree(checkpoint_root)
        checkpoints_deleted = True

    return _write_promotion_report(
        archive_dir,
        repo_id=repo_id,
        revision=revision,
        identity=identity,
        expected=expected,
        checkpoints_deleted=checkpoints_deleted,
    )


def _token_from_environment(name: str) -> str:
    if not name or not name.replace("_", "").isalnum():
        raise HuggingFacePromotionError("token environment variable name is invalid")
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise HuggingFacePromotionError(
            f"required Hugging Face token environment variable {name} is not set"
        )
    return value


def promotion_report_json(report: HuggingFacePromotionReport) -> str:
    """Serialize verified promotion evidence."""

    return json.dumps(asdict(report), indent=2, sort_keys=True) + "\n"


def huggingface_promotion_main(argv: list[str] | None = None) -> NoReturn:
    """CLI entry point for fail-closed Hugging Face adapter promotion."""

    parser = argparse.ArgumentParser(
        description="Archive and verify a completed Tiny Qwen Coder LoRA adapter on Hugging Face"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--create-private-repo", action="store_true")
    parser.add_argument("--delete-checkpoints-after-verify", action="store_true")
    args = parser.parse_args(argv)

    try:
        token = _token_from_environment(args.token_env)
        report = promote_completed_adapter(
            args.output_dir,
            archive_dir=args.archive_dir,
            repo_id=args.repo_id,
            hub=HuggingFaceHub(token=token),
            create_private_repo=args.create_private_repo,
            delete_checkpoints_after_verify=args.delete_checkpoints_after_verify,
        )
    except HuggingFacePromotionError as exc:
        parser.exit(2, f"error: {exc}\n")
    print(promotion_report_json(report), end="")
    raise SystemExit(0)


if __name__ == "__main__":  # pragma: no cover
    huggingface_promotion_main()
