"""Regression tests for fail-closed Hugging Face adapter archival."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from tiny_qwen_coder.huggingface_promotion import (
    ArchiveFileDigest,
    HuggingFacePromotionError,
    PromotionHub,
    promote_completed_adapter,
)


class FakeHub(PromotionHub):
    def __init__(self, *, corrupt_download: str | None = None) -> None:
        self.corrupt_download = corrupt_download
        self.remote: dict[str, bytes] = {}
        self.commit_calls = 0
        self.create_private_repo = False

    def prepare_repo(
        self,
        repo_id: str,
        *,
        adapter_id: str,
        archive_manifest: Path,
        create_private_repo: bool,
    ) -> str | None:
        del repo_id, adapter_id, archive_manifest
        self.create_private_repo = create_private_repo
        return "parent-commit"

    def commit_files(
        self,
        repo_id: str,
        *,
        archive_dir: Path,
        files: tuple[ArchiveFileDigest, ...],
        parent_commit: str | None,
        commit_message: str,
    ) -> str:
        del repo_id, parent_commit, commit_message
        self.commit_calls += 1
        self.remote = {item.path: (archive_dir / item.path).read_bytes() for item in files}
        return "0123456789abcdef0123456789abcdef01234567"

    def download_file(
        self,
        repo_id: str,
        *,
        revision: str,
        filename: str,
        destination_dir: Path,
    ) -> Path:
        del repo_id, revision
        path = destination_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        content = self.remote[filename]
        if filename == self.corrupt_download:
            content += b"corrupt"
        path.write_bytes(content)
        return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _completed_output(root: Path) -> Path:
    output = root / "artifacts" / "train" / "python" / "p0"
    adapter = output / "adapter"
    checkpoint = output / "checkpoints" / "checkpoint-12"
    adapter.mkdir(parents=True)
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text("{}\n", encoding="utf-8")

    _write_json(adapter / "adapter_config.json", {"peft_type": "LORA"})
    (adapter / "adapter_model.safetensors").write_bytes(b"safe-adapter")
    (adapter / "chat_template.jinja").write_text("{{ messages }}\n", encoding="utf-8")
    (adapter / "tokenizer.json").write_text('{"version":"1.0"}\n', encoding="utf-8")
    _write_json(adapter / "tokenizer_config.json", {"model_max_length": 2048})
    (adapter / "training_args.bin").write_bytes(b"pickle-like-training-state")
    (adapter / "README.md").write_text("generic generated card\n", encoding="utf-8")

    _write_json(
        output / "adapter-manifest.json",
        {"adapter_id": "language/python/p0", "training_summary": {"steps": 12}},
    )
    _write_json(output / "dataset-manifest.json", {"manifest_id": "python-p0-v1"})
    _write_json(output / "run-manifest.json", {"git": {"sha": "a" * 40}})
    _write_json(output / "training-config.json", {"config_sha256": "b" * 64})
    (output / "training-metrics.jsonl").write_text('{"loss":1.0}\n', encoding="utf-8")
    _write_json(output / "training-preflight.json", {"status": "passed"})

    persisted_paths = (
        adapter / "adapter_config.json",
        adapter / "adapter_model.safetensors",
        adapter / "chat_template.jinja",
        adapter / "tokenizer.json",
        adapter / "tokenizer_config.json",
        adapter / "training_args.bin",
        adapter / "README.md",
        output / "adapter-manifest.json",
        output / "dataset-manifest.json",
        output / "run-manifest.json",
        output / "training-config.json",
        output / "training-metrics.jsonl",
        output / "training-preflight.json",
    )
    persisted = [
        {
            "path": path.relative_to(output).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(persisted_paths, key=lambda item: item.relative_to(output).as_posix())
    ]
    set_payload = json.dumps(persisted, sort_keys=True, separators=(",", ":"))
    report = {
        "schema_version": 1,
        "adapter_id": "language/python/p0",
        "run_id": "training-python-test",
        "language": "python",
        "global_steps": 12,
        "train_records": 100,
        "validation_records": 10,
        "training_loss": 1.0,
        "validation_loss": 1.1,
        "source_training_config": "configs/train/python/p0.yaml",
        "source_training_config_sha256": "b" * 64,
        "dataset_manifest_id": "python-p0-v1",
        "dataset_manifest_sha256": "c" * 64,
        "final_checkpoint_dir": "checkpoints/checkpoint-12",
        "adapter_dir": "adapter",
        "base_model": {"repository": "Qwen/Qwen3.5-4B", "revision": "d" * 40},
        "persisted_artifacts": persisted,
        "artifact_set_sha256": hashlib.sha256(set_payload.encode("utf-8")).hexdigest(),
    }
    _write_json(output / "training-report.json", report)
    return output


def test_promotion_archives_bounded_safe_file_set_and_verifies_remote(tmp_path: Path) -> None:
    output = _completed_output(tmp_path)
    archive = tmp_path / "durable" / "python-p0"
    hub = FakeHub()

    report = promote_completed_adapter(
        output,
        archive_dir=archive,
        repo_id="user/tiny-qwen-coder-python-p0",
        hub=hub,
        create_private_repo=True,
    )

    assert hub.commit_calls == 1
    assert hub.create_private_repo is True
    assert report.revision == "0123456789abcdef0123456789abcdef01234567"
    assert (archive / "adapter_model.safetensors").read_bytes() == b"safe-adapter"
    assert (archive / "archive-manifest.json").is_file()
    assert (archive / "huggingface-promotion.json").is_file()
    assert "training_args.bin" not in hub.remote
    assert "checkpoints/checkpoint-12/trainer_state.json" not in hub.remote
    assert "README.md" in hub.remote
    assert "adapter_model.safetensors" in hub.remote
    assert "dataset-manifest.json" in hub.remote
    assert (output / "checkpoints" / "checkpoint-12").is_dir()


def test_checkpoint_cleanup_occurs_only_after_exact_remote_verification(tmp_path: Path) -> None:
    output = _completed_output(tmp_path)
    archive = tmp_path / "durable" / "python-p0"
    hub = FakeHub(corrupt_download="adapter_model.safetensors")

    with pytest.raises(HuggingFacePromotionError, match="remote (size|SHA-256) mismatch"):
        promote_completed_adapter(
            output,
            archive_dir=archive,
            repo_id="user/tiny-qwen-coder-python-p0",
            hub=hub,
            delete_checkpoints_after_verify=True,
        )

    assert (output / "checkpoints" / "checkpoint-12").is_dir()
    assert not (archive / "huggingface-promotion.json").exists()


def test_verified_promotion_may_delete_only_checkpoint_tree(tmp_path: Path) -> None:
    output = _completed_output(tmp_path)
    archive = tmp_path / "durable" / "python-p0"

    report = promote_completed_adapter(
        output,
        archive_dir=archive,
        repo_id="user/tiny-qwen-coder-python-p0",
        hub=FakeHub(),
        delete_checkpoints_after_verify=True,
    )

    assert report.checkpoints_deleted is True
    assert not (output / "checkpoints").exists()
    assert (output / "adapter" / "adapter_model.safetensors").is_file()
    assert (archive / "adapter_model.safetensors").is_file()


def test_promotion_rejects_tampered_training_evidence_before_upload(tmp_path: Path) -> None:
    output = _completed_output(tmp_path)
    (output / "adapter" / "adapter_model.safetensors").write_bytes(b"tampered")
    hub = FakeHub()

    with pytest.raises(
        HuggingFacePromotionError, match="persisted artifact (size|SHA-256) mismatch"
    ):
        promote_completed_adapter(
            output,
            archive_dir=tmp_path / "durable" / "python-p0",
            repo_id="user/tiny-qwen-coder-python-p0",
            hub=hub,
        )

    assert hub.commit_calls == 0


def test_promotion_rejects_unreviewed_adapter_files_instead_of_silently_omitting(
    tmp_path: Path,
) -> None:
    output = _completed_output(tmp_path)
    rogue = output / "adapter" / "mystery.bin"
    rogue.write_bytes(b"unknown")

    with pytest.raises(HuggingFacePromotionError, match="unreviewed files"):
        promote_completed_adapter(
            output,
            archive_dir=tmp_path / "durable" / "python-p0",
            repo_id="user/tiny-qwen-coder-python-p0",
            hub=FakeHub(),
        )

    rogue.unlink()
    shutil.rmtree(output)
