from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from transformers import PreTrainedTokenizerBase

import tiny_qwen_coder.training.preflight_loss as preflight_loss
from tiny_qwen_coder.adapters import load_frozen_selective_lora_target_profile
from tiny_qwen_coder.config import LoraConfig, QuantizationConfig
from tiny_qwen_coder.data import (
    LicenseMetadata,
    NormalizedTrainingRecord,
    SourceProvenance,
    TrainingMessage,
)
from tiny_qwen_coder.data.deduplication import normalized_record_fingerprint
from tiny_qwen_coder.model import InspectionTarget
from tiny_qwen_coder.training.plan import (
    AdapterTrainingConfig,
    AdapterTrainingError,
    AdapterTrainingPlan,
    TrainerArtifactPaths,
    TrainingDatasetIdentity,
)
from tiny_qwen_coder.training.preflight_dataset import verify_frozen_training_dataset
from tiny_qwen_coder.training.preflight_hardware import (
    HardwareSnapshot,
    verify_training_hardware,
)
from tiny_qwen_coder.training.preflight_loss import verify_training_loss_mask
from tiny_qwen_coder.training.preflight_output import verify_training_output_path
from tiny_qwen_coder.training.preflight_targets import verify_frozen_lora_targets

_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
_TEMPLATE = "fixture-chat-template"
_TEMPLATE_SHA = hashlib.sha256(_TEMPLATE.encode()).hexdigest()


def _record(text: str) -> NormalizedTrainingRecord:
    return NormalizedTrainingRecord(
        schema_version=1,
        messages=(
            TrainingMessage(role="user", content=f"prompt {text}"),
            TrainingMessage(role="assistant", content=f"answer {text}"),
        ),
        language="python",
        provenance=SourceProvenance(
            source_id="fixture",
            revision="revision-1",
            license=LicenseMetadata(name="MIT"),
        ),
    )


def _content_sha(records: tuple[NormalizedTrainingRecord, ...]) -> str:
    hashes = tuple(normalized_record_fingerprint(record).record_sha256 for record in records)
    payload = json.dumps(hashes, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode(
        "ascii"
    )
    return hashlib.sha256(payload).hexdigest()


def _write_records(path: Path, records: tuple[NormalizedTrainingRecord, ...]) -> None:
    path.write_text(
        "".join(json.dumps(asdict(record), sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _config(
    *, output_dir: str, train: Path, validation: Path, manifest: Path
) -> AdapterTrainingConfig:
    profile = load_frozen_selective_lora_target_profile()
    return AdapterTrainingConfig(
        schema_version=1,
        base_config="configs/base/qwen35-4b.yaml",
        language="python",
        dataset_manifest=str(manifest),
        output_dir=output_dir,
        seed=1729,
        training_mode="qlora_4bit",
        compute_dtype="bfloat16",
        sequence_length=2048,
        micro_batch_size=1,
        gradient_accumulation_steps=8,
        epochs=1.0,
        learning_rate=0.0002,
        scheduler="cosine",
        warmup_ratio=0.03,
        gradient_checkpointing=True,
        loss_mode="assistant_only",
        lora=LoraConfig(
            rank=16,
            alpha=32,
            dropout=0.05,
            bias="none",
            target_strategy="selective",
            target_modules=profile.target_modules,
        ),
        quantization=QuantizationConfig(
            bits=4,
            quant_type="nf4",
            double_quant=True,
            compute_dtype="bfloat16",
        ),
        adapter_family="language",
        adapter_id="language/python/p0",
        train_records=str(train),
        validation_records=str(validation),
    )


def _plan(
    tmp_path: Path,
    *,
    output_dir: str | None = None,
    shared_record: bool = False,
) -> AdapterTrainingPlan:
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    manifest_path = tmp_path / "dataset-manifest.json"
    train_records = (_record("train"),)
    validation_records = (train_records[0],) if shared_record else (_record("validation"),)
    _write_records(train_path, train_records)
    _write_records(validation_path, validation_records)
    manifest_payload = {
        "schema_version": 1,
        "manifest_id": "dataset/python/p0",
        "language": "python",
        "seed": 1729,
        "tokenizer": {
            "repository": "Qwen/Qwen3.5-4B",
            "revision": _REVISION,
            "chat_template_sha256": _TEMPLATE_SHA,
        },
        "counts": {"train_records": 1, "validation_records": 1},
        "checksums": {
            "train_content_sha256": _content_sha(train_records),
            "validation_content_sha256": _content_sha(validation_records),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    manifest_path.with_suffix(".sha256").write_text(
        f"{manifest_sha}  dataset-manifest.json\n", encoding="ascii"
    )
    selected_output = output_dir or str(tmp_path / "repo" / "artifacts" / "train" / "python-p0")
    config = _config(
        output_dir=selected_output,
        train=train_path,
        validation=validation_path,
        manifest=manifest_path,
    )
    artifacts_dir = Path(selected_output)
    return AdapterTrainingPlan(
        config_path=tmp_path / "training.yaml",
        config=config,
        config_sha256="c" * 64,
        language="python",
        target=InspectionTarget(
            config_id="qwen35-4b",
            model_repository="Qwen/Qwen3.5-4B",
            model_revision=_REVISION,
            tokenizer_repository="Qwen/Qwen3.5-4B",
            tokenizer_revision=_REVISION,
            model_load_dtype="bfloat16",
        ),
        dataset=TrainingDatasetIdentity(
            manifest_id="dataset/python/p0",
            language="python",
            tokenizer_repository="Qwen/Qwen3.5-4B",
            tokenizer_revision=_REVISION,
            chat_template_sha256=_TEMPLATE_SHA,
            sha256=manifest_sha,
        ),
        train_records=train_path,
        validation_records=validation_path,
        artifacts=TrainerArtifactPaths(
            output_dir=artifacts_dir,
            checkpoints=artifacts_dir / "checkpoints",
            adapter=artifacts_dir / "adapter",
            dataset_manifest=artifacts_dir / "dataset-manifest.json",
            training_config=artifacts_dir / "training-config.json",
            training_metrics=artifacts_dir / "training-metrics.jsonl",
            run_manifest=artifacts_dir / "run-manifest.json",
            adapter_manifest=artifacts_dir / "adapter-manifest.json",
        ),
    )


def test_dataset_preflight_verifies_manifest_sidecar_and_split_content(tmp_path: Path) -> None:
    evidence = verify_frozen_training_dataset(_plan(tmp_path))

    assert evidence.train_records == 1
    assert evidence.validation_records == 1
    assert evidence.split_overlap_records == 0


def test_dataset_preflight_rejects_manifest_sidecar_drift(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    Path(plan.config.dataset_manifest).with_suffix(".sha256").write_text(
        f"{'0' * 64}  dataset-manifest.json\n", encoding="ascii"
    )

    with pytest.raises(AdapterTrainingError, match="checksum sidecar"):
        verify_frozen_training_dataset(plan)


def test_dataset_preflight_rejects_tampered_records(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    plan.train_records.write_text(
        plan.train_records.read_text(encoding="utf-8").replace("answer train", "tampered"),
        encoding="utf-8",
    )

    with pytest.raises(AdapterTrainingError, match="content checksum"):
        verify_frozen_training_dataset(plan)


def test_dataset_preflight_rejects_cross_split_exact_duplicates(tmp_path: Path) -> None:
    plan = _plan(tmp_path, shared_record=True)

    with pytest.raises(AdapterTrainingError, match="share 1 exact normalized record"):
        verify_frozen_training_dataset(plan)


def test_target_preflight_binds_config_to_frozen_architecture(tmp_path: Path) -> None:
    evidence = verify_frozen_lora_targets(_plan(tmp_path))

    assert evidence.base_revision == _REVISION
    assert evidence.rank == 16
    assert len(evidence.target_modules) == 12
    assert evidence.measured_trainable_parameters == 32464896


def test_target_preflight_rejects_target_drift(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    changed_lora = replace(plan.config.lora, target_modules=("q_proj",))
    changed_config = replace(plan.config, lora=changed_lora)

    with pytest.raises(AdapterTrainingError, match="configured LoRA targets"):
        verify_frozen_lora_targets(replace(plan, config=changed_config))


def test_output_preflight_requires_fresh_artifacts_train_child(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    plan = _plan(tmp_path, output_dir=str(repo / "artifacts" / "train" / "python-p0"))

    evidence = verify_training_output_path(plan, repo_root=repo)

    assert evidence.output_dir.endswith("artifacts/train/python-p0")
    Path(evidence.output_dir).mkdir(parents=True)
    with pytest.raises(AdapterTrainingError, match="already exists"):
        verify_training_output_path(plan, repo_root=repo)


def test_output_preflight_rejects_symlinked_parent(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    outside = tmp_path / "outside"
    repo.mkdir()
    outside.mkdir()
    (repo / "artifacts").symlink_to(outside, target_is_directory=True)
    plan = _plan(tmp_path, output_dir=str(repo / "artifacts" / "train" / "run"))

    with pytest.raises(AdapterTrainingError, match="symlink component"):
        verify_training_output_path(plan, repo_root=repo)


class _Probe:
    def __init__(self, snapshot: HardwareSnapshot) -> None:
        self._snapshot = snapshot

    def snapshot(self) -> HardwareSnapshot:
        return self._snapshot


def test_hardware_preflight_reports_qlora_compatibility(tmp_path: Path) -> None:
    evidence = verify_training_hardware(
        _plan(tmp_path),
        probe=_Probe(
            HardwareSnapshot(
                cuda_available=True,
                device_count=1,
                device_name="RTX fixture",
                total_vram_bytes=16 * 1024**3,
                bf16_supported=True,
                bitsandbytes_available=True,
            )
        ),
    )

    assert evidence.training_mode == "qlora_4bit"
    assert evidence.bf16_supported is True
    assert evidence.bitsandbytes_available is True


def test_hardware_preflight_rejects_qlora_without_bitsandbytes(tmp_path: Path) -> None:
    with pytest.raises(AdapterTrainingError, match="bitsandbytes"):
        verify_training_hardware(
            _plan(tmp_path),
            probe=_Probe(
                HardwareSnapshot(
                    cuda_available=True,
                    device_count=1,
                    device_name="RTX fixture",
                    total_vram_bytes=16 * 1024**3,
                    bf16_supported=True,
                    bitsandbytes_available=False,
                )
            ),
        )


class _Tokenizer:
    chat_template = _TEMPLATE


def _tokenizer() -> PreTrainedTokenizerBase:
    return cast(PreTrainedTokenizerBase, _Tokenizer())


def test_loss_preflight_rejects_dataset_template_drift(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    changed_dataset = replace(plan.dataset, chat_template_sha256="0" * 64)

    with pytest.raises(AdapterTrainingError, match="chat template"):
        verify_training_loss_mask(replace(plan, dataset=changed_dataset), tokenizer=_tokenizer())


def test_loss_preflight_requires_true_assistant_only_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        preflight_loss,
        "build_chat_loss_mask_report",
        lambda *args, **kwargs: SimpleNamespace(
            strategy="completion_only_fallback",
            checkpoint_chat_template_sha256=_TEMPLATE_SHA,
            loss_token_count=2,
            ignored_token_count=2,
        ),
    )

    with pytest.raises(AdapterTrainingError, match="TRL assistant-only"):
        verify_training_loss_mask(_plan(tmp_path), tokenizer=_tokenizer())


def test_loss_preflight_accepts_proven_trl_assistant_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        preflight_loss,
        "build_chat_loss_mask_report",
        lambda *args, **kwargs: SimpleNamespace(
            strategy="trl_assistant_only",
            checkpoint_chat_template_sha256=_TEMPLATE_SHA,
            loss_token_count=2,
            ignored_token_count=2,
        ),
    )

    evidence = verify_training_loss_mask(_plan(tmp_path), tokenizer=_tokenizer())

    assert evidence.strategy == "trl_assistant_only"
    assert evidence.loss_token_count == 2
    assert evidence.ignored_token_count == 2
