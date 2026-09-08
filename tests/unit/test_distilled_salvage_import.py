from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest
import yaml

import tiny_qwen_coder.distilled_salvage as salvage
from tiny_qwen_coder.distilled_salvage import (
    DistilledSalvageImportError,
    import_repaired_v4_2000_corpus,
)


def _record(index: int, *, marker: bool = False) -> dict[str, object]:
    answer = f"print({index})"
    if marker:
        answer = f"<think>hidden</think>\n{answer}"
    return {
        "schema_version": 1,
        "language": "python",
        "messages": [
            {"role": "user", "content": f"Return {index}."},
            {"role": "assistant", "content": answer},
        ],
        "provenance": {
            "source_id": "fixture",
            "revision": "fixture-revision",
            "record_id": str(index),
            "license": {"name": "MIT", "url": None, "attribution": None},
            "split": "train",
            "url": None,
            "source_metadata": [],
        },
        "validation": None,
    }


def _write_jsonl(path: Path, records: tuple[dict[str, object], ...]) -> None:
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def _write_sidecar(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".sha256").write_text(f"{digest}  {path.name}\n", encoding="ascii")


def _build_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    marker: bool = False,
    tamper_train: bool = False,
) -> Path:
    root = tmp_path / salvage._EXPECTED_ROOT
    final = root / "final"
    source_identity = root / "source-identity"
    final.mkdir(parents=True)
    source_identity.mkdir(parents=True)

    accepted = (_record(0, marker=marker), _record(1))
    train = (accepted[0],)
    validation = (accepted[1],)
    accepted_fingerprints = tuple(
        salvage._record_fingerprint(record, context=f"accepted[{index}]")
        for index, record in enumerate(accepted)
    )
    train_fingerprints = tuple(
        salvage._record_fingerprint(record, context=f"train[{index}]")
        for index, record in enumerate(train)
    )
    validation_fingerprints = tuple(
        salvage._record_fingerprint(record, context=f"validation[{index}]")
        for index, record in enumerate(validation)
    )
    memberships = []
    train_hashes = {item[2] for item in train_fingerprints}
    for index, fingerprint in enumerate(accepted_fingerprints):
        prompt_sha256, _, record_sha256, source_id, source_record_id = fingerprint
        memberships.append(
            {
                "partition": "train" if record_sha256 in train_hashes else "validation",
                "prompt_sha256": prompt_sha256,
                "record_sha256": record_sha256,
                "source_id": source_id,
                "source_record_id": source_record_id,
                "unique_index": index,
            }
        )
    manifest = {
        "schema_version": 1,
        "language": "python",
        "seed": 1729,
        "counts": {
            "deduplicated_unique": 2,
            "train_records": 1,
            "validation_records": 1,
        },
        "checksums": {
            "unique_corpus_sha256": salvage._canonical_sha256(
                tuple(item[2] for item in accepted_fingerprints)
            ),
            "train_records_sha256": salvage._canonical_sha256(
                tuple(item[2] for item in train_fingerprints)
            ),
            "validation_records_sha256": salvage._canonical_sha256(
                tuple(item[2] for item in validation_fingerprints)
            ),
            "split_membership_sha256": salvage._canonical_sha256(memberships),
        },
        "memberships": memberships,
        "contamination": {"status": "clean", "findings": []},
    }
    manifest_path = final / "dataset-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_sidecar(manifest_path)
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    _write_jsonl(final / "accepted.jsonl", accepted)
    _write_jsonl(final / "train.jsonl", train)
    _write_jsonl(final / "validation.jsonl", validation)
    if tamper_train:
        _write_jsonl(final / "train.jsonl", (_record(99),))

    (final / "teacher-finalization.json").write_text(
        json.dumps(
            {
                "prepared_unique": 2,
                "train_records": 1,
                "validation_records": 1,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "qualification.json").write_text(
        json.dumps(
            {
                "qualified": True,
                "failed_gates": [],
                "contamination_status": "clean",
                "total_records": 2,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    source_output_sha256 = "a" * 64
    (source_identity / "run-identity.json").write_text(
        json.dumps(
            {
                "policy_id": salvage._EXPECTED_POLICY_ID,
                "output_sha256": source_output_sha256,
                "total_records": 2,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(salvage, "_EXPECTED_MANIFEST_SHA256", manifest_sha256)
    monkeypatch.setattr(salvage, "_EXPECTED_SOURCE_OUTPUT_SHA256", source_output_sha256)
    monkeypatch.setattr(salvage, "_EXPECTED_TOTAL_RECORDS", 2)
    monkeypatch.setattr(salvage, "_EXPECTED_ACCEPTED_RECORDS", 2)
    monkeypatch.setattr(salvage, "_EXPECTED_TRAIN_RECORDS", 1)
    monkeypatch.setattr(salvage, "_EXPECTED_VALIDATION_RECORDS", 1)

    archive_path = tmp_path / "salvage.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(tmp_path).as_posix())
    return archive_path


def test_import_repaired_v4_2000_corpus_accepts_verified_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _build_archive(tmp_path, monkeypatch)
    output = tmp_path / "imported"

    receipt = import_repaired_v4_2000_corpus(archive, output_dir=output)

    assert receipt.accepted_records == 2
    assert receipt.train_records == 1
    assert receipt.validation_records == 1
    assert receipt.contamination_status == "clean"
    assert receipt.reasoning_marker_hits == 0
    assert (output / "import-receipt.json").is_file()
    assert (output / "train.jsonl").is_file()
    assert (output / "validation.jsonl").is_file()
    with pytest.raises(DistilledSalvageImportError, match="refusing to mutate frozen data"):
        import_repaired_v4_2000_corpus(archive, output_dir=output)


def test_import_rejects_hidden_reasoning_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _build_archive(tmp_path, monkeypatch, marker=True)

    with pytest.raises(DistilledSalvageImportError, match="hidden-thinking marker"):
        import_repaired_v4_2000_corpus(archive, output_dir=tmp_path / "imported")


def test_import_rejects_split_content_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _build_archive(tmp_path, monkeypatch, tamper_train=True)

    with pytest.raises(DistilledSalvageImportError, match="accepted corpus"):
        import_repaired_v4_2000_corpus(archive, output_dir=tmp_path / "imported")


def test_p9_007_config_and_evidence_pin_the_frozen_corpus() -> None:
    config_path = Path("configs/train/python/p9_distilled_v4_2000_r8_lr1e5.yaml")
    smoke_path = Path("configs/train/python/p9_distilled_v4_2000_r8_lr1e5_smoke.yaml")
    evidence_path = Path("docs/evidence/P9_007_DISTILLED_V4_2000_CORPUS.json")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    smoke = yaml.safe_load(smoke_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert isinstance(config, dict)
    assert isinstance(smoke, dict)
    assert isinstance(evidence, dict)

    dataset_root = "data/python/qwen38-27b-v4-2000-salvage-v1"
    assert config["dataset_manifest"] == f"{dataset_root}/dataset-manifest.json"
    assert config["train_records"] == f"{dataset_root}/train.jsonl"
    assert config["validation_records"] == f"{dataset_root}/validation.jsonl"
    assert config["training_mode"] == "qlora_4bit"
    assert config["learning_rate"] == 0.00001
    assert config["lora"]["rank"] == 8
    assert smoke["training_config"] == str(config_path)
    assert evidence["dataset_manifest_sha256"] == salvage._EXPECTED_MANIFEST_SHA256
    assert evidence["source_output_sha256"] == salvage._EXPECTED_SOURCE_OUTPUT_SHA256
    assert evidence["final_counts"] == {
        "accepted_records": salvage._EXPECTED_ACCEPTED_RECORDS,
        "train_records": salvage._EXPECTED_TRAIN_RECORDS,
        "validation_records": salvage._EXPECTED_VALIDATION_RECORDS,
    }
