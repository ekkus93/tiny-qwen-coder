"""Frozen-dataset integrity checks for adapter-training preflight."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from tiny_qwen_coder.data import NormalizedTrainingRecord, load_normalized_training_records_jsonl
from tiny_qwen_coder.data.deduplication import normalized_record_fingerprint
from tiny_qwen_coder.training.plan import AdapterTrainingError, AdapterTrainingPlan


@dataclass(frozen=True, slots=True)
class DatasetPreflightEvidence:
    """Verified frozen-dataset evidence required before training."""

    manifest_sha256: str
    sidecar_path: str
    train_records: int
    validation_records: int
    train_content_sha256: str
    validation_content_sha256: str
    split_overlap_records: int


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise AdapterTrainingError(f"{context} must be a mapping")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise AdapterTrainingError(f"{context} keys must be strings")
        result[key] = item
    return result


def _int(mapping: dict[str, object], key: str, *, context: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AdapterTrainingError(f"{context}.{key} must be a non-negative integer")
    return value


def _sha(mapping: dict[str, object], key: str, *, context: str) -> str:
    value = mapping.get(key)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AdapterTrainingError(f"{context}.{key} must be a lowercase SHA-256")
    return value


def _ordered_content_sha256(
    records: tuple[NormalizedTrainingRecord, ...],
) -> tuple[str, tuple[str, ...]]:
    hashes = tuple(normalized_record_fingerprint(record).record_sha256 for record in records)
    return _canonical_sha256(hashes), hashes


def _require_manifest_sidecar(manifest_path: Path, manifest_sha256: str) -> Path:
    sidecar = manifest_path.with_suffix(".sha256")
    try:
        text = sidecar.read_text(encoding="ascii")
    except OSError as exc:
        raise AdapterTrainingError(
            f"could not read dataset manifest checksum {sidecar}: {exc}"
        ) from exc
    expected = f"{manifest_sha256}  {manifest_path.name}\n"
    if text != expected:
        raise AdapterTrainingError(
            "dataset manifest checksum sidecar does not match the configured manifest"
        )
    return sidecar


def verify_frozen_training_dataset(plan: AdapterTrainingPlan) -> DatasetPreflightEvidence:
    """Verify frozen manifest, sidecar, split counts/checksums, and split isolation."""

    manifest_path = Path(plan.config.dataset_manifest)
    try:
        raw_bytes = manifest_path.read_bytes()
        raw: object = json.loads(raw_bytes)
    except OSError as exc:
        raise AdapterTrainingError(
            f"could not read dataset manifest {manifest_path}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise AdapterTrainingError(
            f"dataset manifest {manifest_path} is invalid JSON: {exc}"
        ) from exc

    manifest_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if manifest_sha256 != plan.dataset.sha256:
        raise AdapterTrainingError("dataset manifest changed after training-plan resolution")
    sidecar = _require_manifest_sidecar(manifest_path, manifest_sha256)

    root = _mapping(raw, context="dataset manifest")
    counts = _mapping(root.get("counts"), context="dataset manifest.counts")
    checksums = _mapping(root.get("checksums"), context="dataset manifest.checksums")
    manifest_seed = root.get("seed")
    if isinstance(manifest_seed, bool) or not isinstance(manifest_seed, int):
        raise AdapterTrainingError("dataset manifest.seed must be an integer")
    if manifest_seed != plan.config.seed:
        raise AdapterTrainingError("dataset manifest seed does not match training seed")

    expected_train_count = _int(counts, "train_records", context="dataset manifest.counts")
    expected_validation_count = _int(
        counts, "validation_records", context="dataset manifest.counts"
    )
    expected_train_sha = _sha(
        checksums, "train_content_sha256", context="dataset manifest.checksums"
    )
    expected_validation_sha = _sha(
        checksums, "validation_content_sha256", context="dataset manifest.checksums"
    )

    train_records = load_normalized_training_records_jsonl(
        plan.train_records,
        expected_language=plan.language,
    )
    validation_records = load_normalized_training_records_jsonl(
        plan.validation_records,
        expected_language=plan.language,
    )
    if len(train_records) != expected_train_count:
        raise AdapterTrainingError(
            f"train record count does not match frozen manifest: expected {expected_train_count}, "
            f"got {len(train_records)}"
        )
    if len(validation_records) != expected_validation_count:
        raise AdapterTrainingError(
            "validation record count does not match frozen manifest: "
            f"expected {expected_validation_count}, got {len(validation_records)}"
        )
    if not train_records or not validation_records:
        raise AdapterTrainingError("training and validation splits must both be non-empty")

    train_sha, train_hashes = _ordered_content_sha256(train_records)
    validation_sha, validation_hashes = _ordered_content_sha256(validation_records)
    if train_sha != expected_train_sha:
        raise AdapterTrainingError("train record content checksum does not match frozen manifest")
    if validation_sha != expected_validation_sha:
        raise AdapterTrainingError(
            "validation record content checksum does not match frozen manifest"
        )

    overlap = len(set(train_hashes) & set(validation_hashes))
    if overlap:
        raise AdapterTrainingError(
            f"train/validation splits share {overlap} exact normalized record(s)"
        )
    return DatasetPreflightEvidence(
        manifest_sha256=manifest_sha256,
        sidecar_path=str(sidecar),
        train_records=len(train_records),
        validation_records=len(validation_records),
        train_content_sha256=train_sha,
        validation_content_sha256=validation_sha,
        split_overlap_records=overlap,
    )
