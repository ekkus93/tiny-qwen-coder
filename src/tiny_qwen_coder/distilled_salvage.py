"""Import and verify the frozen repaired V4-2000 teacher corpus for student training."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

_EXPECTED_ROOT = "qwen38-27b-v4-2000-salvage-v1"
_EXPECTED_MANIFEST_SHA256 = "7e299de17cbb62bff3cf5f50c61ad6ad30ca571c9297b9935ff1307cd52e0eb7"
_EXPECTED_SOURCE_OUTPUT_SHA256 = "7f07f7253e98bf8ed295b12d72ebdf03c44b2e08a8d9f74aaa02dce5b760d966"
_EXPECTED_POLICY_ID = "qwen-closing-think-salvage-v1"
_EXPECTED_TOTAL_RECORDS = 2000
_EXPECTED_ACCEPTED_RECORDS = 1557
_EXPECTED_TRAIN_RECORDS = 1479
_EXPECTED_VALIDATION_RECORDS = 78
_REQUIRED_FINAL_FILES = (
    "accepted.jsonl",
    "dataset-manifest.json",
    "dataset-manifest.sha256",
    "teacher-finalization.json",
    "train.jsonl",
    "validation.jsonl",
)
_REASONING_MARKERS = ("<think>", "</think>")


class DistilledSalvageImportError(RuntimeError):
    """Raised when repaired teacher evidence cannot be trusted for student training."""


@dataclass(frozen=True, slots=True)
class DistilledSalvageImportReceipt:
    """Stable receipt emitted after importing one verified repaired teacher corpus."""

    schema_version: int
    source_archive_sha256: str
    source_root: str
    source_output_sha256: str
    dataset_manifest_sha256: str
    qualification_sha256: str
    teacher_finalization_sha256: str
    accepted_records: int
    train_records: int
    validation_records: int
    contamination_status: str
    reasoning_marker_hits: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object, *, ensure_ascii: bool = True) -> str:
    encoding = "ascii" if ensure_ascii else "utf-8"
    payload = json.dumps(
        value,
        ensure_ascii=ensure_ascii,
        separators=(",", ":"),
        sort_keys=True,
    ).encode(encoding)
    return hashlib.sha256(payload).hexdigest()


def _json_object(path: Path, *, context: str) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DistilledSalvageImportError(f"could not read {context} {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise DistilledSalvageImportError(f"{context} is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise DistilledSalvageImportError(f"{context} must be a JSON object")
    return {str(key): item for key, item in value.items()}


def _require_int(mapping: dict[str, object], key: str, *, context: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise DistilledSalvageImportError(f"{context}.{key} must be an integer")
    return value


def _require_string(mapping: dict[str, object], key: str, *, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise DistilledSalvageImportError(f"{context}.{key} must be a non-empty string")
    return value


def _require_mapping(mapping: dict[str, object], key: str, *, context: str) -> dict[str, object]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise DistilledSalvageImportError(f"{context}.{key} must be an object")
    return {str(item_key): item for item_key, item in value.items()}


def _safe_archive_members(archive: zipfile.ZipFile) -> tuple[zipfile.ZipInfo, ...]:
    members = tuple(archive.infolist())
    if not members:
        raise DistilledSalvageImportError("teacher evidence archive is empty")
    names = tuple(member.filename for member in members)
    if len(names) != len(set(names)):
        raise DistilledSalvageImportError("teacher evidence archive contains duplicate paths")
    for member in members:
        path = PurePosixPath(member.filename)
        if path.is_absolute() or ".." in path.parts:
            raise DistilledSalvageImportError(
                f"teacher evidence archive contains unsafe path {member.filename!r}"
            )
        unix_mode = (member.external_attr >> 16) & 0o170000
        if unix_mode == 0o120000:
            raise DistilledSalvageImportError(
                f"teacher evidence archive contains symbolic link {member.filename!r}"
            )
    return members


def _verify_sidecars(root: Path) -> None:
    sidecars = tuple(sorted(root.rglob("*.sha256")))
    if not sidecars:
        raise DistilledSalvageImportError("teacher evidence contains no checksum sidecars")
    for sidecar in sidecars:
        try:
            text = sidecar.read_text(encoding="ascii")
        except OSError as exc:
            raise DistilledSalvageImportError(
                f"could not read checksum sidecar {sidecar}: {exc}"
            ) from exc
        parts = text.rstrip("\n").split("  ", maxsplit=1)
        if len(parts) != 2 or text != f"{parts[0]}  {parts[1]}\n":
            raise DistilledSalvageImportError(
                f"checksum sidecar has non-canonical format: {sidecar}"
            )
        expected, filename = parts
        if PurePosixPath(filename).name != filename:
            raise DistilledSalvageImportError(
                f"checksum sidecar references a non-local payload: {sidecar}"
            )
        target = sidecar.parent / filename
        if not target.is_file():
            raise DistilledSalvageImportError(
                f"checksum sidecar references missing payload {target}"
            )
        if _sha256_file(target) != expected:
            raise DistilledSalvageImportError(f"checksum mismatch for {target}")


def _jsonl_objects(path: Path, *, context: str) -> tuple[dict[str, object], ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DistilledSalvageImportError(f"could not read {context} {path}: {exc}") from exc
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise DistilledSalvageImportError(
                f"{context} contains blank JSONL line at {path}:{line_number}"
            )
        try:
            value: object = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DistilledSalvageImportError(
                f"{context} contains invalid JSON at {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise DistilledSalvageImportError(
                f"{context} record at {path}:{line_number} must be an object"
            )
        records.append({str(key): item for key, item in value.items()})
    if not records:
        raise DistilledSalvageImportError(f"{context} must not be empty: {path}")
    return tuple(records)


def _normalized_text(value: str) -> str:
    value.encode("utf-8", errors="strict")
    return value.removeprefix("\ufeff").replace("\r\n", "\n").replace("\r", "\n")


def _record_fingerprint(
    record: dict[str, object], *, context: str
) -> tuple[str, str, str, str, str | None]:
    messages_value = record.get("messages")
    if not isinstance(messages_value, list) or len(messages_value) < 2:
        raise DistilledSalvageImportError(f"{context}.messages must contain at least two items")
    messages: list[dict[str, str]] = []
    for index, item in enumerate(messages_value):
        if not isinstance(item, dict):
            raise DistilledSalvageImportError(f"{context}.messages[{index}] must be an object")
        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise DistilledSalvageImportError(
                f"{context}.messages[{index}] role/content must be strings"
            )
        messages.append({"role": role, "content": _normalized_text(content)})
    if messages[-1]["role"] != "assistant":
        raise DistilledSalvageImportError(f"{context} must end with an assistant response")

    prompt_sha256 = _canonical_sha256(
        {"schema_version": 1, "messages": messages[:-1]},
        ensure_ascii=False,
    )
    response_sha256 = _canonical_sha256(
        {
            "schema_version": 1,
            "role": "assistant",
            "content": messages[-1]["content"],
        },
        ensure_ascii=False,
    )
    record_sha256 = _canonical_sha256(
        {
            "schema_version": 1,
            "prompt_sha256": prompt_sha256,
            "response_sha256": response_sha256,
        },
        ensure_ascii=False,
    )

    provenance = record.get("provenance")
    if not isinstance(provenance, dict):
        raise DistilledSalvageImportError(f"{context}.provenance must be an object")
    source_id = provenance.get("source_id")
    source_record_id = provenance.get("record_id")
    if not isinstance(source_id, str) or not source_id:
        raise DistilledSalvageImportError(f"{context}.provenance.source_id must be a string")
    if source_record_id is not None and not isinstance(source_record_id, str):
        raise DistilledSalvageImportError(
            f"{context}.provenance.record_id must be a string or null"
        )
    return prompt_sha256, response_sha256, record_sha256, source_id, source_record_id


def _verify_manifest_content(
    manifest: dict[str, object],
    *,
    accepted: tuple[dict[str, object], ...],
    train: tuple[dict[str, object], ...],
    validation: tuple[dict[str, object], ...],
) -> None:
    checksums = _require_mapping(manifest, "checksums", context="dataset manifest")
    accepted_fingerprints = tuple(
        _record_fingerprint(record, context=f"accepted[{index}]")
        for index, record in enumerate(accepted)
    )
    train_fingerprints = tuple(
        _record_fingerprint(record, context=f"train[{index}]") for index, record in enumerate(train)
    )
    validation_fingerprints = tuple(
        _record_fingerprint(record, context=f"validation[{index}]")
        for index, record in enumerate(validation)
    )

    accepted_hashes = tuple(item[2] for item in accepted_fingerprints)
    train_hashes = tuple(item[2] for item in train_fingerprints)
    validation_hashes = tuple(item[2] for item in validation_fingerprints)
    if len(set(accepted_hashes)) != len(accepted_hashes):
        raise DistilledSalvageImportError("accepted corpus contains duplicate normalized records")
    train_set = set(train_hashes)
    validation_set = set(validation_hashes)
    if train_set & validation_set:
        raise DistilledSalvageImportError("train and validation splits overlap")
    if set(accepted_hashes) != train_set | validation_set:
        raise DistilledSalvageImportError("accepted corpus does not equal train plus validation")

    expected_hashes = {
        "unique_corpus_sha256": _canonical_sha256(accepted_hashes),
        "train_records_sha256": _canonical_sha256(train_hashes),
        "validation_records_sha256": _canonical_sha256(validation_hashes),
    }
    for key, computed in expected_hashes.items():
        if _require_string(checksums, key, context="dataset manifest.checksums") != computed:
            raise DistilledSalvageImportError(
                f"dataset manifest {key} does not match JSONL content"
            )

    memberships_value = manifest.get("memberships")
    if not isinstance(memberships_value, list) or len(memberships_value) != len(accepted):
        raise DistilledSalvageImportError("dataset manifest memberships do not align with accepted")
    if _require_string(
        checksums,
        "split_membership_sha256",
        context="dataset manifest.checksums",
    ) != _canonical_sha256(memberships_value):
        raise DistilledSalvageImportError("dataset manifest split-membership checksum drifted")

    for index, (membership_value, fingerprint) in enumerate(
        zip(memberships_value, accepted_fingerprints, strict=True)
    ):
        if not isinstance(membership_value, dict):
            raise DistilledSalvageImportError(f"dataset manifest memberships[{index}] is invalid")
        membership = {str(key): item for key, item in membership_value.items()}
        prompt_sha256, _, record_sha256, source_id, source_record_id = fingerprint
        if membership.get("unique_index") != index:
            raise DistilledSalvageImportError(f"dataset membership index drift at {index}")
        if membership.get("prompt_sha256") != prompt_sha256:
            raise DistilledSalvageImportError(f"dataset membership prompt hash drift at {index}")
        if membership.get("record_sha256") != record_sha256:
            raise DistilledSalvageImportError(f"dataset membership record hash drift at {index}")
        if membership.get("source_id") != source_id:
            raise DistilledSalvageImportError(f"dataset membership source ID drift at {index}")
        if membership.get("source_record_id") != source_record_id:
            raise DistilledSalvageImportError(f"dataset membership source record drift at {index}")
        partition = "train" if record_sha256 in train_set else "validation"
        if membership.get("partition") != partition:
            raise DistilledSalvageImportError(f"dataset membership partition drift at {index}")


def _reasoning_marker_hits(paths: tuple[Path, ...]) -> int:
    hits = 0
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise DistilledSalvageImportError(
                f"could not scan training data {path}: {exc}"
            ) from exc
        hits += sum(text.count(marker) for marker in _REASONING_MARKERS)
    return hits


def _verify_qualification(root: Path) -> tuple[dict[str, object], str]:
    path = root / "qualification.json"
    qualification = _json_object(path, context="qualification")
    if qualification.get("qualified") is not True:
        raise DistilledSalvageImportError("repaired V4-2000 corpus is not training-qualified")
    if qualification.get("failed_gates") != []:
        raise DistilledSalvageImportError("repaired V4-2000 qualification contains failed gates")
    if qualification.get("contamination_status") != "clean":
        raise DistilledSalvageImportError("repaired V4-2000 contamination status is not clean")
    total_records = _require_int(qualification, "total_records", context="qualification")
    if total_records != _EXPECTED_TOTAL_RECORDS:
        raise DistilledSalvageImportError("repaired V4-2000 qualification record count drifted")
    return qualification, _sha256_file(path)


def _verify_source_identity(root: Path) -> dict[str, object]:
    identity = _json_object(root / "source-identity/run-identity.json", context="source identity")
    if _require_string(identity, "policy_id", context="source identity") != _EXPECTED_POLICY_ID:
        raise DistilledSalvageImportError("unexpected reasoning-salvage policy identity")
    if (
        _require_string(identity, "output_sha256", context="source identity")
        != _EXPECTED_SOURCE_OUTPUT_SHA256
    ):
        raise DistilledSalvageImportError("repaired source-output identity drifted")
    total_records = _require_int(identity, "total_records", context="source identity")
    if total_records != _EXPECTED_TOTAL_RECORDS:
        raise DistilledSalvageImportError("repaired source record count drifted")
    return identity


def _verify_final_evidence(root: Path) -> tuple[str, str, int, int, int, str, int]:
    final = root / "final"
    for filename in _REQUIRED_FINAL_FILES:
        if not (final / filename).is_file():
            raise DistilledSalvageImportError(f"repaired corpus is missing final/{filename}")

    manifest_path = final / "dataset-manifest.json"
    manifest_sha256 = _sha256_file(manifest_path)
    if manifest_sha256 != _EXPECTED_MANIFEST_SHA256:
        raise DistilledSalvageImportError(
            "repaired V4-2000 dataset manifest does not match the frozen qualified corpus"
        )
    expected_sidecar = f"{manifest_sha256}  dataset-manifest.json\n"
    if (final / "dataset-manifest.sha256").read_text(encoding="ascii") != expected_sidecar:
        raise DistilledSalvageImportError("dataset manifest checksum sidecar does not match")

    manifest = _json_object(manifest_path, context="dataset manifest")
    counts = _require_mapping(manifest, "counts", context="dataset manifest")
    accepted_records = _require_int(
        counts,
        "deduplicated_unique",
        context="dataset manifest.counts",
    )
    train_records = _require_int(counts, "train_records", context="dataset manifest.counts")
    validation_records = _require_int(
        counts, "validation_records", context="dataset manifest.counts"
    )
    expected = (_EXPECTED_ACCEPTED_RECORDS, _EXPECTED_TRAIN_RECORDS, _EXPECTED_VALIDATION_RECORDS)
    if (accepted_records, train_records, validation_records) != expected:
        raise DistilledSalvageImportError("repaired V4-2000 final record counts drifted")
    accepted = _jsonl_objects(final / "accepted.jsonl", context="accepted corpus")
    train = _jsonl_objects(final / "train.jsonl", context="training split")
    validation = _jsonl_objects(final / "validation.jsonl", context="validation split")
    if len(accepted) != accepted_records:
        raise DistilledSalvageImportError("accepted.jsonl count does not match dataset manifest")
    if len(train) != train_records:
        raise DistilledSalvageImportError("train.jsonl count does not match dataset manifest")
    if len(validation) != validation_records:
        raise DistilledSalvageImportError("validation.jsonl count does not match dataset manifest")
    _verify_manifest_content(
        manifest,
        accepted=accepted,
        train=train,
        validation=validation,
    )

    contamination = _require_mapping(manifest, "contamination", context="dataset manifest")
    contamination_status = _require_string(
        contamination, "status", context="dataset manifest.contamination"
    )
    if contamination_status != "clean" or contamination.get("findings") != []:
        raise DistilledSalvageImportError("dataset manifest contamination evidence is not clean")

    marker_hits = _reasoning_marker_hits(
        (final / "accepted.jsonl", final / "train.jsonl", final / "validation.jsonl")
    )
    if marker_hits:
        raise DistilledSalvageImportError(
            "repaired final corpus still contains "
            f"{marker_hits} hidden-thinking marker occurrence(s)"
        )

    finalization_path = final / "teacher-finalization.json"
    finalization = _json_object(finalization_path, context="teacher finalization")
    finalized_unique = _require_int(
        finalization,
        "prepared_unique",
        context="teacher finalization",
    )
    if finalized_unique != accepted_records:
        raise DistilledSalvageImportError(
            "teacher finalization unique count disagrees with manifest"
        )
    if _require_int(finalization, "train_records", context="teacher finalization") != train_records:
        raise DistilledSalvageImportError(
            "teacher finalization train count disagrees with manifest"
        )
    if (
        _require_int(finalization, "validation_records", context="teacher finalization")
        != validation_records
    ):
        raise DistilledSalvageImportError(
            "teacher finalization validation count disagrees with manifest"
        )
    return (
        manifest_sha256,
        _sha256_file(finalization_path),
        accepted_records,
        train_records,
        validation_records,
        contamination_status,
        marker_hits,
    )


def import_repaired_v4_2000_corpus(
    archive_path: Path,
    *,
    output_dir: Path,
) -> DistilledSalvageImportReceipt:
    """Verify one exported salvage bundle and atomically import its training evidence."""

    if output_dir.exists():
        raise DistilledSalvageImportError(
            f"output directory already exists; refusing to mutate frozen data: {output_dir}"
        )
    if not archive_path.is_file():
        raise DistilledSalvageImportError(
            f"teacher evidence archive does not exist: {archive_path}"
        )
    archive_sha256 = _sha256_file(archive_path)

    with tempfile.TemporaryDirectory(prefix="tqc-v4-2000-salvage-") as temporary:
        scratch = Path(temporary)
        try:
            with zipfile.ZipFile(archive_path) as archive:
                _safe_archive_members(archive)
                archive.extractall(scratch)
        except (OSError, zipfile.BadZipFile) as exc:
            raise DistilledSalvageImportError(f"could not extract teacher evidence: {exc}") from exc

        root = scratch / _EXPECTED_ROOT
        if not root.is_dir():
            raise DistilledSalvageImportError(
                f"teacher evidence archive must contain {_EXPECTED_ROOT!r} at its root"
            )
        _verify_sidecars(root)
        _, qualification_sha256 = _verify_qualification(root)
        source_identity = _verify_source_identity(root)
        (
            manifest_sha256,
            finalization_sha256,
            accepted_records,
            train_records,
            validation_records,
            contamination_status,
            marker_hits,
        ) = _verify_final_evidence(root)

        receipt = DistilledSalvageImportReceipt(
            schema_version=1,
            source_archive_sha256=archive_sha256,
            source_root=_EXPECTED_ROOT,
            source_output_sha256=_require_string(
                source_identity, "output_sha256", context="source identity"
            ),
            dataset_manifest_sha256=manifest_sha256,
            qualification_sha256=qualification_sha256,
            teacher_finalization_sha256=finalization_sha256,
            accepted_records=accepted_records,
            train_records=train_records,
            validation_records=validation_records,
            contamination_status=contamination_status,
            reasoning_marker_hits=marker_hits,
        )

        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = output_dir.with_name(f".{output_dir.name}.tmp-{os.getpid()}")
        if staging.exists():
            raise DistilledSalvageImportError(f"staging path already exists: {staging}")
        staging.mkdir()
        try:
            for filename in _REQUIRED_FINAL_FILES:
                shutil.copy2(root / "final" / filename, staging / filename)
            shutil.copy2(root / "qualification.json", staging / "qualification.json")
            shutil.copy2(
                root / "source-identity/run-identity.json",
                staging / "source-run-identity.json",
            )
            (staging / "import-receipt.json").write_text(
                json.dumps(asdict(receipt), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(staging, output_dir)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    return receipt


def distilled_salvage_import_receipt_json(receipt: DistilledSalvageImportReceipt) -> str:
    """Serialize an import receipt deterministically."""

    return json.dumps(asdict(receipt), indent=2, sort_keys=True) + "\n"
