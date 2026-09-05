"""Deterministic source-stratified subsets for bounded teacher experiments."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from tiny_qwen_coder.data.deduplication import normalized_record_fingerprint
from tiny_qwen_coder.data.loading import load_normalized_training_records_jsonl
from tiny_qwen_coder.data.records import NormalizedTrainingRecord


class TeacherSubsetError(ValueError):
    """Raised when a representative teacher-input subset cannot be created."""


@dataclass(frozen=True, slots=True)
class TeacherSubsetSourceSummary:
    """Population and selected counts for one exact upstream source identity."""

    source_id: str
    revision: str
    population_records: int
    selected_records: int


@dataclass(frozen=True, slots=True)
class TeacherSubsetSummary:
    """Audit record for one deterministic stratified teacher-input subset."""

    schema_version: int
    seed: int
    language: str
    input_path: str
    input_sha256: str
    input_records: int
    output_path: str
    output_sha256: str
    selected_records: int
    sources: tuple[TeacherSubsetSourceSummary, ...]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding=encoding)
    os.replace(temporary, path)


def _source_key(record: NormalizedTrainingRecord) -> tuple[str, str]:
    return record.provenance.source_id, record.provenance.revision


def _source_quotas(
    counts: Counter[tuple[str, str]], *, selected: int, total: int
) -> dict[tuple[str, str], int]:
    quotas = {key: (selected * count) // total for key, count in counts.items()}
    remaining = selected - sum(quotas.values())
    # Allocate the largest fractional remainders first, with stable source-key ties.
    ranked_remainders = sorted(
        counts,
        key=lambda key: (-((selected * counts[key]) % total), key),
    )
    for key in ranked_remainders[:remaining]:
        quotas[key] += 1
    return quotas


def _selection_rank(*, seed: int, index: int, record: NormalizedTrainingRecord) -> bytes:
    fingerprint = normalized_record_fingerprint(record).record_sha256
    payload = f"{seed}\0{index}\0{fingerprint}".encode("ascii")
    return hashlib.sha256(payload).digest()


def select_teacher_input_records(
    records: tuple[NormalizedTrainingRecord, ...],
    *,
    count: int,
    seed: int,
) -> tuple[NormalizedTrainingRecord, ...]:
    """Select an exact-count source-stratified deterministic subset."""

    if seed < 0:
        raise TeacherSubsetError("seed must be non-negative")
    if count <= 0:
        raise TeacherSubsetError("count must be greater than zero")
    if count > len(records):
        raise TeacherSubsetError(
            f"count {count} exceeds input population of {len(records)} records"
        )
    if count == len(records):
        return records

    grouped: dict[tuple[str, str], list[tuple[int, NormalizedTrainingRecord]]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[_source_key(record)].append((index, record))
    counts: Counter[tuple[str, str]] = Counter(
        {key: len(group_records) for key, group_records in grouped.items()}
    )
    quotas = _source_quotas(counts, selected=count, total=len(records))

    selected_indexes: set[int] = set()
    for key in sorted(grouped):
        ranked = sorted(
            grouped[key],
            key=lambda item: (_selection_rank(seed=seed, index=item[0], record=item[1]), item[0]),
        )
        selected_indexes.update(index for index, _record in ranked[: quotas[key]])
    if len(selected_indexes) != count:
        raise TeacherSubsetError(
            f"stratified selection produced {len(selected_indexes)} records; expected {count}"
        )
    return tuple(record for index, record in enumerate(records) if index in selected_indexes)


def write_teacher_input_subset(
    *,
    input_path: Path,
    output_path: Path,
    count: int,
    seed: int = 1729,
    language: str = "python",
) -> TeacherSubsetSummary:
    """Write one sealed deterministic subset plus checksum and audit summary."""

    records = load_normalized_training_records_jsonl(input_path, expected_language=language)
    selected = select_teacher_input_records(records, count=count, seed=seed)
    content = "".join(
        json.dumps(asdict(record), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for record in selected
    )
    _atomic_write(output_path, content)
    output_sha = _file_sha256(output_path)
    checksum_path = output_path.with_suffix(output_path.suffix + ".sha256")
    _atomic_write(
        checksum_path,
        f"{output_sha}  {output_path.name}\n",
        encoding="ascii",
    )

    population_counts = Counter(_source_key(record) for record in records)
    selected_counts = Counter(_source_key(record) for record in selected)
    source_summaries = tuple(
        TeacherSubsetSourceSummary(
            source_id=key[0],
            revision=key[1],
            population_records=population_counts[key],
            selected_records=selected_counts[key],
        )
        for key in sorted(population_counts)
    )
    summary = TeacherSubsetSummary(
        schema_version=1,
        seed=seed,
        language=language,
        input_path=str(input_path),
        input_sha256=_file_sha256(input_path),
        input_records=len(records),
        output_path=str(output_path),
        output_sha256=output_sha,
        selected_records=len(selected),
        sources=source_summaries,
    )
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    _atomic_write(summary_path, json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n")
    return summary


__all__ = [
    "TeacherSubsetError",
    "TeacherSubsetSourceSummary",
    "TeacherSubsetSummary",
    "select_teacher_input_records",
    "write_teacher_input_subset",
]
