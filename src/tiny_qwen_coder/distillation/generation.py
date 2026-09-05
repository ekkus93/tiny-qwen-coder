"""Resumable, shard-oriented teacher generation core."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from tiny_qwen_coder.data.deduplication import normalized_record_fingerprint
from tiny_qwen_coder.data.loading import load_normalized_training_records_jsonl
from tiny_qwen_coder.data.records import (
    NormalizedTrainingRecord,
    SourceProvenance,
    TrainingMessage,
)
from tiny_qwen_coder.distillation.config import (
    TeacherDistillationConfig,
    teacher_distillation_config_sha256,
)


class TeacherGenerationError(RuntimeError):
    """Raised when durable teacher generation cannot continue safely."""


@dataclass(frozen=True, slots=True)
class TeacherCompletion:
    """One raw completion plus bounded inference accounting."""

    text: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int


class TeacherBackend(Protocol):
    """Minimal batch backend used by the resumable orchestration layer."""

    def generate(
        self,
        conversations: tuple[tuple[TrainingMessage, ...], ...],
        *,
        seeds: tuple[int, ...],
    ) -> tuple[TeacherCompletion, ...]:
        """Generate exactly one completion for each conversation."""


@dataclass(frozen=True, slots=True)
class TeacherShardRecord:
    """Auditable durable row for one teacher-generated answer."""

    schema_version: int
    config_sha256: str
    implementation_sha256: str
    input_file_sha256: str
    input_index: int
    input_record_sha256: str
    prompt_sha256: str
    seed: int
    teacher_repository: str
    teacher_revision: str
    raw_completion_sha256: str
    reasoning_sha256: str | None
    reasoning_chars: int
    final_response: str
    final_response_sha256: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True, slots=True)
class TeacherGenerationProgress:
    """Human-readable resume state derived from verified durable shards."""

    schema_version: int
    config_sha256: str
    input_file_sha256: str
    total_records: int
    completed_records: int
    completed_shards: int
    total_shards: int
    updated_at_utc: str


@dataclass(frozen=True, slots=True)
class TeacherGenerationStatus:
    """Verified durable progress available before loading the teacher model."""

    total_records: int
    completed_records: int
    completed_shards: int
    total_shards: int
    missing_shards: tuple[int, ...]
    checkpoint_dir: Path

    @property
    def complete(self) -> bool:
        """Return whether every requested record is durably checkpointed."""

        return not self.missing_shards


@dataclass(frozen=True, slots=True)
class TeacherGenerationResult:
    """Summary returned after all requested shards are already durable."""

    total_records: int
    completed_records: int
    completed_shards: int
    total_shards: int
    checkpoint_dir: Path


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _implementation_sha256() -> str:
    """Bind durable shards to the exact generation/config/backend source implementation."""

    digest = hashlib.sha256()
    for path in sorted(
        (
            Path(__file__),
            Path(__file__).with_name("config.py"),
            Path(__file__).with_name("vllm_backend.py"),
        ),
        key=lambda item: item.name,
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _prompt_messages(record: NormalizedTrainingRecord) -> tuple[TrainingMessage, ...]:
    if not record.messages:
        raise TeacherGenerationError("input record has no messages")
    if record.messages[-1].role != "assistant":
        raise TeacherGenerationError(
            "teacher distillation requires each input record to end with an assistant answer"
        )
    prompt = record.messages[:-1]
    if not prompt or prompt[-1].role != "user":
        raise TeacherGenerationError(
            "teacher distillation requires the answer prefix to end with a user message"
        )
    return prompt


def _prompt_sha256(messages: tuple[TrainingMessage, ...]) -> str:
    return _sha256_text(_canonical_json([asdict(message) for message in messages]))


def _split_reasoning(text: str) -> tuple[str | None, str]:
    """Separate Qwen-style thinking from the trainable final answer.

    Qwen3.8 thinking-mode completions normally contain a leading ``<think>`` block.
    We retain only a hash and character count of that reasoning in durable audit
    records so the student corpus never learns hidden-thought markup.
    """

    stripped = text.strip()
    if not stripped.startswith("<think>"):
        return None, stripped
    closing = stripped.find("</think>")
    if closing < 0:
        raise TeacherGenerationError("teacher completion opened <think> without closing </think>")
    reasoning = stripped[len("<think>") : closing].strip()
    final = stripped[closing + len("</think>") :].strip()
    if not final:
        raise TeacherGenerationError("teacher completion contained no final answer after </think>")
    return reasoning, final


def _shard_bounds(shard_index: int, *, shard_size: int, total: int) -> tuple[int, int]:
    start = shard_index * shard_size
    return start, min(start + shard_size, total)


def _shard_name(shard_index: int) -> str:
    return f"shard-{shard_index:06d}.jsonl"


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def _serialize_shard(records: tuple[TeacherShardRecord, ...]) -> str:
    return "".join(_canonical_json(asdict(record)) + "\n" for record in records)


def _parse_shard(path: Path) -> tuple[dict[str, object], ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise TeacherGenerationError(f"could not read durable shard {path}: {exc}") from exc
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise TeacherGenerationError(f"durable shard {path}:{line_number} is blank")
        try:
            raw: object = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TeacherGenerationError(
                f"durable shard {path}:{line_number} contains invalid JSON: {exc}"
            ) from exc
        if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
            raise TeacherGenerationError(f"durable shard {path}:{line_number} must be a mapping")
        rows.append(raw)
    return tuple(rows)


def _validate_durable_shard(
    path: Path,
    *,
    records: tuple[NormalizedTrainingRecord, ...],
    start: int,
    end: int,
    config_sha256: str,
    implementation_sha256: str,
    input_file_sha256: str,
) -> bool:
    sidecar = path.with_suffix(".sha256")
    if not path.exists():
        if sidecar.exists():
            raise TeacherGenerationError(
                f"durable shard checksum sidecar exists without its payload: {sidecar}"
            )
        return False
    if not sidecar.exists():
        # The checksum sidecar is the commit marker. A payload without it can be
        # left behind by Colab preemption between the two atomic Drive copies.
        return False
    expected_sidecar = f"{_file_sha256(path)}  {path.name}\n"
    if sidecar.read_text(encoding="ascii") != expected_sidecar:
        raise TeacherGenerationError(f"durable shard checksum mismatch: {path}")
    rows = _parse_shard(path)
    if len(rows) != end - start:
        raise TeacherGenerationError(
            f"durable shard {path} has {len(rows)} rows; expected {end - start}"
        )
    for offset, row in enumerate(rows):
        index = start + offset
        expected_fingerprint = normalized_record_fingerprint(records[index]).record_sha256
        expected_prompt = _prompt_sha256(_prompt_messages(records[index]))
        if row.get("schema_version") != 1:
            raise TeacherGenerationError(f"durable shard {path} has unsupported schema_version")
        if row.get("config_sha256") != config_sha256:
            raise TeacherGenerationError(f"durable shard {path} belongs to another config")
        if row.get("implementation_sha256") != implementation_sha256:
            raise TeacherGenerationError(
                f"durable shard {path} belongs to another generator implementation"
            )
        if row.get("input_file_sha256") != input_file_sha256:
            raise TeacherGenerationError(f"durable shard {path} belongs to another input file")
        if row.get("input_index") != index:
            raise TeacherGenerationError(f"durable shard {path} has unexpected input_index")
        if row.get("input_record_sha256") != expected_fingerprint:
            raise TeacherGenerationError(f"durable shard {path} input record fingerprint mismatch")
        if row.get("prompt_sha256") != expected_prompt:
            raise TeacherGenerationError(f"durable shard {path} prompt fingerprint mismatch")
    return True


def _run_identity_payload(
    *,
    config: TeacherDistillationConfig,
    config_sha256: str,
    input_file_sha256: str,
    total_records: int,
) -> str:
    return (
        json.dumps(
            {
                "schema_version": 1,
                "config_sha256": config_sha256,
                "implementation_sha256": _implementation_sha256(),
                "input_file_sha256": input_file_sha256,
                "total_records": total_records,
                "config": asdict(config),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _ensure_run_identity(
    checkpoint_dir: Path,
    *,
    config: TeacherDistillationConfig,
    config_sha256: str,
    input_file_sha256: str,
    total_records: int,
) -> None:
    identity_path = checkpoint_dir / "run-identity.json"
    expected = _run_identity_payload(
        config=config,
        config_sha256=config_sha256,
        input_file_sha256=input_file_sha256,
        total_records=total_records,
    )
    if identity_path.exists():
        actual = identity_path.read_text(encoding="utf-8")
        if actual != expected:
            raise TeacherGenerationError(
                "checkpoint directory belongs to a different config or input corpus"
            )
        return
    _atomic_write_text(identity_path, expected)


def _write_progress(
    checkpoint_dir: Path,
    *,
    config_sha256: str,
    input_file_sha256: str,
    total_records: int,
    completed_records: int,
    completed_shards: int,
    total_shards: int,
) -> None:
    progress = TeacherGenerationProgress(
        schema_version=1,
        config_sha256=config_sha256,
        input_file_sha256=input_file_sha256,
        total_records=total_records,
        completed_records=completed_records,
        completed_shards=completed_shards,
        total_shards=total_shards,
        updated_at_utc=datetime.now(UTC).isoformat(),
    )
    _atomic_write_text(
        checkpoint_dir / "progress.json",
        json.dumps(asdict(progress), indent=2, sort_keys=True) + "\n",
    )


def _build_shard_records(
    *,
    source_records: tuple[NormalizedTrainingRecord, ...],
    source_start: int,
    completions: tuple[TeacherCompletion, ...],
    config: TeacherDistillationConfig,
    config_sha256: str,
    implementation_sha256: str,
    input_file_sha256: str,
) -> tuple[TeacherShardRecord, ...]:
    if len(source_records) != len(completions):
        raise TeacherGenerationError("teacher backend returned a mismatched completion count")
    output: list[TeacherShardRecord] = []
    for offset, (source, completion) in enumerate(zip(source_records, completions, strict=True)):
        index = source_start + offset
        prompt = _prompt_messages(source)
        reasoning, final = _split_reasoning(completion.text)
        if not final.strip():
            raise TeacherGenerationError(
                f"teacher produced an empty final response for input {index}"
            )
        output.append(
            TeacherShardRecord(
                schema_version=1,
                config_sha256=config_sha256,
                implementation_sha256=implementation_sha256,
                input_file_sha256=input_file_sha256,
                input_index=index,
                input_record_sha256=normalized_record_fingerprint(source).record_sha256,
                prompt_sha256=_prompt_sha256(prompt),
                seed=config.generation.seed + index,
                teacher_repository=config.teacher.repository,
                teacher_revision=config.teacher.revision,
                raw_completion_sha256=_sha256_text(completion.text),
                reasoning_sha256=_sha256_text(reasoning) if reasoning is not None else None,
                reasoning_chars=len(reasoning) if reasoning is not None else 0,
                final_response=final,
                final_response_sha256=_sha256_text(final),
                finish_reason=completion.finish_reason,
                prompt_tokens=completion.prompt_tokens,
                completion_tokens=completion.completion_tokens,
            )
        )
    return tuple(output)


def inspect_teacher_generation(
    config: TeacherDistillationConfig,
    *,
    checkpoint_dir: Path,
    input_path: Path | None = None,
    limit: int | None = None,
) -> TeacherGenerationStatus:
    """Validate config/input identity and report missing shards without loading a model."""

    if limit is not None and limit <= 0:
        raise TeacherGenerationError("limit must be greater than zero when provided")
    resolved_input = input_path or Path(config.input_records)
    records = load_normalized_training_records_jsonl(
        resolved_input, expected_language=config.language
    )
    if limit is not None:
        records = records[:limit]
    if not records:
        raise TeacherGenerationError("teacher generation input is empty")

    input_file_sha256 = _file_sha256(resolved_input)
    config_sha256 = teacher_distillation_config_sha256(config)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "shards").mkdir(parents=True, exist_ok=True)
    _ensure_run_identity(
        checkpoint_dir,
        config=config,
        config_sha256=config_sha256,
        input_file_sha256=input_file_sha256,
        total_records=len(records),
    )

    shard_size = config.checkpoint.shard_size
    total_shards = (len(records) + shard_size - 1) // shard_size
    completed_records = 0
    completed_shards = 0
    missing: list[int] = []
    for shard_index in range(total_shards):
        start, end = _shard_bounds(shard_index, shard_size=shard_size, total=len(records))
        path = checkpoint_dir / "shards" / _shard_name(shard_index)
        if _validate_durable_shard(
            path,
            records=records,
            start=start,
            end=end,
            config_sha256=config_sha256,
            implementation_sha256=_implementation_sha256(),
            input_file_sha256=input_file_sha256,
        ):
            completed_shards += 1
            completed_records += end - start
        else:
            missing.append(shard_index)

    _write_progress(
        checkpoint_dir,
        config_sha256=config_sha256,
        input_file_sha256=input_file_sha256,
        total_records=len(records),
        completed_records=completed_records,
        completed_shards=completed_shards,
        total_shards=total_shards,
    )
    return TeacherGenerationStatus(
        total_records=len(records),
        completed_records=completed_records,
        completed_shards=completed_shards,
        total_shards=total_shards,
        missing_shards=tuple(missing),
        checkpoint_dir=checkpoint_dir,
    )


def run_teacher_generation(
    config: TeacherDistillationConfig,
    *,
    backend: TeacherBackend,
    checkpoint_dir: Path,
    work_dir: Path,
    input_path: Path | None = None,
    limit: int | None = None,
) -> TeacherGenerationResult:
    """Generate missing deterministic shards and checkpoint each one durably.

    ``checkpoint_dir`` is intended to point at Google Drive on Colab. A shard is
    considered complete only after both the JSONL file and its SHA-256 sidecar
    exist and re-validate against the exact config and input record fingerprints.
    Consequently an interrupted run simply re-enters this function and skips
    already durable shards.
    """

    if limit is not None and limit <= 0:
        raise TeacherGenerationError("limit must be greater than zero when provided")
    resolved_input = input_path or Path(config.input_records)
    all_records = load_normalized_training_records_jsonl(
        resolved_input, expected_language=config.language
    )
    if limit is not None:
        all_records = all_records[:limit]
    if not all_records:
        raise TeacherGenerationError("teacher generation input is empty")

    input_file_sha256 = _file_sha256(resolved_input)
    config_sha256 = teacher_distillation_config_sha256(config)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "shards").mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "shards").mkdir(parents=True, exist_ok=True)
    _ensure_run_identity(
        checkpoint_dir,
        config=config,
        config_sha256=config_sha256,
        input_file_sha256=input_file_sha256,
        total_records=len(all_records),
    )

    shard_size = config.checkpoint.shard_size
    total_records = len(all_records)
    total_shards = (total_records + shard_size - 1) // shard_size
    completed_records = 0
    completed_shards = 0

    for shard_index in range(total_shards):
        start, end = _shard_bounds(shard_index, shard_size=shard_size, total=total_records)
        shard_name = _shard_name(shard_index)
        durable_path = checkpoint_dir / "shards" / shard_name
        if _validate_durable_shard(
            durable_path,
            records=all_records,
            start=start,
            end=end,
            config_sha256=config_sha256,
            implementation_sha256=_implementation_sha256(),
            input_file_sha256=input_file_sha256,
        ):
            completed_shards += 1
            completed_records += end - start
            continue

        source_records = all_records[start:end]
        conversations = tuple(_prompt_messages(record) for record in source_records)
        seeds = tuple(config.generation.seed + index for index in range(start, end))
        completions = backend.generate(conversations, seeds=seeds)
        shard_records = _build_shard_records(
            source_records=source_records,
            source_start=start,
            completions=completions,
            config=config,
            config_sha256=config_sha256,
            implementation_sha256=_implementation_sha256(),
            input_file_sha256=input_file_sha256,
        )

        local_path = work_dir / "shards" / shard_name
        _atomic_write_text(local_path, _serialize_shard(shard_records))
        local_sidecar = local_path.with_suffix(".sha256")
        _atomic_write_text(local_sidecar, f"{_file_sha256(local_path)}  {local_path.name}\n")
        _copy_atomic(local_path, durable_path)
        _copy_atomic(local_sidecar, durable_path.with_suffix(".sha256"))
        if not _validate_durable_shard(
            durable_path,
            records=all_records,
            start=start,
            end=end,
            config_sha256=config_sha256,
            implementation_sha256=_implementation_sha256(),
            input_file_sha256=input_file_sha256,
        ):
            raise TeacherGenerationError(f"newly copied shard did not validate: {durable_path}")

        completed_shards += 1
        completed_records += end - start
        _write_progress(
            checkpoint_dir,
            config_sha256=config_sha256,
            input_file_sha256=input_file_sha256,
            total_records=total_records,
            completed_records=completed_records,
            completed_shards=completed_shards,
            total_shards=total_shards,
        )

    _write_progress(
        checkpoint_dir,
        config_sha256=config_sha256,
        input_file_sha256=input_file_sha256,
        total_records=total_records,
        completed_records=completed_records,
        completed_shards=completed_shards,
        total_shards=total_shards,
    )
    return TeacherGenerationResult(
        total_records=total_records,
        completed_records=completed_records,
        completed_shards=completed_shards,
        total_shards=total_shards,
        checkpoint_dir=checkpoint_dir,
    )


def load_completed_distilled_records(
    config: TeacherDistillationConfig,
    *,
    checkpoint_dir: Path,
    input_path: Path | None = None,
    limit: int | None = None,
) -> tuple[NormalizedTrainingRecord, ...]:
    """Load a complete checkpoint set and reconstruct teacher-replaced records.

    This is deliberately fail-closed: every expected shard and checksum must be
    present and must bind to the exact source record, config, and input file.
    """

    if limit is not None and limit <= 0:
        raise TeacherGenerationError("limit must be greater than zero when provided")
    resolved_input = input_path or Path(config.input_records)
    records = load_normalized_training_records_jsonl(
        resolved_input, expected_language=config.language
    )
    if limit is not None:
        records = records[:limit]
    input_file_sha256 = _file_sha256(resolved_input)
    config_sha256 = teacher_distillation_config_sha256(config)
    _ensure_run_identity(
        checkpoint_dir,
        config=config,
        config_sha256=config_sha256,
        input_file_sha256=input_file_sha256,
        total_records=len(records),
    )
    shard_size = config.checkpoint.shard_size
    total_shards = (len(records) + shard_size - 1) // shard_size
    output: list[NormalizedTrainingRecord] = []
    for shard_index in range(total_shards):
        start, end = _shard_bounds(shard_index, shard_size=shard_size, total=len(records))
        path = checkpoint_dir / "shards" / _shard_name(shard_index)
        if not _validate_durable_shard(
            path,
            records=records,
            start=start,
            end=end,
            config_sha256=config_sha256,
            implementation_sha256=_implementation_sha256(),
            input_file_sha256=input_file_sha256,
        ):
            raise TeacherGenerationError(f"required durable shard is missing: {path}")
        rows = _parse_shard(path)
        for source, row in zip(records[start:end], rows, strict=True):
            output.append(distilled_record_from_shard(source, row, config=config))
    if len(output) != len(records):
        raise TeacherGenerationError(
            "completed checkpoint set reconstructed the wrong record count"
        )
    return tuple(output)


def distilled_record_from_shard(
    source: NormalizedTrainingRecord,
    shard_row: dict[str, object],
    *,
    config: TeacherDistillationConfig,
) -> NormalizedTrainingRecord:
    """Replace the source answer with the verified teacher final response."""

    final = shard_row.get("final_response")
    if not isinstance(final, str) or not final.strip():
        raise TeacherGenerationError("shard row final_response must be a non-empty string")
    parent_sha = normalized_record_fingerprint(source).record_sha256
    if shard_row.get("input_record_sha256") != parent_sha:
        raise TeacherGenerationError("shard row does not match the source record")
    metadata = dict(source.provenance.source_metadata)
    metadata.update(
        {
            "distillation.config_sha256": teacher_distillation_config_sha256(config),
            "distillation.completion_tokens": str(shard_row.get("completion_tokens", 0)),
            "distillation.final_response_sha256": str(shard_row.get("final_response_sha256", "")),
            "distillation.finish_reason": str(shard_row.get("finish_reason", "unknown")),
            "distillation.parent_record_sha256": parent_sha,
            "distillation.parent_source_id": source.provenance.source_id,
            "distillation.parent_source_revision": source.provenance.revision,
            "distillation.teacher_license": "Apache-2.0",
            "distillation.teacher_repository": config.teacher.repository,
            "distillation.teacher_revision": config.teacher.revision,
            "distillation.prompt_tokens": str(shard_row.get("prompt_tokens", 0)),
            "distillation.reasoning_chars": str(shard_row.get("reasoning_chars", 0)),
        }
    )
    provenance = SourceProvenance(
        source_id=f"teacher-qwen38-27b.{source.provenance.source_id}",
        revision=config.teacher.revision,
        license=source.provenance.license,
        split=source.provenance.split,
        record_id=parent_sha,
        url=source.provenance.url,
        source_metadata=tuple(sorted(metadata.items())),
    )
    return replace(
        source,
        messages=_prompt_messages(source) + (TrainingMessage(role="assistant", content=final),),
        provenance=provenance,
        validation=None,
    )


__all__ = [
    "TeacherBackend",
    "TeacherCompletion",
    "TeacherGenerationError",
    "TeacherGenerationResult",
    "TeacherGenerationStatus",
    "TeacherShardRecord",
    "distilled_record_from_shard",
    "inspect_teacher_generation",
    "load_completed_distilled_records",
    "run_teacher_generation",
]
