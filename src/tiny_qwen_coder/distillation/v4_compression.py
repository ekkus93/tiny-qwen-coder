"""Selective second-pass compression for overlength v3 teacher answers."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from tiny_qwen_coder.data.deduplication import normalized_record_fingerprint
from tiny_qwen_coder.data.length_filtering import tokenize_training_record
from tiny_qwen_coder.data.records import NormalizedTrainingRecord, TrainingMessage
from tiny_qwen_coder.distillation.config import (
    TeacherDistillationConfig,
    teacher_distillation_config_sha256,
)
from tiny_qwen_coder.distillation.generation import TeacherBackend, TeacherCompletion
from tiny_qwen_coder.distillation.input_policy import strip_teacher_input_policy

V4_STUDENT_MAX_TOKENS = 2048
V4_POLICY_ID = "selective-compression-v4"
V4_COMPRESSION_SYSTEM = (
    "You are a precise editor compressing an already-solved programming answer for supervised "
    "fine-tuning. Preserve the draft's correctness, code behavior, required edge cases, and every "
    "detail needed to satisfy the original request. Do not solve a different problem or add new "
    "functionality. Remove repetition, alternatives not requested, and nonessential commentary. "
    "Return only the rewritten final answer; do not discuss the editing process."
)
_ACTIVE_V3_POLICY_KEYS = frozenset(
    {
        "distillation.input_policy",
        "distillation.input_policy_sha256",
        "distillation.student_prompt_tokens",
        "distillation.final_answer_budget_tokens",
        "distillation.student_max_tokens",
        "distillation.answer_budget_reserve_tokens",
        "distillation.answer_budget_ceiling_tokens",
    }
)


class TeacherV4CompressionError(RuntimeError):
    """Raised when selective compression cannot proceed safely."""


@dataclass(frozen=True, slots=True)
class V4CompressionTarget:
    """One normally completed v3 answer that exceeds the student envelope and its budget."""

    input_index: int
    record_id: str
    answer_budget_tokens: int
    original_answer_tokens: int
    original_full_record_tokens: int
    original_final_response_sha256: str


@dataclass(frozen=True, slots=True)
class V4CompressionShardRecord:
    """Durable evidence for one selective compression rewrite."""

    schema_version: int
    compression_config_sha256: str
    implementation_sha256: str
    source_run_identity_sha256: str
    source_input_index: int
    source_record_sha256: str
    source_final_response_sha256: str
    target_answer_budget_tokens: int
    original_answer_tokens: int
    original_full_record_tokens: int
    compression_prompt_sha256: str
    seed: int
    compressed_response: str
    compressed_response_sha256: str
    reasoning_sha256: str | None
    reasoning_chars: int
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    compressed_answer_tokens: int
    compressed_full_record_tokens: int


@dataclass(frozen=True, slots=True)
class V4CompressionStatus:
    """Validated durable progress for the selective compression pass."""

    source_records: int
    compression_targets: int
    completed_targets: int
    completed_shards: int
    total_shards: int
    missing_shards: tuple[int, ...]
    checkpoint_dir: Path

    @property
    def complete(self) -> bool:
        return not self.missing_shards


@dataclass(frozen=True, slots=True)
class V4CompressionSummary:
    """Aggregate before/after evidence written beside the merged v4 corpus."""

    schema_version: int
    source_records: int
    source_stop_records: int
    source_student_length_accepted: int
    compression_targets: int
    compression_stop_records: int
    compressed_within_budget: int
    compressed_student_length_accepted: int
    final_stop_records: int
    final_student_length_accepted: int
    rescued_student_length_records: int


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


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _implementation_sha256() -> str:
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


def _metadata(record: NormalizedTrainingRecord) -> dict[str, str]:
    return dict(record.provenance.source_metadata)


def _required_metadata_int(record: NormalizedTrainingRecord, key: str) -> int:
    value = _metadata(record).get(key)
    if value is None:
        raise TeacherV4CompressionError(f"v3 source record is missing metadata {key!r}")
    try:
        parsed = int(value)
    except ValueError as exc:
        raise TeacherV4CompressionError(f"v3 source metadata {key!r} is not an integer") from exc
    if parsed < 0:
        raise TeacherV4CompressionError(f"v3 source metadata {key!r} must be non-negative")
    return parsed


def _plain_token_count(tokenizer: object, text: str) -> int:
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise TeacherV4CompressionError("student tokenizer does not expose encode")
    encoded = encode(text, add_special_tokens=False)
    if not isinstance(encoded, Sequence) or isinstance(encoded, (str, bytes, bytearray)):
        raise TeacherV4CompressionError("student tokenizer encode did not return a token sequence")
    return len(encoded)


def _student_record(record: NormalizedTrainingRecord) -> NormalizedTrainingRecord:
    return strip_teacher_input_policy(record)


def select_v4_compression_targets(
    records: tuple[NormalizedTrainingRecord, ...],
    *,
    student_tokenizer: object,
    student_max_tokens: int = V4_STUDENT_MAX_TOKENS,
) -> tuple[V4CompressionTarget, ...]:
    """Select only normal-stop v3 records that are overlength and violated their v3 budget."""

    if student_max_tokens != V4_STUDENT_MAX_TOKENS:
        raise TeacherV4CompressionError("v4 selection is frozen to the 2048-token student boundary")
    targets: list[V4CompressionTarget] = []
    for input_index, source in enumerate(records):
        metadata = _metadata(source)
        if metadata.get("distillation.finish_reason") != "stop":
            continue
        student = _student_record(source)
        full_tokens = len(tokenize_training_record(student_tokenizer, student))
        if full_tokens <= student_max_tokens:
            continue
        answer_budget = _required_metadata_int(source, "distillation.final_answer_budget_tokens")
        answer = student.messages[-1].content
        answer_tokens = _plain_token_count(student_tokenizer, answer)
        if answer_tokens <= answer_budget:
            raise TeacherV4CompressionError(
                "student-overlength v3 source obeyed its final-answer budget; investigate token "
                "accounting before using selective compression"
            )
        targets.append(
            V4CompressionTarget(
                input_index=input_index,
                record_id=source.provenance.record_id or f"index-{input_index}",
                answer_budget_tokens=answer_budget,
                original_answer_tokens=answer_tokens,
                original_full_record_tokens=full_tokens,
                original_final_response_sha256=_sha256_text(answer),
            )
        )
    return tuple(targets)


def _compression_conversation(
    source: NormalizedTrainingRecord,
    target: V4CompressionTarget,
) -> tuple[TrainingMessage, ...]:
    student = _student_record(source)
    payload = {
        "target_final_answer_tokens": target.answer_budget_tokens,
        "original_conversation": [asdict(message) for message in student.messages[:-1]],
        "draft_answer": student.messages[-1].content,
        "requirements": [
            "Keep the rewritten final answer at or below the target token budget.",
            "Preserve correctness and all code or instructions necessary to satisfy the request.",
            "Prefer direct code and essential explanation only.",
            "Do not include private reasoning or commentary about compression.",
        ],
    }
    return (
        TrainingMessage(role="system", content=V4_COMPRESSION_SYSTEM),
        TrainingMessage(role="user", content=_canonical_json(payload)),
    )


def _compression_prompt_sha256(messages: tuple[TrainingMessage, ...]) -> str:
    return _sha256_text(_canonical_json([asdict(message) for message in messages]))


def _split_completion(text: str) -> tuple[str | None, str]:
    stripped = text.strip()
    if not stripped.startswith("<think>"):
        return None, stripped
    reasoning_and_close, separator, final = stripped.partition("</think>")
    if not separator:
        raise TeacherV4CompressionError("compression completion opened <think> without closing it")
    reasoning = reasoning_and_close.removeprefix("<think>").strip()
    final = final.strip()
    if not final:
        raise TeacherV4CompressionError("compression completion contained no final answer")
    return reasoning, final


def _source_fingerprint(record: NormalizedTrainingRecord) -> str:
    return normalized_record_fingerprint(record).record_sha256


def _source_run_identity_sha256(source_checkpoint_dir: Path) -> str:
    path = source_checkpoint_dir / "run-identity.json"
    if not path.is_file():
        raise TeacherV4CompressionError(f"v3 source run identity is missing: {path}")
    return _file_sha256(path)


def _run_identity_text(
    *,
    compression_config: TeacherDistillationConfig,
    source_run_identity_sha256: str,
    targets: tuple[V4CompressionTarget, ...],
) -> str:
    return (
        json.dumps(
            {
                "schema_version": 1,
                "policy_id": V4_POLICY_ID,
                "compression_config_sha256": teacher_distillation_config_sha256(compression_config),
                "implementation_sha256": _implementation_sha256(),
                "source_run_identity_sha256": source_run_identity_sha256,
                "target_input_indices": [target.input_index for target in targets],
                "target_source_response_sha256": [
                    target.original_final_response_sha256 for target in targets
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _atomic_write_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding=encoding)
    os.replace(temporary, path)


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def _ensure_run_identity(
    checkpoint_dir: Path,
    *,
    compression_config: TeacherDistillationConfig,
    source_run_identity_sha256: str,
    targets: tuple[V4CompressionTarget, ...],
) -> None:
    expected = _run_identity_text(
        compression_config=compression_config,
        source_run_identity_sha256=source_run_identity_sha256,
        targets=targets,
    )
    path = checkpoint_dir / "run-identity.json"
    if path.exists():
        if path.read_text(encoding="utf-8") != expected:
            raise TeacherV4CompressionError(
                "v4 compression checkpoint belongs to different source evidence or code"
            )
        return
    _atomic_write_text(path, expected)


def _shard_name(shard_index: int) -> str:
    return f"shard-{shard_index:06d}.jsonl"


def _parse_shard(path: Path) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            raise TeacherV4CompressionError(f"blank row in {path}:{line_number}")
        raw: object = json.loads(line)
        if not isinstance(raw, dict):
            raise TeacherV4CompressionError(f"v4 compression row must be a mapping: {path}")
        rows.append({str(key): value for key, value in raw.items()})
    return tuple(rows)


def _row_int(row: dict[str, object], key: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TeacherV4CompressionError(f"compression row field {key!r} must be an integer")
    return value


def _student_with_answer(
    source: NormalizedTrainingRecord,
    answer: str,
) -> NormalizedTrainingRecord:
    student = _student_record(source)
    return replace(
        student,
        messages=student.messages[:-1] + (TrainingMessage(role="assistant", content=answer),),
    )


def _validate_shard(
    path: Path,
    *,
    source_records: tuple[NormalizedTrainingRecord, ...],
    targets: tuple[V4CompressionTarget, ...],
    start: int,
    end: int,
    compression_config: TeacherDistillationConfig,
    source_run_identity_sha256: str,
    student_tokenizer: object,
) -> bool:
    sidecar = path.with_suffix(".sha256")
    if not path.exists():
        if sidecar.exists():
            raise TeacherV4CompressionError(
                f"compression checksum exists without payload: {sidecar}"
            )
        return False
    if not sidecar.exists():
        return False
    expected_sidecar = f"{_file_sha256(path)}  {path.name}\n"
    if sidecar.read_text(encoding="ascii") != expected_sidecar:
        raise TeacherV4CompressionError(f"compression shard checksum mismatch: {path}")
    rows = _parse_shard(path)
    selected = targets[start:end]
    if len(rows) != len(selected):
        raise TeacherV4CompressionError(
            f"compression shard {path} has {len(rows)} rows; expected {len(selected)}"
        )
    config_sha = teacher_distillation_config_sha256(compression_config)
    implementation_sha = _implementation_sha256()
    for target, row in zip(selected, rows, strict=True):
        source = source_records[target.input_index]
        response = row.get("compressed_response")
        if not isinstance(response, str) or not response.strip():
            raise TeacherV4CompressionError(f"compression shard {path} contains an empty response")
        expected_prompt = _compression_prompt_sha256(_compression_conversation(source, target))
        if row.get("schema_version") != 1:
            raise TeacherV4CompressionError(f"compression shard {path} has bad schema_version")
        if row.get("compression_config_sha256") != config_sha:
            raise TeacherV4CompressionError(f"compression shard {path} belongs to another config")
        if row.get("implementation_sha256") != implementation_sha:
            raise TeacherV4CompressionError(
                f"compression shard {path} belongs to another implementation"
            )
        if row.get("source_run_identity_sha256") != source_run_identity_sha256:
            raise TeacherV4CompressionError(
                f"compression shard {path} has wrong v3 source identity"
            )
        if row.get("source_input_index") != target.input_index:
            raise TeacherV4CompressionError(f"compression shard {path} has wrong source index")
        if row.get("source_record_sha256") != _source_fingerprint(source):
            raise TeacherV4CompressionError(
                f"compression shard {path} has wrong source fingerprint"
            )
        if row.get("source_final_response_sha256") != target.original_final_response_sha256:
            raise TeacherV4CompressionError(f"compression shard {path} has wrong source answer")
        if row.get("compression_prompt_sha256") != expected_prompt:
            raise TeacherV4CompressionError(
                f"compression shard {path} has wrong prompt fingerprint"
            )
        if row.get("compressed_response_sha256") != _sha256_text(response):
            raise TeacherV4CompressionError(
                f"compression shard {path} has wrong response fingerprint"
            )
        answer_tokens = _plain_token_count(student_tokenizer, response)
        full_tokens = len(
            tokenize_training_record(student_tokenizer, _student_with_answer(source, response))
        )
        if row.get("compressed_answer_tokens") != answer_tokens:
            raise TeacherV4CompressionError(f"compression shard {path} answer-token count changed")
        if row.get("compressed_full_record_tokens") != full_tokens:
            raise TeacherV4CompressionError(f"compression shard {path} full-token count changed")
    return True


def inspect_v4_compression(
    *,
    source_records: tuple[NormalizedTrainingRecord, ...],
    source_checkpoint_dir: Path,
    compression_config: TeacherDistillationConfig,
    checkpoint_dir: Path,
    student_tokenizer: object,
) -> V4CompressionStatus:
    """Validate source selection and durable compression shards without loading the teacher."""

    targets = select_v4_compression_targets(source_records, student_tokenizer=student_tokenizer)
    source_identity = _source_run_identity_sha256(source_checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "shards").mkdir(parents=True, exist_ok=True)
    _ensure_run_identity(
        checkpoint_dir,
        compression_config=compression_config,
        source_run_identity_sha256=source_identity,
        targets=targets,
    )
    shard_size = compression_config.checkpoint.shard_size
    total_shards = (len(targets) + shard_size - 1) // shard_size
    completed_targets = 0
    completed_shards = 0
    missing: list[int] = []
    for shard_index in range(total_shards):
        start = shard_index * shard_size
        end = min(start + shard_size, len(targets))
        path = checkpoint_dir / "shards" / _shard_name(shard_index)
        if _validate_shard(
            path,
            source_records=source_records,
            targets=targets,
            start=start,
            end=end,
            compression_config=compression_config,
            source_run_identity_sha256=source_identity,
            student_tokenizer=student_tokenizer,
        ):
            completed_shards += 1
            completed_targets += end - start
        else:
            missing.append(shard_index)
    return V4CompressionStatus(
        source_records=len(source_records),
        compression_targets=len(targets),
        completed_targets=completed_targets,
        completed_shards=completed_shards,
        total_shards=total_shards,
        missing_shards=tuple(missing),
        checkpoint_dir=checkpoint_dir,
    )


def _build_shard_records(
    *,
    source_records: tuple[NormalizedTrainingRecord, ...],
    targets: tuple[V4CompressionTarget, ...],
    completions: tuple[TeacherCompletion, ...],
    compression_config: TeacherDistillationConfig,
    source_run_identity_sha256: str,
    student_tokenizer: object,
) -> tuple[V4CompressionShardRecord, ...]:
    if len(targets) != len(completions):
        raise TeacherV4CompressionError(
            "compression backend returned a mismatched completion count"
        )
    rows: list[V4CompressionShardRecord] = []
    config_sha = teacher_distillation_config_sha256(compression_config)
    implementation_sha = _implementation_sha256()
    for target, completion in zip(targets, completions, strict=True):
        source = source_records[target.input_index]
        conversation = _compression_conversation(source, target)
        reasoning, final = _split_completion(completion.text)
        if not final.strip():
            raise TeacherV4CompressionError("compression teacher produced an empty final response")
        answer_tokens = _plain_token_count(student_tokenizer, final)
        full_tokens = len(
            tokenize_training_record(student_tokenizer, _student_with_answer(source, final))
        )
        rows.append(
            V4CompressionShardRecord(
                schema_version=1,
                compression_config_sha256=config_sha,
                implementation_sha256=implementation_sha,
                source_run_identity_sha256=source_run_identity_sha256,
                source_input_index=target.input_index,
                source_record_sha256=_source_fingerprint(source),
                source_final_response_sha256=target.original_final_response_sha256,
                target_answer_budget_tokens=target.answer_budget_tokens,
                original_answer_tokens=target.original_answer_tokens,
                original_full_record_tokens=target.original_full_record_tokens,
                compression_prompt_sha256=_compression_prompt_sha256(conversation),
                seed=compression_config.generation.seed + target.input_index,
                compressed_response=final,
                compressed_response_sha256=_sha256_text(final),
                reasoning_sha256=_sha256_text(reasoning) if reasoning is not None else None,
                reasoning_chars=len(reasoning) if reasoning is not None else 0,
                finish_reason=completion.finish_reason,
                prompt_tokens=completion.prompt_tokens,
                completion_tokens=completion.completion_tokens,
                compressed_answer_tokens=answer_tokens,
                compressed_full_record_tokens=full_tokens,
            )
        )
    return tuple(rows)


def run_v4_compression(
    *,
    source_records: tuple[NormalizedTrainingRecord, ...],
    source_checkpoint_dir: Path,
    compression_config: TeacherDistillationConfig,
    backend: TeacherBackend,
    checkpoint_dir: Path,
    work_dir: Path,
    student_tokenizer: object,
) -> V4CompressionStatus:
    """Generate only missing compression shards and copy them durably."""

    targets = select_v4_compression_targets(source_records, student_tokenizer=student_tokenizer)
    source_identity = _source_run_identity_sha256(source_checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "shards").mkdir(parents=True, exist_ok=True)
    (work_dir / "shards").mkdir(parents=True, exist_ok=True)
    _ensure_run_identity(
        checkpoint_dir,
        compression_config=compression_config,
        source_run_identity_sha256=source_identity,
        targets=targets,
    )
    shard_size = compression_config.checkpoint.shard_size
    total_shards = (len(targets) + shard_size - 1) // shard_size
    for shard_index in range(total_shards):
        start = shard_index * shard_size
        end = min(start + shard_size, len(targets))
        durable = checkpoint_dir / "shards" / _shard_name(shard_index)
        if _validate_shard(
            durable,
            source_records=source_records,
            targets=targets,
            start=start,
            end=end,
            compression_config=compression_config,
            source_run_identity_sha256=source_identity,
            student_tokenizer=student_tokenizer,
        ):
            continue
        selected = targets[start:end]
        conversations = tuple(
            _compression_conversation(source_records[target.input_index], target)
            for target in selected
        )
        seeds = tuple(
            compression_config.generation.seed + target.input_index for target in selected
        )
        completions = backend.generate(conversations, seeds=seeds)
        rows = _build_shard_records(
            source_records=source_records,
            targets=selected,
            completions=completions,
            compression_config=compression_config,
            source_run_identity_sha256=source_identity,
            student_tokenizer=student_tokenizer,
        )
        local = work_dir / "shards" / _shard_name(shard_index)
        _atomic_write_text(
            local,
            "".join(_canonical_json(asdict(row)) + "\n" for row in rows),
        )
        local_sidecar = local.with_suffix(".sha256")
        _atomic_write_text(
            local_sidecar, f"{_file_sha256(local)}  {local.name}\n", encoding="ascii"
        )
        _copy_atomic(local, durable)
        _copy_atomic(local_sidecar, durable.with_suffix(".sha256"))
        if not _validate_shard(
            durable,
            source_records=source_records,
            targets=targets,
            start=start,
            end=end,
            compression_config=compression_config,
            source_run_identity_sha256=source_identity,
            student_tokenizer=student_tokenizer,
        ):
            raise TeacherV4CompressionError(f"new compression shard did not validate: {durable}")
    return inspect_v4_compression(
        source_records=source_records,
        source_checkpoint_dir=source_checkpoint_dir,
        compression_config=compression_config,
        checkpoint_dir=checkpoint_dir,
        student_tokenizer=student_tokenizer,
    )


def _load_completed_rows(
    *,
    status: V4CompressionStatus,
) -> dict[int, dict[str, object]]:
    if not status.complete:
        raise TeacherV4CompressionError(
            "cannot merge v4 corpus until every compression shard is sealed"
        )
    rows: dict[int, dict[str, object]] = {}
    for shard_index in range(status.total_shards):
        path = status.checkpoint_dir / "shards" / _shard_name(shard_index)
        for row in _parse_shard(path):
            index = row.get("source_input_index")
            if not isinstance(index, int) or index in rows:
                raise TeacherV4CompressionError(
                    "compression checkpoint contains invalid source indices"
                )
            rows[index] = row
    if len(rows) != status.compression_targets:
        raise TeacherV4CompressionError(
            "compression checkpoint reconstructed the wrong target count"
        )
    return rows


def _clear_active_policy_metadata(metadata: dict[str, str]) -> None:
    historical = {
        "distillation.v4.source_input_policy": metadata.get("distillation.input_policy", "missing"),
        "distillation.v4.source_input_policy_sha256": metadata.get(
            "distillation.input_policy_sha256", "missing"
        ),
        "distillation.v4.source_answer_budget_tokens": metadata.get(
            "distillation.final_answer_budget_tokens", "missing"
        ),
    }
    for key in _ACTIVE_V3_POLICY_KEYS:
        metadata.pop(key, None)
    metadata.update(historical)


def build_v4_merged_records(
    *,
    source_records: tuple[NormalizedTrainingRecord, ...],
    compression_config: TeacherDistillationConfig,
    status: V4CompressionStatus,
    student_tokenizer: object,
) -> tuple[tuple[NormalizedTrainingRecord, ...], V4CompressionSummary]:
    """Merge compressed targets with untouched student-shaped v3 answers."""

    rows = _load_completed_rows(status=status)
    output: list[NormalizedTrainingRecord] = []
    source_stop = 0
    source_accepted = 0
    compression_stop = 0
    compressed_within_budget = 0
    compressed_accepted = 0
    final_stop = 0
    final_accepted = 0
    rescued = 0
    config_sha = teacher_distillation_config_sha256(compression_config)

    for input_index, source in enumerate(source_records):
        student = _student_record(source)
        metadata = _metadata(student)
        source_finish = metadata.get("distillation.finish_reason", "missing")
        source_full = len(tokenize_training_record(student_tokenizer, student))
        if source_finish == "stop":
            source_stop += 1
            if source_full <= V4_STUDENT_MAX_TOKENS:
                source_accepted += 1
        _clear_active_policy_metadata(metadata)
        metadata.update(
            {
                "distillation.v4.compression_config_sha256": config_sha,
                "distillation.v4.policy_id": V4_POLICY_ID,
                "distillation.v4.source_finish_reason": source_finish,
                "distillation.v4.source_final_response_sha256": _sha256_text(
                    student.messages[-1].content
                ),
                "distillation.v4.source_full_record_tokens": str(source_full),
            }
        )

        row = rows.get(input_index)
        answer = student.messages[-1].content
        if row is None:
            metadata["distillation.v4.compressed"] = "false"
            final_finish = source_finish
        else:
            compressed = row.get("compressed_response")
            if not isinstance(compressed, str) or not compressed.strip():
                raise TeacherV4CompressionError(
                    "sealed compression row has no usable final response"
                )
            answer = compressed
            final_finish = str(row.get("finish_reason", "unknown"))
            budget = _row_int(row, "target_answer_budget_tokens")
            compressed_answer_tokens = _row_int(row, "compressed_answer_tokens")
            compressed_full_tokens = _row_int(row, "compressed_full_record_tokens")
            if final_finish == "stop":
                compression_stop += 1
            if compressed_answer_tokens <= budget:
                compressed_within_budget += 1
            if final_finish == "stop" and compressed_full_tokens <= V4_STUDENT_MAX_TOKENS:
                compressed_accepted += 1
                if source_full > V4_STUDENT_MAX_TOKENS:
                    rescued += 1
            ratio = compressed_answer_tokens / max(1, _row_int(row, "original_answer_tokens"))
            metadata.update(
                {
                    "distillation.v4.compressed": "true",
                    "distillation.v4.compression_finish_reason": final_finish,
                    "distillation.v4.compression_prompt_tokens": str(row["prompt_tokens"]),
                    "distillation.v4.compression_completion_tokens": str(row["completion_tokens"]),
                    "distillation.v4.compression_reasoning_chars": str(row["reasoning_chars"]),
                    "distillation.v4.target_answer_budget_tokens": str(budget),
                    "distillation.v4.original_answer_tokens": str(row["original_answer_tokens"]),
                    "distillation.v4.compressed_answer_tokens": str(compressed_answer_tokens),
                    "distillation.v4.compressed_full_record_tokens": str(compressed_full_tokens),
                    "distillation.v4.compression_ratio": f"{ratio:.8f}",
                    "distillation.v4.original_final_response_sha256": str(
                        row["source_final_response_sha256"]
                    ),
                    "distillation.v4.compressed_final_response_sha256": str(
                        row["compressed_response_sha256"]
                    ),
                }
            )
            metadata["distillation.finish_reason"] = final_finish
            metadata["distillation.prompt_tokens"] = str(row["prompt_tokens"])
            metadata["distillation.completion_tokens"] = str(row["completion_tokens"])
            metadata["distillation.reasoning_chars"] = str(row["reasoning_chars"])
            metadata["distillation.final_response_sha256"] = str(row["compressed_response_sha256"])

        merged = replace(
            student,
            messages=student.messages[:-1] + (TrainingMessage(role="assistant", content=answer),),
            provenance=replace(
                student.provenance,
                source_id=f"teacher-qwen38-27b-v4.{student.provenance.source_id}",
                revision=compression_config.teacher.revision,
                source_metadata=tuple(sorted(metadata.items())),
            ),
            validation=None,
        )
        final_full = len(tokenize_training_record(student_tokenizer, merged))
        if final_finish == "stop":
            final_stop += 1
            if final_full <= V4_STUDENT_MAX_TOKENS:
                final_accepted += 1
        output.append(merged)

    summary = V4CompressionSummary(
        schema_version=1,
        source_records=len(source_records),
        source_stop_records=source_stop,
        source_student_length_accepted=source_accepted,
        compression_targets=status.compression_targets,
        compression_stop_records=compression_stop,
        compressed_within_budget=compressed_within_budget,
        compressed_student_length_accepted=compressed_accepted,
        final_stop_records=final_stop,
        final_student_length_accepted=final_accepted,
        rescued_student_length_records=rescued,
    )
    return tuple(output), summary


def write_v4_merged_records(
    *,
    records: tuple[NormalizedTrainingRecord, ...],
    summary: V4CompressionSummary,
    output_path: Path,
) -> None:
    """Write and seal the deterministic merged v4 corpus and compression summary."""

    content = "".join(_canonical_json(asdict(record)) + "\n" for record in records)
    _atomic_write_text(output_path, content)
    _atomic_write_text(
        output_path.with_suffix(output_path.suffix + ".sha256"),
        f"{_file_sha256(output_path)}  {output_path.name}\n",
        encoding="ascii",
    )
    _atomic_write_text(
        output_path.with_suffix(output_path.suffix + ".summary.json"),
        json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n",
    )


__all__ = [
    "TeacherV4CompressionError",
    "V4CompressionShardRecord",
    "V4CompressionStatus",
    "V4CompressionSummary",
    "V4CompressionTarget",
    "V4_POLICY_ID",
    "V4_STUDENT_MAX_TOKENS",
    "build_v4_merged_records",
    "inspect_v4_compression",
    "run_v4_compression",
    "select_v4_compression_targets",
    "write_v4_merged_records",
]
