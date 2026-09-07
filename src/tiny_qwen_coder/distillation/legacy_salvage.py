"""Fail-closed salvage of legacy Qwen checkpoints affected by closing-only thinking output."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from tiny_qwen_coder.data.deduplication import normalized_record_fingerprint
from tiny_qwen_coder.data.loading import load_normalized_training_records_jsonl
from tiny_qwen_coder.data.records import (
    NormalizedTrainingRecord,
    SourceProvenance,
    TrainingMessage,
)
from tiny_qwen_coder.distillation.config import (
    TeacherDistillationConfig,
    parse_teacher_distillation_config,
    teacher_distillation_config_sha256,
)
from tiny_qwen_coder.distillation.reasoning import (
    THINK_CLOSE,
    THINK_OPEN,
    TeacherReasoningParseError,
    split_qwen_thinking_completion,
)

SALVAGE_POLICY_ID = "qwen-closing-think-salvage-v1"


class LegacyTeacherSalvageError(RuntimeError):
    """Raised when affected legacy evidence cannot be salvaged safely."""


@dataclass(frozen=True, slots=True)
class LegacyTeacherSalvageSummary:
    """Immutable accounting for one sanitized legacy generation checkpoint."""

    schema_version: int
    policy_id: str
    source_records: int
    stop_records: int
    length_records: int
    closing_only_records: int
    wrapped_records: int
    truncated_reasoning_only_records: int
    source_input_sha256: str
    source_run_identity_sha256: str
    source_implementation_sha256: str
    salvage_implementation_sha256: str
    output_sha256: str


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
        (Path(__file__), Path(__file__).with_name("reasoning.py")),
        key=lambda item: item.name,
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _load_mapping(path: Path) -> dict[str, object]:
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LegacyTeacherSalvageError(f"could not read JSON mapping {path}: {exc}") from exc
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise LegacyTeacherSalvageError(f"JSON value must be a string-keyed mapping: {path}")
    return {str(key): value for key, value in raw.items()}


def _required_str(mapping: dict[str, object], key: str, *, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise LegacyTeacherSalvageError(f"{context} field {key!r} must be a non-empty string")
    return value


def _required_int(mapping: dict[str, object], key: str, *, context: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise LegacyTeacherSalvageError(f"{context} field {key!r} must be an integer")
    return value


def _prompt_messages(record: NormalizedTrainingRecord) -> tuple[TrainingMessage, ...]:
    if not record.messages or record.messages[-1].role != "assistant":
        raise LegacyTeacherSalvageError(
            "legacy source record must end with the original assistant answer"
        )
    prompt = record.messages[:-1]
    if not prompt or prompt[-1].role != "user":
        raise LegacyTeacherSalvageError("legacy source prompt must end with a user message")
    return prompt


def _prompt_sha256(messages: tuple[TrainingMessage, ...]) -> str:
    return _sha256_text(_canonical_json([asdict(message) for message in messages]))


def _parse_shard(path: Path) -> tuple[dict[str, object], ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise LegacyTeacherSalvageError(f"could not read legacy shard {path}: {exc}") from exc
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise LegacyTeacherSalvageError(f"legacy shard {path}:{line_number} is blank")
        try:
            raw: object = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LegacyTeacherSalvageError(
                f"legacy shard {path}:{line_number} contains invalid JSON"
            ) from exc
        if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
            raise LegacyTeacherSalvageError(
                f"legacy shard {path}:{line_number} must be a string-keyed mapping"
            )
        rows.append({str(key): value for key, value in raw.items()})
    return tuple(rows)


def _validate_sidecar(path: Path) -> None:
    sidecar = path.with_suffix(".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise LegacyTeacherSalvageError(
            f"legacy shard is not durably sealed: payload={path}, sidecar={sidecar}"
        )
    expected = f"{_file_sha256(path)}  {path.name}\n"
    try:
        actual = sidecar.read_text(encoding="ascii")
    except OSError as exc:
        raise LegacyTeacherSalvageError(f"could not read legacy shard checksum: {sidecar}") from exc
    if actual != expected:
        raise LegacyTeacherSalvageError(f"legacy shard checksum mismatch: {path}")


def _salvage_stored_response(
    stored: str,
    *,
    finish_reason: str,
) -> tuple[str | None, str, str]:
    stripped = stored.strip()
    if THINK_OPEN in stripped or THINK_CLOSE in stripped:
        try:
            reasoning, final = split_qwen_thinking_completion(stripped)
        except TeacherReasoningParseError as exc:
            raise LegacyTeacherSalvageError(f"could not split legacy reasoning: {exc}") from exc
        mode = "wrapped" if stripped.startswith(THINK_OPEN) else "closing-only"
        return reasoning, final, mode
    if finish_reason == "length":
        # The affected Qwen template can spend the entire completion budget inside
        # prefilled thinking. Such rows have no final answer and are rejected later
        # by their frozen finish_reason, but the hidden text must not survive here.
        return stripped or None, "", "truncated-reasoning-only"
    raise LegacyTeacherSalvageError(
        "normal-stop legacy row contains no </think> boundary; refusing to guess where "
        "hidden reasoning ends"
    )


def _salvaged_record(
    source: NormalizedTrainingRecord,
    row: dict[str, object],
    *,
    config: TeacherDistillationConfig,
    source_run_identity_sha256: str,
    source_implementation_sha256: str,
) -> tuple[NormalizedTrainingRecord, str]:
    stored = _required_str(row, "final_response", context="legacy shard")
    stored_sha = _required_str(row, "final_response_sha256", context="legacy shard")
    if stored_sha != _sha256_text(stored):
        raise LegacyTeacherSalvageError("legacy final_response SHA-256 does not match its payload")
    finish_reason = _required_str(row, "finish_reason", context="legacy shard")
    recorded_reasoning_chars = _required_int(row, "reasoning_chars", context="legacy shard")
    recorded_reasoning_sha = row.get("reasoning_sha256")
    if (THINK_OPEN in stored or THINK_CLOSE in stored) and (
        recorded_reasoning_chars != 0 or recorded_reasoning_sha is not None
    ):
        raise LegacyTeacherSalvageError(
            "legacy row contains thinking markup but already records extracted reasoning; "
            "this salvage policy does not apply"
        )

    reasoning, final, mode = _salvage_stored_response(stored, finish_reason=finish_reason)
    if THINK_OPEN in final or THINK_CLOSE in final:
        raise LegacyTeacherSalvageError("salvaged final answer still contains thinking markup")

    parent_sha = normalized_record_fingerprint(source).record_sha256
    metadata = dict(source.provenance.source_metadata)
    metadata.update(
        {
            "distillation.config_sha256": teacher_distillation_config_sha256(config),
            "distillation.completion_tokens": str(
                _required_int(row, "completion_tokens", context="legacy shard")
            ),
            "distillation.final_response_sha256": _sha256_text(final),
            "distillation.finish_reason": finish_reason,
            "distillation.parent_record_sha256": parent_sha,
            "distillation.parent_source_id": source.provenance.source_id,
            "distillation.parent_source_revision": source.provenance.revision,
            "distillation.teacher_license": "Apache-2.0",
            "distillation.teacher_repository": config.teacher.repository,
            "distillation.teacher_revision": config.teacher.revision,
            "distillation.prompt_tokens": str(
                _required_int(row, "prompt_tokens", context="legacy shard")
            ),
            "distillation.reasoning_chars": str(len(reasoning) if reasoning is not None else 0),
            "distillation.salvage.legacy_final_response_sha256": stored_sha,
            "distillation.salvage.marker_mode": mode,
            "distillation.salvage.policy_id": SALVAGE_POLICY_ID,
            "distillation.salvage.source_implementation_sha256": source_implementation_sha256,
            "distillation.salvage.source_run_identity_sha256": source_run_identity_sha256,
        }
    )
    if reasoning is not None:
        metadata["distillation.reasoning_sha256"] = _sha256_text(reasoning)

    provenance = SourceProvenance(
        source_id=f"teacher-qwen38-27b-salvage.{source.provenance.source_id}",
        revision=config.teacher.revision,
        license=source.provenance.license,
        split=source.provenance.split,
        record_id=parent_sha,
        url=source.provenance.url,
        source_metadata=tuple(sorted(metadata.items())),
    )
    record = replace(
        source,
        messages=_prompt_messages(source) + (TrainingMessage(role="assistant", content=final),),
        provenance=provenance,
        validation=None,
    )
    return record, mode


def _write_exact_or_verify(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    if path.exists():
        try:
            actual = path.read_text(encoding=encoding)
        except OSError as exc:
            raise LegacyTeacherSalvageError(
                f"could not read existing salvage output {path}"
            ) from exc
        if actual != content:
            raise LegacyTeacherSalvageError(
                f"existing salvage output differs from deterministic reconstruction: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding=encoding)
    os.replace(temporary, path)


def write_salvaged_teacher_records(
    *,
    config: TeacherDistillationConfig,
    source_input_path: Path,
    legacy_checkpoint_dir: Path,
    output_path: Path,
    identity_dir: Path,
) -> LegacyTeacherSalvageSummary:
    """Validate the frozen legacy checkpoint and deterministically strip leaked reasoning."""

    records = load_normalized_training_records_jsonl(
        source_input_path,
        expected_language=config.language,
    )
    if not records:
        raise LegacyTeacherSalvageError("legacy source input is empty")

    source_input_sha = _file_sha256(source_input_path)
    input_sidecar = source_input_path.with_suffix(source_input_path.suffix + ".sha256")
    if input_sidecar.exists():
        expected = f"{source_input_sha}  {source_input_path.name}\n"
        if input_sidecar.read_text(encoding="ascii") != expected:
            raise LegacyTeacherSalvageError("legacy source-input checksum sidecar mismatch")

    identity_path = legacy_checkpoint_dir / "run-identity.json"
    if not identity_path.is_file():
        raise LegacyTeacherSalvageError(f"legacy run identity is missing: {identity_path}")
    identity = _load_mapping(identity_path)
    if _required_int(identity, "schema_version", context="legacy run identity") != 1:
        raise LegacyTeacherSalvageError("legacy run identity has unsupported schema_version")
    if _required_int(identity, "total_records", context="legacy run identity") != len(records):
        raise LegacyTeacherSalvageError(
            "legacy run identity record count does not match source input"
        )
    if (
        _required_str(identity, "input_file_sha256", context="legacy run identity")
        != source_input_sha
    ):
        raise LegacyTeacherSalvageError("legacy run identity input SHA-256 mismatch")

    raw_config = identity.get("config")
    if not isinstance(raw_config, dict) or any(not isinstance(key, str) for key in raw_config):
        raise LegacyTeacherSalvageError("legacy run identity config must be a string-keyed mapping")
    identity_config = parse_teacher_distillation_config(
        {str(key): value for key, value in raw_config.items()}
    )
    identity_config_sha = _required_str(identity, "config_sha256", context="legacy run identity")
    if teacher_distillation_config_sha256(identity_config) != identity_config_sha:
        raise LegacyTeacherSalvageError(
            "legacy run identity config hash is internally inconsistent"
        )
    if identity_config != config:
        raise LegacyTeacherSalvageError(
            "legacy run identity config does not match requested config"
        )

    source_implementation_sha = _required_str(
        identity, "implementation_sha256", context="legacy run identity"
    )
    source_run_identity_sha = _file_sha256(identity_path)
    shard_size = config.checkpoint.shard_size
    total_shards = (len(records) + shard_size - 1) // shard_size
    output: list[NormalizedTrainingRecord] = []
    modes: Counter[str] = Counter()
    finish_reasons: Counter[str] = Counter()

    for shard_index in range(total_shards):
        start = shard_index * shard_size
        end = min(start + shard_size, len(records))
        path = legacy_checkpoint_dir / "shards" / f"shard-{shard_index:06d}.jsonl"
        _validate_sidecar(path)
        rows = _parse_shard(path)
        if len(rows) != end - start:
            raise LegacyTeacherSalvageError(
                f"legacy shard {path} has {len(rows)} rows; expected {end - start}"
            )
        for offset, row in enumerate(rows):
            input_index = start + offset
            source = records[input_index]
            context = f"legacy shard {path} row {offset}"
            if _required_int(row, "schema_version", context=context) != 1:
                raise LegacyTeacherSalvageError(f"{context} has unsupported schema_version")
            if _required_str(row, "config_sha256", context=context) != identity_config_sha:
                raise LegacyTeacherSalvageError(f"{context} belongs to another config")
            if (
                _required_str(row, "implementation_sha256", context=context)
                != source_implementation_sha
            ):
                raise LegacyTeacherSalvageError(f"{context} belongs to another implementation")
            if _required_str(row, "input_file_sha256", context=context) != source_input_sha:
                raise LegacyTeacherSalvageError(f"{context} belongs to another input file")
            if _required_int(row, "input_index", context=context) != input_index:
                raise LegacyTeacherSalvageError(f"{context} has unexpected input_index")
            if (
                _required_str(row, "input_record_sha256", context=context)
                != normalized_record_fingerprint(source).record_sha256
            ):
                raise LegacyTeacherSalvageError(f"{context} source-record fingerprint mismatch")
            if _required_str(row, "prompt_sha256", context=context) != _prompt_sha256(
                _prompt_messages(source)
            ):
                raise LegacyTeacherSalvageError(f"{context} prompt fingerprint mismatch")
            if _required_int(row, "seed", context=context) != config.generation.seed + input_index:
                raise LegacyTeacherSalvageError(f"{context} seed mismatch")
            if (
                _required_str(row, "teacher_repository", context=context)
                != config.teacher.repository
            ):
                raise LegacyTeacherSalvageError(f"{context} teacher repository mismatch")
            if _required_str(row, "teacher_revision", context=context) != config.teacher.revision:
                raise LegacyTeacherSalvageError(f"{context} teacher revision mismatch")

            salvaged, mode = _salvaged_record(
                source,
                row,
                config=config,
                source_run_identity_sha256=source_run_identity_sha,
                source_implementation_sha256=source_implementation_sha,
            )
            output.append(salvaged)
            modes[mode] += 1
            finish_reasons[_required_str(row, "finish_reason", context=context)] += 1

    if len(output) != len(records):
        raise LegacyTeacherSalvageError("salvage reconstructed the wrong record count")

    content = "".join(_canonical_json(asdict(record)) + "\n" for record in output)
    output_sha = _sha256_text(content)
    _write_exact_or_verify(output_path, content)
    _write_exact_or_verify(
        output_path.with_suffix(output_path.suffix + ".sha256"),
        f"{output_sha}  {output_path.name}\n",
        encoding="ascii",
    )

    summary = LegacyTeacherSalvageSummary(
        schema_version=1,
        policy_id=SALVAGE_POLICY_ID,
        source_records=len(records),
        stop_records=finish_reasons["stop"],
        length_records=finish_reasons["length"],
        closing_only_records=modes["closing-only"],
        wrapped_records=modes["wrapped"],
        truncated_reasoning_only_records=modes["truncated-reasoning-only"],
        source_input_sha256=source_input_sha,
        source_run_identity_sha256=source_run_identity_sha,
        source_implementation_sha256=source_implementation_sha,
        salvage_implementation_sha256=_implementation_sha256(),
        output_sha256=output_sha,
    )
    summary_text = json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n"
    _write_exact_or_verify(
        output_path.with_suffix(output_path.suffix + ".summary.json"),
        summary_text,
    )

    salvage_identity = {
        "schema_version": 1,
        "policy_id": SALVAGE_POLICY_ID,
        "source_run_identity_sha256": source_run_identity_sha,
        "source_input_sha256": source_input_sha,
        "source_implementation_sha256": source_implementation_sha,
        "salvage_implementation_sha256": summary.salvage_implementation_sha256,
        "output_sha256": output_sha,
        "total_records": len(records),
        "summary": asdict(summary),
    }
    _write_exact_or_verify(
        identity_dir / "run-identity.json",
        json.dumps(salvage_identity, indent=2, sort_keys=True) + "\n",
    )
    return summary


__all__ = [
    "LegacyTeacherSalvageError",
    "LegacyTeacherSalvageSummary",
    "SALVAGE_POLICY_ID",
    "write_salvaged_teacher_records",
]
