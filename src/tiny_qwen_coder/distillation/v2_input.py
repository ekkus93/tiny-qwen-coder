"""Deterministic prompt policy for the bounded v2 teacher study."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from tiny_qwen_coder.data.loading import load_normalized_training_records_jsonl
from tiny_qwen_coder.data.records import NormalizedTrainingRecord, TrainingMessage

V2_DISTILLATION_INSTRUCTION = (
    "Distillation requirement: solve the user's task completely, but keep the final answer concise "
    "enough for a Qwen3.5-4B training record with a 2,048-token full-conversation limit. Prefer "
    "direct code and only the explanation needed to satisfy the request. Do not repeat the prompt. "
    "Private reasoning may happen internally, but the final answer must stand on its own."
)
_V2_POLICY_ID = "concise-v2"
_V2_POLICY_SUFFIX = f"\n\n{V2_DISTILLATION_INSTRUCTION}"


class TeacherV2InputError(ValueError):
    """Raised when the bounded v2 prompt policy cannot be applied safely."""


@dataclass(frozen=True, slots=True)
class TeacherV2InputSummary:
    """Identity of one transformed v2 input file."""

    schema_version: int
    input_records: int
    output_records: int
    input_sha256: str
    output_sha256: str
    policy_id: str
    policy_sha256: str


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _policy_sha256() -> str:
    return hashlib.sha256(V2_DISTILLATION_INSTRUCTION.encode("utf-8")).hexdigest()


def _policy_metadata(record: NormalizedTrainingRecord) -> tuple[str | None, str | None]:
    metadata = dict(record.provenance.source_metadata)
    return (
        metadata.get("distillation.input_policy"),
        metadata.get("distillation.input_policy_sha256"),
    )


def apply_v2_teacher_input_policy(record: NormalizedTrainingRecord) -> NormalizedTrainingRecord:
    """Add the teacher-only concise-final-answer policy without changing the user request."""

    if not record.messages or record.messages[-1].role != "assistant":
        raise TeacherV2InputError("v2 teacher input must end with the source assistant answer")
    if _policy_metadata(record) != (None, None):
        raise TeacherV2InputError("v2 teacher input already contains distillation policy metadata")
    prompt = list(record.messages[:-1])
    if not prompt or prompt[-1].role != "user":
        raise TeacherV2InputError("v2 teacher input prompt must end with a user message")

    if prompt[0].role == "system":
        prompt[0] = TrainingMessage(
            role="system",
            content=f"{prompt[0].content.rstrip()}{_V2_POLICY_SUFFIX}",
        )
    else:
        prompt.insert(0, TrainingMessage(role="system", content=V2_DISTILLATION_INSTRUCTION))

    metadata = dict(record.provenance.source_metadata)
    metadata.update(
        {
            "distillation.input_policy": _V2_POLICY_ID,
            "distillation.input_policy_sha256": _policy_sha256(),
        }
    )
    return replace(
        record,
        messages=tuple(prompt) + (record.messages[-1],),
        provenance=replace(record.provenance, source_metadata=tuple(sorted(metadata.items()))),
    )


def strip_v2_teacher_input_policy(record: NormalizedTrainingRecord) -> NormalizedTrainingRecord:
    """Remove the teacher-only v2 instruction before student tokenization or corpus writing.

    Distillation-policy metadata is deliberately retained as provenance. Only the
    synthetic teacher instruction is removed from the conversation that the student
    will train on.
    """

    policy_id, policy_sha256 = _policy_metadata(record)
    if policy_id is None and policy_sha256 is None:
        return record
    if policy_id != _V2_POLICY_ID or policy_sha256 != _policy_sha256():
        raise TeacherV2InputError("distilled record has unknown or corrupted v2 policy metadata")
    if not record.messages or record.messages[-1].role != "assistant":
        raise TeacherV2InputError("v2 distilled record must end with an assistant answer")

    messages = list(record.messages)
    first = messages[0]
    if first.role != "system":
        raise TeacherV2InputError("v2 policy metadata exists without a leading system message")
    if first.content == V2_DISTILLATION_INSTRUCTION:
        del messages[0]
    elif first.content.endswith(_V2_POLICY_SUFFIX):
        original_system = first.content[: -len(_V2_POLICY_SUFFIX)]
        if not original_system:
            raise TeacherV2InputError(
                "v2 policy stripping produced an empty original system message"
            )
        messages[0] = TrainingMessage(role="system", content=original_system)
    else:
        raise TeacherV2InputError("v2 policy metadata does not match the teacher prompt content")

    if not messages or messages[-1].role != "assistant":
        raise TeacherV2InputError("v2 policy stripping removed required training messages")
    return replace(record, messages=tuple(messages))


def write_v2_teacher_input(
    *,
    input_path: Path,
    output_path: Path,
    language: str = "python",
) -> TeacherV2InputSummary:
    """Transform and SHA-256 seal one selected teacher-input subset."""

    records = load_normalized_training_records_jsonl(input_path, expected_language=language)
    if not records:
        raise TeacherV2InputError("v2 teacher input source is empty")
    transformed = tuple(apply_v2_teacher_input_policy(record) for record in records)
    content = "".join(
        json.dumps(asdict(record), sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
        for record in transformed
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, output_path)
    output_sha256 = _file_sha256(output_path)
    output_path.with_suffix(output_path.suffix + ".sha256").write_text(
        f"{output_sha256}  {output_path.name}\n",
        encoding="ascii",
    )
    summary = TeacherV2InputSummary(
        schema_version=1,
        input_records=len(records),
        output_records=len(transformed),
        input_sha256=_file_sha256(input_path),
        output_sha256=output_sha256,
        policy_id=_V2_POLICY_ID,
        policy_sha256=_policy_sha256(),
    )
    output_path.with_suffix(output_path.suffix + ".summary.json").write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


__all__ = [
    "TeacherV2InputError",
    "TeacherV2InputSummary",
    "V2_DISTILLATION_INSTRUCTION",
    "apply_v2_teacher_input_policy",
    "strip_v2_teacher_input_policy",
    "write_v2_teacher_input",
]
