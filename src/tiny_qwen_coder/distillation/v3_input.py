"""Deterministic per-record answer-budget policy for the bounded v3 teacher study."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from tiny_qwen_coder.data.loading import load_normalized_training_records_jsonl
from tiny_qwen_coder.data.records import NormalizedTrainingRecord, TrainingMessage

V3_POLICY_ID = "budgeted-concise-v3"
V3_STUDENT_MAX_TOKENS = 2048
V3_SAFETY_RESERVE_TOKENS = 128
V3_MAX_ANSWER_TOKENS = 1792
V3_DISTILLATION_INSTRUCTION_TEMPLATE = (
    "Distillation requirement: solve the user's task completely. Your final answer has a strict "
    "budget of approximately {answer_budget} tokens. Keep the final answer at or below that budget. "
    "Prefer direct code and only the explanation required to satisfy the request. Do not repeat the "
    "prompt, provide alternative implementations unless requested, or add lengthy commentary. "
    "Private reasoning may happen internally, but only the concise final answer will be used for "
    "student training."
)


class TeacherV3InputError(ValueError):
    """Raised when the bounded v3 prompt policy cannot be applied safely."""


@dataclass(frozen=True, slots=True)
class TeacherV3InputSummary:
    """Identity and budget range for one transformed v3 input file."""

    schema_version: int
    input_records: int
    output_records: int
    input_sha256: str
    output_sha256: str
    policy_id: str
    policy_sha256: str
    student_max_tokens: int
    safety_reserve_tokens: int
    max_answer_tokens: int
    minimum_answer_budget_tokens: int
    maximum_answer_budget_tokens: int


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _policy_sha256() -> str:
    payload = json.dumps(
        {
            "policy_id": V3_POLICY_ID,
            "template": V3_DISTILLATION_INSTRUCTION_TEMPLATE,
            "student_max_tokens": V3_STUDENT_MAX_TOKENS,
            "safety_reserve_tokens": V3_SAFETY_RESERVE_TOKENS,
            "max_answer_tokens": V3_MAX_ANSWER_TOKENS,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _metadata(record: NormalizedTrainingRecord) -> dict[str, str]:
    return dict(record.provenance.source_metadata)


def _require_chat_template(tokenizer: object) -> str:
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template:
        raise TeacherV3InputError("student tokenizer does not expose a usable chat template")
    return template


def _student_prompt_token_count(
    tokenizer: object,
    messages: tuple[TrainingMessage, ...],
) -> int:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise TeacherV3InputError("student tokenizer does not expose apply_chat_template")
    encoded = apply_chat_template(
        [{"role": message.role, "content": message.content} for message in messages],
        tokenize=True,
        add_generation_prompt=True,
        truncation=False,
        return_dict=False,
        chat_template=_require_chat_template(tokenizer),
    )
    if not isinstance(encoded, Sequence) or isinstance(encoded, (str, bytes, bytearray)):
        raise TeacherV3InputError("student prompt tokenization did not return a token sequence")
    return len(encoded)


def _answer_budget(prompt_tokens: int) -> int:
    if prompt_tokens < 1:
        raise TeacherV3InputError("student prompt token count must be positive")
    budget = min(
        V3_MAX_ANSWER_TOKENS,
        V3_STUDENT_MAX_TOKENS - prompt_tokens - V3_SAFETY_RESERVE_TOKENS,
    )
    if budget <= 0:
        raise TeacherV3InputError(
            "source prompt leaves no positive v3 final-answer budget inside the student envelope"
        )
    return budget


def _render_instruction(answer_budget: int) -> str:
    if answer_budget <= 0 or answer_budget > V3_MAX_ANSWER_TOKENS:
        raise TeacherV3InputError("v3 final-answer budget is outside the supported range")
    return V3_DISTILLATION_INSTRUCTION_TEMPLATE.format(answer_budget=answer_budget)


def _policy_suffix(answer_budget: int) -> str:
    return f"\n\n{_render_instruction(answer_budget)}"


def apply_v3_teacher_input_policy(
    record: NormalizedTrainingRecord,
    *,
    student_tokenizer: object,
) -> NormalizedTrainingRecord:
    """Inject a teacher-only answer budget derived from the canonical student tokenizer."""

    if not record.messages or record.messages[-1].role != "assistant":
        raise TeacherV3InputError("v3 teacher input must end with the source assistant answer")
    metadata = _metadata(record)
    if "distillation.input_policy" in metadata or "distillation.input_policy_sha256" in metadata:
        raise TeacherV3InputError("v3 teacher input already contains distillation policy metadata")

    prompt = list(record.messages[:-1])
    if not prompt or prompt[-1].role != "user":
        raise TeacherV3InputError("v3 teacher input prompt must end with a user message")
    prompt_tokens = _student_prompt_token_count(student_tokenizer, tuple(prompt))
    answer_budget = _answer_budget(prompt_tokens)
    instruction = _render_instruction(answer_budget)

    if prompt[0].role == "system":
        prompt[0] = TrainingMessage(
            role="system",
            content=f"{prompt[0].content.rstrip()}{_policy_suffix(answer_budget)}",
        )
    else:
        prompt.insert(0, TrainingMessage(role="system", content=instruction))

    metadata.update(
        {
            "distillation.input_policy": V3_POLICY_ID,
            "distillation.input_policy_sha256": _policy_sha256(),
            "distillation.student_prompt_tokens": str(prompt_tokens),
            "distillation.final_answer_budget_tokens": str(answer_budget),
            "distillation.student_max_tokens": str(V3_STUDENT_MAX_TOKENS),
            "distillation.answer_budget_reserve_tokens": str(V3_SAFETY_RESERVE_TOKENS),
            "distillation.answer_budget_ceiling_tokens": str(V3_MAX_ANSWER_TOKENS),
        }
    )
    return replace(
        record,
        messages=tuple(prompt) + (record.messages[-1],),
        provenance=replace(record.provenance, source_metadata=tuple(sorted(metadata.items()))),
    )


def _required_metadata_int(metadata: dict[str, str], key: str) -> int:
    value = metadata.get(key)
    if value is None:
        raise TeacherV3InputError(f"v3 policy metadata is missing {key!r}")
    try:
        parsed = int(value)
    except ValueError as exc:
        raise TeacherV3InputError(f"v3 policy metadata {key!r} is not an integer") from exc
    if parsed < 0:
        raise TeacherV3InputError(f"v3 policy metadata {key!r} must be non-negative")
    return parsed


def strip_v3_teacher_input_policy(record: NormalizedTrainingRecord) -> NormalizedTrainingRecord:
    """Remove the dynamic teacher-only v3 instruction before student use."""

    metadata = _metadata(record)
    policy_id = metadata.get("distillation.input_policy")
    policy_sha256 = metadata.get("distillation.input_policy_sha256")
    if policy_id is None and policy_sha256 is None:
        return record
    if policy_id != V3_POLICY_ID or policy_sha256 != _policy_sha256():
        raise TeacherV3InputError("distilled record has unknown or corrupted v3 policy metadata")
    if not record.messages or record.messages[-1].role != "assistant":
        raise TeacherV3InputError("v3 distilled record must end with an assistant answer")

    prompt_tokens = _required_metadata_int(metadata, "distillation.student_prompt_tokens")
    answer_budget = _required_metadata_int(metadata, "distillation.final_answer_budget_tokens")
    if _required_metadata_int(metadata, "distillation.student_max_tokens") != V3_STUDENT_MAX_TOKENS:
        raise TeacherV3InputError("v3 student token limit metadata does not match the policy")
    if (
        _required_metadata_int(metadata, "distillation.answer_budget_reserve_tokens")
        != V3_SAFETY_RESERVE_TOKENS
    ):
        raise TeacherV3InputError("v3 safety-reserve metadata does not match the policy")
    if (
        _required_metadata_int(metadata, "distillation.answer_budget_ceiling_tokens")
        != V3_MAX_ANSWER_TOKENS
    ):
        raise TeacherV3InputError("v3 answer-budget ceiling metadata does not match the policy")
    if answer_budget != _answer_budget(prompt_tokens):
        raise TeacherV3InputError("v3 answer-budget metadata is inconsistent with prompt length")

    messages = list(record.messages)
    first = messages[0]
    if first.role != "system":
        raise TeacherV3InputError("v3 policy metadata exists without a leading system message")
    instruction = _render_instruction(answer_budget)
    suffix = _policy_suffix(answer_budget)
    if first.content == instruction:
        del messages[0]
    elif first.content.endswith(suffix):
        original_system = first.content[: -len(suffix)]
        if not original_system:
            raise TeacherV3InputError("v3 policy stripping produced an empty original system message")
        messages[0] = TrainingMessage(role="system", content=original_system)
    else:
        raise TeacherV3InputError("v3 policy metadata does not match the teacher prompt content")

    if not messages or messages[-1].role != "assistant":
        raise TeacherV3InputError("v3 policy stripping removed required training messages")
    return replace(record, messages=tuple(messages))


def write_v3_teacher_input(
    *,
    input_path: Path,
    output_path: Path,
    student_tokenizer: object,
    language: str = "python",
) -> TeacherV3InputSummary:
    """Transform and SHA-256 seal one selected teacher-input subset with per-record budgets."""

    records = load_normalized_training_records_jsonl(input_path, expected_language=language)
    if not records:
        raise TeacherV3InputError("v3 teacher input source is empty")
    transformed = tuple(
        apply_v3_teacher_input_policy(record, student_tokenizer=student_tokenizer)
        for record in records
    )
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
    budgets = tuple(
        _required_metadata_int(_metadata(record), "distillation.final_answer_budget_tokens")
        for record in transformed
    )
    summary = TeacherV3InputSummary(
        schema_version=1,
        input_records=len(records),
        output_records=len(transformed),
        input_sha256=_file_sha256(input_path),
        output_sha256=output_sha256,
        policy_id=V3_POLICY_ID,
        policy_sha256=_policy_sha256(),
        student_max_tokens=V3_STUDENT_MAX_TOKENS,
        safety_reserve_tokens=V3_SAFETY_RESERVE_TOKENS,
        max_answer_tokens=V3_MAX_ANSWER_TOKENS,
        minimum_answer_budget_tokens=min(budgets),
        maximum_answer_budget_tokens=max(budgets),
    )
    output_path.with_suffix(output_path.suffix + ".summary.json").write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


__all__ = [
    "TeacherV3InputError",
    "TeacherV3InputSummary",
    "V3_DISTILLATION_INSTRUCTION_TEMPLATE",
    "V3_MAX_ANSWER_TOKENS",
    "V3_POLICY_ID",
    "V3_SAFETY_RESERVE_TOKENS",
    "V3_STUDENT_MAX_TOKENS",
    "apply_v3_teacher_input_policy",
    "strip_v3_teacher_input_policy",
    "write_v3_teacher_input",
]
