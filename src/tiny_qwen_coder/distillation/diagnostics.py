"""Length diagnostics for durable teacher-generation checkpoints."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from tiny_qwen_coder.data.length_filtering import load_canonical_tokenizer, tokenize_training_record
from tiny_qwen_coder.data.records import NormalizedTrainingRecord, TrainingMessage
from tiny_qwen_coder.distillation.config import (
    TeacherDistillationConfig,
    load_teacher_distillation_config,
)
from tiny_qwen_coder.distillation.generation import load_completed_distilled_records
from tiny_qwen_coder.distillation.input_policy import strip_teacher_input_policy
from tiny_qwen_coder.model.inspection import load_inspection_target


class TeacherDiagnosticsError(RuntimeError):
    """Raised when a teacher checkpoint cannot be diagnosed safely."""


@dataclass(frozen=True, slots=True)
class IntegerDistribution:
    """Compact deterministic nearest-rank distribution for non-negative integers."""

    count: int
    minimum: int | None
    maximum: int | None
    mean: float | None
    p50: int | None
    p90: int | None
    p95: int | None
    p99: int | None


@dataclass(frozen=True, slots=True)
class TeacherLengthDiagnosticRecord:
    """One record's teacher/runtime and student-tokenizer length evidence."""

    input_index: int
    record_id: str
    finish_reason: str
    teacher_prompt_tokens: int
    teacher_completion_tokens: int
    reasoning_chars: int
    teacher_final_answer_tokens: int
    student_prompt_tokens: int
    student_final_answer_tokens: int
    student_full_record_tokens: int
    student_max_tokens: int
    student_length_accepted: bool


@dataclass(frozen=True, slots=True)
class TeacherLengthDiagnosticsSummary:
    """Aggregate evidence used to qualify a bounded distillation experiment."""

    schema_version: int
    total_records: int
    stop_records: int
    stop_rate: float
    stop_and_student_length_accepted_records: int
    student_length_accept_rate_given_stop: float
    finish_reason_counts: tuple[tuple[str, int], ...]
    teacher_prompt_tokens: IntegerDistribution
    teacher_completion_tokens: IntegerDistribution
    reasoning_chars: IntegerDistribution
    teacher_final_answer_tokens: IntegerDistribution
    student_prompt_tokens: IntegerDistribution
    student_final_answer_tokens: IntegerDistribution
    student_full_record_tokens: IntegerDistribution


@dataclass(frozen=True, slots=True)
class TeacherLengthDiagnostics:
    """Detailed and aggregate length diagnostics."""

    records: tuple[TeacherLengthDiagnosticRecord, ...]
    summary: TeacherLengthDiagnosticsSummary


def _nearest_rank(values: Sequence[int], percentile: float) -> int:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _distribution(values: Sequence[int]) -> IntegerDistribution:
    if any(value < 0 for value in values):
        raise TeacherDiagnosticsError("diagnostic lengths must be non-negative")
    if not values:
        return IntegerDistribution(
            count=0,
            minimum=None,
            maximum=None,
            mean=None,
            p50=None,
            p90=None,
            p95=None,
            p99=None,
        )
    return IntegerDistribution(
        count=len(values),
        minimum=min(values),
        maximum=max(values),
        mean=sum(values) / len(values),
        p50=_nearest_rank(values, 0.50),
        p90=_nearest_rank(values, 0.90),
        p95=_nearest_rank(values, 0.95),
        p99=_nearest_rank(values, 0.99),
    )


def _metadata_int(record: NormalizedTrainingRecord, key: str) -> int:
    value = dict(record.provenance.source_metadata).get(key)
    if value is None:
        raise TeacherDiagnosticsError(f"distilled record is missing metadata {key!r}")
    try:
        parsed = int(value)
    except ValueError as exc:
        raise TeacherDiagnosticsError(f"distilled metadata {key!r} is not an integer") from exc
    if parsed < 0:
        raise TeacherDiagnosticsError(f"distilled metadata {key!r} must be non-negative")
    return parsed


def _finish_reason(record: NormalizedTrainingRecord) -> str:
    value = dict(record.provenance.source_metadata).get("distillation.finish_reason")
    if not value:
        raise TeacherDiagnosticsError("distilled record is missing finish-reason metadata")
    return value


def _plain_token_count(tokenizer: object, text: str) -> int:
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise TeacherDiagnosticsError("tokenizer does not expose encode")
    encoded = encode(text, add_special_tokens=False)
    if not isinstance(encoded, Sequence) or isinstance(encoded, (str, bytes, bytearray)):
        raise TeacherDiagnosticsError("tokenizer encode did not return a token sequence")
    return len(encoded)


def _prompt_token_count(tokenizer: object, messages: tuple[TrainingMessage, ...]) -> int:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    template = getattr(tokenizer, "chat_template", None)
    if not callable(apply_chat_template) or not isinstance(template, str) or not template:
        raise TeacherDiagnosticsError("student tokenizer does not expose a usable chat template")
    encoded = apply_chat_template(
        [{"role": message.role, "content": message.content} for message in messages],
        tokenize=True,
        add_generation_prompt=True,
        truncation=False,
        return_dict=False,
        chat_template=template,
    )
    if not isinstance(encoded, Sequence) or isinstance(encoded, (str, bytes, bytearray)):
        raise TeacherDiagnosticsError("chat-template tokenization did not return a token sequence")
    return len(encoded)


def diagnose_teacher_records(
    records: tuple[NormalizedTrainingRecord, ...],
    *,
    teacher_tokenizer: object,
    student_tokenizer: object,
    student_max_tokens: int = 2048,
) -> TeacherLengthDiagnostics:
    """Measure lengths without persisting hidden reasoning content.

    Teacher-runtime counters describe the exact teacher prompt/completion. Student
    counters first remove any teacher-only v2 prompt policy so they describe the
    conversation that would actually be written to the training corpus.
    """

    if student_max_tokens < 1:
        raise TeacherDiagnosticsError("student_max_tokens must be positive")
    diagnostics: list[TeacherLengthDiagnosticRecord] = []
    for input_index, record in enumerate(records):
        if len(record.messages) < 2 or record.messages[-1].role != "assistant":
            raise TeacherDiagnosticsError("distilled record must end with an assistant answer")
        student_record = strip_teacher_input_policy(record)
        answer = student_record.messages[-1].content
        prompt = student_record.messages[:-1]
        full_tokens = len(tokenize_training_record(student_tokenizer, student_record))
        diagnostics.append(
            TeacherLengthDiagnosticRecord(
                input_index=input_index,
                record_id=record.provenance.record_id or f"index-{input_index}",
                finish_reason=_finish_reason(record),
                teacher_prompt_tokens=_metadata_int(record, "distillation.prompt_tokens"),
                teacher_completion_tokens=_metadata_int(record, "distillation.completion_tokens"),
                reasoning_chars=_metadata_int(record, "distillation.reasoning_chars"),
                teacher_final_answer_tokens=_plain_token_count(teacher_tokenizer, answer),
                student_prompt_tokens=_prompt_token_count(student_tokenizer, prompt),
                student_final_answer_tokens=_plain_token_count(student_tokenizer, answer),
                student_full_record_tokens=full_tokens,
                student_max_tokens=student_max_tokens,
                student_length_accepted=full_tokens <= student_max_tokens,
            )
        )

    frozen = tuple(diagnostics)
    finish_counts = Counter(item.finish_reason for item in frozen)
    stop = tuple(item for item in frozen if item.finish_reason == "stop")
    stop_accepted = sum(item.student_length_accepted for item in stop)
    summary = TeacherLengthDiagnosticsSummary(
        schema_version=1,
        total_records=len(frozen),
        stop_records=len(stop),
        stop_rate=(len(stop) / len(frozen) if frozen else 0.0),
        stop_and_student_length_accepted_records=stop_accepted,
        student_length_accept_rate_given_stop=(stop_accepted / len(stop) if stop else 0.0),
        finish_reason_counts=tuple(sorted(finish_counts.items())),
        teacher_prompt_tokens=_distribution(tuple(item.teacher_prompt_tokens for item in frozen)),
        teacher_completion_tokens=_distribution(
            tuple(item.teacher_completion_tokens for item in frozen)
        ),
        reasoning_chars=_distribution(tuple(item.reasoning_chars for item in frozen)),
        teacher_final_answer_tokens=_distribution(
            tuple(item.teacher_final_answer_tokens for item in frozen)
        ),
        student_prompt_tokens=_distribution(tuple(item.student_prompt_tokens for item in frozen)),
        student_final_answer_tokens=_distribution(
            tuple(item.student_final_answer_tokens for item in frozen)
        ),
        student_full_record_tokens=_distribution(
            tuple(item.student_full_record_tokens for item in frozen)
        ),
    )
    return TeacherLengthDiagnostics(records=frozen, summary=summary)


def diagnose_teacher_checkpoint(
    *,
    distillation_config_path: Path,
    checkpoint_dir: Path,
    input_path: Path | None = None,
    base_config: Path = Path("configs/base/qwen35-4b.yaml"),
    local_files_only: bool = False,
    limit: int | None = None,
    student_max_tokens: int = 2048,
) -> TeacherLengthDiagnostics:
    """Load pinned teacher/student tokenizers and diagnose a durable checkpoint."""

    from transformers import AutoTokenizer

    config: TeacherDistillationConfig = load_teacher_distillation_config(distillation_config_path)
    records = load_completed_distilled_records(
        config,
        checkpoint_dir=checkpoint_dir,
        input_path=input_path,
        limit=limit,
    )
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        config.teacher.repository,
        revision=config.teacher.revision,
        trust_remote_code=False,
        local_files_only=local_files_only,
    )
    target = load_inspection_target(base_config)
    student_tokenizer = load_canonical_tokenizer(target, local_files_only=local_files_only)
    return diagnose_teacher_records(
        records,
        teacher_tokenizer=teacher_tokenizer,
        student_tokenizer=student_tokenizer,
        student_max_tokens=student_max_tokens,
    )


def write_teacher_diagnostics(diagnostics: TeacherLengthDiagnostics, output_dir: Path) -> None:
    """Write detailed diagnostics without any hidden-reasoning content."""

    output_dir.mkdir(parents=True, exist_ok=True)
    detail = "".join(
        json.dumps(asdict(record), sort_keys=True, separators=(",", ":")) + "\n"
        for record in diagnostics.records
    )
    (output_dir / "teacher-length-diagnostics.jsonl").write_text(detail, encoding="utf-8")
    (output_dir / "teacher-length-summary.json").write_text(
        json.dumps(asdict(diagnostics.summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "IntegerDistribution",
    "TeacherDiagnosticsError",
    "TeacherLengthDiagnosticRecord",
    "TeacherLengthDiagnostics",
    "TeacherLengthDiagnosticsSummary",
    "diagnose_teacher_checkpoint",
    "diagnose_teacher_records",
    "write_teacher_diagnostics",
]
