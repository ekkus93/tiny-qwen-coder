from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from tiny_qwen_coder.data.records import (
    LicenseMetadata,
    NormalizedTrainingRecord,
    SourceProvenance,
    TrainingMessage,
)
from tiny_qwen_coder.distillation.config import load_teacher_distillation_config
from tiny_qwen_coder.distillation.input_policy import (
    TeacherInputPolicyError,
    strip_teacher_input_policy,
)
from tiny_qwen_coder.distillation.v3_input import (
    V3_MAX_ANSWER_TOKENS,
    V3_POLICY_ID,
    V3_SAFETY_RESERVE_TOKENS,
    V3_STUDENT_MAX_TOKENS,
    apply_v3_teacher_input_policy,
    strip_v3_teacher_input_policy,
    write_v3_teacher_input,
)


class _FakeTokenizer:
    chat_template = "fixture-template"

    def apply_chat_template(self, messages: object, **kwargs: object) -> list[int]:
        assert isinstance(messages, list)
        token_count = 0
        for message in messages:
            assert isinstance(message, dict)
            content = message["content"]
            assert isinstance(content, str)
            token_count += max(1, len(content.split())) + 1
        if kwargs.get("add_generation_prompt") is True:
            token_count += 1
        return list(range(token_count))


def _record(*, user_text: str = "Return one.") -> NormalizedTrainingRecord:
    return NormalizedTrainingRecord(
        schema_version=1,
        messages=(
            TrainingMessage(role="system", content="Write correct Python."),
            TrainingMessage(role="user", content=user_text),
            TrainingMessage(role="assistant", content="print(1)"),
        ),
        language="python",
        provenance=SourceProvenance(
            source_id="fixture",
            revision="fixture-revision",
            license=LicenseMetadata(name="MIT"),
            split="train",
            record_id="fixture-1",
        ),
    )


def _metadata(record: NormalizedTrainingRecord) -> dict[str, str]:
    return dict(record.provenance.source_metadata)


def test_v3_short_prompt_uses_answer_budget_ceiling_and_round_trips() -> None:
    source = _record()
    transformed = apply_v3_teacher_input_policy(source, student_tokenizer=_FakeTokenizer())
    metadata = _metadata(transformed)

    assert metadata["distillation.input_policy"] == V3_POLICY_ID
    assert int(metadata["distillation.final_answer_budget_tokens"]) == V3_MAX_ANSWER_TOKENS
    assert int(metadata["distillation.student_max_tokens"]) == V3_STUDENT_MAX_TOKENS
    assert int(metadata["distillation.answer_budget_reserve_tokens"]) == V3_SAFETY_RESERVE_TOKENS
    assert str(V3_MAX_ANSWER_TOKENS) in transformed.messages[0].content

    distilled = replace(
        transformed,
        messages=transformed.messages[:-1]
        + (TrainingMessage(role="assistant", content="return 1"),),
    )
    stripped = strip_v3_teacher_input_policy(distilled)

    assert stripped.messages[:-1] == source.messages[:-1]
    assert stripped.messages[-1].content == "return 1"
    assert _metadata(stripped)["distillation.input_policy"] == V3_POLICY_ID


def test_v3_longer_prompt_gets_smaller_dynamic_budget() -> None:
    tokenizer = _FakeTokenizer()
    short = apply_v3_teacher_input_policy(_record(), student_tokenizer=tokenizer)
    long = apply_v3_teacher_input_policy(
        _record(user_text=" ".join(f"token-{index}" for index in range(300))),
        student_tokenizer=tokenizer,
    )

    short_metadata = _metadata(short)
    long_metadata = _metadata(long)
    short_budget = int(short_metadata["distillation.final_answer_budget_tokens"])
    long_budget = int(long_metadata["distillation.final_answer_budget_tokens"])
    long_prompt = int(long_metadata["distillation.student_prompt_tokens"])

    assert short_budget == V3_MAX_ANSWER_TOKENS
    assert long_budget < short_budget
    assert long_budget == min(
        V3_MAX_ANSWER_TOKENS,
        V3_STUDENT_MAX_TOKENS - long_prompt - V3_SAFETY_RESERVE_TOKENS,
    )


def test_v3_write_is_sealed_and_reports_budget_range(tmp_path: Path) -> None:
    records = (
        _record(),
        _record(user_text=" ".join(f"token-{index}" for index in range(300))),
    )
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "v3.jsonl"
    input_path.write_text(
        "".join(json.dumps(asdict(record), sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    summary = write_v3_teacher_input(
        input_path=input_path,
        output_path=output_path,
        student_tokenizer=_FakeTokenizer(),
    )

    assert summary.output_records == 2
    assert summary.minimum_answer_budget_tokens < summary.maximum_answer_budget_tokens
    assert summary.maximum_answer_budget_tokens == V3_MAX_ANSWER_TOKENS
    assert output_path.with_suffix(".jsonl.sha256").is_file()
    assert output_path.with_suffix(".jsonl.summary.json").is_file()


def test_generic_policy_stripper_dispatches_v3_and_rejects_unknown_policy() -> None:
    source = _record()
    transformed = apply_v3_teacher_input_policy(source, student_tokenizer=_FakeTokenizer())
    stripped = strip_teacher_input_policy(transformed)
    assert stripped.messages == source.messages

    metadata = _metadata(source)
    metadata["distillation.input_policy"] = "future-policy"
    unknown = replace(
        source,
        provenance=replace(source.provenance, source_metadata=tuple(sorted(metadata.items()))),
    )
    with pytest.raises(TeacherInputPolicyError, match="future-policy"):
        strip_teacher_input_policy(unknown)


def test_v3_config_preserves_v2_teacher_and_generation_contract() -> None:
    v2 = load_teacher_distillation_config(Path("configs/distillation/python/qwen38_27b_v2.yaml"))
    v3 = load_teacher_distillation_config(Path("configs/distillation/python/qwen38_27b_v3.yaml"))

    assert v3.teacher == v2.teacher
    assert v3.generation == v2.generation
    assert v3.generation.reasoning_effort == "high"
    assert v3.generation.max_tokens == 8192
    assert v3.checkpoint == v2.checkpoint
