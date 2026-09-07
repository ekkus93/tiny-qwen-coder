from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

from tiny_qwen_coder.data.records import (
    LicenseMetadata,
    NormalizedTrainingRecord,
    SourceProvenance,
    TrainingMessage,
)
from tiny_qwen_coder.distillation.config import load_teacher_distillation_config
from tiny_qwen_coder.distillation.v3_input import apply_v3_teacher_input_policy
from tiny_qwen_coder.distillation.v4_compression import (
    V4CompressionShardRecord,
    V4CompressionStatus,
    build_v4_merged_records,
    select_v4_compression_targets,
)


class _FakeTokenizer:
    chat_template = "fixture-template"

    def encode(self, text: str, **_: object) -> list[int]:
        return list(range(max(1, len(text.split()))))

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


def _source_record() -> NormalizedTrainingRecord:
    return NormalizedTrainingRecord(
        schema_version=1,
        messages=(
            TrainingMessage(role="system", content="Write correct Python."),
            TrainingMessage(role="user", content="Return the requested value."),
            TrainingMessage(role="assistant", content="return 1"),
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


def _v3_distilled(*, answer_words: int) -> NormalizedTrainingRecord:
    transformed = apply_v3_teacher_input_policy(
        _source_record(),
        student_tokenizer=_FakeTokenizer(),
    )
    metadata = dict(transformed.provenance.source_metadata)
    metadata.update(
        {
            "distillation.finish_reason": "stop",
            "distillation.prompt_tokens": "100",
            "distillation.completion_tokens": str(answer_words),
            "distillation.reasoning_chars": "0",
            "distillation.final_response_sha256": "a" * 64,
        }
    )
    return replace(
        transformed,
        messages=transformed.messages[:-1]
        + (
            TrainingMessage(
                role="assistant",
                content=" ".join(f"token-{index}" for index in range(answer_words)),
            ),
        ),
        provenance=replace(
            transformed.provenance,
            source_metadata=tuple(sorted(metadata.items())),
        ),
    )


def test_v4_config_changes_only_reasoning_effort_from_v3_generation_contract() -> None:
    v3 = load_teacher_distillation_config(Path("configs/distillation/python/qwen38_27b_v3.yaml"))
    v4 = load_teacher_distillation_config(
        Path("configs/distillation/python/qwen38_27b_v4_compression.yaml")
    )

    assert v4.teacher == v3.teacher
    assert replace(v4.generation, reasoning_effort="high") == v3.generation
    assert v4.generation.reasoning_effort == "low"
    assert v4.generation.max_tokens == 8192
    assert v4.checkpoint == v3.checkpoint


def test_v4_selects_only_normal_stop_student_overlength_budget_violations() -> None:
    tokenizer = _FakeTokenizer()
    long = _v3_distilled(answer_words=2200)
    short = _v3_distilled(answer_words=100)

    targets = select_v4_compression_targets((long, short), student_tokenizer=tokenizer)

    assert len(targets) == 1
    assert targets[0].input_index == 0
    assert targets[0].original_answer_tokens == 2200
    assert targets[0].original_full_record_tokens > 2048
    assert targets[0].original_answer_tokens > targets[0].answer_budget_tokens


def test_v4_merge_replaces_only_target_and_clears_active_v3_policy(tmp_path: Path) -> None:
    tokenizer = _FakeTokenizer()
    source = _v3_distilled(answer_words=2200)
    checkpoint = tmp_path / "checkpoint"
    shards = checkpoint / "shards"
    shards.mkdir(parents=True)
    compressed = " ".join(f"short-{index}" for index in range(200))
    row = V4CompressionShardRecord(
        schema_version=1,
        compression_config_sha256="b" * 64,
        implementation_sha256="c" * 64,
        source_run_identity_sha256="d" * 64,
        source_input_index=0,
        source_record_sha256="e" * 64,
        source_final_response_sha256="f" * 64,
        target_answer_budget_tokens=1792,
        original_answer_tokens=2200,
        original_full_record_tokens=2210,
        compression_prompt_sha256="1" * 64,
        seed=1729,
        compressed_response=compressed,
        compressed_response_sha256="2" * 64,
        reasoning_sha256=None,
        reasoning_chars=0,
        finish_reason="stop",
        prompt_tokens=300,
        completion_tokens=200,
        compressed_answer_tokens=200,
        compressed_full_record_tokens=210,
    )
    (shards / "shard-000000.jsonl").write_text(
        json.dumps(asdict(row), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    status = V4CompressionStatus(
        source_records=1,
        compression_targets=1,
        completed_targets=1,
        completed_shards=1,
        total_shards=1,
        missing_shards=(),
        checkpoint_dir=checkpoint,
    )
    config = load_teacher_distillation_config(
        Path("configs/distillation/python/qwen38_27b_v4_compression.yaml")
    )

    merged, summary = build_v4_merged_records(
        source_records=(source,),
        compression_config=config,
        status=status,
        student_tokenizer=tokenizer,
    )

    assert merged[0].messages[-1].content == compressed
    metadata = dict(merged[0].provenance.source_metadata)
    assert "distillation.input_policy" not in metadata
    assert "distillation.input_policy_sha256" not in metadata
    assert metadata["distillation.v4.compressed"] == "true"
    assert metadata["distillation.v4.policy_id"] == "selective-compression-v4"
    assert summary.compression_targets == 1
    assert summary.rescued_student_length_records == 1
    assert summary.final_student_length_accepted == 1
