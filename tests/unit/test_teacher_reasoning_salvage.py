from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from tiny_qwen_coder.data.deduplication import normalized_record_fingerprint
from tiny_qwen_coder.data.loading import load_normalized_training_records_jsonl
from tiny_qwen_coder.data.records import (
    LicenseMetadata,
    NormalizedTrainingRecord,
    SourceProvenance,
    TrainingMessage,
)
from tiny_qwen_coder.distillation.config import (
    load_teacher_distillation_config,
    teacher_distillation_config_sha256,
)
from tiny_qwen_coder.distillation.input_policy import strip_teacher_input_policy
from tiny_qwen_coder.distillation.legacy_salvage import (
    LegacyTeacherSalvageError,
    write_salvaged_teacher_records,
)
from tiny_qwen_coder.distillation.v3_input import apply_v3_teacher_input_policy


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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _file_sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _source(index: int) -> NormalizedTrainingRecord:
    return NormalizedTrainingRecord(
        schema_version=1,
        messages=(
            TrainingMessage(role="system", content="Write correct Python."),
            TrainingMessage(role="user", content=f"Return {index}."),
            TrainingMessage(role="assistant", content=f"print({index})"),
        ),
        language="python",
        provenance=SourceProvenance(
            source_id="fixture",
            revision="fixture-revision",
            license=LicenseMetadata(name="MIT"),
            split="train",
            record_id=str(index),
        ),
    )


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_legacy_fixture(
    tmp_path: Path,
    *,
    second_finish_reason: str = "length",
    second_response: str = "private reasoning truncated",
) -> tuple[Path, Path]:
    config = load_teacher_distillation_config(
        Path("configs/distillation/python/qwen38_27b_v3.yaml")
    )
    sources = tuple(
        apply_v3_teacher_input_policy(_source(index), student_tokenizer=_FakeTokenizer())
        for index in range(2)
    )
    input_path = tmp_path / "p0-2-v3.jsonl"
    input_path.write_text(
        "".join(_canonical_json(asdict(record)) + "\n" for record in sources),
        encoding="utf-8",
    )
    input_sha = _file_sha256(input_path)
    input_path.with_suffix(".jsonl.sha256").write_text(
        f"{input_sha}  {input_path.name}\n",
        encoding="ascii",
    )

    checkpoint = tmp_path / "legacy-checkpoint"
    shards = checkpoint / "shards"
    shards.mkdir(parents=True)
    implementation_sha = "f" * 64
    config_sha = teacher_distillation_config_sha256(config)
    identity = {
        "schema_version": 1,
        "config_sha256": config_sha,
        "implementation_sha256": implementation_sha,
        "input_file_sha256": input_sha,
        "total_records": 2,
        "config": asdict(config),
    }
    (checkpoint / "run-identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    responses = (
        ("private reasoning\n</think>\n\n```python\nprint(1)\n```", "stop"),
        (second_response, second_finish_reason),
    )
    rows: list[dict[str, object]] = []
    for index, (source, (response, finish_reason)) in enumerate(
        zip(sources, responses, strict=True)
    ):
        prompt = source.messages[:-1]
        prompt_sha = _sha256_text(_canonical_json([asdict(message) for message in prompt]))
        rows.append(
            {
                "schema_version": 1,
                "config_sha256": config_sha,
                "implementation_sha256": implementation_sha,
                "input_file_sha256": input_sha,
                "input_index": index,
                "input_record_sha256": normalized_record_fingerprint(source).record_sha256,
                "prompt_sha256": prompt_sha,
                "seed": config.generation.seed + index,
                "teacher_repository": config.teacher.repository,
                "teacher_revision": config.teacher.revision,
                "raw_completion_sha256": _sha256_text(response),
                "reasoning_sha256": None,
                "reasoning_chars": 0,
                "final_response": response,
                "final_response_sha256": _sha256_text(response),
                "finish_reason": finish_reason,
                "prompt_tokens": 100,
                "completion_tokens": 200,
            }
        )
    shard = shards / "shard-000000.jsonl"
    shard.write_text(
        "".join(_canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    shard.with_suffix(".sha256").write_text(
        f"{_file_sha256(shard)}  {shard.name}\n",
        encoding="ascii",
    )
    return input_path, checkpoint


def test_legacy_salvage_strips_closing_only_reasoning_and_preserves_v3_budget(
    tmp_path: Path,
) -> None:
    config = load_teacher_distillation_config(
        Path("configs/distillation/python/qwen38_27b_v3.yaml")
    )
    input_path, checkpoint = _write_legacy_fixture(tmp_path)
    output = tmp_path / "salvaged" / "p0-2-v3-sanitized.jsonl"
    identity_dir = tmp_path / "salvaged" / "identity"

    summary = write_salvaged_teacher_records(
        config=config,
        source_input_path=input_path,
        legacy_checkpoint_dir=checkpoint,
        output_path=output,
        identity_dir=identity_dir,
    )

    assert summary.source_records == 2
    assert summary.stop_records == 1
    assert summary.length_records == 1
    assert summary.closing_only_records == 1
    assert summary.truncated_reasoning_only_records == 1
    records = load_normalized_training_records_jsonl(output, expected_language="python")
    assert records[0].messages[-1].content == "```python\nprint(1)\n```"
    assert records[1].messages[-1].content == ""
    metadata = dict(records[0].provenance.source_metadata)
    assert metadata["distillation.reasoning_chars"] == str(len("private reasoning"))
    assert metadata["distillation.salvage.marker_mode"] == "closing-only"
    assert "distillation.final_answer_budget_tokens" in metadata
    student = strip_teacher_input_policy(records[0])
    assert student.messages[-1].content == "```python\nprint(1)\n```"
    assert "</think>" not in output.read_text(encoding="utf-8")
    identity = json.loads((identity_dir / "run-identity.json").read_text(encoding="utf-8"))
    assert identity["output_sha256"] == _file_sha256(output)


def test_legacy_salvage_fails_closed_on_normal_stop_without_reasoning_boundary(
    tmp_path: Path,
) -> None:
    config = load_teacher_distillation_config(
        Path("configs/distillation/python/qwen38_27b_v3.yaml")
    )
    input_path, checkpoint = _write_legacy_fixture(
        tmp_path,
        second_finish_reason="stop",
        second_response="private reasoning but no close marker",
    )

    with pytest.raises(LegacyTeacherSalvageError, match="no </think> boundary"):
        write_salvaged_teacher_records(
            config=config,
            source_input_path=input_path,
            legacy_checkpoint_dir=checkpoint,
            output_path=tmp_path / "salvaged.jsonl",
            identity_dir=tmp_path / "identity",
        )


def test_legacy_salvage_rejects_corrupt_shard_sidecar(tmp_path: Path) -> None:
    config = load_teacher_distillation_config(
        Path("configs/distillation/python/qwen38_27b_v3.yaml")
    )
    input_path, checkpoint = _write_legacy_fixture(tmp_path)
    (checkpoint / "shards" / "shard-000000.sha256").write_text(
        f"{'0' * 64}  shard-000000.jsonl\n",
        encoding="ascii",
    )

    with pytest.raises(LegacyTeacherSalvageError, match="checksum mismatch"):
        write_salvaged_teacher_records(
            config=config,
            source_input_path=input_path,
            legacy_checkpoint_dir=checkpoint,
            output_path=tmp_path / "salvaged.jsonl",
            identity_dir=tmp_path / "identity",
        )
