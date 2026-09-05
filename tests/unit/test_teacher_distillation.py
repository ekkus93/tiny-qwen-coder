from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tiny_qwen_coder.data.records import (
    LicenseMetadata,
    NormalizedTrainingRecord,
    SourceProvenance,
    TrainingMessage,
)
from tiny_qwen_coder.distillation.config import (
    TeacherCheckpointConfig,
    TeacherDistillationConfig,
    TeacherDistillationConfigError,
    TeacherGenerationConfig,
    TeacherModelConfig,
    TeacherRuntimeConfig,
    parse_teacher_distillation_config,
    teacher_distillation_config_sha256,
)
from tiny_qwen_coder.distillation.generation import (
    TeacherCompletion,
    TeacherGenerationError,
    inspect_teacher_generation,
    load_completed_distilled_records,
    run_teacher_generation,
)

_REVISION = "a" * 40


def _config(*, shard_size: int = 2) -> TeacherDistillationConfig:
    return TeacherDistillationConfig(
        schema_version=1,
        id="python-qwen38-27b-test",
        language="python",
        input_records="unused.jsonl",
        runtime=TeacherRuntimeConfig(
            vllm_version="0.28.0",
            vllm_bnb_plugin_version="0.0.3",
            bitsandbytes_version="0.50.2",
        ),
        teacher=TeacherModelConfig(
            repository="Qwen/Qwen3.8-27B",
            revision=_REVISION,
            backend="vllm",
            dtype="bfloat16",
            quantization="bitsandbytes",
            max_model_len=4096,
            gpu_memory_utilization=0.9,
        ),
        generation=TeacherGenerationConfig(
            thinking=True,
            preserve_thinking=False,
            reasoning_effort="xhigh",
            temperature=1.0,
            top_p=0.95,
            top_k=20,
            min_p=0.0,
            presence_penalty=0.0,
            repetition_penalty=1.0,
            max_tokens=2048,
            seed=1729,
        ),
        checkpoint=TeacherCheckpointConfig(shard_size=shard_size),
    )


def _record(index: int) -> NormalizedTrainingRecord:
    return NormalizedTrainingRecord(
        schema_version=1,
        messages=(
            TrainingMessage(role="system", content="Write correct Python."),
            TrainingMessage(role="user", content=f"Return the integer {index}."),
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


def _write_input(path: Path, count: int = 5) -> None:
    path.write_text(
        "".join(
            json.dumps(asdict(_record(index)), sort_keys=True, separators=(",", ":")) + "\n"
            for index in range(count)
        ),
        encoding="utf-8",
    )


class _FakeBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[int, ...]] = []

    def generate(self, conversations, *, seeds):  # type: ignore[no-untyped-def]
        self.calls.append(seeds)
        return tuple(
            TeacherCompletion(
                text=f"<think>secret reasoning {seed}</think>\n\n```python\nprint({seed})\n```",
                finish_reason="stop",
                prompt_tokens=10,
                completion_tokens=20,
            )
            for seed in seeds
        )


class _FailBackend:
    def generate(self, conversations, *, seeds):  # type: ignore[no-untyped-def]
        raise AssertionError("backend must not be called for complete durable shards")


def test_teacher_config_parser_is_strict_and_semantically_hashed() -> None:
    config = _config()
    raw = asdict(config)

    parsed = parse_teacher_distillation_config(raw)

    assert parsed == config
    assert teacher_distillation_config_sha256(parsed) == teacher_distillation_config_sha256(config)
    raw["unexpected"] = True
    with pytest.raises(TeacherDistillationConfigError, match="unknown field"):
        parse_teacher_distillation_config(raw)


def test_generation_checkpoints_reasoning_free_shards_and_resumes(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    checkpoint = tmp_path / "drive"
    work = tmp_path / "scratch"
    _write_input(input_path)
    backend = _FakeBackend()

    result = run_teacher_generation(
        _config(),
        backend=backend,
        checkpoint_dir=checkpoint,
        work_dir=work,
        input_path=input_path,
    )

    assert result.completed_records == 5
    assert result.completed_shards == 3
    assert backend.calls == [(1729, 1730), (1731, 1732), (1733,)]
    shard_text = (checkpoint / "shards" / "shard-000000.jsonl").read_text(encoding="utf-8")
    assert "secret reasoning" not in shard_text
    assert "<think>" not in shard_text
    assert "reasoning_sha256" in shard_text
    assert "final_response" in shard_text

    status = inspect_teacher_generation(
        _config(), checkpoint_dir=checkpoint, input_path=input_path
    )
    assert status.complete
    assert status.completed_records == 5
    assert status.missing_shards == ()

    resumed = run_teacher_generation(
        _config(),
        backend=_FailBackend(),
        checkpoint_dir=checkpoint,
        work_dir=work,
        input_path=input_path,
    )
    assert resumed.completed_records == 5

    distilled = load_completed_distilled_records(
        _config(), checkpoint_dir=checkpoint, input_path=input_path
    )
    assert len(distilled) == 5
    assert distilled[0].messages[-1].content.startswith("```python")
    assert distilled[0].messages[-1].content != _record(0).messages[-1].content
    metadata = dict(distilled[0].provenance.source_metadata)
    assert metadata["distillation.teacher_repository"] == "Qwen/Qwen3.8-27B"
    assert metadata["distillation.reasoning_chars"] != "0"


def test_generation_recovers_payload_left_before_checksum_commit(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    checkpoint = tmp_path / "drive"
    _write_input(input_path, count=2)
    run_teacher_generation(
        _config(),
        backend=_FakeBackend(),
        checkpoint_dir=checkpoint,
        work_dir=tmp_path / "scratch-first",
        input_path=input_path,
    )
    sidecar = checkpoint / "shards" / "shard-000000.sha256"
    sidecar.unlink()

    status = inspect_teacher_generation(
        _config(), checkpoint_dir=checkpoint, input_path=input_path
    )
    assert not status.complete
    assert status.missing_shards == (0,)

    backend = _FakeBackend()
    resumed = run_teacher_generation(
        _config(),
        backend=backend,
        checkpoint_dir=checkpoint,
        work_dir=tmp_path / "scratch-resume",
        input_path=input_path,
    )
    assert resumed.completed_records == 2
    assert backend.calls == [(1729, 1730)]


def test_generation_fails_closed_on_corrupt_durable_shard(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    checkpoint = tmp_path / "drive"
    _write_input(input_path, count=2)
    run_teacher_generation(
        _config(),
        backend=_FakeBackend(),
        checkpoint_dir=checkpoint,
        work_dir=tmp_path / "scratch",
        input_path=input_path,
    )
    sidecar = checkpoint / "shards" / "shard-000000.sha256"
    sidecar.write_text(f"{'0' * 64}  shard-000000.jsonl\n", encoding="ascii")

    with pytest.raises(TeacherGenerationError, match="checksum mismatch"):
        inspect_teacher_generation(_config(), checkpoint_dir=checkpoint, input_path=input_path)


def test_generation_identity_rejects_changed_config_input_or_limit(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    checkpoint = tmp_path / "drive"
    _write_input(input_path, count=3)
    run_teacher_generation(
        _config(),
        backend=_FakeBackend(),
        checkpoint_dir=checkpoint,
        work_dir=tmp_path / "scratch",
        input_path=input_path,
        limit=2,
    )

    with pytest.raises(TeacherGenerationError, match="different config or input corpus"):
        inspect_teacher_generation(_config(), checkpoint_dir=checkpoint, input_path=input_path)

    changed = replace(
        _config(),
        generation=replace(_config().generation, temperature=0.8),
    )
    with pytest.raises(TeacherGenerationError, match="different config or input corpus"):
        inspect_teacher_generation(
            changed, checkpoint_dir=checkpoint, input_path=input_path, limit=2
        )

    changed_input = tmp_path / "changed.jsonl"
    _write_input(changed_input, count=2)
    changed_input.write_text(
        changed_input.read_text(encoding="utf-8").replace("integer 0", "integer zero"),
        encoding="utf-8",
    )
    with pytest.raises(TeacherGenerationError, match="different config or input corpus"):
        inspect_teacher_generation(
            _config(), checkpoint_dir=checkpoint, input_path=changed_input, limit=2
        )


def test_stratified_teacher_subset_preserves_source_ratio_and_is_nested() -> None:
    from tiny_qwen_coder.distillation.subset import select_teacher_input_records

    records = tuple(
        replace(
            _record(index),
            provenance=replace(
                _record(index).provenance,
                source_id="source-a" if index < 12 else "source-b",
                revision="rev-a" if index < 12 else "rev-b",
            ),
        )
        for index in range(16)
    )

    four = select_teacher_input_records(records, count=4, seed=1729)
    eight = select_teacher_input_records(records, count=8, seed=1729)

    four_sources = [record.provenance.source_id for record in four]
    eight_sources = [record.provenance.source_id for record in eight]
    assert four_sources.count("source-a") == 3
    assert four_sources.count("source-b") == 1
    assert eight_sources.count("source-a") == 6
    assert eight_sources.count("source-b") == 2
    four_ids = {record.provenance.record_id for record in four}
    eight_ids = {record.provenance.record_id for record in eight}
    assert four_ids < eight_ids


def test_teacher_subset_writer_seals_output_and_summary(tmp_path: Path) -> None:
    from tiny_qwen_coder.distillation.subset import write_teacher_input_subset

    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "subset.jsonl"
    records = tuple(
        replace(
            _record(index),
            provenance=replace(
                _record(index).provenance,
                source_id="source-a" if index < 6 else "source-b",
                revision="rev-a" if index < 6 else "rev-b",
            ),
        )
        for index in range(8)
    )
    input_path.write_text(
        "".join(
            json.dumps(asdict(record), sort_keys=True, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )

    summary = write_teacher_input_subset(
        input_path=input_path,
        output_path=output_path,
        count=4,
        seed=1729,
    )

    assert summary.selected_records == 4
    assert output_path.with_suffix(".jsonl.sha256").exists()
    assert output_path.with_suffix(".jsonl.summary.json").exists()
    assert [(item.source_id, item.selected_records) for item in summary.sources] == [
        ("source-a", 3),
        ("source-b", 1),
    ]


def test_vllm_backend_rejects_runtime_version_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    import tiny_qwen_coder.distillation.vllm_backend as backend_module

    installed = {
        "vllm": "0.28.0",
        "vllm-bnb-plugin": "0.0.3",
        "bitsandbytes": "0.50.2",
    }
    monkeypatch.setattr(
        backend_module.importlib.metadata,
        "version",
        lambda distribution: installed[distribution],
    )
    backend_module.VllmTeacherBackend._verify_runtime(_config())

    installed["vllm"] = "0.28.1"
    with pytest.raises(TeacherGenerationError, match="vllm==0.28.0"):
        backend_module.VllmTeacherBackend._verify_runtime(_config())


def test_teacher_finalization_prefilter_rejects_truncation_and_bad_python() -> None:
    from tiny_qwen_coder.distillation.finalize import _prefilter_candidates

    def candidate(answer: str, finish_reason: str, record_id: str) -> NormalizedTrainingRecord:
        source = _record(int(record_id))
        metadata = tuple(sorted((("distillation.finish_reason", finish_reason),)))
        return replace(
            source,
            messages=source.messages[:-1] + (TrainingMessage(role="assistant", content=answer),),
            provenance=replace(source.provenance, source_metadata=metadata),
        )

    valid = candidate("```python\nprint(1)\n```", "stop", "1")
    truncated = candidate("```python\nprint(2)\n```", "length", "2")
    invalid = candidate("```python\ndef broken(:\n    pass\n```", "stop", "3")

    accepted, finish_rejections, quality_rejections = _prefilter_candidates(
        (valid, truncated, invalid)
    )

    assert accepted == (valid,)
    assert finish_rejections == {"length": 1}
    assert sum(quality_rejections.values()) == 1
    assert next(iter(quality_rejections)).startswith("reason=syntax_error")


def test_vllm_backend_uses_text_only_qwen_thinking_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tiny_qwen_coder.distillation.vllm_backend as backend_module

    installed = {
        "vllm": "0.28.0",
        "vllm-bnb-plugin": "0.0.3",
        "bitsandbytes": "0.50.2",
    }
    monkeypatch.setattr(
        backend_module.importlib.metadata,
        "version",
        lambda distribution: installed[distribution],
    )
    module = ModuleType("vllm")
    llm_init: dict[str, object] = {}
    chat_call: dict[str, object] = {}
    sampling_calls: list[dict[str, object]] = []

    class FakeSamplingParams:
        def __init__(self, **kwargs: object) -> None:
            sampling_calls.append(kwargs)

    class FakeLLM:
        def __init__(self, **kwargs: object) -> None:
            llm_init.update(kwargs)

        def chat(self, messages: object, **kwargs: object) -> list[object]:
            chat_call["messages"] = messages
            chat_call.update(kwargs)
            return [
                SimpleNamespace(
                    outputs=[
                        SimpleNamespace(
                            text="<think>reason</think>\n\nprint(1)",
                            token_ids=[1, 2],
                            finish_reason="stop",
                        )
                    ],
                    prompt_token_ids=[3, 4, 5],
                )
            ]

    module.LLM = FakeLLM  # type: ignore[attr-defined]
    module.SamplingParams = FakeSamplingParams  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", module)

    backend = backend_module.VllmTeacherBackend(_config())
    completion = backend.generate(
        ((TrainingMessage(role="user", content="write code"),),),
        seeds=(1729,),
    )[0]

    assert llm_init["quantization"] == "bitsandbytes"
    assert llm_init["language_model_only"] is True
    assert llm_init["enable_prefix_caching"] is True
    assert llm_init["max_num_seqs"] == 2
    assert sampling_calls[0]["temperature"] == 1.0
    assert sampling_calls[0]["seed"] == 1729
    assert chat_call["chat_template_kwargs"] == {
        "enable_thinking": True,
        "preserve_thinking": False,
        "reasoning_effort": "xhigh",
    }
    assert completion.finish_reason == "stop"
    assert completion.prompt_tokens == 3
    assert completion.completion_tokens == 2
