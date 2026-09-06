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
from tiny_qwen_coder.distillation.diagnostics import diagnose_teacher_records
from tiny_qwen_coder.distillation.qualification import qualify_teacher_study
from tiny_qwen_coder.distillation.v2_input import (
    V2_DISTILLATION_INSTRUCTION,
    apply_v2_teacher_input_policy,
    write_v2_teacher_input,
)
from tiny_qwen_coder.evaluation.python_protected_examples import load_python_protected_examples
from tiny_qwen_coder.languages.python import load_python_protected_benchmark_registry


def _record() -> NormalizedTrainingRecord:
    return NormalizedTrainingRecord(
        schema_version=1,
        messages=(
            TrainingMessage(role="system", content="Write correct Python."),
            TrainingMessage(role="user", content="Return one."),
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


def test_v2_input_policy_preserves_user_and_seed_answer_and_is_sealed(tmp_path: Path) -> None:
    source = _record()
    transformed = apply_v2_teacher_input_policy(source)

    assert transformed.messages[-2:] == source.messages[-2:]
    assert V2_DISTILLATION_INSTRUCTION in transformed.messages[0].content
    metadata = dict(transformed.provenance.source_metadata)
    assert metadata["distillation.input_policy"] == "concise-v2"
    assert len(metadata["distillation.input_policy_sha256"]) == 64

    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "v2.jsonl"
    input_path.write_text(json.dumps(asdict(source), sort_keys=True) + "\n", encoding="utf-8")
    summary = write_v2_teacher_input(input_path=input_path, output_path=output_path)

    assert summary.output_records == 1
    assert output_path.with_suffix(".jsonl.sha256").exists()
    assert output_path.with_suffix(".jsonl.summary.json").exists()


def test_v2_config_changes_reasoning_effort_without_changing_generation_cap() -> None:
    v1 = load_teacher_distillation_config(Path("configs/distillation/python/qwen38_27b_v1.yaml"))
    v2 = load_teacher_distillation_config(Path("configs/distillation/python/qwen38_27b_v2.yaml"))

    assert v1.generation.reasoning_effort == "xhigh"
    assert v2.generation.reasoning_effort == "high"
    assert v2.generation.max_tokens == v1.generation.max_tokens == 8192
    assert v2.teacher == v1.teacher


class _FakeTokenizer:
    chat_template = "fixture-template"

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        del add_special_tokens
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


def test_teacher_length_diagnostics_separate_runtime_and_student_lengths() -> None:
    source = _record()
    metadata = tuple(
        sorted(
            {
                "distillation.finish_reason": "stop",
                "distillation.prompt_tokens": "17",
                "distillation.completion_tokens": "23",
                "distillation.reasoning_chars": "91",
            }.items()
        )
    )
    distilled = replace(source, provenance=replace(source.provenance, source_metadata=metadata))

    result = diagnose_teacher_records(
        (distilled,),
        teacher_tokenizer=_FakeTokenizer(),
        student_tokenizer=_FakeTokenizer(),
        student_max_tokens=2048,
    )

    row = result.records[0]
    assert row.teacher_prompt_tokens == 17
    assert row.teacher_completion_tokens == 23
    assert row.reasoning_chars == 91
    assert row.teacher_final_answer_tokens > 0
    assert row.student_prompt_tokens > 0
    assert row.student_final_answer_tokens > 0
    assert row.student_full_record_tokens > 0
    assert row.student_length_accepted
    assert result.summary.stop_rate == 1.0
    assert result.summary.student_length_accept_rate_given_stop == 1.0


def _humaneval_rows() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "task_id": f"HumanEval/{index}",
            "prompt": f'def task_{index}():\n    """Return {index}."""\n',
            "canonical_solution": f"    return {index}\n",
            "test": f"def check(candidate):\n    assert candidate() == {index}\n",
            "entry_point": f"task_{index}",
        }
        for index in range(164)
    )


def _mbpp_rows() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "task_id": task_id,
            "text": f"Return {task_id}.",
            "code": f"def task_{task_id}():\n    return {task_id}\n",
            "test_list": [
                f"assert task_{task_id}() == {task_id}",
                f"assert isinstance(task_{task_id}(), int)",
                f"assert task_{task_id}() >= 0",
            ],
            "test_setup_code": "",
            "challenge_test_list": [],
        }
        for task_id in range(11, 511)
    )


def test_python_protected_example_loader_covers_every_registered_benchmark() -> None:
    registry = load_python_protected_benchmark_registry()

    def humaneval_loader(*args: object, **kwargs: object) -> tuple[dict[str, object], ...]:
        del args, kwargs
        return _humaneval_rows()

    def mbpp_loader(*args: object, **kwargs: object) -> tuple[dict[str, object], ...]:
        del args, kwargs
        return _mbpp_rows()

    examples = load_python_protected_examples(
        registry,
        humaneval_dataset_loader=humaneval_loader,
        mbpp_dataset_loader=mbpp_loader,
    )

    ids = {example.benchmark_id for example in examples}
    assert ids == {"humaneval", "mbpp", "repository-holdout"}
    assert sum(example.benchmark_id == "humaneval" for example in examples) == 164
    assert sum(example.benchmark_id == "mbpp" for example in examples) == 500
    assert sum(example.benchmark_id == "repository-holdout" for example in examples) == 11
    assert all(
        example.solution is not None
        for example in examples
        if example.benchmark_id in {"humaneval", "mbpp"}
    )
    assert all(
        example.solution is None
        for example in examples
        if example.benchmark_id == "repository-holdout"
    )


def test_bounded_teacher_study_requires_rates_and_clean_contamination() -> None:
    diagnostics: dict[str, object] = {
        "total_records": 200,
        "stop_rate": 0.95,
        "student_length_accept_rate_given_stop": 0.85,
    }
    clean_manifest: dict[str, object] = {"contamination": {"status": "clean"}}

    qualified = qualify_teacher_study(
        diagnostics_summary=diagnostics,
        dataset_manifest=clean_manifest,
    )

    assert qualified.qualified
    assert qualified.failed_gates == ()

    failed = qualify_teacher_study(
        diagnostics_summary={
            **diagnostics,
            "stop_rate": 0.89,
            "student_length_accept_rate_given_stop": 0.79,
        },
        dataset_manifest={"contamination": {"status": "not_run"}},
    )
    assert not failed.qualified
    assert failed.failed_gates == (
        "stop_rate",
        "student_length_accept_rate_given_stop",
        "contamination",
    )
