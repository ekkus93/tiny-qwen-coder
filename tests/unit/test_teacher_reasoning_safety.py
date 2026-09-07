from __future__ import annotations

import sys
from dataclasses import replace
from importlib import metadata as importlib_metadata
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tiny_qwen_coder.data.records import (
    LicenseMetadata,
    NormalizedTrainingRecord,
    SourceProvenance,
    TrainingMessage,
)
from tiny_qwen_coder.distillation.config import load_teacher_distillation_config
from tiny_qwen_coder.distillation.finalize import (
    TeacherFinalizationError,
    _require_no_reasoning_markers,
)
from tiny_qwen_coder.distillation.reasoning import (
    normalize_qwen_thinking_completion,
    split_qwen_thinking_completion,
)


def _record(answer: str) -> NormalizedTrainingRecord:
    return NormalizedTrainingRecord(
        schema_version=1,
        messages=(
            TrainingMessage(role="user", content="Return one."),
            TrainingMessage(role="assistant", content=answer),
        ),
        language="python",
        provenance=SourceProvenance(
            source_id="fixture",
            revision="fixture-revision",
            license=LicenseMetadata(name="MIT"),
            record_id="fixture-1",
        ),
    )


def test_qwen_closing_only_thinking_is_normalized_and_split() -> None:
    raw = "private reasoning\n</think>\n\n```python\nprint(1)\n```"

    normalized = normalize_qwen_thinking_completion(raw)
    reasoning, final = split_qwen_thinking_completion(raw)

    assert normalized.startswith("<think>private reasoning")
    assert reasoning == "private reasoning"
    assert final == "```python\nprint(1)\n```"
    assert "<think>" not in final
    assert "</think>" not in final


def test_finalization_fails_closed_on_any_assistant_thinking_marker() -> None:
    with pytest.raises(TeacherFinalizationError, match="hidden-thinking markup"):
        _require_no_reasoning_markers((_record("reasoning\n</think>\nprint(1)"),))

    with pytest.raises(TeacherFinalizationError, match="hidden-thinking markup"):
        _require_no_reasoning_markers((_record("<think>reasoning</think>\nprint(1)"),))

    _require_no_reasoning_markers((_record("```python\nprint(1)\n```"),))


def test_vllm_backend_restores_prefilled_qwen_think_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tiny_qwen_coder.distillation.vllm_backend as backend_module

    config = load_teacher_distillation_config(
        Path("configs/distillation/python/qwen38_27b_v3.yaml")
    )
    monkeypatch.setattr(
        importlib_metadata,
        "version",
        lambda distribution: {"vllm": "0.28.0"}[distribution],
    )

    module = ModuleType("vllm")

    class FakeSamplingParams:
        def __init__(self, **_: object) -> None:
            pass

    class FakeLLM:
        def __init__(self, **_: object) -> None:
            pass

        def chat(self, messages: object, **_: object) -> list[object]:
            assert isinstance(messages, list)
            return [
                SimpleNamespace(
                    outputs=[
                        SimpleNamespace(
                            text="private reasoning\n</think>\n\nprint(1)",
                            token_ids=[1, 2, 3],
                            finish_reason="stop",
                        )
                    ],
                    prompt_token_ids=[4, 5],
                )
            ]

    module.LLM = FakeLLM  # type: ignore[attr-defined]
    module.SamplingParams = FakeSamplingParams  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", module)

    backend = backend_module.VllmTeacherBackend(replace(config))
    completion = backend.generate(
        ((TrainingMessage(role="user", content="write code"),),),
        seeds=(1729,),
    )[0]

    assert completion.text == "<think>private reasoning\n</think>\n\nprint(1)"
    reasoning, final = split_qwen_thinking_completion(completion.text)
    assert reasoning == "private reasoning"
    assert final == "print(1)"
