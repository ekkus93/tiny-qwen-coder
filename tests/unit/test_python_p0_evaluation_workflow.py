from __future__ import annotations

from pathlib import Path

import yaml

_WORKFLOW = Path(".github/workflows/python-p0-evaluation.yml")


def _workflow() -> tuple[str, dict[str, object]]:
    text = _WORKFLOW.read_text(encoding="utf-8")
    parsed = yaml.safe_load(text)
    assert isinstance(parsed, dict)
    return text, parsed


def test_p8_001_workflow_separates_gpu_generation_from_hosted_scoring() -> None:
    text, parsed = _workflow()
    jobs = parsed["jobs"]
    assert isinstance(jobs, dict)
    generate = jobs["generate"]
    score = jobs["score"]
    assert isinstance(generate, dict)
    assert isinstance(score, dict)
    assert generate["runs-on"] == ["self-hosted", "linux", "x64"]
    assert score["runs-on"] == "ubuntu-24.04"
    assert "tiny_qwen_coder.evaluation.python_p0 generate" in text
    assert "tiny_qwen_coder.evaluation.python_p0 score" in text
    assert "tiny_qwen_coder.evaluation.python_p0 verify" in text


def test_p8_001_workflow_pins_p7_adapter_and_p6_baseline() -> None:
    text, _ = _workflow()
    assert 'P7_006_RUN_ID: "33422910444"' in text
    assert (
        'P7_006_ARTIFACT_NAME: '
        '"python-p0-full-training-02df92a9c2d347b9fb013dc25714fe066c6bcafe"'
    ) in text
    assert 'P6_005_RUN_ID: "33301242379"' in text
    assert (
        'P6_005_ARTIFACT_NAME: '
        '"python-base-baseline-da537443ab80b1380bee0fc3c7d9d01ca0574f35"'
    ) in text
    assert "c94606250e112f72362eb883a55f7b2c8af854d445f6bb6194352c2806a8f276" in text
    assert "training-python-20260831T180916446466Z-02df92a9-eafc119d" in text


def test_p8_001_workflow_uses_same_digest_pinned_execution_image_as_p6() -> None:
    text, _ = _workflow()
    assert (
        "python:3.11.14-slim@sha256:"
        "c8271b1f627d0068857dce5b53e14a9558603b527e46f1f901722f935b786a39"
    ) in text
    assert "general-tool-regression" not in text


def test_p8_001_workflow_is_manual_only_and_uses_compact_retention() -> None:
    text, parsed = _workflow()
    trigger = parsed.get("on")
    if trigger is None:
        trigger = parsed.get(True)
    assert trigger == {"workflow_dispatch": None}
    assert "push:" not in text
    assert text.count("retention-days: 7") == 2
    assert text.count("retention-days: 3") == 2
    assert "adapter_model.safetensors\n" not in text.split(
        "Upload compact GPU generation evidence", 1
    )[1]
