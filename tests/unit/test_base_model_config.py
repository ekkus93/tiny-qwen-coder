"""Contract tests for the canonical shared base-model configuration."""

import re
from pathlib import Path

CONFIG_PATH = Path("configs/base/qwen35-4b.yaml")
EXPECTED_REPOSITORY = "Qwen/Qwen3.5-4B"
EXPECTED_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def _config_text() -> str:
    return CONFIG_PATH.read_text(encoding="utf-8")


def test_canonical_base_uses_exact_repository_and_revision() -> None:
    text = _config_text()

    assert f"repository: {EXPECTED_REPOSITORY}" in text
    assert text.count(f"revision: {EXPECTED_REVISION}") == 2
    assert re.fullmatch(r"[0-9a-f]{40}", EXPECTED_REVISION)


def test_canonical_base_forbids_floating_revisions() -> None:
    text = _config_text()

    assert "revision_policy: immutable_commit" in text
    assert "allow_floating_revision: false" in text
    assert not re.search(r"revision:\s*(main|master|latest)\s*$", text, re.MULTILINE)


def test_precision_policy_is_measured_and_frozen() -> None:
    text = _config_text()

    assert "model_load_dtype: bfloat16" in text
    assert "lora_compute_dtype: bfloat16" in text
    assert "training_mode_policy: measure_then_freeze" in text
    assert "preferred_training_mode: bf16_lora" in text
    assert "fallback_training_mode: qlora_4bit" in text
    assert "quantization: conditional" in text


def test_text_specialization_freezes_vision_components() -> None:
    text = _config_text()

    assert "scope: text_code_only" in text
    assert "freeze_vision_components: true" in text
