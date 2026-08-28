"""Contract tests for the canonical shared base-model configuration."""

import re
from pathlib import Path

CONFIG_PATH = Path("configs/base/qwen35-0.8b.yaml")
EXPECTED_REPOSITORY = "Qwen/Qwen3.5-0.8B"
EXPECTED_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"


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


def test_tokenizer_and_precision_policies_are_explicit() -> None:
    text = _config_text()

    assert "revision_policy: match_model_revision" in text
    assert "chat_template_source: pinned_checkpoint" in text
    assert "model_load_dtype: bfloat16" in text
    assert "canonical_lora_training_dtype: bfloat16" in text
    assert "quantization: none" in text
    assert "qlora: false" in text
