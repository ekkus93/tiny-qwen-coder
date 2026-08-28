"""P4-006 tests for the frozen base/adapter evaluation protocol."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tiny_qwen_coder.config import EvaluationConfig, ExecutionConfig, GenerationConfig
from tiny_qwen_coder.evaluation.settings import (
    EvaluationSettingsError,
    evaluation_settings_json,
    evaluation_settings_sha256,
    load_evaluation_settings,
    load_frozen_evaluation_settings,
    validate_evaluation_config_settings,
)

_SETTINGS_PATH = Path("configs/eval/canonical_generation_v1.yaml")
_EXPECTED_SHA256 = "8660c83a561dc5d9896de2fbf5471ecec5c9763d46dcc16891b24f55e6262591"


def _evaluation_config(*, adapter_id: str | None = None) -> EvaluationConfig:
    settings = load_frozen_evaluation_settings()
    frozen = settings.generation
    return EvaluationConfig(
        schema_version=1,
        base_config="configs/base/qwen35-4b.yaml",
        language="python",
        adapter_id=adapter_id,
        suites=("general_tool_regression_v1",),
        output_dir="artifacts/eval/python",
        seed=settings.seed,
        generation=GenerationConfig(
            temperature=frozen.temperature,
            top_p=frozen.top_p,
            top_k=frozen.top_k,
            max_new_tokens=frozen.max_new_tokens,
            prompt_version=frozen.prompt_version,
        ),
        execution=ExecutionConfig(timeout_seconds=10.0, network_enabled=False),
    )


def test_canonical_settings_are_frozen_and_semantically_pinned() -> None:
    settings = load_frozen_evaluation_settings()

    assert settings.settings_id == "canonical_evaluation"
    assert settings.settings_version == 1
    assert settings.frozen is True
    assert settings.seed == 1729
    assert settings.generation.decoding_strategy == "greedy"
    assert settings.generation.temperature == 0.0
    assert settings.generation.top_p == 1.0
    assert settings.generation.top_k == 0
    assert settings.generation.max_new_tokens == 512
    assert settings.generation.stop_policy == "eos_or_max_new_tokens"
    assert settings.generation.prompt_version == "canonical-evaluation-prompt-v1"
    assert settings.generation.chat_template_version.endswith(":checkpoint")
    assert evaluation_settings_sha256(settings) == _EXPECTED_SHA256


def test_settings_json_is_deterministic() -> None:
    first = load_frozen_evaluation_settings()
    second = load_frozen_evaluation_settings()

    assert evaluation_settings_json(first) == evaluation_settings_json(second)
    assert evaluation_settings_sha256(first) == evaluation_settings_sha256(second)


def test_base_and_adapter_configs_accept_the_same_frozen_settings() -> None:
    settings = load_frozen_evaluation_settings()
    base = _evaluation_config(adapter_id=None)
    adapter = _evaluation_config(adapter_id="language/python/p0")

    assert validate_evaluation_config_settings(base, settings) == _EXPECTED_SHA256
    assert validate_evaluation_config_settings(adapter, settings) == _EXPECTED_SHA256
    assert base.generation == adapter.generation
    assert base.seed == adapter.seed == settings.seed


def test_seed_drift_is_rejected() -> None:
    settings = load_frozen_evaluation_settings()
    config = replace(_evaluation_config(), seed=settings.seed + 1)

    with pytest.raises(EvaluationSettingsError, match="does not match frozen seed"):
        validate_evaluation_config_settings(config, settings)


def test_generation_drift_is_rejected() -> None:
    settings = load_frozen_evaluation_settings()
    changed_generation = replace(_evaluation_config().generation, max_new_tokens=256)
    config = replace(_evaluation_config(), generation=changed_generation)

    with pytest.raises(EvaluationSettingsError, match="max_new_tokens"):
        validate_evaluation_config_settings(config, settings)


def test_unfrozen_comparison_settings_are_rejected() -> None:
    settings = replace(load_evaluation_settings(_SETTINGS_PATH), frozen=False)

    with pytest.raises(EvaluationSettingsError, match="must be frozen"):
        validate_evaluation_config_settings(_evaluation_config(), settings)


def test_unknown_settings_fields_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "settings.yaml"
    path.write_text(
        _SETTINGS_PATH.read_text(encoding="utf-8") + "surprise: forbidden\n",
        encoding="utf-8",
    )

    with pytest.raises(EvaluationSettingsError, match=r"unknown field\(s\): surprise"):
        load_evaluation_settings(path)


def test_canonical_content_drift_requires_explicit_versioned_freeze(tmp_path: Path) -> None:
    path = tmp_path / "settings.yaml"
    original = _SETTINGS_PATH.read_text(encoding="utf-8")
    path.write_text(
        original.replace("max_new_tokens: 512", "max_new_tokens: 256"),
        encoding="utf-8",
    )

    with pytest.raises(EvaluationSettingsError, match="fingerprint mismatch"):
        load_frozen_evaluation_settings(path)


def test_greedy_generation_rejects_sampling_knob_drift(tmp_path: Path) -> None:
    original = _SETTINGS_PATH.read_text(encoding="utf-8")
    replacements = (
        ("temperature: 0", "temperature: 0.1", "temperature=0"),
        ("top_p: 1", "top_p: 0.95", "top_p=1"),
        ("top_k: 0", "top_k: 20", "top_k=0"),
    )
    for old, new, message in replacements:
        path = tmp_path / f"{message}.yaml"
        path.write_text(original.replace(old, new), encoding="utf-8")
        with pytest.raises(EvaluationSettingsError, match=message):
            load_evaluation_settings(path)
