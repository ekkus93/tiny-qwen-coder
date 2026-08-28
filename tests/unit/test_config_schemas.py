"""Contract tests for strict training/evaluation/runtime configuration schemas."""

from pathlib import Path

import pytest

from tiny_qwen_coder.config import (
    ConfigError,
    canonical_config_json,
    load_data_preparation_config,
    load_evaluation_config,
    load_lora_training_config,
    load_runtime_config,
)


def _write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_data_preparation_schema_is_language_neutral(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "data.yaml",
        """
schema_version: 1
language: rust
source_configs:
  - configs/data/rust/source-a.yaml
output_dir: data/rust/p0
seed: 1729
validation_fraction: 0.05
min_tokens: 16
max_tokens: 2048
truncation_policy: reject
deduplicate: true
""",
    )

    config = load_data_preparation_config(path)

    assert config.language == "rust"
    assert config.source_configs == ("configs/data/rust/source-a.yaml",)
    assert config.max_tokens == 2048
    assert config.deduplicate is True


def test_bf16_lora_training_schema_parses_without_quantization(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "train.yaml",
        """
schema_version: 1
base_config: configs/base/qwen35-4b.yaml
language: python
dataset_manifest: artifacts/datasets/python/p0/manifest.json
output_dir: outputs/language/python/p0
seed: 1729
training_mode: bf16_lora
compute_dtype: bfloat16
sequence_length: 2048
micro_batch_size: 1
gradient_accumulation_steps: 16
epochs: 1
learning_rate: 0.0002
scheduler: cosine
warmup_ratio: 0.03
gradient_checkpointing: true
loss_mode: assistant_only
lora:
  rank: 16
  alpha: 32
  dropout: 0.05
  bias: none
  target_strategy: selective
  target_modules:
    - q_proj
    - v_proj
""",
    )

    config = load_lora_training_config(path)

    assert config.training_mode == "bf16_lora"
    assert config.quantization is None
    assert config.lora.rank == 16
    assert config.lora.target_modules == ("q_proj", "v_proj")


def test_qlora_training_schema_requires_explicit_quantization(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "train.yaml",
        """
schema_version: 1
base_config: configs/base/qwen35-4b.yaml
language: python
dataset_manifest: artifacts/datasets/python/p0/manifest.json
output_dir: outputs/language/python/p0
seed: 1729
training_mode: qlora_4bit
compute_dtype: bfloat16
sequence_length: 2048
micro_batch_size: 1
gradient_accumulation_steps: 16
epochs: 1
learning_rate: 0.0002
scheduler: cosine
warmup_ratio: 0.03
gradient_checkpointing: true
loss_mode: assistant_only
lora:
  rank: 16
  alpha: 32
  dropout: 0.05
  bias: none
  target_strategy: all_linear
  target_modules: []
""",
    )

    with pytest.raises(ConfigError, match="requires quantization"):
        load_lora_training_config(path)


def test_evaluation_schema_captures_frozen_generation_and_execution(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "eval.yaml",
        """
schema_version: 1
base_config: configs/base/qwen35-4b.yaml
language: typescript
adapter_id: null
suites:
  - eval/typescript/smoke-v1
output_dir: artifacts/eval/typescript/base
seed: 1729
generation:
  temperature: 0
  top_p: 1
  top_k: 0
  max_new_tokens: 512
  prompt_version: typescript-v1
execution:
  timeout_seconds: 10
  network_enabled: false
""",
    )

    config = load_evaluation_config(path)

    assert config.adapter_id is None
    assert config.generation.temperature == 0.0
    assert config.execution.network_enabled is False


def test_runtime_schema_enforces_adapter_selection_contract(tmp_path: Path) -> None:
    valid = _write(
        tmp_path,
        "runtime.yaml",
        """
schema_version: 1
base_config: configs/base/qwen35-4b.yaml
selection_mode: explicit
adapter_id: language/python/recommended
preload_adapters:
  - language/python/recommended
adapter_search_paths:
  - outputs/language
compatibility_policy: strict
allow_auto_detection: false
""",
    )

    config = load_runtime_config(valid)

    assert config.selection_mode == "explicit"
    assert config.adapter_id == "language/python/recommended"

    invalid = _write(
        tmp_path,
        "runtime-invalid.yaml",
        """
schema_version: 1
base_config: configs/base/qwen35-4b.yaml
selection_mode: explicit
preload_adapters: []
adapter_search_paths:
  - outputs/language
compatibility_policy: strict
allow_auto_detection: false
""",
    )

    with pytest.raises(ConfigError, match="requires adapter_id"):
        load_runtime_config(invalid)


def test_unknown_fields_fail_closed_at_root_and_nested_levels(tmp_path: Path) -> None:
    root_unknown = _write(
        tmp_path,
        "data-unknown.yaml",
        """
schema_version: 1
language: python
source_configs:
  - configs/data/python/source.yaml
output_dir: data/python/p0
seed: 1
validation_fraction: 0.05
min_tokens: 0
max_tokens: 2048
truncation_policy: reject
deduplicate: true
surprise: forbidden
""",
    )

    with pytest.raises(ConfigError, match=r"unknown field\(s\): surprise"):
        load_data_preparation_config(root_unknown)

    nested_unknown = _write(
        tmp_path,
        "eval-unknown.yaml",
        """
schema_version: 1
base_config: configs/base/qwen35-4b.yaml
language: python
suites:
  - eval/python/smoke-v1
output_dir: artifacts/eval/python/base
seed: 1
generation:
  temperature: 0
  top_p: 1
  top_k: 0
  max_new_tokens: 64
  prompt_version: python-v1
  unregistered_knob: 42
execution:
  timeout_seconds: 5
  network_enabled: false
""",
    )

    with pytest.raises(ConfigError, match=r"unknown field\(s\): unregistered_knob"):
        load_evaluation_config(nested_unknown)


def test_equivalent_yaml_parses_and_canonicalizes_deterministically(tmp_path: Path) -> None:
    first = _write(
        tmp_path,
        "first.yaml",
        """
schema_version: 1
language: python
source_configs: [configs/data/python/source.yaml]
output_dir: data/python/p0
seed: 1729
validation_fraction: 0.05
min_tokens: 16
max_tokens: 2048
truncation_policy: reject
deduplicate: true
""",
    )
    second = _write(
        tmp_path,
        "second.yaml",
        """
deduplicate: true
max_tokens: 2048
min_tokens: 16
truncation_policy: reject
validation_fraction: 0.050
seed: 1729
output_dir: data/python/p0
source_configs:
  - configs/data/python/source.yaml
language: python
schema_version: 1
""",
    )

    first_config = load_data_preparation_config(first)
    second_config = load_data_preparation_config(second)

    assert first_config == second_config
    assert canonical_config_json(first_config) == canonical_config_json(second_config)


def test_invalid_scalar_types_are_not_coerced(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "invalid-types.yaml",
        """
schema_version: 1
language: python
source_configs:
  - configs/data/python/source.yaml
output_dir: data/python/p0
seed: "1729"
validation_fraction: 0.05
min_tokens: 0
max_tokens: 2048
truncation_policy: reject
deduplicate: true
""",
    )

    with pytest.raises(ConfigError, match="data.seed must be an integer"):
        load_data_preparation_config(path)
