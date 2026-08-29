"""Concrete P6-001 Python protected-benchmark registrations."""

from __future__ import annotations

from pathlib import Path

import pytest

from tiny_qwen_coder.config import DataPreparationConfig
from tiny_qwen_coder.evaluation import (
    ProtectedBenchmarkRegistrationError,
    ProtectedBenchmarkTrainingSelectionError,
    load_protected_benchmark_config,
)
from tiny_qwen_coder.languages.python import (
    load_python_plugin,
    load_python_protected_benchmark_registry,
)

_HUMANEVAL_REVISION = "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544"
_MBPP_REVISION = "4bb6404fdc6cacfda99d4ac4205087b89d32030c"


def _data_config(*source_configs: str) -> DataPreparationConfig:
    return DataPreparationConfig(
        schema_version=1,
        language="python",
        source_configs=source_configs,
        output_dir="data/python/test",
        seed=1729,
        validation_fraction=0.05,
        min_tokens=1,
        max_tokens=2048,
        truncation_policy="reject",
        deduplicate=True,
    )


def test_python_plugin_registers_all_p6_protected_benchmark_ids() -> None:
    plugin = load_python_plugin()

    assert tuple(item.id for item in plugin.spec.protected_benchmarks) == (
        "humaneval",
        "mbpp",
        "repository-holdout",
    )
    assert set(plugin.spec.config.config_refs.data_sources).isdisjoint(
        plugin.spec.config.config_refs.evaluation
    )


def test_python_registry_pins_external_benchmarks_and_repo_holdout() -> None:
    registry = load_python_protected_benchmark_registry()

    humaneval = registry.resolve("python", "humaneval")
    assert humaneval.dataset_id == "openai/openai_humaneval"
    assert humaneval.dataset_revision == _HUMANEVAL_REVISION
    assert humaneval.source_configs == ("configs/eval/python/humaneval.yaml",)

    mbpp = registry.resolve("python", "mbpp")
    assert mbpp.dataset_id == "google-research-datasets/mbpp"
    assert mbpp.dataset_revision == _MBPP_REVISION
    assert mbpp.source_configs == ("configs/eval/python/mbpp.yaml",)

    holdout = registry.resolve("python", "repository-holdout")
    assert holdout.dataset_id == "repository://tiny-qwen-coder/python-holdout"
    assert holdout.dataset_revision == "repository-holdout-v1"
    assert holdout.source_configs == ("configs/eval/python/repository_holdout.yaml",)


def test_python_registry_allows_only_normal_training_source_selectors() -> None:
    plugin = load_python_plugin()
    registry = load_python_protected_benchmark_registry()

    registry.assert_plugin_registration_matches(plugin)
    registry.assert_sft_config_allowed(_data_config(*plugin.spec.config.config_refs.data_sources))


@pytest.mark.parametrize(
    ("benchmark_id", "selector"),
    [
        ("humaneval", "openai/openai_humaneval"),
        ("humaneval", "configs/eval/python/humaneval.yaml"),
        ("mbpp", "google-research-datasets/mbpp"),
        ("mbpp", "configs/eval/python/mbpp.yaml"),
        ("repository-holdout", "repository://tiny-qwen-coder/python-holdout"),
        ("repository-holdout", "configs/eval/python/repository_holdout.yaml"),
    ],
)
def test_every_python_protected_selector_is_inaccessible_to_sft(
    benchmark_id: str,
    selector: str,
) -> None:
    registry = load_python_protected_benchmark_registry()

    with pytest.raises(
        ProtectedBenchmarkTrainingSelectionError,
        match=rf"python/{benchmark_id}",
    ):
        registry.assert_sft_config_allowed(_data_config(selector))


def test_python_protected_configs_are_strict_and_self_protecting() -> None:
    for path_text in load_python_plugin().spec.config.config_refs.evaluation:
        path = Path(path_text)
        benchmark = load_protected_benchmark_config(path)
        assert benchmark.language == "python"
        assert benchmark.source_configs == (path_text,)


def test_protected_benchmark_config_rejects_unknown_fields(tmp_path: Path) -> None:
    source = Path("configs/eval/python/humaneval.yaml")
    path = tmp_path / "humaneval.yaml"
    path.write_text(source.read_text(encoding="utf-8") + "surprise: forbidden\n", encoding="utf-8")

    with pytest.raises(ProtectedBenchmarkRegistrationError, match=r"unknown field\(s\): surprise"):
        load_protected_benchmark_config(path)
