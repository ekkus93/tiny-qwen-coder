"""Contract tests for the portable adapter-manifest schema."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from tiny_qwen_coder.adapters.manifest import (
    AdapterManifestError,
    adapter_manifest_json,
    load_adapter_manifest,
    parse_adapter_manifest,
)


def _manifest_mapping(language: str = "python") -> dict[str, object]:
    return {
        "schema_version": 1,
        "adapter_id": f"language/{language}/p0-r16-40k",
        "family": "language",
        "language": language,
        "created_at_utc": "2026-08-28T10:30:00Z",
        "base_model": {
            "repository": "Qwen/Qwen3.5-4B",
            "revision": "a" * 40,
        },
        "tokenizer": {
            "repository": "Qwen/Qwen3.5-4B",
            "revision": "a" * 40,
            "chat_template": {
                "identifier": "qwen35-4b-pinned-checkpoint",
                "sha256": "b" * 64,
            },
        },
        "training": {
            "run_id": f"training-{language}-p0",
            "git_sha": "c" * 40,
            "config_sha256": "d" * 64,
            "seed": 42,
            "transformers_version": "5.16.1",
            "peft_version": "0.20.0",
        },
        "datasets": [
            {
                "manifest_id": f"dataset/{language}/p0",
                "sha256": "e" * 64,
            }
        ],
        "lora": {
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
            "bias": "none",
            "target_strategy": "selective",
            "target_modules": [
                "model.language_model.layers.0.self_attn.q_proj",
                "model.language_model.layers.0.mlp.gate_proj",
            ],
            "trainable_parameters": 32_464_896,
        },
        "training_summary": {
            "precision": "bfloat16",
            "sequence_length": 2048,
            "optimizer": {
                "name": "adamw_torch",
                "settings": [
                    {"name": "learning_rate", "value": 0.0002},
                    {"name": "weight_decay", "value": 0.0},
                ],
            },
            "scheduler": {
                "name": "cosine",
                "settings": [{"name": "warmup_ratio", "value": 0.03}],
            },
            "steps": 1000,
            "epochs": 1.0,
            "peak_vram_bytes": 12_000_000_000,
        },
        "validation_metrics": [
            {
                "name": "validation_loss",
                "value": 1.234,
                "split": "validation",
                "unit": None,
            }
        ],
        "evaluation_artifacts": [f"evaluation/{language}/p0/humaneval.json"],
    }


@pytest.mark.parametrize("language", ["python", "typescript", "rust", "go"])
def test_schema_represents_current_and_future_languages_without_changes(language: str) -> None:
    manifest = parse_adapter_manifest(_manifest_mapping(language))

    assert manifest.language == language
    assert manifest.adapter_id == f"language/{language}/p0-r16-40k"
    assert manifest.base_model.revision == "a" * 40
    assert manifest.tokenizer.chat_template.sha256 == "b" * 64
    assert manifest.training.git_sha == "c" * 40
    assert manifest.training.config_sha256 == "d" * 64
    assert manifest.datasets[0].manifest_id == f"dataset/{language}/p0"
    assert manifest.lora.trainable_parameters == 32_464_896


def test_manifest_records_resolved_targets_even_for_symbolic_all_linear() -> None:
    raw = _manifest_mapping()
    lora = raw["lora"]
    assert isinstance(lora, dict)
    lora["target_strategy"] = "all_linear"
    lora["target_modules"] = [
        "model.language_model.layers.0.self_attn.q_proj",
        "model.visual.blocks.0.attn.qkv",
    ]
    lora["trainable_parameters"] = 38_993_920

    manifest = parse_adapter_manifest(raw)

    assert manifest.lora.target_strategy == "all_linear"
    assert manifest.lora.target_modules[-1] == "model.visual.blocks.0.attn.qkv"
    assert manifest.lora.trainable_parameters == 38_993_920


def test_adapter_id_must_encode_family_and_language() -> None:
    raw = _manifest_mapping("rust")
    raw["adapter_id"] = "language/python/p0-r16-40k"

    with pytest.raises(AdapterManifestError, match="must begin with family/language"):
        parse_adapter_manifest(raw)


def test_unknown_fields_fail_closed_at_root_and_nested_levels() -> None:
    root = _manifest_mapping()
    root["surprise"] = True
    with pytest.raises(AdapterManifestError, match="unknown field.*surprise"):
        parse_adapter_manifest(root)

    nested = _manifest_mapping()
    tokenizer = nested["tokenizer"]
    assert isinstance(tokenizer, dict)
    tokenizer["surprise"] = True
    with pytest.raises(AdapterManifestError, match="unknown field.*surprise"):
        parse_adapter_manifest(nested)


def test_immutable_git_and_content_hashes_are_required() -> None:
    bad_git = _manifest_mapping()
    training = bad_git["training"]
    assert isinstance(training, dict)
    training["git_sha"] = "main"
    with pytest.raises(AdapterManifestError, match="40-character Git SHA"):
        parse_adapter_manifest(bad_git)

    bad_dataset = _manifest_mapping()
    datasets = bad_dataset["datasets"]
    assert isinstance(datasets, list)
    dataset = datasets[0]
    assert isinstance(dataset, dict)
    dataset["sha256"] = "not-a-hash"
    with pytest.raises(AdapterManifestError, match="64-character SHA-256"):
        parse_adapter_manifest(bad_dataset)


def test_resolved_target_modules_are_required_and_unique() -> None:
    missing = _manifest_mapping()
    lora = missing["lora"]
    assert isinstance(lora, dict)
    lora["target_modules"] = []
    with pytest.raises(AdapterManifestError, match="resolved trained module names"):
        parse_adapter_manifest(missing)

    duplicated = _manifest_mapping()
    lora = duplicated["lora"]
    assert isinstance(lora, dict)
    lora["target_modules"] = ["q_proj", "q_proj"]
    with pytest.raises(AdapterManifestError, match="must not contain duplicates"):
        parse_adapter_manifest(duplicated)


def test_dataset_and_metric_identities_must_be_unambiguous() -> None:
    duplicate_dataset = _manifest_mapping()
    datasets = duplicate_dataset["datasets"]
    assert isinstance(datasets, list)
    datasets.append(copy.deepcopy(datasets[0]))
    with pytest.raises(AdapterManifestError, match="dataset manifest IDs must be unique"):
        parse_adapter_manifest(duplicate_dataset)

    duplicate_metric = _manifest_mapping()
    metrics = duplicate_metric["validation_metrics"]
    assert isinstance(metrics, list)
    metrics.append(copy.deepcopy(metrics[0]))
    with pytest.raises(AdapterManifestError, match="validation metric names must be unique"):
        parse_adapter_manifest(duplicate_metric)


def test_yaml_load_and_json_serialization_are_deterministic(tmp_path: Path) -> None:
    raw = _manifest_mapping("typescript")
    path = tmp_path / "adapter-manifest.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    first = load_adapter_manifest(path)
    second = load_adapter_manifest(path)
    first_json = adapter_manifest_json(first)

    assert first == second
    assert first_json == adapter_manifest_json(second)
    payload = json.loads(first_json)
    assert payload["adapter_id"] == "language/typescript/p0-r16-40k"
    assert payload["lora"]["target_modules"][0].endswith("q_proj")
