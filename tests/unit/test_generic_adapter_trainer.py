from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from tiny_qwen_coder.config import ConfigError
from tiny_qwen_coder.data import (
    LicenseMetadata,
    NormalizedTrainingRecord,
    SourceProvenance,
    TrainingMessage,
)
from tiny_qwen_coder.identities import AdapterIdentity, BaseModelIdentity
from tiny_qwen_coder.languages import LanguageRegistry, load_language_plugin
from tiny_qwen_coder.reporting import (
    DependencyVersions,
    GitMetadata,
    HostMetadata,
    RunManifest,
)
from tiny_qwen_coder.training import (
    AdapterTrainingError,
    AdapterTrainingPlan,
    create_completed_adapter_manifest,
    resolve_adapter_training_plan,
    training_rows,
)

_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
_TEMPLATE_SHA = "f" * 64


def _record(*, language: str = "rust") -> NormalizedTrainingRecord:
    return NormalizedTrainingRecord(
        schema_version=1,
        messages=(
            TrainingMessage(role="system", content="Be precise."),
            TrainingMessage(role="user", content="Write an identity function."),
            TrainingMessage(role="assistant", content="fn identity<T>(value: T) -> T { value }"),
        ),
        language=language,
        provenance=SourceProvenance(
            source_id="fixture",
            revision="revision-1",
            license=LicenseMetadata(name="MIT"),
        ),
    )


def _write_rust_language_config(path: Path) -> None:
    path.write_text(
        """schema_version: 1
id: rust
aliases:
  - rs
extensions:
  - .rs
repository_detection:
  files:
    - Cargo.toml
  directories:
    - src
  globs:
    - "**/*.rs"
system_prompt:
  version: rust-v1
  text: Write correct Rust.
config_refs:
  data_sources:
    - configs/data/rust/example.yaml
  evaluation:
    - configs/eval/rust/example.yaml
hooks:
  validator: fixtures.rust:validate
  executor: fixtures.rust:execute
""",
        encoding="utf-8",
    )


def _write_dataset_manifest(path: Path, *, revision: str = _REVISION) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifest_id": "dataset/rust/p0",
                "language": "rust",
                "tokenizer": {
                    "repository": "Qwen/Qwen3.5-4B",
                    "revision": revision,
                    "chat_template_sha256": _TEMPLATE_SHA,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_training_config(
    path: Path,
    *,
    dataset_manifest: Path,
    output_dir: Path,
    loss_mode: str = "assistant_only",
) -> None:
    path.write_text(
        f"""schema_version: 1
base_config: configs/base/qwen35-4b.yaml
language: rust
adapter_family: language
adapter_id: language/rust/p0
dataset_manifest: {dataset_manifest}
train_records: {dataset_manifest.parent / "train.jsonl"}
validation_records: {dataset_manifest.parent / "validation.jsonl"}
output_dir: {output_dir}
seed: 1337
training_mode: qlora_4bit
compute_dtype: bfloat16
sequence_length: 2048
micro_batch_size: 1
gradient_accumulation_steps: 8
epochs: 1.0
learning_rate: 0.0002
scheduler: cosine
warmup_ratio: 0.03
gradient_checkpointing: true
loss_mode: {loss_mode}
lora:
  rank: 16
  alpha: 32
  dropout: 0.05
  bias: none
  target_strategy: selective
  target_modules:
    - q_proj
    - v_proj
quantization:
  bits: 4
  quant_type: nf4
  double_quant: true
  compute_dtype: bfloat16
""",
        encoding="utf-8",
    )


def _rust_registry(tmp_path: Path) -> LanguageRegistry:
    config = tmp_path / "rust.yaml"
    _write_rust_language_config(config)
    return LanguageRegistry((load_language_plugin(config),))


def test_training_rows_support_both_generic_loss_modes() -> None:
    record = _record()

    assistant_rows = training_rows((record,), loss_mode="assistant_only")
    completion_rows = training_rows((record,), loss_mode="completion_only")

    assert assistant_rows[0]["messages"] == [
        {"role": "system", "content": "Be precise."},
        {"role": "user", "content": "Write an identity function."},
        {"role": "assistant", "content": "fn identity<T>(value: T) -> T { value }"},
    ]
    assert completion_rows[0]["prompt"] == [
        {"role": "system", "content": "Be precise."},
        {"role": "user", "content": "Write an identity function."},
    ]
    assert completion_rows[0]["completion"] == [
        {"role": "assistant", "content": "fn identity<T>(value: T) -> T { value }"}
    ]


def test_training_rows_reject_completion_without_prompt() -> None:
    record = NormalizedTrainingRecord(
        schema_version=1,
        messages=(TrainingMessage(role="assistant", content="answer"),),
        language="rust",
        provenance=_record().provenance,
    )

    with pytest.raises(AdapterTrainingError, match="non-empty prompt"):
        training_rows((record,), loss_mode="completion_only")


def test_resolve_training_plan_uses_registry_and_frozen_revisions(tmp_path: Path) -> None:
    manifest = tmp_path / "dataset-manifest.json"
    config = tmp_path / "training.yaml"
    _write_dataset_manifest(manifest)
    _write_training_config(config, dataset_manifest=manifest, output_dir=tmp_path / "run")

    plan = resolve_adapter_training_plan(config, registry=_rust_registry(tmp_path))

    assert plan.language == "rust"
    assert plan.config.adapter_id == "language/rust/p0"
    assert plan.target.model_revision == _REVISION
    assert plan.target.tokenizer_revision == _REVISION
    assert plan.dataset.manifest_id == "dataset/rust/p0"
    assert plan.dataset.sha256 == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert plan.artifacts.adapter == tmp_path / "run" / "adapter"
    assert plan.artifacts.training_metrics == tmp_path / "run" / "training-metrics.jsonl"


def test_resolve_training_plan_rejects_dataset_revision_drift(tmp_path: Path) -> None:
    manifest = tmp_path / "dataset-manifest.json"
    config = tmp_path / "training.yaml"
    _write_dataset_manifest(manifest, revision="a" * 40)
    _write_training_config(config, dataset_manifest=manifest, output_dir=tmp_path / "run")

    with pytest.raises(AdapterTrainingError, match="tokenizer revision"):
        resolve_adapter_training_plan(config, registry=_rust_registry(tmp_path))


def test_training_config_rejects_adapter_identity_for_another_language(tmp_path: Path) -> None:
    manifest = tmp_path / "dataset-manifest.json"
    config = tmp_path / "training.yaml"
    _write_dataset_manifest(manifest)
    _write_training_config(config, dataset_manifest=manifest, output_dir=tmp_path / "run")
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "adapter_id: language/rust/p0", "adapter_id: language/python/p0"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="adapter_family/language"):
        resolve_adapter_training_plan(config, registry=_rust_registry(tmp_path))


class _FakeParameter:
    def __init__(self, size: int, *, requires_grad: bool) -> None:
        self._size = size
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return self._size


class _FakeLoraLayer:
    lora_A = object()
    lora_B = object()


class _FakeModel:
    def get_base_model(self) -> _FakeModel:
        return self

    def named_modules(self) -> Iterator[tuple[str, object]]:
        yield "", self
        yield "layer.q_proj", _FakeLoraLayer()

    def parameters(self) -> Iterator[_FakeParameter]:
        yield _FakeParameter(4, requires_grad=True)
        yield _FakeParameter(10, requires_grad=False)


class _FakeTokenizer:
    chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"


def _run_manifest(plan: AdapterTrainingPlan) -> RunManifest:
    return RunManifest(
        schema_version=1,
        run_id="training-rust-fixture",
        run_kind="training",
        created_at_utc="2026-08-30T10:00:00.000000Z",
        git=GitMetadata(sha="b" * 40, dirty=False),
        base_model=BaseModelIdentity(
            repository=plan.target.model_repository,
            revision=plan.target.model_revision,
            tokenizer_repository=plan.target.tokenizer_repository,
            tokenizer_revision=plan.target.tokenizer_revision,
        ),
        language=plan.language,
        adapter=AdapterIdentity(
            family=plan.config.adapter_family,
            adapter_id=plan.config.adapter_id,
        ),
        seed=1337,
        dependencies=DependencyVersions(
            accelerate="1",
            bitsandbytes="1",
            datasets="1",
            numpy="1",
            peft="1",
            pyyaml="1",
            tiny_qwen_coder="1",
            torch="1",
            transformers="1",
            trl="1",
        ),
        host=HostMetadata(
            hostname="host",
            system="Linux",
            release="1",
            machine="x86_64",
            python_version="3.11",
            cuda_available=True,
            cuda_runtime="13.0",
            gpus=(),
        ),
    )


def test_completed_manifest_uses_measured_targets_and_trainable_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "dataset-manifest.json"
    config = tmp_path / "training.yaml"
    _write_dataset_manifest(manifest)
    _write_training_config(
        config,
        dataset_manifest=manifest,
        output_dir=tmp_path / "run",
        loss_mode="completion_only",
    )
    plan = resolve_adapter_training_plan(config, registry=_rust_registry(tmp_path))

    def fake_version(name: str) -> str:
        return f"{name}-version"

    monkeypatch.setattr("importlib.metadata.version", fake_version)
    adapter_manifest = create_completed_adapter_manifest(
        plan,
        _run_manifest(plan),
        tokenizer=_FakeTokenizer(),
        model=_FakeModel(),
        global_steps=12,
        validation_loss=1.25,
        peak_vram_bytes=1234,
    )

    assert adapter_manifest.language == "rust"
    assert adapter_manifest.lora.target_modules == ("layer.q_proj",)
    assert adapter_manifest.lora.trainable_parameters == 4
    assert adapter_manifest.training_summary.steps == 12
    assert adapter_manifest.validation_metrics[0].value == 1.25
