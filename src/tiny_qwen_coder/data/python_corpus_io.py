"""Materialization and serialization helpers for the canonical Python P0 corpus."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from tiny_qwen_coder.data.length_filtering import load_canonical_tokenizer
from tiny_qwen_coder.data.python_corpus import (
    PythonP0CorpusResult,
    build_python_p0_corpus,
)
from tiny_qwen_coder.data.python_corpus_config import (
    DEFAULT_PYTHON_P0_CONFIG,
    PythonP0CorpusConfig,
    PythonP0CorpusError,
    load_python_p0_config,
)
from tiny_qwen_coder.data.python_p0_manifest import (
    PythonP0DatasetManifest,
    create_python_p0_dataset_manifest,
    python_p0_dataset_manifest_sha256,
    split_python_p0_corpus,
    write_python_p0_dataset_manifest,
)
from tiny_qwen_coder.data.records import NormalizedTrainingRecord
from tiny_qwen_coder.data.source_config import DatasetSourceConfig, load_dataset_source_config
from tiny_qwen_coder.data.splitting import DeduplicatedDatasetSplit
from tiny_qwen_coder.languages.python import load_python_plugin
from tiny_qwen_coder.model.inspection import load_inspection_target
from tiny_qwen_coder.reporting.dataset_manifest import ContaminationSummary
from tiny_qwen_coder.reporting.manifest import GitMetadata

_SCHEMA_VERSION = 1
_DEFAULT_BASE_CONFIG = Path("configs/base/qwen35-4b.yaml")


@dataclass(frozen=True, slots=True)
class FrozenPythonP0Artifacts:
    """Materialized records and audit files for one frozen Python P0 corpus."""

    result: PythonP0CorpusResult
    split: DeduplicatedDatasetSplit
    manifest: PythonP0DatasetManifest
    accepted_path: Path
    composition_path: Path
    train_path: Path
    validation_path: Path
    manifest_path: Path
    manifest_checksum_path: Path
    manifest_sha256: str


def _load_sources(
    config_path: Path,
) -> tuple[PythonP0CorpusConfig, dict[str, DatasetSourceConfig]]:
    config = load_python_p0_config(config_path)
    sources: dict[str, DatasetSourceConfig] = {}
    for budget in config.sources:
        source = load_dataset_source_config(Path(budget.source_config))
        if source.id != budget.id:
            raise PythonP0CorpusError(
                f"source config {budget.source_config!r} resolved to {source.id!r}, "
                f"expected {budget.id!r}"
            )
        if source.language != config.language:
            raise PythonP0CorpusError(
                f"source {source.id!r} language {source.language!r} does not match "
                f"corpus language {config.language!r}"
            )
        sources[source.id] = source
    return config, sources


def build_canonical_python_p0(
    *,
    config_path: Path = DEFAULT_PYTHON_P0_CONFIG,
    base_config: Path = _DEFAULT_BASE_CONFIG,
    local_files_only: bool = False,
) -> PythonP0CorpusResult:
    """Load pinned sources/tokenizer and construct the canonical Python P0 corpus."""

    config, sources = _load_sources(config_path)
    plugin = load_python_plugin()
    target = load_inspection_target(base_config)
    tokenizer = load_canonical_tokenizer(target, local_files_only=local_files_only)
    return build_python_p0_corpus(
        config,
        plugin=plugin,
        tokenizer=tokenizer,
        target=target,
        source_configs=sources,
    )


def python_p0_summary_json(result: PythonP0CorpusResult) -> str:
    """Serialize measured P5-005 composition without embedding training content."""

    payload = {
        "schema_version": _SCHEMA_VERSION,
        "id": result.config.id,
        "target_total": result.config.target_total,
        "accepted_total": result.accepted_total,
        "shortfall": result.shortfall,
        "fill_accepted": result.fill_accepted,
        "sources": [asdict(item) for item in result.source_stats],
        "rejection_counts": [
            {"stage": item.stage.value, "reason": item.reason, "count": item.count}
            for item in result.rejection_counts
        ],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_records_jsonl(
    records: tuple[NormalizedTrainingRecord, ...],
    destination: Path,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    asdict(record),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            handle.write("\n")
    temporary.replace(destination)
    return destination


def _write_text_atomic(destination: Path, content: str, *, encoding: str = "utf-8") -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(content, encoding=encoding)
    temporary.replace(destination)
    return destination


def write_python_p0_jsonl(result: PythonP0CorpusResult, path: Path | None = None) -> Path:
    """Write accepted pre-split P0 records deterministically as UTF-8 JSONL."""

    destination = path or Path(result.config.output_jsonl)
    return _write_records_jsonl(result.accepted_records, destination)


def write_python_p0_split(
    split: DeduplicatedDatasetSplit,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write deterministic linkage-safe train and validation JSONL files."""

    train_path = _write_records_jsonl(split.train_records, output_dir / "train.jsonl")
    validation_path = _write_records_jsonl(
        split.validation_records, output_dir / "validation.jsonl"
    )
    return train_path, validation_path


def materialize_canonical_python_p0(
    *,
    config_path: Path = DEFAULT_PYTHON_P0_CONFIG,
    base_config: Path = _DEFAULT_BASE_CONFIG,
    local_files_only: bool = False,
) -> tuple[PythonP0CorpusResult, Path, Path]:
    """Build P0 and write deterministic pre-split records plus composition summary."""

    result = build_canonical_python_p0(
        config_path=config_path,
        base_config=base_config,
        local_files_only=local_files_only,
    )
    output = write_python_p0_jsonl(result)
    summary_path = output.with_name("composition.json")
    summary_path.write_text(python_p0_summary_json(result) + "\n", encoding="utf-8")
    return result, output, summary_path


def freeze_python_p0_result(
    result: PythonP0CorpusResult,
    *,
    source_configs: dict[str, DatasetSourceConfig],
    output_dir: Path | None = None,
    contamination: ContaminationSummary | None = None,
    git: GitMetadata | None = None,
    repo_root: Path = Path("."),
) -> FrozenPythonP0Artifacts:
    """Split, validate, then atomically write one measured Python P0 result."""

    destination = output_dir or Path(result.config.output_jsonl).parent
    split = split_python_p0_corpus(result)
    manifest = create_python_p0_dataset_manifest(
        result,
        source_configs=source_configs,
        split=split,
        contamination=contamination,
        repo_root=repo_root,
        git=git,
    )

    accepted_path = write_python_p0_jsonl(
        result, destination / Path(result.config.output_jsonl).name
    )
    composition_path = _write_text_atomic(
        destination / "composition.json", python_p0_summary_json(result) + "\n"
    )
    train_path, validation_path = write_python_p0_split(split, destination)
    manifest_files = write_python_p0_dataset_manifest(manifest, destination)
    return FrozenPythonP0Artifacts(
        result=result,
        split=split,
        manifest=manifest,
        accepted_path=accepted_path,
        composition_path=composition_path,
        train_path=train_path,
        validation_path=validation_path,
        manifest_path=manifest_files.manifest,
        manifest_checksum_path=manifest_files.checksum,
        manifest_sha256=python_p0_dataset_manifest_sha256(manifest),
    )


def freeze_canonical_python_p0(
    *,
    config_path: Path = DEFAULT_PYTHON_P0_CONFIG,
    base_config: Path = _DEFAULT_BASE_CONFIG,
    local_files_only: bool = False,
    contamination: ContaminationSummary | None = None,
    repo_root: Path = Path("."),
) -> FrozenPythonP0Artifacts:
    """Build and freeze the canonical Python P0 records, split, and manifest."""

    result = build_canonical_python_p0(
        config_path=config_path,
        base_config=base_config,
        local_files_only=local_files_only,
    )
    config, sources = _load_sources(config_path)
    if config != result.config:
        raise PythonP0CorpusError("reloaded P0 config does not match measured corpus config")
    return freeze_python_p0_result(
        result,
        source_configs=sources,
        contamination=contamination,
        repo_root=repo_root,
    )

