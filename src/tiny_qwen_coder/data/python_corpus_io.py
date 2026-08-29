"""Materialization and serialization helpers for the canonical Python P0 corpus."""

from __future__ import annotations

import json
from dataclasses import asdict
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
from tiny_qwen_coder.data.source_config import DatasetSourceConfig, load_dataset_source_config
from tiny_qwen_coder.languages.python import load_python_plugin
from tiny_qwen_coder.model.inspection import load_inspection_target

_SCHEMA_VERSION = 1
_DEFAULT_BASE_CONFIG = Path("configs/base/qwen35-4b.yaml")


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


def write_python_p0_jsonl(result: PythonP0CorpusResult, path: Path | None = None) -> Path:
    """Write accepted pre-split P0 records deterministically as UTF-8 JSONL."""

    destination = path or Path(result.config.output_jsonl)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in result.accepted_records:
            handle.write(
                json.dumps(
                    asdict(record),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            handle.write("\n")
    return destination


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
