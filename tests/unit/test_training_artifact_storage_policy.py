"""Regression tests for bounded GitHub Actions training-artifact storage."""

from __future__ import annotations

from pathlib import Path

_FULL_WORKFLOW = Path(".github/workflows/python-p0-full-training.yml")
_SMOKE_WORKFLOW = Path(".github/workflows/python-p0-smoke-training.yml")


def _section(text: str, start: str, end: str) -> str:
    return text.split(start, 1)[1].split(end, 1)[0]


def test_full_training_is_manual_only_and_uploads_compact_recovery_bundle() -> None:
    workflow = _FULL_WORKFLOW.read_text(encoding="utf-8")
    trigger_block = workflow.split("permissions:", 1)[0]

    assert "workflow_dispatch:" in trigger_block
    assert "\n  push:" not in trigger_block

    upload = _section(
        workflow,
        "- name: Upload compact P7-006 recovery bundle",
        "- name: Upload compact partial evidence on failure",
    )
    assert "artifacts/train/python/p0/adapter/" in upload
    assert "artifacts/train/python/p0/training-report.json" in upload
    assert "artifacts/train/python/p0/\n" not in upload
    assert "checkpoints/" not in upload
    assert "retention-days: 7" in upload

    partial = workflow.split("- name: Upload compact partial evidence on failure", 1)[1]
    assert "artifacts/train/python/p0/checkpoints" not in partial
    assert "retention-days: 3" in partial

    assert "Canonical local output remains at:" in workflow


def test_smoke_training_uploads_evidence_only_and_does_not_self_trigger() -> None:
    workflow = _SMOKE_WORKFLOW.read_text(encoding="utf-8")
    trigger_block = workflow.split("permissions:", 1)[0]

    assert "workflow_dispatch:" in trigger_block
    assert "\n  push:" in trigger_block
    assert '".github/workflows/python-p0-smoke-training.yml"' not in trigger_block

    upload = _section(
        workflow,
        "- name: Upload compact smoke evidence",
        "- name: Upload compact partial smoke evidence on failure",
    )
    assert "smoke-training-report.json" not in upload or "*.json" in upload
    assert "adapter/adapter_config.json" in upload
    assert "adapter_model" not in upload
    assert "checkpoints/" not in upload
    assert "artifacts/train/python/p0-smoke/\n" not in upload
    assert "retention-days: 7" in upload

    partial = workflow.split("- name: Upload compact partial smoke evidence on failure", 1)[1]
    assert "adapter_model" not in partial
    assert "checkpoints/" not in partial
    assert "retention-days: 3" in partial
