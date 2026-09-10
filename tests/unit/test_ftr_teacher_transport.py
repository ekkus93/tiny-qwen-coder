"""CPU-only FTR-202 transport-integrity tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tiny_qwen_coder.evaluation._ftr_teacher_transport import (
    FTRTeacherTransportError,
    build_generation_handoff,
    verify_generation_handoff,
    write_generation_handoff,
)

_SHA = "a" * 40


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_stage(root: Path, *, artifact_path: str = "humaneval/responses.jsonl") -> Path:
    artifact = root / artifact_path
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text('{"response":"ok"}\n', encoding="utf-8")
    stage = root / "generation-stage.json"
    stage.write_text(
        json.dumps(
            {
                "source_git_sha": _SHA,
                "artifacts": [{"path": artifact_path, "sha256": _sha256(artifact)}],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return artifact


def _build(root: Path) -> dict[str, object]:
    handoff = build_generation_handoff(
        generation_dir=root,
        source_sha=_SHA,
        gpu={"gpu_name": "NVIDIA A100-SXM4-80GB"},
    )
    write_generation_handoff(generation_dir=root, handoff=handoff)
    return handoff


def test_handoff_round_trip_verifies_all_artifact_hashes(tmp_path: Path) -> None:
    _write_stage(tmp_path)
    handoff = _build(tmp_path)

    verified = verify_generation_handoff(
        generation_dir=tmp_path,
        expected_source_sha=_SHA,
    )

    assert verified == handoff
    assert verified["artifact_count"] == 1
    assert verified["candidate_execution_performed"] is False


def test_transport_verification_rejects_artifact_tampering(tmp_path: Path) -> None:
    artifact = _write_stage(tmp_path)
    _build(tmp_path)
    artifact.write_text('{"response":"tampered"}\n', encoding="utf-8")

    with pytest.raises(FTRTeacherTransportError, match="digest mismatch"):
        verify_generation_handoff(
            generation_dir=tmp_path,
            expected_source_sha=_SHA,
        )


def test_transport_verification_rejects_handoff_stage_tampering(tmp_path: Path) -> None:
    _write_stage(tmp_path)
    _build(tmp_path)
    stage = tmp_path / "generation-stage.json"
    payload = json.loads(stage.read_text(encoding="utf-8"))
    payload["extra"] = "tamper"
    stage.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(FTRTeacherTransportError, match="generation-stage digest mismatch"):
        verify_generation_handoff(
            generation_dir=tmp_path,
            expected_source_sha=_SHA,
        )


def test_handoff_rejects_parent_traversal_artifact_paths(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.jsonl"
    outside.write_text("outside\n", encoding="utf-8")
    stage = tmp_path / "generation-stage.json"
    stage.write_text(
        json.dumps(
            {
                "source_git_sha": _SHA,
                "artifacts": [{"path": "../outside.jsonl", "sha256": _sha256(outside)}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(FTRTeacherTransportError, match="within the generation directory"):
        build_generation_handoff(
            generation_dir=tmp_path,
            source_sha=_SHA,
            gpu={"gpu_name": "NVIDIA A100-SXM4-80GB"},
        )


def test_transport_verification_rejects_executed_handoff_claim(tmp_path: Path) -> None:
    _write_stage(tmp_path)
    _build(tmp_path)
    handoff_path = tmp_path / "FTR_202_GENERATION_HANDOFF.json"
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    handoff["candidate_execution_performed"] = True
    handoff_path.write_text(json.dumps(handoff, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(FTRTeacherTransportError, match="generated code was already executed"):
        verify_generation_handoff(
            generation_dir=tmp_path,
            expected_source_sha=_SHA,
        )
