"""Transport-integrity helpers for FTR-202 teacher generation evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

HANDOFF_FILENAME = "FTR_202_GENERATION_HANDOFF.json"


class FTRTeacherTransportError(RuntimeError):
    """Raised when transported FTR-202 generation evidence is invalid."""


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise FTRTeacherTransportError(f"could not hash transport artifact: {path}") from exc


def _mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise FTRTeacherTransportError(f"{context} must be a mapping")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise FTRTeacherTransportError(f"{context} keys must be strings")
        result[key] = item
    return result


def _read_json(path: Path, *, context: str) -> dict[str, object]:
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FTRTeacherTransportError(f"could not read {context}: {path}") from exc
    return _mapping(raw, context=context)


def _validate_source_sha(value: object, *, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FTRTeacherTransportError(f"{context} must be a lowercase 40-character SHA")
    return value


def _validate_digest(value: object, *, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FTRTeacherTransportError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _relative_artifact_path(value: object, *, context: str) -> Path:
    if not isinstance(value, str) or not value:
        raise FTRTeacherTransportError(f"{context} path must be a non-empty string")
    path = Path(value)
    if path.is_absolute() or path == Path(".") or ".." in path.parts:
        raise FTRTeacherTransportError(f"{context} path must remain within the generation directory")
    return path


def _artifact_inventory(
    generation_dir: Path,
    raw_artifacts: object,
    *,
    context: str,
) -> list[dict[str, str]]:
    if not isinstance(raw_artifacts, list):
        raise FTRTeacherTransportError(f"{context} artifact inventory must be a list")
    root = generation_dir.resolve()
    inventory: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, raw_item in enumerate(raw_artifacts):
        item = _mapping(raw_item, context=f"{context} artifact[{index}]")
        relative = _relative_artifact_path(
            item.get("path"), context=f"{context} artifact[{index}]"
        )
        relative_text = relative.as_posix()
        if relative_text in seen:
            raise FTRTeacherTransportError(f"duplicate transport artifact: {relative_text}")
        seen.add(relative_text)
        expected_digest = _validate_digest(
            item.get("sha256"), context=f"{context} artifact[{index}].sha256"
        )
        candidate = generation_dir / relative
        if candidate.is_symlink():
            raise FTRTeacherTransportError(f"transport artifact must not be a symlink: {relative_text}")
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise FTRTeacherTransportError(f"transport artifact is missing: {relative_text}") from exc
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise FTRTeacherTransportError(
                f"transport artifact escapes generation directory: {relative_text}"
            )
        observed_digest = _sha256(resolved)
        if observed_digest != expected_digest:
            raise FTRTeacherTransportError(f"transport artifact digest mismatch: {relative_text}")
        inventory.append({"path": relative_text, "sha256": observed_digest})
    if not inventory:
        raise FTRTeacherTransportError(f"{context} artifact inventory must not be empty")
    return inventory


def build_generation_handoff(
    *, generation_dir: Path, source_sha: str, gpu: Mapping[str, object]
) -> dict[str, object]:
    """Build a self-verifying handoff manifest beside generated teacher evidence."""

    source_sha = _validate_source_sha(source_sha, context="source_git_sha")
    stage_path = generation_dir / "generation-stage.json"
    stage = _read_json(stage_path, context="generation stage")
    if stage.get("source_git_sha") != source_sha:
        raise FTRTeacherTransportError("generation-stage source SHA does not match this checkout")
    artifacts = _artifact_inventory(
        generation_dir,
        stage.get("artifacts"),
        context="generation stage",
    )
    return {
        "schema_version": 1,
        "task_id": "FTR-202",
        "source_git_sha": source_sha,
        "generation_stage_sha256": _sha256(stage_path),
        "teacher_generation_directory": generation_dir.as_posix(),
        "gpu": dict(gpu),
        "artifacts": artifacts,
        "artifact_count": len(artifacts),
        "scoring_performed": False,
        "candidate_execution_performed": False,
        "resume_policy": "checkpoint files are persisted directly to the mounted durable root",
    }


def write_generation_handoff(
    *, generation_dir: Path, handoff: Mapping[str, object]
) -> Path:
    """Write the FTR-202 handoff inside the transported generation directory."""

    path = generation_dir / HANDOFF_FILENAME
    path.write_text(json.dumps(dict(handoff), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def verify_generation_handoff(
    *, generation_dir: Path, expected_source_sha: str
) -> dict[str, object]:
    """Verify all transported teacher artifacts before any generated code is executed."""

    expected_source_sha = _validate_source_sha(
        expected_source_sha, context="expected_source_git_sha"
    )
    handoff = _read_json(generation_dir / HANDOFF_FILENAME, context="FTR-202 handoff")
    if handoff.get("schema_version") != 1 or handoff.get("task_id") != "FTR-202":
        raise FTRTeacherTransportError("unexpected FTR-202 handoff schema or task ID")
    if handoff.get("source_git_sha") != expected_source_sha:
        raise FTRTeacherTransportError("handoff source SHA does not match scoring checkout")
    if handoff.get("scoring_performed") is not False:
        raise FTRTeacherTransportError("handoff claims scoring was already performed")
    if handoff.get("candidate_execution_performed") is not False:
        raise FTRTeacherTransportError("handoff claims generated code was already executed")

    stage_path = generation_dir / "generation-stage.json"
    stage = _read_json(stage_path, context="generation stage")
    if stage.get("source_git_sha") != expected_source_sha:
        raise FTRTeacherTransportError("generation-stage source SHA does not match scoring checkout")
    expected_stage_digest = _validate_digest(
        handoff.get("generation_stage_sha256"), context="handoff generation_stage_sha256"
    )
    if _sha256(stage_path) != expected_stage_digest:
        raise FTRTeacherTransportError("generation-stage digest mismatch after transport")

    handoff_artifacts = _artifact_inventory(
        generation_dir,
        handoff.get("artifacts"),
        context="handoff",
    )
    stage_artifacts = _artifact_inventory(
        generation_dir,
        stage.get("artifacts"),
        context="generation stage",
    )
    if handoff_artifacts != stage_artifacts:
        raise FTRTeacherTransportError("handoff and generation-stage artifact inventories differ")
    artifact_count = handoff.get("artifact_count")
    if isinstance(artifact_count, bool) or not isinstance(artifact_count, int):
        raise FTRTeacherTransportError("handoff artifact_count must be an integer")
    if artifact_count != len(handoff_artifacts):
        raise FTRTeacherTransportError("handoff artifact_count does not match inventory")
    return handoff
