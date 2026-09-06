"""Qualification gates for bounded teacher-distillation studies."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from tiny_qwen_coder.reporting.dataset_manifest import ContaminationStatus


class TeacherStudyQualificationError(ValueError):
    """Raised when study evidence is malformed or incomplete."""


@dataclass(frozen=True, slots=True)
class TeacherStudyQualification:
    """Mechanical go/no-go result for scaling a bounded teacher study."""

    schema_version: int
    total_records: int
    stop_rate: float
    minimum_stop_rate: float
    student_length_accept_rate_given_stop: float
    minimum_student_length_accept_rate_given_stop: float
    contamination_status: str
    qualified: bool
    failed_gates: tuple[str, ...]


def _mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TeacherStudyQualificationError(f"{context} must be a string-keyed mapping")
    return value


def _number(mapping: dict[str, object], key: str, *, context: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TeacherStudyQualificationError(f"{context}.{key} must be numeric")
    return float(value)


def _integer(mapping: dict[str, object], key: str, *, context: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TeacherStudyQualificationError(f"{context}.{key} must be a non-negative integer")
    return value


def qualify_teacher_study(
    *,
    diagnostics_summary: dict[str, object],
    dataset_manifest: dict[str, object],
    minimum_stop_rate: float = 0.90,
    minimum_student_length_accept_rate_given_stop: float = 0.80,
) -> TeacherStudyQualification:
    """Require bounded generation quality and a clean contamination result."""

    for field_name, value in (
        ("minimum_stop_rate", minimum_stop_rate),
        (
            "minimum_student_length_accept_rate_given_stop",
            minimum_student_length_accept_rate_given_stop,
        ),
    ):
        if not 0.0 < value <= 1.0:
            raise TeacherStudyQualificationError(f"{field_name} must be in (0, 1]")

    total_records = _integer(diagnostics_summary, "total_records", context="diagnostics")
    if total_records == 0:
        raise TeacherStudyQualificationError("diagnostics.total_records must be positive")
    stop_rate = _number(diagnostics_summary, "stop_rate", context="diagnostics")
    student_accept_rate = _number(
        diagnostics_summary,
        "student_length_accept_rate_given_stop",
        context="diagnostics",
    )
    if not 0.0 <= stop_rate <= 1.0:
        raise TeacherStudyQualificationError("diagnostics.stop_rate must be in [0, 1]")
    if not 0.0 <= student_accept_rate <= 1.0:
        raise TeacherStudyQualificationError(
            "diagnostics.student_length_accept_rate_given_stop must be in [0, 1]"
        )

    contamination = _mapping(dataset_manifest.get("contamination"), context="manifest.contamination")
    raw_status = contamination.get("status")
    if not isinstance(raw_status, str):
        raise TeacherStudyQualificationError("manifest.contamination.status must be a string")
    try:
        contamination_status = ContaminationStatus(raw_status)
    except ValueError as exc:
        raise TeacherStudyQualificationError(
            f"unsupported contamination status {raw_status!r}"
        ) from exc

    failed: list[str] = []
    if stop_rate < minimum_stop_rate:
        failed.append("stop_rate")
    if student_accept_rate < minimum_student_length_accept_rate_given_stop:
        failed.append("student_length_accept_rate_given_stop")
    if contamination_status is not ContaminationStatus.CLEAN:
        failed.append("contamination")

    return TeacherStudyQualification(
        schema_version=1,
        total_records=total_records,
        stop_rate=stop_rate,
        minimum_stop_rate=minimum_stop_rate,
        student_length_accept_rate_given_stop=student_accept_rate,
        minimum_student_length_accept_rate_given_stop=(
            minimum_student_length_accept_rate_given_stop
        ),
        contamination_status=str(contamination_status),
        qualified=not failed,
        failed_gates=tuple(failed),
    )


def qualify_teacher_study_from_paths(
    *,
    diagnostics_summary_path: Path,
    dataset_manifest_path: Path,
    minimum_stop_rate: float = 0.90,
    minimum_student_length_accept_rate_given_stop: float = 0.80,
) -> TeacherStudyQualification:
    """Load JSON evidence and qualify one bounded study."""

    try:
        diagnostics_raw: object = json.loads(diagnostics_summary_path.read_text(encoding="utf-8"))
        manifest_raw: object = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TeacherStudyQualificationError("could not read bounded-study evidence") from exc
    return qualify_teacher_study(
        diagnostics_summary=_mapping(diagnostics_raw, context="diagnostics"),
        dataset_manifest=_mapping(manifest_raw, context="manifest"),
        minimum_stop_rate=minimum_stop_rate,
        minimum_student_length_accept_rate_given_stop=(
            minimum_student_length_accept_rate_given_stop
        ),
    )


__all__ = [
    "TeacherStudyQualification",
    "TeacherStudyQualificationError",
    "qualify_teacher_study",
    "qualify_teacher_study_from_paths",
]
