"""Dispatch stripping for versioned teacher-only distillation input policies."""

from __future__ import annotations

from tiny_qwen_coder.data.records import NormalizedTrainingRecord
from tiny_qwen_coder.distillation.v2_input import strip_v2_teacher_input_policy
from tiny_qwen_coder.distillation.v3_input import V3_POLICY_ID, strip_v3_teacher_input_policy


class TeacherInputPolicyError(ValueError):
    """Raised when a persisted teacher-input policy cannot be identified safely."""


def strip_teacher_input_policy(record: NormalizedTrainingRecord) -> NormalizedTrainingRecord:
    """Remove a recognized teacher-only policy before student tokenization or corpus writing."""

    metadata = dict(record.provenance.source_metadata)
    policy_id = metadata.get("distillation.input_policy")
    policy_sha256 = metadata.get("distillation.input_policy_sha256")
    if policy_id is None and policy_sha256 is None:
        return record
    if policy_id is None or policy_sha256 is None:
        raise TeacherInputPolicyError("incomplete distillation input policy metadata")
    if policy_id == "concise-v2":
        return strip_v2_teacher_input_policy(record)
    if policy_id == V3_POLICY_ID:
        return strip_v3_teacher_input_policy(record)
    raise TeacherInputPolicyError(f"unsupported distillation input policy: {policy_id!r}")


__all__ = ["TeacherInputPolicyError", "strip_teacher_input_policy"]
