"""FTR-202 benchmark protocol and FTR-203 precommit validation."""

from __future__ import annotations

from pathlib import Path

from tiny_qwen_coder.evaluation._ftr_teacher_common import (
    FTRTeacherSuperiorityError,
    expect_int,
    expect_number,
    expect_str,
    read_json,
    read_yaml,
    score_pair,
    sha256_file,
    strict_mapping,
)

PROTOCOL_PATH = Path("configs/eval/python/ftr_202_teacher_benchmark_v1.json")
GATE_PATH = Path("configs/eval/python/ftr_203_teacher_superiority_gate_v1.json")
_FTR201_EVIDENCE = Path("docs/evidence/FTR_201_DIRECT_TEACHER_EVALUATION_SUPPORT.json")
_FTR201_EVAL = Path("configs/eval/python/ftr_201_teacher_direct_v1.yaml")
_EXPECTED_TEACHER = ("Qwen/Qwen3.8-27B", "72a217afab8029b39e4af1c7273a829995a3dbaf")
_EXPECTED_EXECUTION_IMAGE = (
    "python:3.11.14-slim@sha256:"
    "c8271b1f627d0068857dce5b53e14a9558603b527e46f1f901722f935b786a39"
)
_EXPECTED_BASE_SCORES = {
    "humaneval": (128, 164),
    "mbpp": (290, 500),
    "repository_holdout": (6, 11),
    "public_coding": (418, 664),
    "combined_coding": (424, 675),
}
_EXPECTED_PRIMARY = {
    "base_passed": 418,
    "total_tasks": 664,
    "minimum_net_improvement_tasks": 54,
    "minimum_teacher_passed": 472,
}


def _validate_source_sha(source_git_sha: str) -> None:
    valid = len(source_git_sha) == 40 and all(
        character in "0123456789abcdef" for character in source_git_sha
    )
    if not valid:
        raise FTRTeacherSuperiorityError("source_git_sha must be a lowercase 40-character SHA")


def audit_teacher_benchmark_protocol(*, repo_root: Path, source_git_sha: str) -> dict[str, object]:
    """Validate FTR-202 protocol and prove the FTR-203 gate was frozen with it."""

    _validate_source_sha(source_git_sha)
    protocol_path = repo_root / PROTOCOL_PATH
    gate_path = repo_root / GATE_PATH
    protocol = read_json(protocol_path, context="FTR-202 protocol")
    gate = read_json(gate_path, context="FTR-203 gate")
    ftr201 = read_json(repo_root / _FTR201_EVIDENCE, context="FTR-201 evidence")
    teacher_eval = read_yaml(repo_root / _FTR201_EVAL, context="FTR-201 teacher evaluation")

    if expect_str(protocol, "task_id", context="FTR-202 protocol") != "FTR-202":
        raise FTRTeacherSuperiorityError("unexpected FTR-202 task_id")
    if expect_str(gate, "task_id", context="FTR-203 gate") != "FTR-203":
        raise FTRTeacherSuperiorityError("unexpected FTR-203 task_id")
    if ftr201.get("support_ready") is not True:
        raise FTRTeacherSuperiorityError("FTR-201 direct teacher support is not frozen as ready")

    teacher = strict_mapping(protocol.get("teacher"), context="FTR-202 teacher")
    observed_teacher = (
        expect_str(teacher, "repository", context="FTR-202 teacher"),
        expect_str(teacher, "revision", context="FTR-202 teacher"),
    )
    if observed_teacher != _EXPECTED_TEACHER:
        raise FTRTeacherSuperiorityError("FTR-202 teacher identity drifted")
    if expect_str(teacher_eval, "output_dir", context="FTR-201 evaluation") != (
        "artifacts/eval/python/ftr-201-teacher-direct-v1"
    ):
        raise FTRTeacherSuperiorityError("FTR-201 teacher artifact root drifted")

    base = strict_mapping(protocol.get("frozen_base_reference"), context="FTR-202 frozen base")
    scores = strict_mapping(base.get("scores"), context="FTR-202 frozen base scores")
    for suite_id, expected_score in _EXPECTED_BASE_SCORES.items():
        if (
            score_pair(scores.get(suite_id), context=f"FTR-202 base score {suite_id}")
            != expected_score
        ):
            raise FTRTeacherSuperiorityError(f"frozen base score drifted for {suite_id}")

    holdout = strict_mapping(
        protocol.get("repository_holdout_policy"), context="FTR-202 holdout policy"
    )
    if (
        holdout.get("eligible") is not True
        or holdout.get("may_tune_after_observation") is not False
    ):
        raise FTRTeacherSuperiorityError("repository holdout must remain one-shot and non-tuning")

    generation = strict_mapping(protocol.get("generation"), context="FTR-202 generation")
    scoring = strict_mapping(protocol.get("scoring"), context="FTR-202 scoring")
    minimum_bytes = expect_int(generation, "minimum_cuda_total_bytes", context="FTR-202 generation")
    if minimum_bytes < 75 * 1024**3 or generation.get("execute_generated_code") is not False:
        raise FTRTeacherSuperiorityError("FTR-202 A100 generation isolation contract drifted")
    if (
        scoring.get("require_oci_isolation") is not True
        or scoring.get("network_enabled") is not False
        or scoring.get("execute_with_drive_or_cloud_credentials_mounted") is not False
    ):
        raise FTRTeacherSuperiorityError("FTR-202 scoring isolation contract drifted")
    if expect_str(scoring, "oci_runtime", context="FTR-202 scoring") != "docker":
        raise FTRTeacherSuperiorityError("FTR-202 must reproduce the historical Docker runtime")
    if expect_str(scoring, "execution_image", context="FTR-202 scoring") != _EXPECTED_EXECUTION_IMAGE:
        raise FTRTeacherSuperiorityError("FTR-202 execution image drifted from the frozen baseline")
    if scoring.get("require_preloaded_execution_image") is not True:
        raise FTRTeacherSuperiorityError("FTR-202 must require the pinned image before scoring")

    if gate.get("precommitted_before_teacher_results") is not True:
        raise FTRTeacherSuperiorityError("FTR-203 gate must be precommitted")
    if expect_int(gate, "fixed_teacher_comparisons", context="FTR-203 gate") != 1:
        raise FTRTeacherSuperiorityError("FTR-203 must cover exactly one fixed teacher comparison")
    primary = strict_mapping(gate.get("primary_capability"), context="FTR-203 primary")
    for key, expected_value in _EXPECTED_PRIMARY.items():
        if expect_int(primary, key, context="FTR-203 primary") != expected_value:
            raise FTRTeacherSuperiorityError(f"FTR-203 primary {key} drifted")
    if expect_number(primary, "minimum_absolute_pass_rate_gain", context="FTR-203 primary") != 0.08:
        raise FTRTeacherSuperiorityError("FTR-203 absolute gain floor drifted")

    checks = {
        "ftr_201_support_frozen": True,
        "exact_teacher_identity_pinned": True,
        "frozen_base_reference_pinned": True,
        "a100_80gb_generation_required": True,
        "generation_does_not_execute_candidates": True,
        "isolated_network_disabled_scoring_required": True,
        "historical_docker_runtime_pinned": True,
        "execution_image_digest_pinned": True,
        "cloud_credentials_excluded_from_candidate_execution": True,
        "repository_holdout_one_shot_status_frozen": True,
        "ftr_203_gate_precommitted": True,
        "eight_point_public_coding_margin_frozen": True,
    }
    return {
        "schema_version": 1,
        "task_id": "FTR-202/FTR-203",
        "source_git_sha": source_git_sha,
        "protocol_path": PROTOCOL_PATH.as_posix(),
        "protocol_sha256": sha256_file(protocol_path),
        "gate_path": GATE_PATH.as_posix(),
        "gate_sha256": sha256_file(gate_path),
        "checks": checks,
        "protocol_ready": all(checks.values()),
    }
