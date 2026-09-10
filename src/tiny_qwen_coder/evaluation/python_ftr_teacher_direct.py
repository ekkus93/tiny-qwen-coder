"""FTR-201 direct Qwen3.8-27B evaluation support.

This module intentionally delegates generation and scoring to the canonical
unchanged-model baseline stages.  Only the pinned model/tokenizer identity and
artifact destination differ, so prompt construction, decoding, normalization,
execution, and evidence formats remain shared with the Qwen3.5-4B baseline.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, cast

import yaml

from tiny_qwen_coder.config import EvaluationConfig, load_evaluation_config
from tiny_qwen_coder.evaluation._baseline_provenance import load_baseline_base_model_identity
from tiny_qwen_coder.evaluation.settings import (
    evaluation_settings_sha256,
    load_frozen_evaluation_settings,
    validate_evaluation_config_settings,
)
from tiny_qwen_coder.identities import BaseModelIdentity

if TYPE_CHECKING:
    from tiny_qwen_coder.evaluation._baseline_types import PythonBaselineManifest

_TASK_ID = "FTR-201"
_SCHEMA_VERSION = 1
_BASE_EVALUATION = Path("configs/eval/python/base_baseline_v1.yaml")
_TEACHER_EVALUATION = Path("configs/eval/python/ftr_201_teacher_direct_v1.yaml")
_TEACHER_BASE = Path("configs/base/qwen38-27b-teacher.yaml")
_TEACHER_DISTILLATION = Path("configs/distillation/python/qwen38_27b_v1.yaml")
_EXPECTED_OUTPUT = Path("artifacts/eval/python/ftr-201-teacher-direct-v1")
_TRAINING_CONFIG_ROOTS = (
    Path("configs/data"),
    Path("configs/distillation"),
    Path("configs/train"),
)


class FTRTeacherDirectError(RuntimeError):
    """Raised when the direct-teacher evaluation contract is not comparable."""


def _resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _load_evaluation(repo_root: Path, path: Path) -> EvaluationConfig:
    return load_evaluation_config(_resolve(repo_root, path))


def _load_identity(repo_root: Path, path: Path) -> BaseModelIdentity:
    return load_baseline_base_model_identity(_resolve(repo_root, path))


def _load_distillation_teacher_identity(repo_root: Path) -> BaseModelIdentity:
    path = _resolve(repo_root, _TEACHER_DISTILLATION)
    try:
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise FTRTeacherDirectError(f"could not read pinned teacher config {path}") from exc
    if not isinstance(raw, dict):
        raise FTRTeacherDirectError("pinned teacher config must be a mapping")
    teacher = raw.get("teacher")
    if not isinstance(teacher, dict):
        raise FTRTeacherDirectError("pinned teacher config must contain teacher mapping")
    repository = teacher.get("repository")
    revision = teacher.get("revision")
    if not isinstance(repository, str) or not repository.strip():
        raise FTRTeacherDirectError("pinned teacher repository is invalid")
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(char not in "0123456789abcdef" for char in revision)
    ):
        raise FTRTeacherDirectError("pinned teacher revision is invalid")
    return BaseModelIdentity(
        repository=repository,
        revision=revision,
        tokenizer_repository=repository,
        tokenizer_revision=revision,
    )


def _evaluation_semantics(config: EvaluationConfig) -> dict[str, object]:
    """Return only fields that affect benchmark prompts, generation, or scoring."""

    return {
        "language": config.language,
        "adapter_id": config.adapter_id,
        "suites": list(config.suites),
        "seed": config.seed,
        "generation": asdict(config.generation),
        "execution": asdict(config.execution),
    }


def _training_input_references(repo_root: Path, artifact_root: Path) -> tuple[str, ...]:
    needle = artifact_root.as_posix()
    references: list[str] = []
    for relative_root in _TRAINING_CONFIG_ROOTS:
        root = repo_root / relative_root
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                raise FTRTeacherDirectError(f"could not inspect training config {path}") from exc
            if needle in text:
                references.append(path.relative_to(repo_root).as_posix())
    return tuple(references)


def audit_teacher_direct_support(
    *,
    repo_root: Path,
    source_git_sha: str,
    teacher_evaluation_path: Path = _TEACHER_EVALUATION,
) -> dict[str, object]:
    """Prove that FTR-201 changes identity only, not evaluation semantics."""

    if len(source_git_sha) != 40 or any(char not in "0123456789abcdef" for char in source_git_sha):
        raise FTRTeacherDirectError("source_git_sha must be a lowercase 40-character SHA")

    base_eval = _load_evaluation(repo_root, _BASE_EVALUATION)
    teacher_eval = _load_evaluation(repo_root, teacher_evaluation_path)
    base_identity = _load_identity(repo_root, Path(base_eval.base_config))
    teacher_identity = _load_identity(repo_root, Path(teacher_eval.base_config))
    frozen_settings = load_frozen_evaluation_settings(
        _resolve(repo_root, Path("configs/eval/canonical_generation_v1.yaml"))
    )

    expected_teacher = _load_distillation_teacher_identity(repo_root)
    if teacher_identity != expected_teacher:
        raise FTRTeacherDirectError(
            "direct-evaluation teacher identity must exactly match the pinned distillation teacher"
        )
    if Path(teacher_eval.base_config) != _TEACHER_BASE:
        raise FTRTeacherDirectError("teacher evaluation must use the dedicated FTR-201 base config")

    base_semantics = _evaluation_semantics(base_eval)
    teacher_semantics = _evaluation_semantics(teacher_eval)
    if teacher_semantics != base_semantics:
        raise FTRTeacherDirectError(
            "teacher evaluation semantics drifted from the canonical unchanged-base evaluation"
        )

    base_settings_sha = validate_evaluation_config_settings(base_eval, frozen_settings)
    teacher_settings_sha = validate_evaluation_config_settings(teacher_eval, frozen_settings)
    frozen_settings_sha = evaluation_settings_sha256(frozen_settings)
    if base_settings_sha != frozen_settings_sha or teacher_settings_sha != frozen_settings_sha:
        raise FTRTeacherDirectError(
            "teacher/base evaluation settings do not match the frozen contract"
        )

    artifact_root = Path(teacher_eval.output_dir)
    if artifact_root != _EXPECTED_OUTPUT:
        raise FTRTeacherDirectError(
            f"teacher output_dir must remain {_EXPECTED_OUTPUT.as_posix()!r}"
        )
    if artifact_root.parts[:2] != ("artifacts", "eval"):
        raise FTRTeacherDirectError("teacher benchmark evidence must live under artifacts/eval")
    training_references = _training_input_references(repo_root, artifact_root)
    if training_references:
        raise FTRTeacherDirectError(
            "teacher benchmark artifacts are referenced by training/data configs: "
            + ", ".join(training_references)
        )

    checks = {
        "pinned_teacher_identity_exact": True,
        "tokenizer_identity_exact": True,
        "canonical_prompt_construction_shared": True,
        "generation_semantics_equal_to_base": True,
        "scoring_semantics_equal_to_base": True,
        "frozen_evaluation_settings_equal": True,
        "exact_response_checkpointing_shared": True,
        "execution_evidence_writers_shared": True,
        "benchmark_artifacts_outside_training_inputs": True,
    }
    return {
        "schema_version": _SCHEMA_VERSION,
        "task_id": _TASK_ID,
        "source_git_sha": source_git_sha,
        "base_model": asdict(base_identity),
        "teacher_model": asdict(teacher_identity),
        "base_evaluation_config": _BASE_EVALUATION.as_posix(),
        "teacher_evaluation_config": teacher_evaluation_path.as_posix(),
        "teacher_distillation_config": _TEACHER_DISTILLATION.as_posix(),
        "teacher_artifact_root": artifact_root.as_posix(),
        "evaluation_semantics": teacher_semantics,
        "evaluation_settings_sha256": frozen_settings_sha,
        "shared_generation_stage": (
            "tiny_qwen_coder.evaluation._baseline_stages."
            "generate_canonical_python_base_baseline_stage"
        ),
        "shared_scoring_stage": (
            "tiny_qwen_coder.evaluation._baseline_stages.score_canonical_python_base_baseline_stage"
        ),
        "training_input_references": list(training_references),
        "checks": checks,
        "support_ready": all(checks.values()),
    }


def write_support_report(path: Path, report: Mapping[str, object]) -> Path:
    """Write a deterministic machine-readable FTR-201 support report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _canonical_generate_stage(*, config_path: Path, device_index: int, repo_root: Path) -> Path:
    from tiny_qwen_coder.evaluation._baseline_stages import (
        generate_canonical_python_base_baseline_stage,
    )

    return generate_canonical_python_base_baseline_stage(
        config_path=config_path,
        device_index=device_index,
        repo_root=repo_root,
    )


def _canonical_score_stage(*, config_path: Path, repo_root: Path) -> PythonBaselineManifest:
    from tiny_qwen_coder.evaluation._baseline_stages import (
        score_canonical_python_base_baseline_stage,
    )

    return score_canonical_python_base_baseline_stage(
        config_path=config_path,
        repo_root=repo_root,
    )


def generate_teacher_stage(
    *,
    repo_root: Path = Path("."),
    config_path: Path = _TEACHER_EVALUATION,
    device_index: int = 0,
) -> Path:
    """Generate exact teacher responses with the canonical baseline generation stage."""

    return _canonical_generate_stage(
        config_path=config_path,
        device_index=device_index,
        repo_root=repo_root,
    )


def score_teacher_stage(
    *,
    repo_root: Path = Path("."),
    config_path: Path = _TEACHER_EVALUATION,
) -> PythonBaselineManifest:
    """Score teacher responses with the canonical protected-benchmark execution stage."""

    return _canonical_score_stage(config_path=config_path, repo_root=repo_root)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FTR-201 direct Qwen3.8-27B evaluation support")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="verify teacher/base evaluation comparability")
    audit.add_argument("--repo-root", type=Path, default=Path("."))
    audit.add_argument("--source-git-sha", required=True)
    audit.add_argument("--output", type=Path, default=None)

    generate = subparsers.add_parser("generate", help="generate direct teacher benchmark responses")
    generate.add_argument("--repo-root", type=Path, default=Path("."))
    generate.add_argument("--config", type=Path, default=_TEACHER_EVALUATION)
    generate.add_argument("--device-index", type=int, default=0)

    score = subparsers.add_parser("score", help="score transported direct teacher responses")
    score.add_argument("--repo-root", type=Path, default=Path("."))
    score.add_argument("--config", type=Path, default=_TEACHER_EVALUATION)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    command = cast(str, args.command)
    repo_root = cast(Path, args.repo_root)
    if command == "audit":
        report = audit_teacher_direct_support(
            repo_root=repo_root,
            source_git_sha=cast(str, args.source_git_sha),
        )
        output = cast(Path | None, args.output)
        if output is not None:
            write_support_report(output, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if command == "generate":
        path = generate_teacher_stage(
            repo_root=repo_root,
            config_path=cast(Path, args.config),
            device_index=cast(int, args.device_index),
        )
        print(path)
        return
    from tiny_qwen_coder.evaluation._baseline_artifacts import python_baseline_manifest_json

    manifest = score_teacher_stage(
        repo_root=repo_root,
        config_path=cast(Path, args.config),
    )
    print(python_baseline_manifest_json(manifest), end="")


if __name__ == "__main__":
    main()
