"""CPU-only contract tests for FTR-201 direct teacher evaluation support."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from tiny_qwen_coder.evaluation.python_ftr_teacher_direct import (
    FTRTeacherDirectError,
    audit_teacher_direct_support,
    generate_teacher_stage,
    score_teacher_stage,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SHA = "e01df13742aa5cdfd27c8083e4dcd1261078f485"


def test_ftr_201_contract_reuses_canonical_evaluation_semantics() -> None:
    report = audit_teacher_direct_support(repo_root=_REPO_ROOT, source_git_sha=_SHA)

    assert report["support_ready"] is True
    teacher = cast(dict[str, object], report["teacher_model"])
    base = cast(dict[str, object], report["base_model"])
    semantics = cast(dict[str, object], report["evaluation_semantics"])
    checks = cast(dict[str, object], report["checks"])

    assert teacher["repository"] == "Qwen/Qwen3.8-27B"
    assert teacher["revision"] == "72a217afab8029b39e4af1c7273a829995a3dbaf"
    assert teacher["tokenizer_repository"] == teacher["repository"]
    assert teacher["tokenizer_revision"] == teacher["revision"]
    assert base["repository"] == "Qwen/Qwen3.5-4B"
    assert semantics["suites"] == [
        "humaneval",
        "mbpp",
        "repository-holdout",
        "general-tool-regression",
    ]
    assert semantics["seed"] == 1729
    assert cast(dict[str, object], semantics["generation"])["max_new_tokens"] == 512
    assert cast(dict[str, object], semantics["generation"])["temperature"] == 0.0
    assert report["training_input_references"] == []
    assert all(value is True for value in checks.values())


def test_ftr_201_rejects_semantic_drift(tmp_path: Path) -> None:
    canonical = (_REPO_ROOT / "configs/eval/python/ftr_201_teacher_direct_v1.yaml").read_text(
        encoding="utf-8"
    )
    drifted = canonical.replace("max_new_tokens: 512", "max_new_tokens: 1024")
    config = tmp_path / "drifted.yaml"
    config.write_text(drifted, encoding="utf-8")

    with pytest.raises(FTRTeacherDirectError, match="semantics drifted"):
        audit_teacher_direct_support(
            repo_root=_REPO_ROOT,
            source_git_sha=_SHA,
            teacher_evaluation_path=config,
        )


def test_ftr_201_generation_delegates_to_canonical_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tiny_qwen_coder.evaluation.python_ftr_teacher_direct as ftr

    observed: dict[str, object] = {}

    def fake_generate(*, config_path: Path, device_index: int, repo_root: Path) -> Path:
        observed.update(
            config_path=config_path,
            device_index=device_index,
            repo_root=repo_root,
        )
        return Path("artifact/generation-stage.json")

    monkeypatch.setattr(ftr, "_canonical_generate_stage", fake_generate)

    result = generate_teacher_stage(repo_root=Path("repo"), device_index=2)

    assert result == Path("artifact/generation-stage.json")
    assert observed == {
        "config_path": Path("configs/eval/python/ftr_201_teacher_direct_v1.yaml"),
        "device_index": 2,
        "repo_root": Path("repo"),
    }


def test_ftr_201_scoring_delegates_to_canonical_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tiny_qwen_coder.evaluation.python_ftr_teacher_direct as ftr

    sentinel = object()
    observed: dict[str, object] = {}

    def fake_score(*, config_path: Path, repo_root: Path) -> object:
        observed.update(config_path=config_path, repo_root=repo_root)
        return sentinel

    monkeypatch.setattr(ftr, "_canonical_score_stage", fake_score)

    result = score_teacher_stage(repo_root=Path("repo"))

    assert result is sentinel
    assert observed == {
        "config_path": Path("configs/eval/python/ftr_201_teacher_direct_v1.yaml"),
        "repo_root": Path("repo"),
    }
