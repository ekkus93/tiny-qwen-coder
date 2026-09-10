"""CPU-only regression tests for the FTR-103 parity audit."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

from tiny_qwen_coder.evaluation.python_ftr_generation_evaluation_parity import (
    GeneratorPath,
    _generator_report,
    _json_diff,
    audit_generation_evaluation_parity,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE_SHA = "c10d3fb1f90c2fee93f206e7232ef1329e4f7e57"


def test_json_diff_is_field_specific_and_machine_readable() -> None:
    differences = _json_diff(
        {"generation": {"max_new_tokens": 512, "do_sample": False}},
        {"generation": {"max_new_tokens": 256, "do_sample": False}},
    )

    assert differences == [
        {
            "field": "generation.max_new_tokens",
            "expected": 512,
            "observed": 256,
        }
    ]


def test_full_repository_parity_audit_passes() -> None:
    report = audit_generation_evaluation_parity(
        repo_root=_REPO_ROOT,
        source_git_sha=_FIXTURE_SHA,
    )

    assert report["parity_passed"] is True
    assert report["audit_mode"] == "static_ast_observational_no_production_inference_changes"
    checks = cast(Mapping[str, object], report["checks"])
    assert all(value is True for value in checks.values())

    generation_paths = cast(list[object], report["generation_paths"])
    assert len(generation_paths) == 5
    for raw_path in generation_paths:
        path = cast(Mapping[str, object], raw_path)
        assert path["parity"] is True
        assert path["base_model_load_contract_exact"] is True
        assert path["tokenizer_repository_and_revision_exact"] is True
        assert path["greedy_decoding_guard_present"] is True
        assert path["contract_differences_from_unchanged_base"] == []

    differences = cast(list[object], report["intentional_differences"])
    assert len(differences) == 1
    difference = cast(Mapping[str, object], differences[0])
    assert difference["field"] == "execution.isolation_backend"


def test_generator_audit_detects_generation_parameter_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tiny_qwen_coder.evaluation.python_ftr_generation_evaluation_parity as ftr

    source = (_REPO_ROOT / "src/tiny_qwen_coder/evaluation/_baseline_generation.py").read_text(
        encoding="utf-8"
    )
    source = source.replace('"num_beams": 1', '"num_beams": 2')
    target = tmp_path / "generator.py"
    target.write_text(source, encoding="utf-8")
    monkeypatch.setattr(
        ftr,
        "_GENERATORS",
        (
            GeneratorPath(
                path_id="unchanged-base",
                source="generator.py",
                class_name="HuggingFaceBaselineGenerator",
            ),
        ),
    )

    rows, passed = _generator_report(tmp_path)

    assert passed is False
    row = cast(Mapping[str, object], rows[0])
    differences = cast(list[object], row["contract_differences_from_frozen_expected"])
    assert differences == [
        {
            "field": "generation_kwargs.num_beams",
            "expected": 1,
            "observed": 2,
        }
    ]


def test_generator_audit_detects_chat_template_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tiny_qwen_coder.evaluation.python_ftr_generation_evaluation_parity as ftr

    source = (_REPO_ROOT / "src/tiny_qwen_coder/evaluation/_baseline_generation.py").read_text(
        encoding="utf-8"
    )
    source = source.replace("add_generation_prompt=True", "add_generation_prompt=False")
    target = tmp_path / "generator.py"
    target.write_text(source, encoding="utf-8")
    monkeypatch.setattr(
        ftr,
        "_GENERATORS",
        (
            GeneratorPath(
                path_id="unchanged-base",
                source="generator.py",
                class_name="HuggingFaceBaselineGenerator",
            ),
        ),
    )

    rows, passed = _generator_report(tmp_path)

    assert passed is False
    row = cast(Mapping[str, object], rows[0])
    differences = cast(list[object], row["contract_differences_from_frozen_expected"])
    assert differences == [
        {
            "field": "chat_template.kwargs.add_generation_prompt",
            "expected": True,
            "observed": False,
        }
    ]
