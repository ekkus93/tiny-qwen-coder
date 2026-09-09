"""CPU tests for FTR-101 unchanged-base reproduction comparison."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tiny_qwen_coder.evaluation.python_ftr_base_reproduction import (
    FTRBaseReproductionError,
    compare_reproduction,
    tokenizer_identity_payload,
)
from tiny_qwen_coder.evaluation.settings import (
    FrozenEvaluationSettings,
    FrozenGenerationSettings,
)


def _settings() -> FrozenEvaluationSettings:
    return FrozenEvaluationSettings(
        schema_version=1,
        settings_id="canonical_evaluation",
        settings_version=1,
        frozen=True,
        seed=1729,
        generation=FrozenGenerationSettings(
            decoding_strategy="greedy",
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            max_new_tokens=512,
            stop_policy="eos_or_max_new_tokens",
            prompt_version="canonical-evaluation-prompt-v1",
            chat_template_version=(
                "Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a:checkpoint"
            ),
        ),
    )


class _FakeTokenizer:
    chat_template: str | None = "template"
    eos_token: str | None = "<eos>"
    eos_token_id: int | list[int] | None = 151645
    pad_token: str | None = "<pad>"
    pad_token_id: int | None = 151643


def test_tokenizer_identity_records_template_and_stop_contract() -> None:
    tokenizer = _FakeTokenizer()

    payload = tokenizer_identity_payload(
        tokenizer,
        settings=_settings(),
        repository="Qwen/Qwen3.5-4B",
        revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    )

    assert payload["chat_template_sha256"] == (
        "5cde0f1298f41f7d1c8b907a36992a7a513225a2615bd6e307bf1a9149b06b40"
    )
    assert payload["seed"] == 1729
    generation = payload["generation"]
    assert isinstance(generation, dict)
    assert generation["decoding_strategy"] == "greedy"
    assert generation["enable_thinking"] is False
    stop = payload["stop_conditions"]
    assert isinstance(stop, dict)
    assert stop["policy"] == "eos_or_max_new_tokens"
    assert stop["tokenizer_eos_token_ids"] == [151645]


def _manifest(source_sha: str) -> dict[str, object]:
    base_model = {
        "repository": "Qwen/Qwen3.5-4B",
        "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "tokenizer_repository": "Qwen/Qwen3.5-4B",
        "tokenizer_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    }
    return {
        "source_git_sha": source_sha,
        "base_model": base_model,
        "evaluation_config_sha256": "eval",
        "evaluation_settings_sha256": "settings",
        "system_prompt_sha256": "system",
        "system_prompt_version": "python-v1",
        "generation_contract_sha256": "generation",
        "artifact_set_sha256": "artifact-set",
    }


def _result(problem_id: str, *, generated_text: str, passed: bool) -> dict[str, object]:
    return {
        "problem_id": problem_id,
        "generated_text": generated_text,
        "generated_code": generated_text,
        "generation": {
            "prompt_tokens": 10,
            "generated_tokens": 5,
            "latency_seconds": 1.0,
            "tokens_per_second": 5.0,
        },
        "parse_status": "passed",
        "compile_status": "passed",
        "error_category": "none",
        "tests": {"passed": int(passed), "total": 1},
    }


def _aggregate(suite_id: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "benchmark_id": suite_id,
        "base_model": _manifest("a" * 40)["base_model"],
        "adapter": {"family": None, "adapter_id": None},
        "dataset_id": f"dataset/{suite_id}",
        "dataset_revision": "revision",
        "evaluation_settings_sha256": "settings",
        "execution_image": "python@sha256:image",
        "prompt_version": f"{suite_id}-prompt-v1",
        "runner_source_sha256": f"runner-{suite_id}",
        "total_problems": 1,
        "passed": 1,
        "failed": 0,
        "pass_at_1": 1.0,
        "harness_errors": 0,
        "timed_out": 0,
        "results_sha256": "dynamic",
    }


def _runtime_identity_files(root: Path) -> None:
    (root / "provenance.json").write_text(
        json.dumps(
            {
                "python_version": "3.11.14",
                "cuda_runtime": "13.0",
                "dependencies": [["torch", "2.13.0"]],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "runtime-metadata.json").write_text(
        json.dumps(
            {
                "torch_version": "2.13.0+cu130",
                "transformers_version": "5.16.1",
                "model_class": "Qwen3_5ForConditionalGeneration",
                "parameter_dtypes": ["torch.bfloat16"],
                "resolved_model_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
                "gpu_name": "NVIDIA GeForce RTX 4070 Ti SUPER",
                "gpu_compute_capability": "8.9",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_fixture(root: Path, *, source_sha: str, suffix: str = "") -> None:
    root.mkdir(parents=True)
    (root / "baseline-manifest.json").write_text(
        json.dumps(_manifest(source_sha)) + "\n", encoding="utf-8"
    )
    _runtime_identity_files(root)
    for suite_id, subdir, problem_id in (
        ("humaneval", "humaneval", "HumanEval/0"),
        ("mbpp", "mbpp", "MBPP/1"),
        ("repository-holdout", "repository-holdout", "repository-holdout/one"),
    ):
        suite_dir = root / subdir
        suite_dir.mkdir()
        row = _result(problem_id, generated_text=f"code{suffix}", passed=True)
        (suite_dir / f"{subdir}-results.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        (suite_dir / f"{subdir}-aggregate.json").write_text(
            json.dumps(_aggregate(suite_id)) + "\n", encoding="utf-8"
        )


def _protocol_config(path: Path, *, frozen_source_sha: str) -> str:
    payload = {
        "schema_version": 1,
        "id": "python-ftr-101-base-reproduction-v1",
        "task_id": "FTR-101",
        "frozen_baseline": {
            "workflow_run_id": 33301242379,
            "source_git_sha": frozen_source_sha,
            "artifact_id": 9729636096,
            "artifact_name": ("python-base-baseline-da537443ab80b1380bee0fc3c7d9d01ca0574f35"),
            "artifact_digest": (
                "sha256:bcc08b94e0204e19d38fe28d0771a597687cbe44ad08af4759233a3c824e2e21"
            ),
            "artifact_set_sha256": "artifact-set",
            "chat_template_sha256": (
                "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"
            ),
            "scores": {
                "humaneval": [1, 1],
                "mbpp": [1, 1],
                "repository-holdout": [1, 1],
                "combined": [3, 3],
            },
        },
        "comparison": {},
    }
    raw = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(raw, encoding="utf-8")
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _identity(path: Path, *, protocol_sha256: str) -> None:
    path.write_text(
        json.dumps(
            {
                "chat_template_sha256": (
                    "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"
                ),
                "ftr_101_protocol_config_sha256": protocol_sha256,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_compare_reproduction_accepts_exact_task_records(tmp_path: Path) -> None:
    frozen = tmp_path / "frozen"
    current = tmp_path / "current"
    _write_fixture(frozen, source_sha="a" * 40)
    _write_fixture(current, source_sha="b" * 40)
    protocol = tmp_path / "protocol.json"
    protocol_sha = _protocol_config(protocol, frozen_source_sha="a" * 40)
    identity = tmp_path / "identity.json"
    _identity(identity, protocol_sha256=protocol_sha)

    payload = compare_reproduction(
        frozen_dir=frozen,
        current_dir=current,
        identity_path=identity,
        output=tmp_path / "report.json",
        expected_current_source_sha="b" * 40,
        protocol_config_path=protocol,
    )

    assert payload["status"] == "exact_reproduction"
    assert payload["exact_task_record_reproduction"] is True
    assert payload["target_language"] == {"frozen": 3, "current": 3, "total": 3}


def test_compare_reproduction_fails_on_generated_text_drift(tmp_path: Path) -> None:
    frozen = tmp_path / "frozen"
    current = tmp_path / "current"
    _write_fixture(frozen, source_sha="a" * 40)
    _write_fixture(current, source_sha="b" * 40, suffix=" changed")
    protocol = tmp_path / "protocol.json"
    protocol_sha = _protocol_config(protocol, frozen_source_sha="a" * 40)
    identity = tmp_path / "identity.json"
    _identity(identity, protocol_sha256=protocol_sha)

    with pytest.raises(
        FTRBaseReproductionError,
        match="task-level output/scoring or harness-contract drift",
    ):
        compare_reproduction(
            frozen_dir=frozen,
            current_dir=current,
            identity_path=identity,
            output=tmp_path / "report.json",
            expected_current_source_sha="b" * 40,
            protocol_config_path=protocol,
        )

    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "drift"
    assert all(suite["mismatches"] for suite in report["suites"])


def test_compare_reproduction_fails_on_source_sha_drift(tmp_path: Path) -> None:
    frozen = tmp_path / "frozen"
    current = tmp_path / "current"
    _write_fixture(frozen, source_sha="a" * 40)
    _write_fixture(current, source_sha="b" * 40)
    protocol = tmp_path / "protocol.json"
    protocol_sha = _protocol_config(protocol, frozen_source_sha="a" * 40)
    identity = tmp_path / "identity.json"
    _identity(identity, protocol_sha256=protocol_sha)

    with pytest.raises(FTRBaseReproductionError, match="current baseline source SHA mismatch"):
        compare_reproduction(
            frozen_dir=frozen,
            current_dir=current,
            identity_path=identity,
            output=tmp_path / "report.json",
            expected_current_source_sha="c" * 40,
            protocol_config_path=protocol,
        )
