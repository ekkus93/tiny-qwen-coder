from __future__ import annotations

import json
from pathlib import Path

import pytest

from tiny_qwen_coder.config import ExecutionConfig
from tiny_qwen_coder.data.records import (
    LicenseMetadata,
    NormalizedTrainingRecord,
    SourceProvenance,
    TrainingMessage,
)
from tiny_qwen_coder.distillation.semantic_contracts import (
    MIN_ASSERTIONS,
    SemanticContract,
    SemanticContractError,
    extract_candidate_code,
    parse_semantic_contract,
    prepare_semantic_contract_record,
    validate_semantic_contract_static,
    verify_candidate_against_contract,
)
from tiny_qwen_coder.evaluation.execution import DirectExecutionHarness


def _record(
    answer: str = "```python\ndef add_one(x):\n    return x + 1\n```",
) -> NormalizedTrainingRecord:
    return NormalizedTrainingRecord(
        schema_version=1,
        messages=(
            TrainingMessage(role="system", content="Write Python."),
            TrainingMessage(role="user", content="Implement add_one(x), returning x plus one."),
            TrainingMessage(role="assistant", content=answer),
        ),
        language="python",
        provenance=SourceProvenance(
            source_id="example/source",
            revision="abc123",
            license=LicenseMetadata(name="MIT"),
            record_id="42",
        ),
    )


def _tests() -> str:
    return "\n".join(f"assert add_one({value}) == {value + 1}" for value in range(-4, 4))


def _contract(*, verifiable: bool = True) -> str:
    payload: dict[str, object] = {
        "schema_version": 1,
        "verifiable": verifiable,
        "entry_point": "add_one" if verifiable else None,
        "reference_solution": "def add_one(x):\n    return x + 1" if verifiable else None,
        "tests": _tests() if verifiable else None,
        "reason": "standalone deterministic function" if verifiable else "requires external state",
    }
    return json.dumps(payload)


def test_prepare_contract_input_hides_candidate_answer() -> None:
    source = _record("SECRET-CANDIDATE-ANSWER")
    prepared = prepare_semantic_contract_record(source)

    assert all("SECRET-CANDIDATE-ANSWER" not in message.content for message in prepared.messages)
    assert prepared.messages[-1].content == "semantic-contract-placeholder"
    metadata = dict(prepared.provenance.source_metadata)
    assert len(metadata["semantic_contract.candidate_response_sha256"]) == 64
    assert metadata["semantic_contract.policy_id"] == "python-semantic-contract-v1"


def test_parse_contract_accepts_historical_closing_only_reasoning_transport() -> None:
    parsed = parse_semantic_contract("hidden reasoning\n</think>\n" + _contract())
    assert parsed.verifiable is True
    assert parsed.entry_point == "add_one"


def test_static_contract_requires_minimum_assertions() -> None:
    contract = SemanticContract(
        schema_version=1,
        verifiable=True,
        entry_point="add_one",
        reference_solution="def add_one(x):\n    return x + 1",
        tests="assert add_one(1) == 2",
        reason="test",
    )
    passed, reason, count = validate_semantic_contract_static(contract)
    assert passed is False
    assert reason == "too_few_entry_point_assertions:1"
    assert count == 1
    assert MIN_ASSERTIONS == 8


def test_static_contract_rejects_file_access() -> None:
    contract = SemanticContract(
        schema_version=1,
        verifiable=True,
        entry_point="add_one",
        reference_solution="def add_one(x):\n    open('x')\n    return x + 1",
        tests=_tests(),
        reason="test",
    )
    passed, reason, _ = validate_semantic_contract_static(contract)
    assert passed is False
    assert reason == "forbidden_call:open"


def test_candidate_extraction_selects_only_entry_point_block() -> None:
    candidate = """Explanation.
```python
def helper(x):
    return x
```
More.
```python
def add_one(x):
    return x + 1
```
"""
    code = extract_candidate_code(candidate, entry_point="add_one")
    assert "def add_one" in code
    assert "def helper" not in code


def test_correct_candidate_passes_independent_contract(tmp_path: Path) -> None:
    result = verify_candidate_against_contract(
        _record(),
        _contract(),
        index=0,
        harness=DirectExecutionHarness(temp_root=tmp_path, allow_reduced_isolation=True),
        execution=ExecutionConfig(timeout_seconds=5.0, network_enabled=False),
    )
    assert result.reference_passed is True
    assert result.candidate_passed is True
    assert result.accepted is True
    assert result.assertion_count == 8


def test_wrong_candidate_fails_independent_contract(tmp_path: Path) -> None:
    result = verify_candidate_against_contract(
        _record("```python\ndef add_one(x):\n    return x - 1\n```"),
        _contract(),
        index=0,
        harness=DirectExecutionHarness(temp_root=tmp_path, allow_reduced_isolation=True),
        execution=ExecutionConfig(timeout_seconds=5.0, network_enabled=False),
    )
    assert result.reference_passed is True
    assert result.candidate_passed is False
    assert result.accepted is False
    assert result.reason == "candidate_test_failed"


def test_unverifiable_contract_rejects_without_execution(tmp_path: Path) -> None:
    result = verify_candidate_against_contract(
        _record(),
        _contract(verifiable=False),
        index=0,
        harness=DirectExecutionHarness(temp_root=tmp_path, allow_reduced_isolation=True),
        execution=ExecutionConfig(timeout_seconds=5.0, network_enabled=False),
    )
    assert result.accepted is False
    assert result.reason == "unverifiable"
    assert result.reference_passed is False


def test_candidate_with_duplicate_entrypoint_blocks_fails_closed() -> None:
    with pytest.raises(SemanticContractError, match="exactly one"):
        extract_candidate_code(
            "```python\ndef add_one(x): return x + 1\n```\n"
            "```python\ndef add_one(x): return x + 1\n```",
            entry_point="add_one",
        )
