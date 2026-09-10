from __future__ import annotations

import json

import pytest

from tiny_qwen_coder.pvrl.tool_contract import (
    BashToolArguments,
    EditToolArguments,
    GlobToolArguments,
    LeafLiteContractError,
    LeafLiteProfile,
    LeafLiteResultStatus,
    LeafLiteToolName,
    ReadToolArguments,
    WriteToolArguments,
    create_leaf_lite_tool_result,
    leaf_lite_contract_json,
    leaf_lite_tool_call_json,
    leaf_lite_tool_definitions,
    leaf_lite_tool_result_from_json,
    leaf_lite_tool_result_json,
    parse_leaf_lite_tool_call,
    repository_leaf_lite_contract,
    self_contained_leaf_lite_contract,
)


def _call(tool: str, arguments: dict[str, object]) -> str:
    return json.dumps({"tool": tool, "arguments": arguments})


def test_repository_profile_freezes_exact_five_tool_vocabulary() -> None:
    contract = repository_leaf_lite_contract()

    assert contract.profile is LeafLiteProfile.REPOSITORY
    assert contract.tool_names == (
        LeafLiteToolName.READ,
        LeafLiteToolName.WRITE,
        LeafLiteToolName.EDIT,
        LeafLiteToolName.GLOB,
        LeafLiteToolName.BASH,
    )
    assert len(contract.contract_sha256) == 64
    serialized = leaf_lite_contract_json(contract)
    payload = json.loads(serialized)
    assert payload["tool_names"] == ["read", "write", "edit", "glob", "bash"]
    assert payload["grader_operations_exposed"] is False
    assert [item["name"] for item in payload["tool_definitions"]] == [
        "read",
        "write",
        "edit",
        "glob",
        "bash",
    ]
    assert payload["bash_semantics"]["persistent_shell_state"] is False
    assert payload["contract_sha256"] == contract.contract_sha256


def test_self_contained_v0_removes_edit_and_glob_without_aliases() -> None:
    contract = self_contained_leaf_lite_contract()

    assert contract.tool_names == (
        LeafLiteToolName.READ,
        LeafLiteToolName.WRITE,
        LeafLiteToolName.BASH,
    )
    for disabled in ("edit", "glob"):
        with pytest.raises(LeafLiteContractError, match="not enabled"):
            parse_leaf_lite_tool_call(_call(disabled, {"path": "solution.py"}), contract)
    for alias in ("shell", "run", "cat", "patch", "grade", "run_hidden"):
        with pytest.raises(LeafLiteContractError, match="unknown policy tool"):
            parse_leaf_lite_tool_call(_call(alias, {}), contract)


def test_read_call_schema_defaults_and_canonical_serialization() -> None:
    contract = repository_leaf_lite_contract()

    call = parse_leaf_lite_tool_call(_call("read", {"path": "src/main.py"}), contract)

    assert call.tool is LeafLiteToolName.READ
    assert call.arguments == ReadToolArguments(path="src/main.py", start_line=1, max_lines=200)
    assert leaf_lite_tool_call_json(call) == (
        '{"arguments":{"max_lines":200,"path":"src/main.py","start_line":1},"tool":"read"}\n'
    )


def test_write_call_schema_is_atomic_full_file_content_and_bounded() -> None:
    contract = repository_leaf_lite_contract()
    call = parse_leaf_lite_tool_call(
        _call("write", {"path": "solution.py", "content": "print('ok')\n"}),
        contract,
    )

    assert call.arguments == WriteToolArguments(path="solution.py", content="print('ok')\n")
    oversized = "x" * (contract.max_write_bytes + 1)
    with pytest.raises(LeafLiteContractError, match="byte limit"):
        parse_leaf_lite_tool_call(
            _call("write", {"path": "solution.py", "content": oversized}),
            contract,
        )


def test_edit_call_schema_is_exact_count_checked_replacement() -> None:
    contract = repository_leaf_lite_contract()
    call = parse_leaf_lite_tool_call(
        _call(
            "edit",
            {
                "path": "src/main.py",
                "old_text": "return None",
                "new_text": "return value",
                "expected_replacements": 2,
            },
        ),
        contract,
    )

    assert call.arguments == EditToolArguments(
        path="src/main.py",
        old_text="return None",
        new_text="return value",
        expected_replacements=2,
    )
    with pytest.raises(LeafLiteContractError, match="must not be empty"):
        parse_leaf_lite_tool_call(
            _call("edit", {"path": "x.py", "old_text": "", "new_text": "x"}),
            contract,
        )


def test_glob_call_schema_has_one_deterministic_pattern_surface() -> None:
    contract = repository_leaf_lite_contract()
    call = parse_leaf_lite_tool_call(_call("glob", {"pattern": "src/**/*.py"}), contract)

    assert call.arguments == GlobToolArguments(pattern="src/**/*.py")
    with pytest.raises(LeafLiteContractError, match="schema"):
        parse_leaf_lite_tool_call(
            _call("glob", {"pattern": "**/*.py", "cwd": "src"}),
            contract,
        )


def test_bash_call_schema_uses_fixed_contract_timeout_and_no_per_call_override() -> None:
    contract = repository_leaf_lite_contract()
    call = parse_leaf_lite_tool_call(_call("bash", {"command": "python -m pytest -q"}), contract)

    assert call.arguments == BashToolArguments(command="python -m pytest -q")
    assert contract.bash_timeout_seconds == 20
    with pytest.raises(LeafLiteContractError, match="schema"):
        parse_leaf_lite_tool_call(
            _call("bash", {"command": "pytest", "timeout_seconds": 600}),
            contract,
        )


@pytest.mark.parametrize(
    "path",
    [
        "/grader/hidden_tests.py",
        "../grader/hidden_tests.py",
        "grader/hidden_tests.py",
        "reference/solution.py",
        ".pvrl/limit_runner.py",
        "src//main.py",
        "src\\main.py",
    ],
)
def test_file_tools_cannot_address_hidden_or_reserved_namespaces(path: str) -> None:
    contract = repository_leaf_lite_contract()

    with pytest.raises(LeafLiteContractError):
        parse_leaf_lite_tool_call(_call("read", {"path": path}), contract)


def test_glob_cannot_explicitly_target_hidden_or_reserved_namespaces() -> None:
    contract = repository_leaf_lite_contract()

    for pattern in ("grader/**", "reference/**/*.py", ".pvrl/*", "../grader/*"):
        with pytest.raises(LeafLiteContractError):
            parse_leaf_lite_tool_call(_call("glob", {"pattern": pattern}), contract)


def test_tool_call_parser_is_strict_about_top_level_and_argument_keys() -> None:
    contract = repository_leaf_lite_contract()

    with pytest.raises(LeafLiteContractError, match="unexpected"):
        parse_leaf_lite_tool_call(
            json.dumps({"tool": "read", "arguments": {"path": "x.py"}, "id": "surprise"}),
            contract,
        )
    with pytest.raises(LeafLiteContractError, match="unexpected"):
        parse_leaf_lite_tool_call(
            _call("write", {"path": "x.py", "content": "x", "append": True}),
            contract,
        )


def test_policy_tool_definitions_are_profile_exact_and_schema_closed() -> None:
    repository = repository_leaf_lite_contract()
    compact = self_contained_leaf_lite_contract()

    repository_definitions = leaf_lite_tool_definitions(repository)
    compact_definitions = leaf_lite_tool_definitions(compact)

    assert [item["name"] for item in repository_definitions] == [
        "read",
        "write",
        "edit",
        "glob",
        "bash",
    ]
    assert [item["name"] for item in compact_definitions] == ["read", "write", "bash"]
    for definition in repository_definitions:
        parameters = definition["parameters"]
        assert isinstance(parameters, dict)
        assert parameters["additionalProperties"] is False
    bash_definition = repository_definitions[-1]
    assert "20s" in str(bash_definition["description"])


def test_result_serialization_round_trips_for_each_tool() -> None:
    contract = repository_leaf_lite_contract()
    results = (
        create_leaf_lite_tool_result(
            contract,
            tool=LeafLiteToolName.READ,
            status=LeafLiteResultStatus.OK,
            output="1: value\n",
        ),
        create_leaf_lite_tool_result(
            contract,
            tool=LeafLiteToolName.WRITE,
            status=LeafLiteResultStatus.OK,
            output="written\n",
        ),
        create_leaf_lite_tool_result(
            contract,
            tool=LeafLiteToolName.EDIT,
            status=LeafLiteResultStatus.ERROR,
            output="replacement count mismatch\n",
        ),
        create_leaf_lite_tool_result(
            contract,
            tool=LeafLiteToolName.GLOB,
            status=LeafLiteResultStatus.OK,
            output="src/a.py\nsrc/b.py\n",
        ),
        create_leaf_lite_tool_result(
            contract,
            tool=LeafLiteToolName.BASH,
            status=LeafLiteResultStatus.ERROR,
            output="failed\n",
            exit_code=2,
        ),
    )

    for result in results:
        serialized = leaf_lite_tool_result_json(result)
        assert leaf_lite_tool_result_from_json(serialized, contract) == result
        assert serialized.endswith("\n")


def test_output_truncation_is_deterministic_utf8_prefix_and_marker_bounded() -> None:
    contract = repository_leaf_lite_contract()
    raw = "λ" * contract.max_output_bytes

    first = create_leaf_lite_tool_result(
        contract,
        tool=LeafLiteToolName.READ,
        status=LeafLiteResultStatus.OK,
        output=raw,
    )
    second = create_leaf_lite_tool_result(
        contract,
        tool=LeafLiteToolName.READ,
        status=LeafLiteResultStatus.OK,
        output=raw,
    )

    assert first == second
    assert first.truncated is True
    assert first.output.endswith(contract.truncation_marker)
    assert len(first.output.encode("utf-8")) <= contract.max_output_bytes
    assert leaf_lite_tool_result_from_json(leaf_lite_tool_result_json(first), contract) == first


def test_result_status_semantics_separate_bash_execution_from_structured_tools() -> None:
    contract = repository_leaf_lite_contract()

    with pytest.raises(LeafLiteContractError, match="exit_code 0"):
        create_leaf_lite_tool_result(
            contract,
            tool=LeafLiteToolName.BASH,
            status=LeafLiteResultStatus.OK,
            output="",
            exit_code=1,
        )
    with pytest.raises(LeafLiteContractError, match="non-zero"):
        create_leaf_lite_tool_result(
            contract,
            tool=LeafLiteToolName.BASH,
            status=LeafLiteResultStatus.ERROR,
            output="",
            exit_code=0,
        )
    with pytest.raises(LeafLiteContractError, match="bash-only"):
        create_leaf_lite_tool_result(
            contract,
            tool=LeafLiteToolName.READ,
            status=LeafLiteResultStatus.TIMEOUT,
            output="",
        )


def test_result_parser_rejects_forged_truncation_and_oversized_output() -> None:
    contract = repository_leaf_lite_contract()

    forged = json.dumps(
        {
            "tool": "read",
            "status": "ok",
            "output": "partial",
            "exit_code": None,
            "truncated": True,
        }
    )
    with pytest.raises(LeafLiteContractError, match="frozen marker"):
        leaf_lite_tool_result_from_json(forged, contract)

    oversized = json.dumps(
        {
            "tool": "read",
            "status": "ok",
            "output": "x" * (contract.max_output_bytes + 1),
            "exit_code": None,
            "truncated": False,
        }
    )
    with pytest.raises(LeafLiteContractError, match="byte limit"):
        leaf_lite_tool_result_from_json(oversized, contract)
