"""Frozen Leaf-lite tool-call/result contract for PVRL policy trajectories."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import PurePosixPath
from typing import TypeAlias

_SCHEMA_VERSION = 1
_MARKER = "\n...[output truncated]\n"
_RESERVED = (".pvrl", "grader", "reference")
_REPOSITORY = ("read", "write", "edit", "glob", "bash")
_SELF_CONTAINED = ("read", "write", "bash")


class LeafLiteContractError(ValueError):
    """Invalid Leaf-lite contract material, policy call, or tool result."""


class LeafLiteProfile(StrEnum):
    SELF_CONTAINED_V0 = "self_contained_v0"
    REPOSITORY = "repository"


class LeafLiteToolName(StrEnum):
    READ = "read"
    WRITE = "write"
    EDIT = "edit"
    GLOB = "glob"
    BASH = "bash"


class LeafLiteResultStatus(StrEnum):
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    RESOURCE_LIMIT = "resource_limit"


@dataclass(frozen=True, slots=True)
class LeafLiteToolContract:
    schema_version: int
    contract_id: str
    profile: LeafLiteProfile
    tool_names: tuple[LeafLiteToolName, ...]
    bash_timeout_seconds: int = 20
    max_output_bytes: int = 65_536
    read_default_max_lines: int = 200
    read_max_lines: int = 2_000
    glob_max_results: int = 1_000
    max_write_bytes: int = 1_048_576
    truncation_marker: str = _MARKER

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise LeafLiteContractError("unsupported Leaf-lite schema_version")
        if not self.contract_id or self.contract_id != self.contract_id.strip():
            raise LeafLiteContractError(
                "contract_id must be a non-empty trimmed string"
            )
        if not isinstance(self.profile, LeafLiteProfile):
            raise LeafLiteContractError("profile must be a LeafLiteProfile")
        if any(not isinstance(item, LeafLiteToolName) for item in self.tool_names):
            raise LeafLiteContractError(
                "tool_names must contain LeafLiteToolName values"
            )
        names = (
            _REPOSITORY
            if self.profile is LeafLiteProfile.REPOSITORY
            else _SELF_CONTAINED
        )
        expected = tuple(LeafLiteToolName(item) for item in names)
        if self.tool_names != expected:
            raise LeafLiteContractError(
                f"{self.profile.value} tools must exactly equal {names!r}"
            )
        limits = (
            self.bash_timeout_seconds,
            self.max_output_bytes,
            self.read_default_max_lines,
            self.read_max_lines,
            self.glob_max_results,
            self.max_write_bytes,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in limits
        ):
            raise LeafLiteContractError(
                "Leaf-lite numeric limits must be positive integers"
            )
        if self.read_default_max_lines > self.read_max_lines:
            raise LeafLiteContractError("read default cannot exceed read maximum")
        if (
            not self.truncation_marker
            or len(self.truncation_marker.encode()) >= self.max_output_bytes
        ):
            raise LeafLiteContractError(
                "truncation marker must fit inside the output bound"
            )

    @property
    def contract_sha256(self) -> str:
        return hashlib.sha256(_canonical(_contract_payload(self))).hexdigest()


@dataclass(frozen=True, slots=True)
class ReadToolArguments:
    path: str
    start_line: int = 1
    max_lines: int = 200


@dataclass(frozen=True, slots=True)
class WriteToolArguments:
    path: str
    content: str


@dataclass(frozen=True, slots=True)
class EditToolArguments:
    path: str
    old_text: str
    new_text: str
    expected_replacements: int = 1


@dataclass(frozen=True, slots=True)
class GlobToolArguments:
    pattern: str


@dataclass(frozen=True, slots=True)
class BashToolArguments:
    command: str


LeafLiteArguments: TypeAlias = (
    ReadToolArguments
    | WriteToolArguments
    | EditToolArguments
    | GlobToolArguments
    | BashToolArguments
)


@dataclass(frozen=True, slots=True)
class LeafLiteToolCall:
    tool: LeafLiteToolName
    arguments: LeafLiteArguments


@dataclass(frozen=True, slots=True)
class LeafLiteToolResult:
    tool: LeafLiteToolName
    status: LeafLiteResultStatus
    output: str
    exit_code: int | None
    truncated: bool

    def __post_init__(self) -> None:
        if not isinstance(self.tool, LeafLiteToolName) or not isinstance(
            self.status, LeafLiteResultStatus
        ):
            raise LeafLiteContractError("result tool/status must use Leaf-lite enums")
        if self.tool is not LeafLiteToolName.BASH:
            if self.status in {
                LeafLiteResultStatus.TIMEOUT,
                LeafLiteResultStatus.RESOURCE_LIMIT,
            }:
                raise LeafLiteContractError(
                    "timeout/resource_limit statuses are bash-only"
                )
            if self.exit_code is not None:
                raise LeafLiteContractError("non-bash results cannot carry exit_code")
            return
        if self.status is LeafLiteResultStatus.OK and self.exit_code != 0:
            raise LeafLiteContractError("successful bash result must have exit_code 0")
        if self.status is LeafLiteResultStatus.ERROR and (
            self.exit_code is None or self.exit_code == 0
        ):
            raise LeafLiteContractError("bash error result requires non-zero exit_code")
        if (
            self.status
            in {LeafLiteResultStatus.TIMEOUT, LeafLiteResultStatus.RESOURCE_LIMIT}
            and self.exit_code == 0
        ):
            raise LeafLiteContractError(
                "timed out/resource-limited bash cannot exit zero"
            )


def repository_leaf_lite_contract() -> LeafLiteToolContract:
    return _contract("leaf-lite-v1-repository", LeafLiteProfile.REPOSITORY, _REPOSITORY)


def self_contained_leaf_lite_contract() -> LeafLiteToolContract:
    return _contract(
        "leaf-lite-v1-self-contained-v0",
        LeafLiteProfile.SELF_CONTAINED_V0,
        _SELF_CONTAINED,
    )


def _contract(
    contract_id: str, profile: LeafLiteProfile, names: tuple[str, ...]
) -> LeafLiteToolContract:
    return LeafLiteToolContract(
        schema_version=_SCHEMA_VERSION,
        contract_id=contract_id,
        profile=profile,
        tool_names=tuple(LeafLiteToolName(item) for item in names),
    )


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")


def _object(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise LeafLiteContractError(f"{context} must be a JSON object")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise LeafLiteContractError(f"{context} keys must be strings")
        result[key] = item
    return result


def _keys(value: Mapping[str, object], expected: set[str], context: str) -> None:
    actual = set(value)
    if actual != expected:
        raise LeafLiteContractError(
            f"{context} keys do not match schema; missing={sorted(expected - actual)!r}, "
            f"unexpected={sorted(actual - expected)!r}"
        )


def _string(value: Mapping[str, object], key: str, context: str) -> str:
    item = value.get(key)
    if not isinstance(item, str):
        raise LeafLiteContractError(f"{context}.{key} must be a string")
    return item


def _integer(value: Mapping[str, object], key: str, context: str) -> int:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, int):
        raise LeafLiteContractError(f"{context}.{key} must be an integer")
    return item


def _path(value: str, *, glob: bool = False) -> str:
    label = "glob pattern" if glob else "tool path"
    if not value or not value.strip() or "\x00" in value or "\\" in value:
        raise LeafLiteContractError(f"{label} must be normalized relative POSIX text")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or value != parsed.as_posix() or ".." in parsed.parts:
        raise LeafLiteContractError(
            f"{label} cannot be absolute, non-normalized, or traverse"
        )
    if parsed.parts[0] in _RESERVED:
        raise LeafLiteContractError(
            f"{label} targets a grader/reference/reserved namespace"
        )
    return value


def parse_leaf_lite_tool_call(
    text: str, contract: LeafLiteToolContract
) -> LeafLiteToolCall:
    """Strictly parse one model-emitted JSON tool call under a frozen profile."""

    try:
        raw: object = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LeafLiteContractError("tool call is not valid JSON") from exc
    payload = _object(raw, "tool call")
    _keys(payload, {"tool", "arguments"}, "tool call")
    raw_tool = _string(payload, "tool", "tool call")
    try:
        tool = LeafLiteToolName(raw_tool)
    except ValueError as exc:
        raise LeafLiteContractError(f"unknown policy tool {raw_tool!r}") from exc
    if tool not in contract.tool_names:
        raise LeafLiteContractError(
            f"tool {tool.value!r} is not enabled by this profile"
        )
    args = _object(payload["arguments"], f"{tool.value} arguments")
    parsed: LeafLiteArguments
    if tool is LeafLiteToolName.READ:
        allowed = {"path", "start_line", "max_lines"}
        if "path" not in args or set(args) - allowed:
            raise LeafLiteContractError("read arguments do not match schema")
        start = 1 if "start_line" not in args else _integer(args, "start_line", "read")
        maximum = (
            contract.read_default_max_lines
            if "max_lines" not in args
            else _integer(args, "max_lines", "read")
        )
        if start < 1 or not 1 <= maximum <= contract.read_max_lines:
            raise LeafLiteContractError("read range exceeds contract bounds")
        parsed = ReadToolArguments(_path(_string(args, "path", "read")), start, maximum)
    elif tool is LeafLiteToolName.WRITE:
        _keys(args, {"path", "content"}, "write arguments")
        content = _string(args, "content", "write")
        if len(content.encode()) > contract.max_write_bytes:
            raise LeafLiteContractError("write.content exceeds contract byte limit")
        parsed = WriteToolArguments(_path(_string(args, "path", "write")), content)
    elif tool is LeafLiteToolName.EDIT:
        allowed = {"path", "old_text", "new_text", "expected_replacements"}
        if not {"path", "old_text", "new_text"} <= set(args) or set(args) - allowed:
            raise LeafLiteContractError("edit arguments do not match schema")
        old = _string(args, "old_text", "edit")
        count = (
            1
            if "expected_replacements" not in args
            else _integer(args, "expected_replacements", "edit")
        )
        if not old:
            raise LeafLiteContractError("edit.old_text must not be empty")
        if not 1 <= count <= 100:
            raise LeafLiteContractError(
                "edit.expected_replacements must be between 1 and 100"
            )
        parsed = EditToolArguments(
            _path(_string(args, "path", "edit")),
            old,
            _string(args, "new_text", "edit"),
            count,
        )
    elif tool is LeafLiteToolName.GLOB:
        _keys(args, {"pattern"}, "glob arguments")
        parsed = GlobToolArguments(_path(_string(args, "pattern", "glob"), glob=True))
    else:
        _keys(args, {"command"}, "bash arguments")
        command = _string(args, "command", "bash")
        if not command or not command.strip() or "\x00" in command:
            raise LeafLiteContractError(
                "bash.command must be non-empty and contain no NUL"
            )
        parsed = BashToolArguments(command)
    return LeafLiteToolCall(tool, parsed)


def leaf_lite_tool_call_json(call: LeafLiteToolCall) -> str:
    return (
        _canonical(
            {"tool": call.tool.value, "arguments": asdict(call.arguments)}
        ).decode()
        + "\n"
    )


def create_leaf_lite_tool_result(
    contract: LeafLiteToolContract,
    *,
    tool: LeafLiteToolName,
    status: LeafLiteResultStatus,
    output: str,
    exit_code: int | None = None,
) -> LeafLiteToolResult:
    """Create a result using deterministic UTF-8 prefix truncation."""

    if tool not in contract.tool_names:
        raise LeafLiteContractError("result tool is not enabled by this profile")
    encoded = output.encode()
    truncated = len(encoded) > contract.max_output_bytes
    if truncated:
        budget = contract.max_output_bytes - len(contract.truncation_marker.encode())
        output = encoded[:budget].decode(errors="ignore") + contract.truncation_marker
    return LeafLiteToolResult(tool, status, output, exit_code, truncated)


def leaf_lite_tool_result_json(result: LeafLiteToolResult) -> str:
    payload = asdict(result)
    payload["tool"] = result.tool.value
    payload["status"] = result.status.value
    return _canonical(payload).decode() + "\n"


def leaf_lite_tool_result_from_json(
    text: str, contract: LeafLiteToolContract
) -> LeafLiteToolResult:
    try:
        raw: object = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LeafLiteContractError("tool result is not valid JSON") from exc
    payload = _object(raw, "tool result")
    _keys(
        payload, {"tool", "status", "output", "exit_code", "truncated"}, "tool result"
    )
    try:
        tool = LeafLiteToolName(_string(payload, "tool", "tool result"))
        status = LeafLiteResultStatus(_string(payload, "status", "tool result"))
    except ValueError as exc:
        raise LeafLiteContractError("unknown tool-result tool/status") from exc
    if tool not in contract.tool_names:
        raise LeafLiteContractError("tool result is not enabled by this profile")
    output = _string(payload, "output", "tool result")
    if len(output.encode()) > contract.max_output_bytes:
        raise LeafLiteContractError("tool result output exceeds contract byte limit")
    exit_code = payload["exit_code"]
    if exit_code is not None and (
        isinstance(exit_code, bool) or not isinstance(exit_code, int)
    ):
        raise LeafLiteContractError("tool result exit_code must be integer or null")
    truncated = payload["truncated"]
    if not isinstance(truncated, bool):
        raise LeafLiteContractError("tool result truncated must be boolean")
    if truncated and not output.endswith(contract.truncation_marker):
        raise LeafLiteContractError("truncated result must end with frozen marker")
    return LeafLiteToolResult(tool, status, output, exit_code, truncated)


def _definition(
    name: str,
    description: str,
    required: tuple[str, ...],
    properties: dict[str, object],
) -> dict[str, object]:
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": list(required),
            "properties": properties,
        },
    }


def leaf_lite_tool_definitions(
    contract: LeafLiteToolContract,
) -> tuple[dict[str, object], ...]:
    """Return the exact model-visible JSON schemas driven by the parser contract."""

    definitions = {
        LeafLiteToolName.READ: _definition(
            "read",
            "Read bounded lines from one candidate-visible UTF-8 file.",
            ("path",),
            {
                "path": {"type": "string"},
                "start_line": {"type": "integer", "minimum": 1, "default": 1},
                "max_lines": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": contract.read_max_lines,
                    "default": contract.read_default_max_lines,
                },
            },
        ),
        LeafLiteToolName.WRITE: _definition(
            "write",
            "Atomically create or replace one candidate-visible UTF-8 file.",
            ("path", "content"),
            {"path": {"type": "string"}, "content": {"type": "string"}},
        ),
        LeafLiteToolName.EDIT: _definition(
            "edit",
            "Replace exact text with an exact expected replacement count.",
            ("path", "old_text", "new_text"),
            {
                "path": {"type": "string"},
                "old_text": {"type": "string", "minLength": 1},
                "new_text": {"type": "string"},
                "expected_replacements": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 1,
                },
            },
        ),
        LeafLiteToolName.GLOB: _definition(
            "glob",
            "List matching candidate-visible paths in lexical order.",
            ("pattern",),
            {"pattern": {"type": "string"}},
        ),
        LeafLiteToolName.BASH: _definition(
            "bash",
            f"Run one fresh-shell command in the candidate workspace; fixed timeout {contract.bash_timeout_seconds}s.",
            ("command",),
            {"command": {"type": "string", "minLength": 1}},
        ),
    }
    return tuple(definitions[name] for name in contract.tool_names)


def _contract_payload(contract: LeafLiteToolContract) -> dict[str, object]:
    payload = asdict(contract)
    payload["profile"] = contract.profile.value
    payload["tool_names"] = [item.value for item in contract.tool_names]
    payload["tool_definitions"] = list(leaf_lite_tool_definitions(contract))
    payload["grader_operations_exposed"] = False
    payload["workspace_path_policy"] = {
        "relative_posix_only": True,
        "reserved_prefixes": list(_RESERVED),
    }
    payload["bash_semantics"] = {
        "cwd": "candidate_workspace",
        "fresh_process_per_call": True,
        "persistent_shell_state": False,
        "network_policy": "inherit_pvrl_environment_disabled_policy",
        "resource_policy": "inherit_pvrl_environment_bounds",
    }
    payload["write_semantics"] = "atomic_create_or_replace_utf8_text"
    payload["edit_semantics"] = "exact_text_replacement_with_required_count"
    payload["glob_semantics"] = "candidate_visible_paths_sorted_lexically"
    payload["result_serialization"] = (
        "canonical_ascii_json_sorted_keys_compact_plus_newline"
    )
    payload["output_truncation"] = "utf8_prefix_then_frozen_marker"
    return payload


def leaf_lite_contract_json(contract: LeafLiteToolContract) -> str:
    payload = _contract_payload(contract)
    payload["contract_sha256"] = contract.contract_sha256
    return json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"


__all__ = [
    "BashToolArguments",
    "EditToolArguments",
    "GlobToolArguments",
    "LeafLiteArguments",
    "LeafLiteContractError",
    "LeafLiteProfile",
    "LeafLiteResultStatus",
    "LeafLiteToolCall",
    "LeafLiteToolContract",
    "LeafLiteToolName",
    "LeafLiteToolResult",
    "ReadToolArguments",
    "WriteToolArguments",
    "create_leaf_lite_tool_result",
    "leaf_lite_contract_json",
    "leaf_lite_tool_call_json",
    "leaf_lite_tool_definitions",
    "leaf_lite_tool_result_from_json",
    "leaf_lite_tool_result_json",
    "parse_leaf_lite_tool_call",
    "repository_leaf_lite_contract",
    "self_contained_leaf_lite_contract",
]
