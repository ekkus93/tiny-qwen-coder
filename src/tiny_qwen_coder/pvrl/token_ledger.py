"""Exact append-only Qwen3.5 token ledger for PVRL multi-turn trajectories."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import cast

from tiny_qwen_coder.identities import BaseModelIdentity

_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class TokenLedgerError(RuntimeError):
    """Invalid or inconsistent PVRL token-ledger state."""


class TokenLedgerStateError(TokenLedgerError):
    """A ledger operation was attempted in the wrong state."""


class TokenLedgerForkError(TokenLedgerError):
    """A proposed continuation changed historical token IDs."""


class PromptAppendSource(StrEnum):
    INITIAL = "initial"
    USER = "user"
    TOOL = "tool"


class AssistantSegmentKind(StrEnum):
    TEXT = "text"
    REASONING = "reasoning"
    ACTION = "action"


class TokenLedgerState(StrEnum):
    AWAITING_OUTPUT = "awaiting_output"
    AWAITING_PROMPT_APPEND = "awaiting_prompt_append"


@dataclass(frozen=True, slots=True)
class Qwen35TokenLedgerIdentity:
    """Tokenizer/template/tool identity governing one exact-token session."""

    base_model: BaseModelIdentity
    chat_template_sha256: str
    leaf_lite_contract_sha256: str
    enable_thinking: bool
    compaction_enabled: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.base_model, BaseModelIdentity):
            raise TokenLedgerError("base_model must be a BaseModelIdentity")
        _sha(self.chat_template_sha256, "chat_template_sha256")
        _sha(self.leaf_lite_contract_sha256, "leaf_lite_contract_sha256")
        if not isinstance(self.enable_thinking, bool):
            raise TokenLedgerError("enable_thinking must be boolean")
        if not isinstance(self.compaction_enabled, bool):
            raise TokenLedgerError("compaction_enabled must be boolean")
        if self.compaction_enabled:
            raise TokenLedgerError("PVRL v0 token ledger does not support context compaction")


@dataclass(frozen=True, slots=True)
class AssistantTokenSegment:
    """Half-open semantic label over exact policy-emitted output IDs."""

    kind: AssistantSegmentKind
    start: int
    end: int

    def __post_init__(self) -> None:
        if not isinstance(self.kind, AssistantSegmentKind):
            raise TokenLedgerError("assistant segment kind must be an AssistantSegmentKind")
        if not _plain_int(self.start) or not _plain_int(self.end):
            raise TokenLedgerError("assistant segment boundaries must be integers")
        if self.start < 0 or self.end <= self.start:
            raise TokenLedgerError("assistant segment must satisfy 0 <= start < end")


@dataclass(frozen=True, slots=True)
class PromptTokenRecord:
    """Exact model prompt and the newly rendered suffix that created it."""

    turn_index: int
    source: PromptAppendSource
    prompt_ids: tuple[int, ...]
    appended_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        _index(self.turn_index, "prompt turn_index")
        if not isinstance(self.source, PromptAppendSource):
            raise TokenLedgerError("prompt source must be a PromptAppendSource")
        _tokens(self.prompt_ids, "prompt_ids")
        _tokens(self.appended_ids, "appended_ids")


@dataclass(frozen=True, slots=True)
class AssistantOutputRecord:
    """Exact raw policy output IDs; decoded text is deliberately absent."""

    turn_index: int
    output_ids: tuple[int, ...]
    segments: tuple[AssistantTokenSegment, ...]

    def __post_init__(self) -> None:
        _index(self.turn_index, "output turn_index")
        _tokens(self.output_ids, "output_ids")
        _segments(self.output_ids, self.segments)


@dataclass(frozen=True, slots=True)
class TokenLedgerSnapshot:
    """Immutable exact-token snapshot suitable for trajectory provenance."""

    schema_version: int
    session_id: str
    identity: Qwen35TokenLedgerIdentity
    prompts: tuple[PromptTokenRecord, ...]
    outputs: tuple[AssistantOutputRecord, ...]
    state: TokenLedgerState

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise TokenLedgerError("unsupported token-ledger schema_version")
        _trimmed(self.session_id, "session_id")
        if not isinstance(self.identity, Qwen35TokenLedgerIdentity):
            raise TokenLedgerError("identity must be a Qwen35TokenLedgerIdentity")
        if not isinstance(self.state, TokenLedgerState):
            raise TokenLedgerError("state must be a TokenLedgerState")
        _chain(self.prompts, self.outputs, self.state)

    @property
    def ledger_sha256(self) -> str:
        return hashlib.sha256(_canonical(_snapshot_payload(self))).hexdigest()


class ExactQwen35TokenLedger:
    """Append-only ledger that never reconstructs historical assistant text."""

    def __init__(
        self,
        *,
        session_id: str,
        identity: Qwen35TokenLedgerIdentity,
        initial_prompt_ids: Sequence[int],
    ) -> None:
        _trimmed(session_id, "session_id")
        initial = _token_tuple(initial_prompt_ids, "initial_prompt_ids")
        self._session_id = session_id
        self._identity = identity
        self._prompts = [PromptTokenRecord(0, PromptAppendSource.INITIAL, initial, initial)]
        self._outputs: list[AssistantOutputRecord] = []

    @property
    def state(self) -> TokenLedgerState:
        if len(self._outputs) == len(self._prompts):
            return TokenLedgerState.AWAITING_PROMPT_APPEND
        return TokenLedgerState.AWAITING_OUTPUT

    @property
    def current_prompt_ids(self) -> tuple[int, ...]:
        return self._prompts[-1].prompt_ids

    @property
    def previous_output_ids(self) -> tuple[int, ...] | None:
        return self._outputs[-1].output_ids if self._outputs else None

    def record_assistant_output(
        self,
        output_ids: Sequence[int],
        *,
        segments: Sequence[AssistantTokenSegment] | None = None,
    ) -> AssistantOutputRecord:
        """Record rollout-engine IDs directly, with no decode/normalize/re-tokenize step."""

        if self.state is not TokenLedgerState.AWAITING_OUTPUT:
            raise TokenLedgerStateError("cannot record another output before appending next prompt")
        exact = _token_tuple(output_ids, "output_ids")
        exact_segments = (
            tuple(segments)
            if segments is not None
            else (AssistantTokenSegment(AssistantSegmentKind.TEXT, 0, len(exact)),)
        )
        record = AssistantOutputRecord(len(self._outputs), exact, exact_segments)
        self._outputs.append(record)
        return record

    def append_prompt_delta(
        self, appended_ids: Sequence[int], *, source: PromptAppendSource
    ) -> PromptTokenRecord:
        """Append only newly rendered user/tool observation plus generation-prompt IDs."""

        self._require_append(source)
        delta = _token_tuple(appended_ids, "appended_ids")
        prefix = self._prompts[-1].prompt_ids + self._outputs[-1].output_ids
        record = PromptTokenRecord(len(self._prompts), source, prefix + delta, delta)
        self._prompts.append(record)
        return record

    def adopt_verified_next_prompt(
        self, next_prompt_ids: Sequence[int], *, source: PromptAppendSource
    ) -> PromptTokenRecord:
        """Accept external prompt IDs only if their historical prefix is byte-for-byte exact."""

        self._require_append(source)
        candidate = _token_tuple(next_prompt_ids, "next_prompt_ids")
        prefix = self._prompts[-1].prompt_ids + self._outputs[-1].output_ids
        if len(candidate) <= len(prefix) or candidate[: len(prefix)] != prefix:
            raise TokenLedgerForkError(
                "next prompt violates exact uncompacted-history prefix invariant"
            )
        return self.append_prompt_delta(candidate[len(prefix) :], source=source)

    def _require_append(self, source: PromptAppendSource) -> None:
        if self.state is not TokenLedgerState.AWAITING_PROMPT_APPEND:
            raise TokenLedgerStateError("cannot append prompt material before recording output")
        if source is PromptAppendSource.INITIAL:
            raise TokenLedgerError("only user/tool sources are valid after the initial prompt")
        if not isinstance(source, PromptAppendSource):
            raise TokenLedgerError("prompt source must be a PromptAppendSource")

    def snapshot(self) -> TokenLedgerSnapshot:
        return TokenLedgerSnapshot(
            _SCHEMA_VERSION,
            self._session_id,
            self._identity,
            tuple(self._prompts),
            tuple(self._outputs),
            self.state,
        )

    @classmethod
    def from_snapshot(cls, snapshot: TokenLedgerSnapshot) -> ExactQwen35TokenLedger:
        if not isinstance(snapshot, TokenLedgerSnapshot):
            raise TokenLedgerError("snapshot must be a TokenLedgerSnapshot")
        ledger = cls(
            session_id=snapshot.session_id,
            identity=snapshot.identity,
            initial_prompt_ids=snapshot.prompts[0].prompt_ids,
        )
        ledger._prompts = list(snapshot.prompts)
        ledger._outputs = list(snapshot.outputs)
        if ledger.state is not snapshot.state:
            raise TokenLedgerError("restored token-ledger state drifted")
        return ledger


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    """Hash an exact ordered token-ID sequence."""

    return hashlib.sha256(_canonical(list(_token_tuple(token_ids, "token_ids")))).hexdigest()


def token_ledger_snapshot_json(snapshot: TokenLedgerSnapshot) -> str:
    """Serialize exact IDs deterministically as canonical ASCII JSON plus newline."""

    return _canonical(_snapshot_payload(snapshot)).decode("ascii") + "\n"


def token_ledger_snapshot_from_json(text: str) -> TokenLedgerSnapshot:
    """Strictly parse and revalidate a persisted exact token ledger."""

    try:
        raw: object = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TokenLedgerError("token ledger is not valid JSON") from exc
    row = _object(raw, "token ledger")
    _keys(
        row,
        {"schema_version", "session_id", "identity", "prompts", "outputs", "state"},
        "token ledger",
    )
    prompts = _list(row.get("prompts"), "token ledger.prompts")
    outputs = _list(row.get("outputs"), "token ledger.outputs")
    try:
        state = TokenLedgerState(_string(row, "state", "token ledger"))
    except ValueError as exc:
        raise TokenLedgerError("token ledger.state is unknown") from exc
    return TokenLedgerSnapshot(
        schema_version=_integer(row, "schema_version", "token ledger"),
        session_id=_string(row, "session_id", "token ledger"),
        identity=_identity(row.get("identity")),
        prompts=tuple(_prompt(item) for item in prompts),
        outputs=tuple(_output(item) for item in outputs),
        state=state,
    )


def _chain(
    prompts: tuple[PromptTokenRecord, ...],
    outputs: tuple[AssistantOutputRecord, ...],
    state: TokenLedgerState,
) -> None:
    if not prompts or len(outputs) not in {len(prompts) - 1, len(prompts)}:
        raise TokenLedgerError("token ledger prompt/output counts are inconsistent")
    for index, prompt in enumerate(prompts):
        if not isinstance(prompt, PromptTokenRecord) or prompt.turn_index != index:
            raise TokenLedgerError("prompt records must be contiguous PromptTokenRecord values")
        if index == 0:
            if (
                prompt.source is not PromptAppendSource.INITIAL
                or prompt.prompt_ids != prompt.appended_ids
            ):
                raise TokenLedgerError("initial prompt must be entirely initial appended material")
            continue
        if prompt.source is PromptAppendSource.INITIAL:
            raise TokenLedgerError("only the first prompt may use initial source")
        expected = (
            prompts[index - 1].prompt_ids + outputs[index - 1].output_ids + prompt.appended_ids
        )
        if prompt.prompt_ids != expected:
            raise TokenLedgerForkError(
                "persisted prompt violates exact uncompacted-history prefix invariant"
            )
    for index, output in enumerate(outputs):
        if not isinstance(output, AssistantOutputRecord) or output.turn_index != index:
            raise TokenLedgerError("output records must be contiguous AssistantOutputRecord values")
    expected_state = (
        TokenLedgerState.AWAITING_PROMPT_APPEND
        if len(outputs) == len(prompts)
        else TokenLedgerState.AWAITING_OUTPUT
    )
    if state is not expected_state:
        raise TokenLedgerError("token ledger state is inconsistent with prompt/output counts")


def _segments(output_ids: tuple[int, ...], segments: tuple[AssistantTokenSegment, ...]) -> None:
    if not segments:
        raise TokenLedgerError("assistant output must contain at least one exact token segment")
    position = 0
    for segment in segments:
        if not isinstance(segment, AssistantTokenSegment) or segment.start != position:
            raise TokenLedgerError(
                "assistant token segments must be contiguous and non-overlapping"
            )
        if segment.end > len(output_ids):
            raise TokenLedgerError("assistant token segment exceeds output length")
        position = segment.end
    if position != len(output_ids):
        raise TokenLedgerError(
            "assistant token segments must cover every output token exactly once"
        )


def _plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _index(value: int, name: str) -> None:
    if not _plain_int(value) or value < 0:
        raise TokenLedgerError(f"{name} must be a non-negative integer")


def _tokens(values: tuple[int, ...], name: str) -> None:
    if not values:
        raise TokenLedgerError(f"{name} must not be empty")
    for index, token_id in enumerate(values):
        if not _plain_int(token_id) or token_id < 0:
            raise TokenLedgerError(f"{name}[{index}] must be a non-negative integer")


def _token_tuple(values: Sequence[int], name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TokenLedgerError(f"{name} must be an integer sequence")
    result = tuple(values)
    _tokens(result, name)
    return result


def _sha(value: str, name: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise TokenLedgerError(f"{name} must be a lowercase SHA-256 digest")


def _trimmed(value: str, name: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TokenLedgerError(f"{name} must be a non-empty trimmed string")


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode(
        "ascii"
    )


def _snapshot_payload(snapshot: TokenLedgerSnapshot) -> dict[str, object]:
    identity = asdict(snapshot.identity)
    prompts = []
    for prompt in snapshot.prompts:
        item = asdict(prompt)
        item["source"] = prompt.source.value
        prompts.append(item)
    outputs = []
    for output in snapshot.outputs:
        item = asdict(output)
        item["segments"] = [
            {"kind": segment.kind.value, "start": segment.start, "end": segment.end}
            for segment in output.segments
        ]
        outputs.append(item)
    return {
        "schema_version": snapshot.schema_version,
        "session_id": snapshot.session_id,
        "identity": identity,
        "prompts": prompts,
        "outputs": outputs,
        "state": snapshot.state.value,
    }


def _object(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TokenLedgerError(f"{context} must be a string-keyed JSON object")
    return cast(dict[str, object], dict(value))


def _list(value: object, context: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TokenLedgerError(f"{context} must be a list")
    return cast(Sequence[object], value)


def _keys(row: Mapping[str, object], expected: set[str], context: str) -> None:
    if set(row) != expected:
        raise TokenLedgerError(f"{context} keys do not match schema")


def _string(row: Mapping[str, object], key: str, context: str) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise TokenLedgerError(f"{context}.{key} must be a string")
    return value


def _integer(row: Mapping[str, object], key: str, context: str) -> int:
    value = row.get(key)
    if not _plain_int(value):
        raise TokenLedgerError(f"{context}.{key} must be an integer")
    return cast(int, value)


def _boolean(row: Mapping[str, object], key: str, context: str) -> bool:
    value = row.get(key)
    if not isinstance(value, bool):
        raise TokenLedgerError(f"{context}.{key} must be boolean")
    return value


def _identity(value: object) -> Qwen35TokenLedgerIdentity:
    row = _object(value, "token ledger.identity")
    _keys(
        row,
        {
            "base_model",
            "chat_template_sha256",
            "leaf_lite_contract_sha256",
            "enable_thinking",
            "compaction_enabled",
        },
        "token ledger.identity",
    )
    base = _object(row.get("base_model"), "token ledger.identity.base_model")
    _keys(
        base, {"repository", "revision", "tokenizer_repository", "tokenizer_revision"}, "base_model"
    )
    try:
        base_model = BaseModelIdentity(
            repository=_string(base, "repository", "base_model"),
            revision=_string(base, "revision", "base_model"),
            tokenizer_repository=_string(base, "tokenizer_repository", "base_model"),
            tokenizer_revision=_string(base, "tokenizer_revision", "base_model"),
        )
    except ValueError as exc:
        raise TokenLedgerError("token ledger base-model identity is invalid") from exc
    return Qwen35TokenLedgerIdentity(
        base_model=base_model,
        chat_template_sha256=_string(row, "chat_template_sha256", "token ledger.identity"),
        leaf_lite_contract_sha256=_string(
            row, "leaf_lite_contract_sha256", "token ledger.identity"
        ),
        enable_thinking=_boolean(row, "enable_thinking", "token ledger.identity"),
        compaction_enabled=_boolean(row, "compaction_enabled", "token ledger.identity"),
    )


def _prompt(value: object) -> PromptTokenRecord:
    row = _object(value, "prompt record")
    _keys(row, {"turn_index", "source", "prompt_ids", "appended_ids"}, "prompt record")
    try:
        source = PromptAppendSource(_string(row, "source", "prompt record"))
    except ValueError as exc:
        raise TokenLedgerError("prompt record.source is unknown") from exc
    return PromptTokenRecord(
        _integer(row, "turn_index", "prompt record"),
        source,
        _json_tokens(row.get("prompt_ids"), "prompt record.prompt_ids"),
        _json_tokens(row.get("appended_ids"), "prompt record.appended_ids"),
    )


def _output(value: object) -> AssistantOutputRecord:
    row = _object(value, "output record")
    _keys(row, {"turn_index", "output_ids", "segments"}, "output record")
    segments = []
    for value in _list(row.get("segments"), "output record.segments"):
        item = _object(value, "assistant segment")
        _keys(item, {"kind", "start", "end"}, "assistant segment")
        try:
            kind = AssistantSegmentKind(_string(item, "kind", "assistant segment"))
        except ValueError as exc:
            raise TokenLedgerError("assistant segment.kind is unknown") from exc
        segments.append(
            AssistantTokenSegment(
                kind,
                _integer(item, "start", "assistant segment"),
                _integer(item, "end", "assistant segment"),
            )
        )
    return AssistantOutputRecord(
        _integer(row, "turn_index", "output record"),
        _json_tokens(row.get("output_ids"), "output record.output_ids"),
        tuple(segments),
    )


def _json_tokens(value: object, context: str) -> tuple[int, ...]:
    return _token_tuple(cast(Sequence[int], _list(value, context)), context)
