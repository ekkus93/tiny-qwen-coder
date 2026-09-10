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
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class TokenLedgerError(RuntimeError):
    """Base class for invalid or inconsistent PVRL token-ledger state."""


class TokenLedgerStateError(TokenLedgerError):
    """Raised when a ledger operation is attempted in the wrong state."""


class TokenLedgerForkError(TokenLedgerError):
    """Raised when proposed continuation IDs do not preserve exact history."""


class PromptAppendSource(StrEnum):
    """Origin of newly rendered prompt material after one assistant turn."""

    INITIAL = "initial"
    USER = "user"
    TOOL = "tool"


class AssistantSegmentKind(StrEnum):
    """Non-textual labels over exact policy-emitted output token spans."""

    TEXT = "text"
    REASONING = "reasoning"
    ACTION = "action"


class TokenLedgerState(StrEnum):
    """Next legal operation for an exact token ledger."""

    AWAITING_OUTPUT = "awaiting_output"
    AWAITING_PROMPT_APPEND = "awaiting_prompt_append"


@dataclass(frozen=True, slots=True)
class Qwen35TokenLedgerIdentity:
    """Immutable tokenizer/template/tool identity governing one ledger session."""

    base_model: BaseModelIdentity
    chat_template_sha256: str
    leaf_lite_contract_sha256: str
    enable_thinking: bool
    compaction_enabled: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.base_model, BaseModelIdentity):
            raise TokenLedgerError("base_model must be a BaseModelIdentity")
        _require_sha256(self.chat_template_sha256, "chat_template_sha256")
        _require_sha256(self.leaf_lite_contract_sha256, "leaf_lite_contract_sha256")
        if not isinstance(self.enable_thinking, bool):
            raise TokenLedgerError("enable_thinking must be boolean")
        if not isinstance(self.compaction_enabled, bool):
            raise TokenLedgerError("compaction_enabled must be boolean")
        if self.compaction_enabled:
            raise TokenLedgerError(
                "PVRL v0 token ledger does not support context compaction"
            )


@dataclass(frozen=True, slots=True)
class AssistantTokenSegment:
    """Half-open semantic label over exact assistant output token IDs."""

    kind: AssistantSegmentKind
    start: int
    end: int

    def __post_init__(self) -> None:
        if not isinstance(self.kind, AssistantSegmentKind):
            raise TokenLedgerError(
                "assistant segment kind must be an AssistantSegmentKind"
            )
        if (
            isinstance(self.start, bool)
            or not isinstance(self.start, int)
            or isinstance(self.end, bool)
            or not isinstance(self.end, int)
        ):
            raise TokenLedgerError("assistant segment boundaries must be integers")
        if self.start < 0 or self.end <= self.start:
            raise TokenLedgerError("assistant segment must satisfy 0 <= start < end")


@dataclass(frozen=True, slots=True)
class PromptTokenRecord:
    """Exact prompt IDs for one generation plus the newly appended suffix IDs."""

    turn_index: int
    source: PromptAppendSource
    prompt_ids: tuple[int, ...]
    appended_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_index(self.turn_index, "prompt turn_index")
        if not isinstance(self.source, PromptAppendSource):
            raise TokenLedgerError("prompt source must be a PromptAppendSource")
        _validate_token_ids(self.prompt_ids, field_name="prompt_ids")
        _validate_token_ids(self.appended_ids, field_name="appended_ids")

    @property
    def prompt_sha256(self) -> str:
        return token_ids_sha256(self.prompt_ids)

    @property
    def appended_sha256(self) -> str:
        return token_ids_sha256(self.appended_ids)


@dataclass(frozen=True, slots=True)
class AssistantOutputRecord:
    """Exact raw policy output IDs and optional semantic span labels."""

    turn_index: int
    output_ids: tuple[int, ...]
    segments: tuple[AssistantTokenSegment, ...]

    def __post_init__(self) -> None:
        _require_index(self.turn_index, "output turn_index")
        _validate_token_ids(self.output_ids, field_name="output_ids")
        _validate_segments(self.output_ids, self.segments)

    @property
    def output_sha256(self) -> str:
        return token_ids_sha256(self.output_ids)


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
        if (
            not isinstance(self.session_id, str)
            or not self.session_id
            or self.session_id != self.session_id.strip()
        ):
            raise TokenLedgerError("session_id must be a non-empty trimmed string")
        if not isinstance(self.identity, Qwen35TokenLedgerIdentity):
            raise TokenLedgerError("identity must be a Qwen35TokenLedgerIdentity")
        if not isinstance(self.state, TokenLedgerState):
            raise TokenLedgerError("state must be a TokenLedgerState")
        _validate_snapshot_chain(self.prompts, self.outputs, self.state)

    @property
    def ledger_sha256(self) -> str:
        return hashlib.sha256(_canonical_bytes(_snapshot_payload(self))).hexdigest()


class ExactQwen35TokenLedger:
    """Stateful append-only ledger that never reconstructs historical assistant text."""

    def __init__(
        self,
        *,
        session_id: str,
        identity: Qwen35TokenLedgerIdentity,
        initial_prompt_ids: Sequence[int],
    ) -> None:
        if (
            not isinstance(session_id, str)
            or not session_id
            or session_id != session_id.strip()
        ):
            raise TokenLedgerError("session_id must be a non-empty trimmed string")
        initial = _token_tuple(initial_prompt_ids, field_name="initial_prompt_ids")
        self._session_id = session_id
        self._identity = identity
        self._prompts = [
            PromptTokenRecord(
                turn_index=0,
                source=PromptAppendSource.INITIAL,
                prompt_ids=initial,
                appended_ids=initial,
            )
        ]
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
        if not self._outputs:
            return None
        return self._outputs[-1].output_ids

    def record_assistant_output(
        self,
        output_ids: Sequence[int],
        *,
        segments: Sequence[AssistantTokenSegment] | None = None,
    ) -> AssistantOutputRecord:
        """Record exact rollout-engine IDs without decoding or normalizing them."""

        if self.state is not TokenLedgerState.AWAITING_OUTPUT:
            raise TokenLedgerStateError(
                "cannot record another output before appending next prompt"
            )
        exact = _token_tuple(output_ids, field_name="output_ids")
        exact_segments = (
            tuple(segments)
            if segments is not None
            else (AssistantTokenSegment(AssistantSegmentKind.TEXT, 0, len(exact)),)
        )
        record = AssistantOutputRecord(
            turn_index=len(self._outputs),
            output_ids=exact,
            segments=exact_segments,
        )
        self._outputs.append(record)
        return record

    def append_prompt_delta(
        self,
        appended_ids: Sequence[int],
        *,
        source: PromptAppendSource,
    ) -> PromptTokenRecord:
        """Append only newly rendered user/tool observation + generation-prompt IDs."""

        if self.state is not TokenLedgerState.AWAITING_PROMPT_APPEND:
            raise TokenLedgerStateError(
                "cannot append prompt material before recording output"
            )
        if source is PromptAppendSource.INITIAL:
            raise TokenLedgerError(
                "only user/tool sources are valid after the initial prompt"
            )
        delta = _token_tuple(appended_ids, field_name="appended_ids")
        prefix = self._prompts[-1].prompt_ids + self._outputs[-1].output_ids
        next_prompt = prefix + delta
        record = PromptTokenRecord(
            turn_index=len(self._prompts),
            source=source,
            prompt_ids=next_prompt,
            appended_ids=delta,
        )
        self._prompts.append(record)
        return record

    def adopt_verified_next_prompt(
        self,
        next_prompt_ids: Sequence[int],
        *,
        source: PromptAppendSource,
    ) -> PromptTokenRecord:
        """Adopt externally constructed IDs only when exact historical prefix is unchanged."""

        if self.state is not TokenLedgerState.AWAITING_PROMPT_APPEND:
            raise TokenLedgerStateError(
                "cannot adopt next prompt before recording output"
            )
        if source is PromptAppendSource.INITIAL:
            raise TokenLedgerError(
                "only user/tool sources are valid after the initial prompt"
            )
        candidate = _token_tuple(next_prompt_ids, field_name="next_prompt_ids")
        prefix = self._prompts[-1].prompt_ids + self._outputs[-1].output_ids
        if len(candidate) <= len(prefix) or candidate[: len(prefix)] != prefix:
            raise TokenLedgerForkError(
                "next prompt violates exact uncompacted-history prefix invariant"
            )
        return self.append_prompt_delta(candidate[len(prefix) :], source=source)

    def snapshot(self) -> TokenLedgerSnapshot:
        return TokenLedgerSnapshot(
            schema_version=_SCHEMA_VERSION,
            session_id=self._session_id,
            identity=self._identity,
            prompts=tuple(self._prompts),
            outputs=tuple(self._outputs),
            state=self.state,
        )

    @classmethod
    def from_snapshot(cls, snapshot: TokenLedgerSnapshot) -> ExactQwen35TokenLedger:
        """Restore exact integer arrays after strict chain validation."""

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
    """Hash an exact ordered token-ID sequence using canonical JSON encoding."""

    exact = _token_tuple(token_ids, field_name="token_ids")
    return hashlib.sha256(_canonical_bytes(list(exact))).hexdigest()


def token_ledger_snapshot_json(snapshot: TokenLedgerSnapshot) -> str:
    """Serialize exact token IDs and metadata deterministically."""

    return _canonical_bytes(_snapshot_payload(snapshot)).decode("ascii") + "\n"


def token_ledger_snapshot_from_json(text: str) -> TokenLedgerSnapshot:
    """Strictly parse and revalidate a persisted exact token ledger."""

    try:
        raw: object = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TokenLedgerError("token ledger is not valid JSON") from exc
    payload = _mapping(raw, "token ledger")
    _require_keys(
        payload,
        {"schema_version", "session_id", "identity", "prompts", "outputs", "state"},
        "token ledger",
    )
    schema_version = _integer(payload, "schema_version", "token ledger")
    session_id = _string(payload, "session_id", "token ledger")
    identity = _identity_from_payload(payload.get("identity"))
    prompts_value = payload.get("prompts")
    outputs_value = payload.get("outputs")
    if isinstance(prompts_value, (str, bytes)) or not isinstance(
        prompts_value, Sequence
    ):
        raise TokenLedgerError("token ledger.prompts must be a list")
    if isinstance(outputs_value, (str, bytes)) or not isinstance(
        outputs_value, Sequence
    ):
        raise TokenLedgerError("token ledger.outputs must be a list")
    prompts = tuple(_prompt_from_payload(item) for item in prompts_value)
    outputs = tuple(_output_from_payload(item) for item in outputs_value)
    try:
        state = TokenLedgerState(_string(payload, "state", "token ledger"))
    except ValueError as exc:
        raise TokenLedgerError("token ledger.state is unknown") from exc
    return TokenLedgerSnapshot(
        schema_version=schema_version,
        session_id=session_id,
        identity=identity,
        prompts=prompts,
        outputs=outputs,
        state=state,
    )


def _validate_snapshot_chain(
    prompts: tuple[PromptTokenRecord, ...],
    outputs: tuple[AssistantOutputRecord, ...],
    state: TokenLedgerState,
) -> None:
    if not prompts:
        raise TokenLedgerError("token ledger must contain an initial prompt")
    if len(outputs) not in {len(prompts) - 1, len(prompts)}:
        raise TokenLedgerError("token ledger prompt/output counts are inconsistent")
    for index, prompt in enumerate(prompts):
        if not isinstance(prompt, PromptTokenRecord):
            raise TokenLedgerError("prompts must contain PromptTokenRecord values")
        if prompt.turn_index != index:
            raise TokenLedgerError("prompt turn indices must be contiguous")
        if index == 0:
            if (
                prompt.source is not PromptAppendSource.INITIAL
                or prompt.prompt_ids != prompt.appended_ids
            ):
                raise TokenLedgerError(
                    "initial prompt must be entirely initial appended material"
                )
            continue
        if prompt.source is PromptAppendSource.INITIAL:
            raise TokenLedgerError("only the first prompt may use initial source")
        previous_output = outputs[index - 1]
        expected = (
            prompts[index - 1].prompt_ids
            + previous_output.output_ids
            + prompt.appended_ids
        )
        if prompt.prompt_ids != expected:
            raise TokenLedgerForkError(
                "persisted prompt violates exact uncompacted-history prefix invariant"
            )
    for index, output in enumerate(outputs):
        if not isinstance(output, AssistantOutputRecord):
            raise TokenLedgerError(
                "outputs must contain AssistantOutputRecord values"
            )
        if output.turn_index != index:
            raise TokenLedgerError("output turn indices must be contiguous")
    expected_state = (
        TokenLedgerState.AWAITING_PROMPT_APPEND
        if len(outputs) == len(prompts)
        else TokenLedgerState.AWAITING_OUTPUT
    )
    if state is not expected_state:
        raise TokenLedgerError(
            "token ledger state is inconsistent with prompt/output counts"
        )


def _validate_segments(
    output_ids: tuple[int, ...], segments: tuple[AssistantTokenSegment, ...]
) -> None:
    if not segments:
        raise TokenLedgerError(
            "assistant output must contain at least one exact token segment"
        )
    expected_start = 0
    for segment in segments:
        if not isinstance(segment, AssistantTokenSegment):
            raise TokenLedgerError(
                "segments must contain AssistantTokenSegment values"
            )
        if segment.start != expected_start:
            raise TokenLedgerError(
                "assistant token segments must be contiguous and non-overlapping"
            )
        if segment.end > len(output_ids):
            raise TokenLedgerError("assistant token segment exceeds output length")
        expected_start = segment.end
    if expected_start != len(output_ids):
        raise TokenLedgerError(
            "assistant token segments must cover every output token exactly once"
        )


def _token_tuple(values: Sequence[int], *, field_name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TokenLedgerError(f"{field_name} must be an integer sequence")
    result = tuple(values)
    _validate_token_ids(result, field_name=field_name)
    return result


def _validate_token_ids(values: tuple[int, ...], *, field_name: str) -> None:
    if not values:
        raise TokenLedgerError(f"{field_name} must not be empty")
    for index, token_id in enumerate(values):
        if (
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
        ):
            raise TokenLedgerError(
                f"{field_name}[{index}] must be a non-negative integer"
            )


def _require_index(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TokenLedgerError(f"{field_name} must be a non-negative integer")


def _require_sha256(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise TokenLedgerError(f"{field_name} must be a lowercase SHA-256 digest")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")


def _snapshot_payload(snapshot: TokenLedgerSnapshot) -> dict[str, object]:
    identity = asdict(snapshot.identity)
    prompts = []
    for prompt in snapshot.prompts:
        row = asdict(prompt)
        row["source"] = prompt.source.value
        prompts.append(row)
    outputs = []
    for output in snapshot.outputs:
        row = asdict(output)
        row["segments"] = [
            {"kind": segment.kind.value, "start": segment.start, "end": segment.end}
            for segment in output.segments
        ]
        outputs.append(row)
    return {
        "schema_version": snapshot.schema_version,
        "session_id": snapshot.session_id,
        "identity": identity,
        "prompts": prompts,
        "outputs": outputs,
        "state": snapshot.state.value,
    }


def _mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TokenLedgerError(f"{context} must be a JSON object")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TokenLedgerError(f"{context} keys must be strings")
        result[key] = item
    return result


def _require_keys(
    value: Mapping[str, object], expected: set[str], context: str
) -> None:
    if set(value) != expected:
        raise TokenLedgerError(f"{context} keys do not match schema")


def _string(value: Mapping[str, object], key: str, context: str) -> str:
    item = value.get(key)
    if not isinstance(item, str):
        raise TokenLedgerError(f"{context}.{key} must be a string")
    return item


def _integer(value: Mapping[str, object], key: str, context: str) -> int:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, int):
        raise TokenLedgerError(f"{context}.{key} must be an integer")
    return item


def _boolean(value: Mapping[str, object], key: str, context: str) -> bool:
    item = value.get(key)
    if not isinstance(item, bool):
        raise TokenLedgerError(f"{context}.{key} must be boolean")
    return item


def _identity_from_payload(value: object) -> Qwen35TokenLedgerIdentity:
    row = _mapping(value, "token ledger.identity")
    _require_keys(
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
    base = _mapping(row.get("base_model"), "token ledger.identity.base_model")
    _require_keys(
        base,
        {"repository", "revision", "tokenizer_repository", "tokenizer_revision"},
        "token ledger.identity.base_model",
    )
    try:
        base_model = BaseModelIdentity(
            repository=_string(base, "repository", "token ledger.identity.base_model"),
            revision=_string(base, "revision", "token ledger.identity.base_model"),
            tokenizer_repository=_string(
                base, "tokenizer_repository", "token ledger.identity.base_model"
            ),
            tokenizer_revision=_string(
                base, "tokenizer_revision", "token ledger.identity.base_model"
            ),
        )
    except ValueError as exc:
        raise TokenLedgerError("token ledger base-model identity is invalid") from exc
    return Qwen35TokenLedgerIdentity(
        base_model=base_model,
        chat_template_sha256=_string(
            row, "chat_template_sha256", "token ledger.identity"
        ),
        leaf_lite_contract_sha256=_string(
            row, "leaf_lite_contract_sha256", "token ledger.identity"
        ),
        enable_thinking=_boolean(row, "enable_thinking", "token ledger.identity"),
        compaction_enabled=_boolean(row, "compaction_enabled", "token ledger.identity"),
    )


def _prompt_from_payload(value: object) -> PromptTokenRecord:
    row = _mapping(value, "prompt record")
    _require_keys(
        row, {"turn_index", "source", "prompt_ids", "appended_ids"}, "prompt record"
    )
    try:
        source = PromptAppendSource(_string(row, "source", "prompt record"))
    except ValueError as exc:
        raise TokenLedgerError("prompt record.source is unknown") from exc
    return PromptTokenRecord(
        turn_index=_integer(row, "turn_index", "prompt record"),
        source=source,
        prompt_ids=_json_token_tuple(row.get("prompt_ids"), "prompt record.prompt_ids"),
        appended_ids=_json_token_tuple(
            row.get("appended_ids"), "prompt record.appended_ids"
        ),
    )


def _output_from_payload(value: object) -> AssistantOutputRecord:
    row = _mapping(value, "output record")
    _require_keys(row, {"turn_index", "output_ids", "segments"}, "output record")
    segments_value = row.get("segments")
    if isinstance(segments_value, (str, bytes)) or not isinstance(
        segments_value, Sequence
    ):
        raise TokenLedgerError("output record.segments must be a list")
    segments: list[AssistantTokenSegment] = []
    for item in segments_value:
        segment = _mapping(item, "assistant segment")
        _require_keys(segment, {"kind", "start", "end"}, "assistant segment")
        try:
            kind = AssistantSegmentKind(_string(segment, "kind", "assistant segment"))
        except ValueError as exc:
            raise TokenLedgerError("assistant segment.kind is unknown") from exc
        segments.append(
            AssistantTokenSegment(
                kind=kind,
                start=_integer(segment, "start", "assistant segment"),
                end=_integer(segment, "end", "assistant segment"),
            )
        )
    return AssistantOutputRecord(
        turn_index=_integer(row, "turn_index", "output record"),
        output_ids=_json_token_tuple(row.get("output_ids"), "output record.output_ids"),
        segments=tuple(segments),
    )


def _json_token_tuple(value: object, context: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TokenLedgerError(f"{context} must be a list")
    return _token_tuple(cast(Sequence[int], value), field_name=context)
