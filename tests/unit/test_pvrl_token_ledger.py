from __future__ import annotations

import json
from dataclasses import replace

import pytest

from tiny_qwen_coder.identities import BaseModelIdentity
from tiny_qwen_coder.pvrl.token_ledger import (
    AssistantSegmentKind,
    AssistantTokenSegment,
    ExactQwen35TokenLedger,
    PromptAppendSource,
    Qwen35TokenLedgerIdentity,
    TokenLedgerError,
    TokenLedgerForkError,
    TokenLedgerSnapshot,
    TokenLedgerState,
    TokenLedgerStateError,
    token_ids_sha256,
    token_ledger_snapshot_from_json,
    token_ledger_snapshot_json,
)

_SHA40 = "a" * 40
_SHA64 = "b" * 64
_OTHER64 = "c" * 64


def _identity() -> Qwen35TokenLedgerIdentity:
    return Qwen35TokenLedgerIdentity(
        base_model=BaseModelIdentity(
            repository="Qwen/Qwen3.5-4B",
            revision=_SHA40,
            tokenizer_repository="Qwen/Qwen3.5-4B",
            tokenizer_revision=_SHA40,
        ),
        chat_template_sha256=_SHA64,
        leaf_lite_contract_sha256=_OTHER64,
        enable_thinking=True,
    )


def _ledger() -> ExactQwen35TokenLedger:
    return ExactQwen35TokenLedger(
        session_id="session-001",
        identity=_identity(),
        initial_prompt_ids=(10, 11, 12),
    )


def test_initial_prompt_is_exact_and_awaits_output() -> None:
    ledger = _ledger()

    assert ledger.current_prompt_ids == (10, 11, 12)
    assert ledger.previous_output_ids is None
    assert ledger.state is TokenLedgerState.AWAITING_OUTPUT
    snapshot = ledger.snapshot()
    assert snapshot.prompts[0].appended_ids == snapshot.prompts[0].prompt_ids
    assert snapshot.prompts[0].source is PromptAppendSource.INITIAL


def test_raw_assistant_ids_are_preserved_without_text_round_trip() -> None:
    ledger = _ledger()
    output = (101, 202, 303, 404)

    record = ledger.record_assistant_output(output)

    assert record.output_ids == output
    assert ledger.previous_output_ids == output
    assert record.segments == (AssistantTokenSegment(AssistantSegmentKind.TEXT, 0, 4),)
    assert ledger.state is TokenLedgerState.AWAITING_PROMPT_APPEND


def test_append_delta_constructs_exact_uncompacted_prefix() -> None:
    ledger = _ledger()
    ledger.record_assistant_output((20, 21))

    prompt = ledger.append_prompt_delta((30, 31, 32), source=PromptAppendSource.TOOL)

    assert prompt.prompt_ids == (10, 11, 12, 20, 21, 30, 31, 32)
    assert prompt.appended_ids == (30, 31, 32)
    assert ledger.current_prompt_ids == prompt.prompt_ids
    assert ledger.state is TokenLedgerState.AWAITING_OUTPUT


def test_verified_next_prompt_accepts_only_exact_history_prefix() -> None:
    ledger = _ledger()
    ledger.record_assistant_output((20, 21))

    record = ledger.adopt_verified_next_prompt(
        (10, 11, 12, 20, 21, 40, 41),
        source=PromptAppendSource.USER,
    )

    assert record.appended_ids == (40, 41)


def test_retemplated_historical_assistant_ids_are_fatal_fork() -> None:
    ledger = _ledger()
    ledger.record_assistant_output((20, 21, 22))

    # Simulates a chat-template round trip that changes one historical assistant token.
    retemplated = (10, 11, 12, 20, 999, 22, 30)
    with pytest.raises(TokenLedgerForkError, match="prefix invariant"):
        ledger.adopt_verified_next_prompt(retemplated, source=PromptAppendSource.TOOL)

    assert ledger.current_prompt_ids == (10, 11, 12)
    assert ledger.previous_output_ids == (20, 21, 22)


def test_normalizing_assistant_whitespace_cannot_replace_original_ids() -> None:
    ledger = _ledger()
    # Treat 501/502 as exact whitespace-sensitive output tokens.
    ledger.record_assistant_output((50, 501, 502, 51))

    normalized_history = (10, 11, 12, 50, 503, 51, 60)
    with pytest.raises(TokenLedgerForkError):
        ledger.adopt_verified_next_prompt(normalized_history, source=PromptAppendSource.USER)


def test_structured_tool_call_tokens_remain_exact_action_tokens() -> None:
    ledger = _ledger()
    exact_tool_call_ids = (700, 701, 702, 703, 704, 705)

    record = ledger.record_assistant_output(
        exact_tool_call_ids,
        segments=(AssistantTokenSegment(AssistantSegmentKind.ACTION, 0, 6),),
    )

    assert record.output_ids == exact_tool_call_ids
    assert record.segments[0].kind is AssistantSegmentKind.ACTION


def test_reasoning_and_action_spans_partition_exact_output() -> None:
    ledger = _ledger()
    output_ids = (1, 2, 3, 4, 5, 6, 7)

    record = ledger.record_assistant_output(
        output_ids,
        segments=(
            AssistantTokenSegment(AssistantSegmentKind.REASONING, 0, 4),
            AssistantTokenSegment(AssistantSegmentKind.ACTION, 4, 7),
        ),
    )

    assert record.output_ids[record.segments[0].start : record.segments[0].end] == (
        1,
        2,
        3,
        4,
    )
    assert record.output_ids[record.segments[1].start : record.segments[1].end] == (
        5,
        6,
        7,
    )


def test_segment_gaps_overlaps_and_partial_coverage_fail_closed() -> None:
    ledger = _ledger()
    bad_sets = (
        (AssistantTokenSegment(AssistantSegmentKind.REASONING, 1, 3),),
        (
            AssistantTokenSegment(AssistantSegmentKind.REASONING, 0, 2),
            AssistantTokenSegment(AssistantSegmentKind.ACTION, 1, 3),
        ),
        (AssistantTokenSegment(AssistantSegmentKind.TEXT, 0, 2),),
    )
    for segments in bad_sets:
        with pytest.raises(TokenLedgerError):
            ledger.record_assistant_output((1, 2, 3), segments=segments)


def test_state_machine_prevents_double_output_and_prompt_without_output() -> None:
    ledger = _ledger()
    with pytest.raises(TokenLedgerStateError):
        ledger.append_prompt_delta((9,), source=PromptAppendSource.USER)

    ledger.record_assistant_output((20,))
    with pytest.raises(TokenLedgerStateError):
        ledger.record_assistant_output((21,))


def test_initial_source_cannot_be_reused_for_continuation() -> None:
    ledger = _ledger()
    ledger.record_assistant_output((20,))

    with pytest.raises(TokenLedgerError, match="user/tool"):
        ledger.append_prompt_delta((30,), source=PromptAppendSource.INITIAL)


def test_non_positive_or_boolean_token_ids_fail_closed() -> None:
    with pytest.raises(TokenLedgerError):
        ExactQwen35TokenLedger(
            session_id="x",
            identity=_identity(),
            initial_prompt_ids=(1, -1),
        )
    with pytest.raises(TokenLedgerError):
        ExactQwen35TokenLedger(
            session_id="x",
            identity=_identity(),
            initial_prompt_ids=(1, True),
        )


def test_empty_prompt_output_and_append_are_rejected() -> None:
    with pytest.raises(TokenLedgerError):
        ExactQwen35TokenLedger(session_id="x", identity=_identity(), initial_prompt_ids=())
    ledger = _ledger()
    with pytest.raises(TokenLedgerError):
        ledger.record_assistant_output(())
    ledger.record_assistant_output((2,))
    with pytest.raises(TokenLedgerError):
        ledger.append_prompt_delta((), source=PromptAppendSource.USER)


def test_context_compaction_is_explicitly_disabled_in_v0() -> None:
    with pytest.raises(TokenLedgerError, match="does not support context compaction"):
        replace(_identity(), compaction_enabled=True)


def test_snapshot_round_trip_preserves_every_exact_integer() -> None:
    ledger = _ledger()
    ledger.record_assistant_output(
        (101, 102, 103),
        segments=(
            AssistantTokenSegment(AssistantSegmentKind.REASONING, 0, 1),
            AssistantTokenSegment(AssistantSegmentKind.ACTION, 1, 3),
        ),
    )
    ledger.append_prompt_delta((201, 202), source=PromptAppendSource.TOOL)
    ledger.record_assistant_output((301, 302))
    snapshot = ledger.snapshot()

    serialized = token_ledger_snapshot_json(snapshot)
    restored = token_ledger_snapshot_from_json(serialized)

    assert restored == snapshot
    assert restored.ledger_sha256 == snapshot.ledger_sha256
    assert token_ledger_snapshot_json(restored) == serialized


def test_snapshot_json_is_canonical_compact_ascii_plus_newline() -> None:
    snapshot = _ledger().snapshot()
    serialized = token_ledger_snapshot_json(snapshot)

    assert serialized.endswith("\n")
    assert " " not in serialized
    assert json.loads(serialized)["session_id"] == "session-001"


def test_persisted_retemplating_tamper_is_rejected() -> None:
    ledger = _ledger()
    ledger.record_assistant_output((20, 21))
    ledger.append_prompt_delta((30,), source=PromptAppendSource.TOOL)
    payload = json.loads(token_ledger_snapshot_json(ledger.snapshot()))
    payload["prompts"][1]["prompt_ids"][4] = 999

    with pytest.raises(TokenLedgerForkError, match="prefix invariant"):
        token_ledger_snapshot_from_json(json.dumps(payload))


def test_persisted_state_tamper_is_rejected() -> None:
    payload = json.loads(token_ledger_snapshot_json(_ledger().snapshot()))
    payload["state"] = "awaiting_prompt_append"

    with pytest.raises(TokenLedgerError, match="state is inconsistent"):
        token_ledger_snapshot_from_json(json.dumps(payload))


def test_unknown_json_fields_fail_closed() -> None:
    payload = json.loads(token_ledger_snapshot_json(_ledger().snapshot()))
    payload["assistant_text"] = "must never become authoritative"

    with pytest.raises(TokenLedgerError, match="keys do not match schema"):
        token_ledger_snapshot_from_json(json.dumps(payload))


def test_restore_from_snapshot_continues_append_only_without_retokenizing_history() -> None:
    ledger = _ledger()
    ledger.record_assistant_output((20, 21))
    snapshot = ledger.snapshot()

    restored = ExactQwen35TokenLedger.from_snapshot(snapshot)
    restored.append_prompt_delta((30, 31), source=PromptAppendSource.TOOL)

    assert restored.current_prompt_ids == (10, 11, 12, 20, 21, 30, 31)


def test_token_hash_is_order_sensitive_and_stable() -> None:
    assert token_ids_sha256((1, 2, 3)) == token_ids_sha256([1, 2, 3])
    assert token_ids_sha256((1, 2, 3)) != token_ids_sha256((1, 3, 2))


def test_identity_binds_template_tool_contract_and_thinking_mode() -> None:
    base = _ledger().snapshot()
    variants = (
        replace(base.identity, chat_template_sha256="d" * 64),
        replace(base.identity, leaf_lite_contract_sha256="e" * 64),
        replace(base.identity, enable_thinking=False),
    )
    assert (
        len(
            {
                base.ledger_sha256,
                *(replace(base, identity=item).ledger_sha256 for item in variants),
            }
        )
        == 4
    )


def test_snapshot_rejects_manual_cross_turn_fork_even_if_counts_look_valid() -> None:
    ledger = _ledger()
    ledger.record_assistant_output((20,))
    ledger.append_prompt_delta((30,), source=PromptAppendSource.USER)
    snapshot = ledger.snapshot()
    bad_prompt = replace(snapshot.prompts[1], prompt_ids=(10, 11, 12, 999, 30))

    with pytest.raises(TokenLedgerForkError):
        TokenLedgerSnapshot(
            schema_version=snapshot.schema_version,
            session_id=snapshot.session_id,
            identity=snapshot.identity,
            prompts=(snapshot.prompts[0], bad_prompt),
            outputs=snapshot.outputs,
            state=snapshot.state,
        )


@pytest.mark.parametrize(
    ("case", "emitted", "retemplated"),
    [
        ("assistant_text", (100, 101, 102), (100, 999, 102)),
        ("tool_call", (200, 201, 202, 203), (200, 201, 998, 203)),
        ("whitespace", (300, 301, 302), (300, 997, 302)),
        ("structured_arguments", (400, 401, 402, 403), (400, 401, 996, 403)),
        ("reasoning_block", (500, 501, 502, 503, 504), (500, 501, 995, 503, 504)),
    ],
)
def test_regression_retemplating_drift_cases_fail_closed(
    case: str, emitted: tuple[int, ...], retemplated: tuple[int, ...]
) -> None:
    ledger = _ledger()
    ledger.record_assistant_output(emitted)

    candidate = ledger.current_prompt_ids + retemplated + (600, 601)
    with pytest.raises(TokenLedgerForkError, match="prefix invariant"):
        ledger.adopt_verified_next_prompt(candidate, source=PromptAppendSource.TOOL)

    assert case
    assert ledger.previous_output_ids == emitted
