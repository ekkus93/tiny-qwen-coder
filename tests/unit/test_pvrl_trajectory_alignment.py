from dataclasses import replace

import pytest

from tiny_qwen_coder.identities import AdapterIdentity, BaseModelIdentity
from tiny_qwen_coder.pvrl.token_ledger import (
    AssistantOutputRecord,
    AssistantSegmentKind,
    AssistantTokenSegment,
    PromptAppendSource,
    PromptTokenRecord,
    Qwen35TokenLedgerIdentity,
    TokenLedgerSnapshot,
    TokenLedgerState,
)
from tiny_qwen_coder.pvrl.trajectory_alignment import (
    AlignedRolloutTrajectory,
    AlignedTrainingTrajectory,
    RolloutLogProbEvidence,
    RolloutPolicyIdentity,
    TrajectoryAlignmentError,
    TrajectoryAlignmentMismatchError,
    TrajectoryRewardAttribution,
    ZeroActionTrajectoryError,
    align_rollout_trajectory,
    bind_rollout_log_probs,
    bind_trajectory_reward,
    create_trajectory_alignment_audit,
    trajectory_alignment_audit_json,
)


def _base(char: str = "a") -> BaseModelIdentity:
    return BaseModelIdentity(
        repository="Qwen/Qwen3.5-4B",
        revision=char * 40,
        tokenizer_repository="Qwen/Qwen3.5-4B",
        tokenizer_revision="b" * 40,
    )


def _policy(*, base: BaseModelIdentity | None = None) -> RolloutPolicyIdentity:
    exact_base = base or _base()
    return RolloutPolicyIdentity(
        base_model=exact_base,
        adapter=AdapterIdentity(family=None, adapter_id=None),
        checkpoint_id=exact_base.revision,
    )


def _ledger() -> TokenLedgerSnapshot:
    prompt0 = (10, 11, 12)
    output0 = (20, 21, 22, 23)
    delta1 = (30, 31)
    prompt1 = prompt0 + output0 + delta1
    output1 = (40, 41, 42)
    return TokenLedgerSnapshot(
        schema_version=1,
        session_id="session-001",
        identity=Qwen35TokenLedgerIdentity(
            base_model=_base(),
            chat_template_sha256="c" * 64,
            leaf_lite_contract_sha256="d" * 64,
            enable_thinking=True,
        ),
        prompts=(
            PromptTokenRecord(0, PromptAppendSource.INITIAL, prompt0, prompt0),
            PromptTokenRecord(1, PromptAppendSource.TOOL, prompt1, delta1),
        ),
        outputs=(
            AssistantOutputRecord(
                0,
                output0,
                (
                    AssistantTokenSegment(AssistantSegmentKind.REASONING, 0, 2),
                    AssistantTokenSegment(AssistantSegmentKind.ACTION, 2, 4),
                ),
            ),
            AssistantOutputRecord(
                1,
                output1,
                (
                    AssistantTokenSegment(AssistantSegmentKind.TEXT, 0, 1),
                    AssistantTokenSegment(AssistantSegmentKind.ACTION, 1, 3),
                ),
            ),
        ),
        state=TokenLedgerState.AWAITING_PROMPT_APPEND,
    )


def _aligned(*, log_probs: bool = True) -> AlignedRolloutTrajectory:
    aligned = align_rollout_trajectory(_ledger(), policy=_policy())
    if not log_probs:
        return aligned
    return bind_rollout_log_probs(aligned, [-0.1] * len(aligned.action_token_ids))


def _training(*, log_probs: bool = True) -> AlignedTrainingTrajectory:
    return bind_trajectory_reward(
        _aligned(log_probs=log_probs),
        reward=1.0,
        reward_evidence_sha256="e" * 64,
    )


def test_exact_final_input_action_ids_and_mask_are_ledger_derived() -> None:
    aligned = _aligned()
    assert aligned.input_ids == (10, 11, 12, 20, 21, 22, 23, 30, 31, 40, 41, 42)
    assert aligned.action_token_ids == (20, 21, 22, 23, 40, 41, 42)
    assert aligned.action_mask == (
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        False,
        False,
        True,
        True,
        True,
    )


def test_loss_mask_recovers_exact_actions_in_generation_order() -> None:
    aligned = _aligned()
    masked = tuple(
        token
        for token, selected in zip(aligned.input_ids, aligned.action_mask, strict=True)
        if selected
    )
    assert masked == aligned.action_token_ids


def test_tool_observation_tokens_are_context_only() -> None:
    aligned = _aligned()
    assert aligned.input_ids[7:9] == (30, 31)
    assert aligned.action_mask[7:9] == (False, False)


def test_turn_boundaries_cover_exact_assistant_outputs() -> None:
    aligned = _aligned()
    assert [(turn.input_start, turn.input_end) for turn in aligned.turns] == [
        (3, 7),
        (9, 12),
    ]
    assert [(turn.action_start, turn.action_end) for turn in aligned.turns] == [
        (0, 4),
        (4, 7),
    ]


def test_reasoning_text_and_action_segments_map_exactly() -> None:
    aligned = _aligned()
    assert [segment.kind for segment in aligned.segments] == [
        AssistantSegmentKind.REASONING,
        AssistantSegmentKind.ACTION,
        AssistantSegmentKind.TEXT,
        AssistantSegmentKind.ACTION,
    ]
    for segment in aligned.segments:
        assert (
            aligned.input_ids[segment.input_start : segment.input_end]
            == aligned.action_token_ids[segment.action_start : segment.action_end]
        )
        assert all(aligned.action_mask[segment.input_start : segment.input_end])


def test_log_probs_bind_trajectory_policy_actions_and_count() -> None:
    aligned = _aligned()
    evidence = aligned.rollout_log_probs
    assert evidence is not None
    assert evidence.trajectory_id == aligned.trajectory_id
    assert evidence.policy_sha256 == aligned.policy.policy_sha256
    assert len(evidence.log_probs) == len(aligned.action_token_ids)


def test_log_prob_count_mismatch_rejected_without_repair() -> None:
    aligned = _aligned(log_probs=False)
    with pytest.raises(TrajectoryAlignmentMismatchError):
        bind_rollout_log_probs(aligned, [-0.1] * (len(aligned.action_token_ids) - 1))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.01])
def test_invalid_log_prob_rejected(bad: float) -> None:
    aligned = _aligned(log_probs=False)
    values = [-0.1] * len(aligned.action_token_ids)
    values[0] = bad
    with pytest.raises(TrajectoryAlignmentError):
        bind_rollout_log_probs(aligned, values)


def test_log_probs_from_other_trajectory_rejected() -> None:
    aligned = _aligned(log_probs=False)
    foreign = RolloutLogProbEvidence(
        trajectory_id="f" * 64,
        policy_sha256=aligned.policy.policy_sha256,
        action_token_ids_sha256="e" * 64,
        log_probs=tuple([-0.1] * len(aligned.action_token_ids)),
    )
    with pytest.raises(TrajectoryAlignmentMismatchError):
        replace(aligned, rollout_log_probs=foreign)


def test_log_probs_from_other_policy_rejected() -> None:
    aligned = _aligned(log_probs=False)
    foreign = RolloutLogProbEvidence(
        trajectory_id=aligned.trajectory_id,
        policy_sha256="f" * 64,
        action_token_ids_sha256="e" * 64,
        log_probs=tuple([-0.1] * len(aligned.action_token_ids)),
    )
    with pytest.raises(TrajectoryAlignmentMismatchError):
        replace(aligned, rollout_log_probs=foreign)


def test_policy_base_model_must_match_ledger() -> None:
    with pytest.raises(TrajectoryAlignmentMismatchError):
        align_rollout_trajectory(_ledger(), policy=_policy(base=_base("f")))


def test_base_only_checkpoint_must_match_base_revision() -> None:
    with pytest.raises(TrajectoryAlignmentMismatchError):
        RolloutPolicyIdentity(
            base_model=_base(),
            adapter=AdapterIdentity(family=None, adapter_id=None),
            checkpoint_id="f" * 40,
        )


def test_adapted_policy_requires_content_addressed_checkpoint() -> None:
    with pytest.raises(TrajectoryAlignmentError):
        RolloutPolicyIdentity(
            base_model=_base(),
            adapter=AdapterIdentity(family="python", adapter_id="adapter-v1"),
            checkpoint_id="f" * 40,
        )
    policy = RolloutPolicyIdentity(
        base_model=_base(),
        adapter=AdapterIdentity(family="python", adapter_id="adapter-v1"),
        checkpoint_id="f" * 64,
    )
    assert len(policy.policy_sha256) == 64


def test_checkpoint_change_changes_trajectory_id() -> None:
    ledger = _ledger()
    one = RolloutPolicyIdentity(
        base_model=_base(),
        adapter=AdapterIdentity(family="python", adapter_id="a"),
        checkpoint_id="e" * 64,
    )
    two = replace(one, checkpoint_id="f" * 64)
    assert (
        align_rollout_trajectory(ledger, policy=one).trajectory_id
        != align_rollout_trajectory(ledger, policy=two).trajectory_id
    )


def test_reward_is_bound_to_exact_trajectory() -> None:
    training = _training()
    assert training.reward.trajectory_id == training.alignment.trajectory_id


def test_foreign_reward_attribution_is_rejected() -> None:
    aligned = _aligned()
    reward = TrajectoryRewardAttribution("f" * 64, 1.0, "e" * 64)
    with pytest.raises(TrajectoryAlignmentMismatchError):
        AlignedTrainingTrajectory(aligned, reward)


def test_input_id_tampering_is_rejected() -> None:
    aligned = _aligned()
    with pytest.raises(TrajectoryAlignmentMismatchError):
        replace(aligned, input_ids=aligned.input_ids[:-1] + (999,))


def test_action_id_tampering_is_rejected() -> None:
    aligned = _aligned()
    with pytest.raises(TrajectoryAlignmentMismatchError):
        replace(aligned, action_token_ids=aligned.action_token_ids[:-1] + (999,))


def test_action_mask_tampering_is_rejected() -> None:
    aligned = _aligned()
    with pytest.raises(TrajectoryAlignmentMismatchError):
        replace(aligned, action_mask=aligned.action_mask[:-1] + (False,))


def test_turn_boundary_tampering_is_rejected() -> None:
    aligned = _aligned()
    turns = (replace(aligned.turns[0], input_end=6, action_end=3),) + aligned.turns[1:]
    with pytest.raises((TrajectoryAlignmentError, TrajectoryAlignmentMismatchError)):
        replace(aligned, turns=turns)


def test_incomplete_and_zero_action_ledgers_are_rejected() -> None:
    ledger = _ledger()
    incomplete = replace(
        ledger,
        outputs=ledger.outputs[:1],
        state=TokenLedgerState.AWAITING_OUTPUT,
    )
    with pytest.raises(ZeroActionTrajectoryError):
        align_rollout_trajectory(incomplete, policy=_policy())
    empty = replace(
        ledger,
        prompts=ledger.prompts[:1],
        outputs=(),
        state=TokenLedgerState.AWAITING_OUTPUT,
    )
    with pytest.raises(ZeroActionTrajectoryError):
        align_rollout_trajectory(empty, policy=_policy())


def test_compact_audit_omits_raw_training_arrays() -> None:
    audit = create_trajectory_alignment_audit(_training())
    text = trajectory_alignment_audit_json(audit)
    assert '"input_ids":' not in text
    assert '"action_token_ids":' not in text
    assert '"action_mask":' not in text
    assert '"log_probs":' not in text
    assert audit.action_token_count == 7
    assert audit.rollout_log_prob_count == 7


def test_compact_audit_is_deterministic_and_content_addressed() -> None:
    first = create_trajectory_alignment_audit(_training())
    second = create_trajectory_alignment_audit(_training())
    assert trajectory_alignment_audit_json(first) == trajectory_alignment_audit_json(second)
    assert first.audit_sha256 == second.audit_sha256


def test_audit_without_log_probs_records_explicit_absence() -> None:
    audit = create_trajectory_alignment_audit(_training(log_probs=False))
    assert audit.rollout_log_prob_count == 0
    assert audit.rollout_log_probs_sha256 is None


def test_audit_rejects_count_and_mask_drift() -> None:
    audit = create_trajectory_alignment_audit(_training())
    with pytest.raises(TrajectoryAlignmentMismatchError):
        replace(audit, rollout_log_prob_count=6)
    with pytest.raises(TrajectoryAlignmentMismatchError):
        replace(audit, action_mask_sha256="f" * 64)
