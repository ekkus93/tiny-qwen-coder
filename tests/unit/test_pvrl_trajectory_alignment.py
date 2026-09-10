from dataclasses import replace
from pathlib import Path

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
    token_ids_sha256,
)
from tiny_qwen_coder.pvrl.trajectory_alignment import (
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
    trajectory_alignment_audit_from_json,
    trajectory_alignment_audit_json,
)


def _base(suffix: str = "a") -> BaseModelIdentity:
    return BaseModelIdentity(
        repository="Qwen/Qwen3.5-4B",
        revision=suffix * 40,
        tokenizer_repository="Qwen/Qwen3.5-4B",
        tokenizer_revision="b" * 40,
    )


def _ledger(*, base: BaseModelIdentity | None = None) -> TokenLedgerSnapshot:
    base = base or _base()
    prompt0 = (10, 11, 12)
    output0 = (20, 21, 22, 23)
    delta1 = (30, 31)
    prompt1 = prompt0 + output0 + delta1
    output1 = (40, 41, 42)
    return TokenLedgerSnapshot(
        schema_version=1,
        session_id="session-001",
        identity=Qwen35TokenLedgerIdentity(
            base_model=base,
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


def _policy(*, base: BaseModelIdentity | None = None) -> RolloutPolicyIdentity:
    base = base or _base()
    return RolloutPolicyIdentity(
        base_model=base,
        adapter=AdapterIdentity(family=None, adapter_id=None),
        checkpoint_id=base.revision,
    )


def _log_probs(
    action_ids: tuple[int, ...],
    *,
    trajectory_id: str,
    policy: RolloutPolicyIdentity | None = None,
) -> RolloutLogProbEvidence:
    return RolloutLogProbEvidence(
        trajectory_id=trajectory_id,
        policy=policy or _policy(),
        action_token_ids=action_ids,
        log_probs=tuple(-0.1 - (index * 0.01) for index in range(len(action_ids))),
    )


def _aligned(*, with_log_probs: bool = True):
    ledger = _ledger()
    policy = _policy()
    bare = align_rollout_trajectory(ledger, policy=policy)
    if not with_log_probs:
        return bare
    return bind_rollout_log_probs(
        bare, tuple(-0.1 - (index * 0.01) for index in range(bare.action_token_count))
    )


def _training(*, with_log_probs: bool = True):
    return bind_trajectory_reward(
        _aligned(with_log_probs=with_log_probs),
        reward=1.0,
        reward_evidence_sha256="e" * 64,
    )


def test_alignment_derives_exact_final_input_action_ids_and_mask() -> None:
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


def test_masked_ids_are_exact_sampled_actions_in_generation_order() -> None:
    aligned = _aligned()
    masked = tuple(
        token
        for token, selected in zip(aligned.input_ids, aligned.action_mask, strict=True)
        if selected
    )
    assert masked == aligned.action_token_ids


def test_tool_observation_tokens_are_context_not_targets() -> None:
    aligned = _aligned()
    assert aligned.input_ids[7:9] == (30, 31)
    assert aligned.action_mask[7:9] == (False, False)


def test_turn_boundaries_bind_exact_prompt_and_output_hashes() -> None:
    aligned = _aligned()
    assert [(turn.input_start, turn.input_end) for turn in aligned.turns] == [
        (3, 7),
        (9, 12),
    ]
    assert [(turn.action_start, turn.action_end) for turn in aligned.turns] == [
        (0, 4),
        (4, 7),
    ]
    assert aligned.turns[0].prompt_ids_sha256 == token_ids_sha256((10, 11, 12))
    assert aligned.turns[0].output_ids_sha256 == token_ids_sha256((20, 21, 22, 23))
    assert aligned.turns[1].prompt_ids_sha256 == token_ids_sha256(aligned.input_ids[:9])
    assert aligned.turns[1].output_ids_sha256 == token_ids_sha256((40, 41, 42))


def test_semantic_segments_bind_absolute_exact_token_ranges() -> None:
    aligned = _aligned()
    assert [
        (segment.kind, segment.input_start, segment.input_end)
        for segment in aligned.segments
    ] == [
        (AssistantSegmentKind.REASONING, 3, 5),
        (AssistantSegmentKind.ACTION, 5, 7),
        (AssistantSegmentKind.TEXT, 9, 10),
        (AssistantSegmentKind.ACTION, 10, 12),
    ]
    assert [
        (segment.action_start, segment.action_end) for segment in aligned.segments
    ] == [
        (0, 2),
        (2, 4),
        (4, 5),
        (5, 7),
    ]
    for segment in aligned.segments:
        input_slice = aligned.input_ids[segment.input_start : segment.input_end]
        action_slice = aligned.action_token_ids[
            segment.action_start : segment.action_end
        ]
        assert input_slice == action_slice
        assert segment.token_ids_sha256 == token_ids_sha256(input_slice)


def test_all_policy_generated_segment_kinds_receive_loss() -> None:
    aligned = _aligned()
    for segment in aligned.segments:
        assert all(aligned.action_mask[segment.input_start : segment.input_end])


def test_log_prob_binding_factory_uses_exact_trajectory_identity() -> None:
    bare = _aligned(with_log_probs=False)
    bound = bind_rollout_log_probs(bare, tuple(-0.5 for _ in bare.action_token_ids))
    assert bound.rollout_log_probs is not None
    assert bound.rollout_log_probs.trajectory_id == bare.trajectory_id
    assert bound.rollout_log_probs.policy == bare.policy
    assert bound.rollout_log_probs.action_token_ids == bare.action_token_ids


def test_log_probs_bind_exact_policy_and_action_ids() -> None:
    aligned = _aligned()
    assert aligned.rollout_log_probs is not None
    assert aligned.rollout_log_probs.policy == aligned.policy
    assert aligned.rollout_log_probs.action_token_ids == aligned.action_token_ids
    assert len(aligned.rollout_log_probs.log_probs) == aligned.action_token_count


def test_log_prob_count_mismatch_is_rejected() -> None:
    with pytest.raises(TrajectoryAlignmentMismatchError):
        RolloutLogProbEvidence(
            trajectory_id="f" * 64,
            policy=_policy(),
            action_token_ids=(1, 2),
            log_probs=(-0.1,),
        )


def test_non_finite_or_positive_log_prob_is_rejected() -> None:
    for value in (float("nan"), float("inf"), 0.1):
        with pytest.raises(TrajectoryAlignmentError):
            RolloutLogProbEvidence(
                trajectory_id="f" * 64,
                policy=_policy(),
                action_token_ids=(1,),
                log_probs=(value,),
            )


def test_log_probs_from_other_checkpoint_are_rejected() -> None:
    aligned = _aligned(with_log_probs=False)
    adapted_policy = RolloutPolicyIdentity(
        base_model=_base(),
        adapter=AdapterIdentity(family="python", adapter_id="adapter-v1"),
        checkpoint_id="f" * 64,
    )
    evidence = _log_probs(
        aligned.action_token_ids,
        trajectory_id=aligned.trajectory_id,
        policy=adapted_policy,
    )
    with pytest.raises(TrajectoryAlignmentMismatchError):
        replace(aligned, rollout_log_probs=evidence)


def test_log_probs_for_other_sampled_tokens_are_rejected() -> None:
    aligned = _aligned(with_log_probs=False)
    wrong_ids = aligned.action_token_ids[:-1] + (999,)
    evidence = _log_probs(
        wrong_ids, trajectory_id=aligned.trajectory_id, policy=aligned.policy
    )
    with pytest.raises(TrajectoryAlignmentMismatchError):
        replace(aligned, rollout_log_probs=evidence)


def test_log_probs_from_other_prompt_history_are_rejected() -> None:
    aligned = _aligned(with_log_probs=False)
    evidence = _log_probs(
        aligned.action_token_ids, trajectory_id="f" * 64, policy=aligned.policy
    )
    with pytest.raises(TrajectoryAlignmentMismatchError):
        replace(aligned, rollout_log_probs=evidence)


def test_policy_base_must_match_ledger_base() -> None:
    with pytest.raises(TrajectoryAlignmentMismatchError):
        align_rollout_trajectory(_ledger(), policy=_policy(base=_base("f")))


def test_base_only_policy_checkpoint_must_equal_base_revision() -> None:
    with pytest.raises(TrajectoryAlignmentError):
        RolloutPolicyIdentity(
            base_model=_base(),
            adapter=AdapterIdentity(family=None, adapter_id=None),
            checkpoint_id="f" * 40,
        )


def test_adapter_policy_accepts_exact_content_addressed_checkpoint() -> None:
    policy = RolloutPolicyIdentity(
        base_model=_base(),
        adapter=AdapterIdentity(family="python", adapter_id="checkpoint-25"),
        checkpoint_id="f" * 64,
    )
    assert len(policy.policy_sha256) == 64


def test_adapter_policy_rejects_non_sha256_checkpoint_identity() -> None:
    with pytest.raises(TrajectoryAlignmentError):
        RolloutPolicyIdentity(
            base_model=_base(),
            adapter=AdapterIdentity(family="python", adapter_id="checkpoint-25"),
            checkpoint_id="f" * 40,
        )


def test_different_policy_checkpoint_changes_trajectory_id() -> None:
    ledger = _ledger()
    p1 = RolloutPolicyIdentity(
        base_model=_base(),
        adapter=AdapterIdentity(family="python", adapter_id="a"),
        checkpoint_id="e" * 64,
    )
    p2 = replace(p1, checkpoint_id="f" * 64)
    assert (
        align_rollout_trajectory(ledger, policy=p1).trajectory_id
        != align_rollout_trajectory(ledger, policy=p2).trajectory_id
    )


def test_log_prob_values_do_not_change_sampled_trajectory_identity() -> None:
    aligned = _aligned(with_log_probs=False)
    first = replace(
        aligned,
        rollout_log_probs=RolloutLogProbEvidence(
            aligned.trajectory_id,
            aligned.policy,
            aligned.action_token_ids,
            tuple(-0.1 for _ in aligned.action_token_ids),
        ),
    )
    second = replace(
        aligned,
        rollout_log_probs=RolloutLogProbEvidence(
            aligned.trajectory_id,
            aligned.policy,
            aligned.action_token_ids,
            tuple(-0.2 for _ in aligned.action_token_ids),
        ),
    )
    assert first.trajectory_id == second.trajectory_id
    assert first.rollout_log_probs is not None
    assert second.rollout_log_probs is not None
    assert (
        first.rollout_log_probs.evidence_sha256
        != second.rollout_log_probs.evidence_sha256
    )


def test_action_mask_rejects_integer_aliases_for_booleans() -> None:
    aligned = _aligned()
    bad_mask = tuple(1 if value else 0 for value in aligned.action_mask)
    with pytest.raises(TrajectoryAlignmentError):
        replace(aligned, action_mask=bad_mask)


@pytest.mark.parametrize(
    "field", ["input_ids", "action_token_ids", "action_mask", "turns", "segments"]
)
def test_manual_alignment_tampering_is_rejected(field: str) -> None:
    aligned = _aligned()
    if field == "input_ids":
        value = aligned.input_ids[:-1] + (999,)
    elif field == "action_token_ids":
        value = aligned.action_token_ids[:-1] + (999,)
    elif field == "action_mask":
        value = aligned.action_mask[:-1] + (False,)
    elif field == "turns":
        value = (replace(aligned.turns[0], input_end=6, action_end=3),) + aligned.turns[
            1:
        ]
    else:
        value = (
            replace(aligned.segments[0], input_end=4, action_end=1),
        ) + aligned.segments[1:]
    with pytest.raises((TrajectoryAlignmentError, TrajectoryAlignmentMismatchError)):
        replace(aligned, **{field: value})


def test_incomplete_ledger_is_rejected() -> None:
    ledger = _ledger()
    incomplete = replace(
        ledger,
        outputs=ledger.outputs[:1],
        state=TokenLedgerState.AWAITING_OUTPUT,
    )
    with pytest.raises(ZeroActionTrajectoryError):
        align_rollout_trajectory(incomplete, policy=_policy())


def test_zero_output_ledger_is_rejected() -> None:
    ledger = _ledger()
    empty = replace(
        ledger,
        prompts=ledger.prompts[:1],
        outputs=(),
        state=TokenLedgerState.AWAITING_OUTPUT,
    )
    with pytest.raises(ZeroActionTrajectoryError):
        align_rollout_trajectory(empty, policy=_policy())


def test_reward_is_bound_to_exact_trajectory_id() -> None:
    training = _training()
    assert training.reward.trajectory_id == training.alignment.trajectory_id
    assert training.reward.reward == 1.0


def test_reward_attribution_for_other_trajectory_is_rejected() -> None:
    aligned = _aligned()
    wrong = TrajectoryRewardAttribution(
        trajectory_id="f" * 64,
        reward=1.0,
        reward_evidence_sha256="e" * 64,
    )
    with pytest.raises(TrajectoryAlignmentMismatchError):
        AlignedTrainingTrajectory(alignment=aligned, reward=wrong)


def test_non_finite_reward_is_rejected() -> None:
    with pytest.raises(TrajectoryAlignmentError):
        TrajectoryRewardAttribution(
            trajectory_id="f" * 64,
            reward=float("nan"),
            reward_evidence_sha256="e" * 64,
        )


def test_compact_audit_contains_hashes_and_boundaries_not_raw_arrays() -> None:
    audit = create_trajectory_alignment_audit(_training())
    text = trajectory_alignment_audit_json(audit)
    assert '"input_ids":' not in text
    assert '"action_token_ids":' not in text
    assert '"action_mask":' not in text
    assert '"log_probs":' not in text
    assert audit.action_token_count == 7
    assert [(turn.input_start, turn.input_end) for turn in audit.turns] == [
        (3, 7),
        (9, 12),
    ]
    assert audit.rollout_log_prob_count == 7
    assert audit.rollout_log_probs_sha256 is not None


def test_compact_audit_round_trips_deterministically() -> None:
    audit = create_trajectory_alignment_audit(_training())
    text = trajectory_alignment_audit_json(audit)
    restored = trajectory_alignment_audit_from_json(text)
    assert restored == audit
    assert trajectory_alignment_audit_json(restored) == text
    assert restored.audit_sha256 == audit.audit_sha256


def test_representative_alignment_audit_fixture_is_frozen() -> None:
    audit = create_trajectory_alignment_audit(_training())
    fixture = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "pvrl_203_alignment_audit_v1.json"
    )
    assert fixture.read_text(encoding="utf-8") == trajectory_alignment_audit_json(audit)


def test_audit_without_log_probs_records_explicit_absence() -> None:
    audit = create_trajectory_alignment_audit(_training(with_log_probs=False))
    assert audit.rollout_log_prob_count == 0
    assert audit.rollout_log_probs_sha256 is None


def test_audit_parser_rejects_unknown_fields() -> None:
    text = trajectory_alignment_audit_json(
        create_trajectory_alignment_audit(_training())
    )
    bad = text[:-2] + ',"unexpected":true}\n'
    with pytest.raises(TrajectoryAlignmentError):
        trajectory_alignment_audit_from_json(bad)


def test_audit_rejects_log_prob_count_not_equal_action_count() -> None:
    audit = create_trajectory_alignment_audit(_training())
    with pytest.raises(TrajectoryAlignmentMismatchError):
        replace(audit, rollout_log_prob_count=audit.action_token_count - 1)


def test_audit_rejects_trajectory_id_not_derived_from_ledger_and_policy() -> None:
    audit = create_trajectory_alignment_audit(_training())
    with pytest.raises(TrajectoryAlignmentMismatchError):
        replace(audit, trajectory_id="f" * 64)


def test_audit_rejects_turn_or_segment_boundary_drift() -> None:
    audit = create_trajectory_alignment_audit(_training())
    with pytest.raises(TrajectoryAlignmentMismatchError):
        replace(
            audit,
            turns=(replace(audit.turns[0], input_end=6, action_end=3),)
            + audit.turns[1:],
        )
    with pytest.raises(TrajectoryAlignmentMismatchError):
        replace(
            audit,
            segments=(replace(audit.segments[0], input_end=4, action_end=1),)
            + audit.segments[1:],
        )


def test_audit_rejects_action_mask_hash_not_implied_by_turn_spans() -> None:
    audit = create_trajectory_alignment_audit(_training())
    with pytest.raises(TrajectoryAlignmentMismatchError):
        replace(audit, action_mask_sha256="f" * 64)


def test_audit_hash_changes_with_reward_evidence() -> None:
    audit = create_trajectory_alignment_audit(_training())
    changed = replace(audit, reward_evidence_sha256="f" * 64)
    assert audit.audit_sha256 != changed.audit_sha256


def test_audit_constructor_rejects_zero_action_count() -> None:
    audit = create_trajectory_alignment_audit(_training())
    with pytest.raises(ZeroActionTrajectoryError):
        replace(audit, action_token_count=0)
