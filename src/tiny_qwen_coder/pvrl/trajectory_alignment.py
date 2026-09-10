"""Fail-closed rollout/log-prob/loss-mask alignment for PVRL trajectories."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace

from tiny_qwen_coder.identities import AdapterIdentity, BaseModelIdentity
from tiny_qwen_coder.pvrl.token_ledger import (
    AssistantSegmentKind,
    TokenLedgerSnapshot,
    TokenLedgerState,
    token_ids_sha256,
)

_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CHECKPOINT = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


class TrajectoryAlignmentError(RuntimeError):
    """Invalid trajectory material or alignment evidence."""


class TrajectoryAlignmentMismatchError(TrajectoryAlignmentError):
    """Stored rollout material disagrees with the exact token ledger."""


class ZeroActionTrajectoryError(TrajectoryAlignmentError):
    """Trajectory contains no policy-generated tokens that can receive loss."""


@dataclass(frozen=True, slots=True)
class RolloutPolicyIdentity:
    """Exact base/adapter/checkpoint identity that generated one rollout."""

    base_model: BaseModelIdentity
    adapter: AdapterIdentity
    checkpoint_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.base_model, BaseModelIdentity):
            raise TrajectoryAlignmentError("base_model must be BaseModelIdentity")
        if not isinstance(self.adapter, AdapterIdentity):
            raise TrajectoryAlignmentError("adapter must be AdapterIdentity")
        if not isinstance(self.checkpoint_id, str) or not _CHECKPOINT.fullmatch(
            self.checkpoint_id
        ):
            raise TrajectoryAlignmentError("checkpoint_id must be immutable 40/64-hex")
        if self.adapter.adapter_id is None:
            if self.checkpoint_id != self.base_model.revision:
                raise TrajectoryAlignmentMismatchError(
                    "base-only checkpoint_id must equal base revision"
                )
        elif not _SHA256.fullmatch(self.checkpoint_id):
            raise TrajectoryAlignmentError("adapted checkpoint_id must be SHA-256")

    @property
    def policy_sha256(self) -> str:
        return _hash_json(_policy_payload(self))


@dataclass(frozen=True, slots=True)
class ActionTurnBoundary:
    """Exact input and flattened-action spans for one assistant turn."""

    turn_index: int
    input_start: int
    input_end: int
    action_start: int
    action_end: int
    prompt_ids_sha256: str
    output_ids_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "turn_index",
            "input_start",
            "input_end",
            "action_start",
            "action_end",
        ):
            _non_negative_int(getattr(self, name), name)
        if self.input_end <= self.input_start or self.action_end <= self.action_start:
            raise TrajectoryAlignmentError("turn spans must be non-empty")
        if self.input_end - self.input_start != self.action_end - self.action_start:
            raise TrajectoryAlignmentMismatchError(
                "turn input/action span lengths differ"
            )
        _sha(self.prompt_ids_sha256, "prompt_ids_sha256")
        _sha(self.output_ids_sha256, "output_ids_sha256")


@dataclass(frozen=True, slots=True)
class AlignedActionSegment:
    """Semantic ledger segment mapped to input and flattened-action coordinates."""

    turn_index: int
    segment_index: int
    kind: AssistantSegmentKind
    input_start: int
    input_end: int
    action_start: int
    action_end: int
    token_ids_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "turn_index",
            "segment_index",
            "input_start",
            "input_end",
            "action_start",
            "action_end",
        ):
            _non_negative_int(getattr(self, name), name)
        if not isinstance(self.kind, AssistantSegmentKind):
            raise TrajectoryAlignmentError("kind must be AssistantSegmentKind")
        if self.input_end <= self.input_start or self.action_end <= self.action_start:
            raise TrajectoryAlignmentError("segment spans must be non-empty")
        if self.input_end - self.input_start != self.action_end - self.action_start:
            raise TrajectoryAlignmentMismatchError(
                "segment input/action span lengths differ"
            )
        _sha(self.token_ids_sha256, "token_ids_sha256")


@dataclass(frozen=True, slots=True)
class RolloutLogProbEvidence:
    """Per-action log probabilities bound to exact trajectory and policy identity."""

    trajectory_id: str
    policy_sha256: str
    action_token_ids_sha256: str
    log_probs: tuple[float, ...]

    def __post_init__(self) -> None:
        _sha(self.trajectory_id, "trajectory_id")
        _sha(self.policy_sha256, "policy_sha256")
        _sha(self.action_token_ids_sha256, "action_token_ids_sha256")
        if not self.log_probs:
            raise TrajectoryAlignmentError("log_probs must not be empty")
        for index, value in enumerate(self.log_probs):
            if not isinstance(value, float) or not math.isfinite(value) or value > 0.0:
                raise TrajectoryAlignmentError(
                    f"log_probs[{index}] must be a finite float <= 0"
                )

    @property
    def evidence_sha256(self) -> str:
        return _hash_json(
            {
                "trajectory_id": self.trajectory_id,
                "policy_sha256": self.policy_sha256,
                "action_token_ids_sha256": self.action_token_ids_sha256,
                "log_probs": list(self.log_probs),
            }
        )


@dataclass(frozen=True, slots=True)
class AlignedRolloutTrajectory:
    """Ledger-derived rollout arrays eligible for reward binding."""

    schema_version: int
    ledger: TokenLedgerSnapshot
    policy: RolloutPolicyIdentity
    input_ids: tuple[int, ...]
    action_token_ids: tuple[int, ...]
    action_mask: tuple[bool, ...]
    turns: tuple[ActionTurnBoundary, ...]
    segments: tuple[AlignedActionSegment, ...]
    rollout_log_probs: RolloutLogProbEvidence | None = None

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise TrajectoryAlignmentError("unsupported alignment schema_version")
        if not isinstance(self.ledger, TokenLedgerSnapshot):
            raise TrajectoryAlignmentError("ledger must be TokenLedgerSnapshot")
        if not isinstance(self.policy, RolloutPolicyIdentity):
            raise TrajectoryAlignmentError("policy must be RolloutPolicyIdentity")
        _validate_alignment(self)

    @property
    def ledger_sha256(self) -> str:
        return self.ledger.ledger_sha256

    @property
    def trajectory_id(self) -> str:
        return _hash_json(
            {
                "schema_version": _SCHEMA_VERSION,
                "ledger_sha256": self.ledger_sha256,
                "policy_sha256": self.policy.policy_sha256,
            }
        )


@dataclass(frozen=True, slots=True)
class TrajectoryRewardAttribution:
    """Reward evidence bound to one exact trajectory identity."""

    trajectory_id: str
    reward: float
    reward_evidence_sha256: str

    def __post_init__(self) -> None:
        _sha(self.trajectory_id, "trajectory_id")
        if not isinstance(self.reward, float) or not math.isfinite(self.reward):
            raise TrajectoryAlignmentError("reward must be a finite float")
        _sha(self.reward_evidence_sha256, "reward_evidence_sha256")


@dataclass(frozen=True, slots=True)
class AlignedTrainingTrajectory:
    """Training-admissible rollout plus exact reward attribution."""

    alignment: AlignedRolloutTrajectory
    reward: TrajectoryRewardAttribution

    def __post_init__(self) -> None:
        if not isinstance(self.alignment, AlignedRolloutTrajectory):
            raise TrajectoryAlignmentError("alignment must be AlignedRolloutTrajectory")
        if not isinstance(self.reward, TrajectoryRewardAttribution):
            raise TrajectoryAlignmentError("reward must be TrajectoryRewardAttribution")
        if self.reward.trajectory_id != self.alignment.trajectory_id:
            raise TrajectoryAlignmentMismatchError(
                "reward belongs to another trajectory"
            )


@dataclass(frozen=True, slots=True)
class TrajectoryAlignmentAuditEvidence:
    """Compact, content-addressed proof of PVRL-203 alignment."""

    schema_version: int
    trajectory_id: str
    session_id: str
    ledger_sha256: str
    policy_sha256: str
    input_token_count: int
    input_ids_sha256: str
    action_token_count: int
    action_token_ids_sha256: str
    action_mask_sha256: str
    turns: tuple[ActionTurnBoundary, ...]
    segments: tuple[AlignedActionSegment, ...]
    rollout_log_prob_count: int
    rollout_log_probs_sha256: str | None
    reward: float
    reward_evidence_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise TrajectoryAlignmentError("unsupported audit schema_version")
        for name in (
            "trajectory_id",
            "ledger_sha256",
            "policy_sha256",
            "input_ids_sha256",
            "action_token_ids_sha256",
            "action_mask_sha256",
            "reward_evidence_sha256",
        ):
            _sha(getattr(self, name), name)
        if not isinstance(self.session_id, str) or not self.session_id.strip():
            raise TrajectoryAlignmentError("session_id must be non-empty")
        _positive_int(self.input_token_count, "input_token_count")
        _positive_int(self.action_token_count, "action_token_count")
        _non_negative_int(self.rollout_log_prob_count, "rollout_log_prob_count")
        if self.action_token_count > self.input_token_count:
            raise TrajectoryAlignmentMismatchError("action count exceeds input count")
        if self.rollout_log_probs_sha256 is None:
            if self.rollout_log_prob_count != 0:
                raise TrajectoryAlignmentMismatchError(
                    "log-prob count present without evidence"
                )
        else:
            _sha(self.rollout_log_probs_sha256, "rollout_log_probs_sha256")
            if self.rollout_log_prob_count != self.action_token_count:
                raise TrajectoryAlignmentMismatchError("log-prob/action counts differ")
        if not isinstance(self.reward, float) or not math.isfinite(self.reward):
            raise TrajectoryAlignmentError("reward must be finite")
        _validate_audit_spans(self)

    @property
    def audit_sha256(self) -> str:
        return _hash_json(_audit_payload(self))


def align_rollout_trajectory(
    ledger: TokenLedgerSnapshot,
    *,
    policy: RolloutPolicyIdentity,
) -> AlignedRolloutTrajectory:
    """Derive the only valid training arrays from the exact PVRL-202 ledger."""

    input_ids, action_ids, mask, turns, segments = _derive_alignment(ledger)
    return AlignedRolloutTrajectory(
        schema_version=_SCHEMA_VERSION,
        ledger=ledger,
        policy=policy,
        input_ids=input_ids,
        action_token_ids=action_ids,
        action_mask=mask,
        turns=turns,
        segments=segments,
    )


def bind_rollout_log_probs(
    alignment: AlignedRolloutTrajectory,
    log_probs: Sequence[float],
) -> AlignedRolloutTrajectory:
    """Attach exact sampled-token log probabilities without padding or truncation."""

    if isinstance(log_probs, (str, bytes)):
        raise TrajectoryAlignmentError("log_probs must be numeric sequence")
    exact = tuple(log_probs)
    if len(exact) != len(alignment.action_token_ids):
        raise TrajectoryAlignmentMismatchError("log-prob/action counts differ")
    evidence = RolloutLogProbEvidence(
        trajectory_id=alignment.trajectory_id,
        policy_sha256=alignment.policy.policy_sha256,
        action_token_ids_sha256=token_ids_sha256(alignment.action_token_ids),
        log_probs=exact,
    )
    return replace(alignment, rollout_log_probs=evidence)


def bind_trajectory_reward(
    alignment: AlignedRolloutTrajectory,
    *,
    reward: float,
    reward_evidence_sha256: str,
) -> AlignedTrainingTrajectory:
    """Bind grader reward evidence to the exact trajectory ID."""

    return AlignedTrainingTrajectory(
        alignment=alignment,
        reward=TrajectoryRewardAttribution(
            trajectory_id=alignment.trajectory_id,
            reward=reward,
            reward_evidence_sha256=reward_evidence_sha256,
        ),
    )


def create_trajectory_alignment_audit(
    trajectory: AlignedTrainingTrajectory,
) -> TrajectoryAlignmentAuditEvidence:
    """Persist compact hashes/counts/spans, never raw prompt/action arrays."""

    alignment = trajectory.alignment
    log_probs = alignment.rollout_log_probs
    return TrajectoryAlignmentAuditEvidence(
        schema_version=_SCHEMA_VERSION,
        trajectory_id=alignment.trajectory_id,
        session_id=alignment.ledger.session_id,
        ledger_sha256=alignment.ledger_sha256,
        policy_sha256=alignment.policy.policy_sha256,
        input_token_count=len(alignment.input_ids),
        input_ids_sha256=token_ids_sha256(alignment.input_ids),
        action_token_count=len(alignment.action_token_ids),
        action_token_ids_sha256=token_ids_sha256(alignment.action_token_ids),
        action_mask_sha256=_mask_sha256(alignment.action_mask),
        turns=alignment.turns,
        segments=alignment.segments,
        rollout_log_prob_count=0 if log_probs is None else len(log_probs.log_probs),
        rollout_log_probs_sha256=None if log_probs is None else log_probs.evidence_sha256,
        reward=trajectory.reward.reward,
        reward_evidence_sha256=trajectory.reward.reward_evidence_sha256,
    )


def trajectory_alignment_audit_json(audit: TrajectoryAlignmentAuditEvidence) -> str:
    """Serialize compact audit evidence deterministically."""

    return _canonical_json(_audit_payload(audit)) + "\n"


def _validate_alignment(value: AlignedRolloutTrajectory) -> None:
    if value.ledger.identity.base_model != value.policy.base_model:
        raise TrajectoryAlignmentMismatchError("policy base model differs from ledger")
    expected = _derive_alignment(value.ledger)
    actual = (
        value.input_ids,
        value.action_token_ids,
        value.action_mask,
        value.turns,
        value.segments,
    )
    if actual != expected:
        raise TrajectoryAlignmentMismatchError(
            "stored training arrays differ from exact ledger"
        )
    masked = tuple(
        token_id
        for token_id, selected in zip(value.input_ids, value.action_mask, strict=True)
        if selected
    )
    if masked != value.action_token_ids:
        raise TrajectoryAlignmentMismatchError(
            "loss mask is not 1:1 with action tokens"
        )
    log_probs = value.rollout_log_probs
    if log_probs is not None:
        if log_probs.trajectory_id != value.trajectory_id:
            raise TrajectoryAlignmentMismatchError(
                "log-probs belong to another trajectory"
            )
        if log_probs.policy_sha256 != value.policy.policy_sha256:
            raise TrajectoryAlignmentMismatchError("log-probs belong to another policy")
        if log_probs.action_token_ids_sha256 != token_ids_sha256(
            value.action_token_ids
        ):
            raise TrajectoryAlignmentMismatchError(
                "log-probs bind different action tokens"
            )
        if len(log_probs.log_probs) != len(value.action_token_ids):
            raise TrajectoryAlignmentMismatchError("log-prob/action counts differ")


def _derive_alignment(
    ledger: TokenLedgerSnapshot,
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[bool, ...],
    tuple[ActionTurnBoundary, ...],
    tuple[AlignedActionSegment, ...],
]:
    if not isinstance(ledger, TokenLedgerSnapshot):
        raise TrajectoryAlignmentError("ledger must be TokenLedgerSnapshot")
    if ledger.state is not TokenLedgerState.AWAITING_PROMPT_APPEND or not ledger.outputs:
        raise ZeroActionTrajectoryError("trajectory must end after assistant output")
    if len(ledger.prompts) != len(ledger.outputs):
        raise TrajectoryAlignmentMismatchError("ledger is not a completed trajectory")

    input_ids = ledger.prompts[-1].prompt_ids + ledger.outputs[-1].output_ids
    action_ids: list[int] = []
    mask = [False] * len(input_ids)
    turns: list[ActionTurnBoundary] = []
    segments: list[AlignedActionSegment] = []
    action_cursor = 0

    for turn_index, (prompt, output) in enumerate(
        zip(ledger.prompts, ledger.outputs, strict=True)
    ):
        input_start = len(prompt.prompt_ids)
        input_end = input_start + len(output.output_ids)
        action_start = action_cursor
        action_end = action_start + len(output.output_ids)
        if input_end > len(input_ids) or input_ids[input_start:input_end] != output.output_ids:
            raise TrajectoryAlignmentMismatchError(
                "output IDs are absent at ledger turn boundary"
            )
        mask[input_start:input_end] = [True] * len(output.output_ids)
        action_ids.extend(output.output_ids)
        turns.append(
            ActionTurnBoundary(
                turn_index=turn_index,
                input_start=input_start,
                input_end=input_end,
                action_start=action_start,
                action_end=action_end,
                prompt_ids_sha256=token_ids_sha256(prompt.prompt_ids),
                output_ids_sha256=token_ids_sha256(output.output_ids),
            )
        )
        for segment_index, segment in enumerate(output.segments):
            segment_ids = output.output_ids[segment.start : segment.end]
            segments.append(
                AlignedActionSegment(
                    turn_index=turn_index,
                    segment_index=segment_index,
                    kind=segment.kind,
                    input_start=input_start + segment.start,
                    input_end=input_start + segment.end,
                    action_start=action_start + segment.start,
                    action_end=action_start + segment.end,
                    token_ids_sha256=token_ids_sha256(segment_ids),
                )
            )
        action_cursor = action_end

    exact_actions = tuple(action_ids)
    if not exact_actions:
        raise ZeroActionTrajectoryError("trajectory has zero action tokens")
    exact_mask = tuple(mask)
    masked = tuple(
        token_id
        for token_id, selected in zip(input_ids, exact_mask, strict=True)
        if selected
    )
    if masked != exact_actions:
        raise TrajectoryAlignmentMismatchError(
            "derived mask does not recover exact actions"
        )
    return input_ids, exact_actions, exact_mask, tuple(turns), tuple(segments)


def _validate_audit_spans(audit: TrajectoryAlignmentAuditEvidence) -> None:
    if not audit.turns or not audit.segments:
        raise TrajectoryAlignmentError("audit requires turn and segment spans")
    action_cursor = 0
    mask = [False] * audit.input_token_count
    for turn_index, turn in enumerate(audit.turns):
        if turn.turn_index != turn_index or turn.action_start != action_cursor:
            raise TrajectoryAlignmentMismatchError(
                "audit turn order/action spans drifted"
            )
        if turn.input_end > audit.input_token_count or turn.action_end > audit.action_token_count:
            raise TrajectoryAlignmentMismatchError(
                "audit turn span escapes token counts"
            )
        mask[turn.input_start : turn.input_end] = [True] * (
            turn.input_end - turn.input_start
        )
        action_cursor = turn.action_end
    if action_cursor != audit.action_token_count:
        raise TrajectoryAlignmentMismatchError("audit turns do not cover all actions")
    if _mask_sha256(tuple(mask)) != audit.action_mask_sha256:
        raise TrajectoryAlignmentMismatchError("audit turn spans do not reproduce mask")

    segment_action_cursor = 0
    for segment in audit.segments:
        if segment.action_start != segment_action_cursor:
            raise TrajectoryAlignmentMismatchError("audit segment action spans drifted")
        segment_action_cursor = segment.action_end
    if segment_action_cursor != audit.action_token_count:
        raise TrajectoryAlignmentMismatchError(
            "audit segments do not cover all actions"
        )


def _policy_payload(policy: RolloutPolicyIdentity) -> dict[str, object]:
    return {
        "base_model": asdict(policy.base_model),
        "adapter": asdict(policy.adapter),
        "checkpoint_id": policy.checkpoint_id,
    }


def _audit_payload(audit: TrajectoryAlignmentAuditEvidence) -> dict[str, object]:
    return {
        "schema_version": audit.schema_version,
        "trajectory_id": audit.trajectory_id,
        "session_id": audit.session_id,
        "ledger_sha256": audit.ledger_sha256,
        "policy_sha256": audit.policy_sha256,
        "input_token_count": audit.input_token_count,
        "input_ids_sha256": audit.input_ids_sha256,
        "action_token_count": audit.action_token_count,
        "action_token_ids_sha256": audit.action_token_ids_sha256,
        "action_mask_sha256": audit.action_mask_sha256,
        "turns": [asdict(turn) for turn in audit.turns],
        "segments": [
            {**asdict(segment), "kind": segment.kind.value} for segment in audit.segments
        ],
        "rollout_log_prob_count": audit.rollout_log_prob_count,
        "rollout_log_probs_sha256": audit.rollout_log_probs_sha256,
        "reward": audit.reward,
        "reward_evidence_sha256": audit.reward_evidence_sha256,
    }


def _mask_sha256(mask: tuple[bool, ...]) -> str:
    if not mask or any(not isinstance(value, bool) for value in mask):
        raise TrajectoryAlignmentError("action mask must be a non-empty boolean tuple")
    return _hash_json([1 if value else 0 for value in mask])


def _sha(value: object, name: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise TrajectoryAlignmentError(f"{name} must be lowercase SHA-256")


def _non_negative_int(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise TrajectoryAlignmentError(f"{name} must be non-negative integer")


def _positive_int(value: object, name: str) -> None:
    _non_negative_int(value, name)
    if value == 0:
        raise ZeroActionTrajectoryError(f"{name} must be positive")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _hash_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()
