"""Fail-closed rollout/log-prob/loss-mask alignment for exact PVRL trajectories."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import cast

from tiny_qwen_coder.identities import AdapterIdentity, BaseModelIdentity
from tiny_qwen_coder.pvrl.token_ledger import (
    AssistantSegmentKind,
    TokenLedgerSnapshot,
    TokenLedgerState,
    token_ids_sha256,
)

_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CHECKPOINT_ID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


class TrajectoryAlignmentError(RuntimeError):
    """Invalid trajectory material or alignment evidence."""


class TrajectoryAlignmentMismatchError(TrajectoryAlignmentError):
    """Trajectory arrays or provenance disagree with exact rollout material."""


class ZeroActionTrajectoryError(TrajectoryAlignmentError):
    """A trajectory contains no policy-generated tokens that can receive loss."""


@dataclass(frozen=True, slots=True)
class RolloutPolicyIdentity:
    """Exact policy/checkpoint identity that generated a rollout."""

    base_model: BaseModelIdentity
    adapter: AdapterIdentity
    checkpoint_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.base_model, BaseModelIdentity):
            raise TrajectoryAlignmentError(
                "policy base_model must be a BaseModelIdentity"
            )
        if not isinstance(self.adapter, AdapterIdentity):
            raise TrajectoryAlignmentError("policy adapter must be an AdapterIdentity")
        if not isinstance(self.checkpoint_id, str) or not _CHECKPOINT_ID.fullmatch(
            self.checkpoint_id
        ):
            raise TrajectoryAlignmentError(
                "policy checkpoint_id must be an immutable lowercase 40- or 64-hex identity"
            )
        if self.adapter.adapter_id is None:
            if self.checkpoint_id != self.base_model.revision:
                raise TrajectoryAlignmentError(
                    "base-only rollout checkpoint_id must equal the exact base revision"
                )
        elif not _SHA256.fullmatch(self.checkpoint_id):
            raise TrajectoryAlignmentError(
                "adapted rollout checkpoint_id must be a lowercase SHA-256 digest"
            )

    @property
    def policy_sha256(self) -> str:
        return hashlib.sha256(_canonical(_policy_payload(self))).hexdigest()


@dataclass(frozen=True, slots=True)
class RolloutLogProbEvidence:
    """Sampled-token log probabilities bound to one exact trajectory and policy."""

    trajectory_id: str
    policy: RolloutPolicyIdentity
    action_token_ids: tuple[int, ...]
    log_probs: tuple[float, ...]

    def __post_init__(self) -> None:
        _sha(self.trajectory_id, "log-prob trajectory_id")
        if not isinstance(self.policy, RolloutPolicyIdentity):
            raise TrajectoryAlignmentError(
                "log-prob policy must be a RolloutPolicyIdentity"
            )
        _tokens(self.action_token_ids, "log-prob action_token_ids")
        if not isinstance(self.log_probs, tuple) or not self.log_probs:
            raise TrajectoryAlignmentError(
                "rollout log_probs must be a non-empty tuple"
            )
        if len(self.log_probs) != len(self.action_token_ids):
            raise TrajectoryAlignmentMismatchError(
                "rollout log-prob count must equal sampled action-token count"
            )
        for index, value in enumerate(self.log_probs):
            if not isinstance(value, float) or not math.isfinite(value) or value > 0.0:
                raise TrajectoryAlignmentError(
                    f"rollout log_probs[{index}] must be a finite float <= 0"
                )

    @property
    def evidence_sha256(self) -> str:
        return hashlib.sha256(
            _canonical(
                {
                    "trajectory_id": self.trajectory_id,
                    "policy": _policy_payload(self.policy),
                    "action_token_ids": list(self.action_token_ids),
                    "log_probs": list(self.log_probs),
                }
            )
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class ActionTurnBoundary:
    """Exact mapping between one assistant turn, model input, and flat action arrays."""

    turn_index: int
    prompt_token_count: int
    prompt_ids_sha256: str
    input_start: int
    input_end: int
    action_start: int
    action_end: int
    output_ids_sha256: str

    def __post_init__(self) -> None:
        _index(self.turn_index, "turn_index")
        _index(self.prompt_token_count, "prompt_token_count")
        _index(self.input_start, "turn input_start")
        _index(self.input_end, "turn input_end")
        _index(self.action_start, "turn action_start")
        _index(self.action_end, "turn action_end")
        if (
            self.input_start != self.prompt_token_count
            or self.input_end <= self.input_start
        ):
            raise TrajectoryAlignmentError(
                "action turn must begin at its prompt boundary and contain input tokens"
            )
        if self.action_end <= self.action_start:
            raise TrajectoryAlignmentError(
                "action turn must contain sampled action tokens"
            )
        if self.input_end - self.input_start != self.action_end - self.action_start:
            raise TrajectoryAlignmentMismatchError(
                "turn input span and flat action span must have equal length"
            )
        _sha(self.prompt_ids_sha256, "prompt_ids_sha256")
        _sha(self.output_ids_sha256, "output_ids_sha256")


@dataclass(frozen=True, slots=True)
class AlignedActionSegment:
    """One semantic ledger segment mapped to model-input and flat-action positions."""

    turn_index: int
    segment_index: int
    kind: AssistantSegmentKind
    input_start: int
    input_end: int
    action_start: int
    action_end: int
    token_ids_sha256: str

    def __post_init__(self) -> None:
        _index(self.turn_index, "segment turn_index")
        _index(self.segment_index, "segment_index")
        if not isinstance(self.kind, AssistantSegmentKind):
            raise TrajectoryAlignmentError(
                "segment kind must be an AssistantSegmentKind"
            )
        _index(self.input_start, "segment input_start")
        _index(self.input_end, "segment input_end")
        _index(self.action_start, "segment action_start")
        _index(self.action_end, "segment action_end")
        if self.input_end <= self.input_start or self.action_end <= self.action_start:
            raise TrajectoryAlignmentError("aligned action segment must contain tokens")
        if self.input_end - self.input_start != self.action_end - self.action_start:
            raise TrajectoryAlignmentMismatchError(
                "segment input span and flat action span must have equal length"
            )
        _sha(self.token_ids_sha256, "segment token_ids_sha256")


@dataclass(frozen=True, slots=True)
class AlignedRolloutTrajectory:
    """Exact ledger-derived training view before reward attribution."""

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
            raise TrajectoryAlignmentError(
                "unsupported trajectory-alignment schema_version"
            )
        if not isinstance(self.ledger, TokenLedgerSnapshot):
            raise TrajectoryAlignmentError("ledger must be a TokenLedgerSnapshot")
        if not isinstance(self.policy, RolloutPolicyIdentity):
            raise TrajectoryAlignmentError("policy must be a RolloutPolicyIdentity")
        _validate_alignment(self)

    @property
    def ledger_sha256(self) -> str:
        return self.ledger.ledger_sha256

    @property
    def trajectory_id(self) -> str:
        return _trajectory_id(self.ledger_sha256, self.policy.policy_sha256)

    @property
    def action_token_count(self) -> int:
        return len(self.action_token_ids)


@dataclass(frozen=True, slots=True)
class TrajectoryRewardAttribution:
    """Reward value explicitly bound to one exact trajectory identity."""

    trajectory_id: str
    reward: float
    reward_evidence_sha256: str

    def __post_init__(self) -> None:
        _sha(self.trajectory_id, "reward trajectory_id")
        if not isinstance(self.reward, float) or not math.isfinite(self.reward):
            raise TrajectoryAlignmentError("reward must be a finite float")
        _sha(self.reward_evidence_sha256, "reward_evidence_sha256")


@dataclass(frozen=True, slots=True)
class AlignedTrainingTrajectory:
    """Training-admissible trajectory with exact-token alignment and bound reward."""

    alignment: AlignedRolloutTrajectory
    reward: TrajectoryRewardAttribution

    def __post_init__(self) -> None:
        if not isinstance(self.alignment, AlignedRolloutTrajectory):
            raise TrajectoryAlignmentError(
                "alignment must be an AlignedRolloutTrajectory"
            )
        if not isinstance(self.reward, TrajectoryRewardAttribution):
            raise TrajectoryAlignmentError(
                "reward must be a TrajectoryRewardAttribution"
            )
        if self.reward.trajectory_id != self.alignment.trajectory_id:
            raise TrajectoryAlignmentMismatchError(
                "reward attribution does not match the exact trajectory identity"
            )


@dataclass(frozen=True, slots=True)
class TrajectoryAlignmentAuditEvidence:
    """Compact persisted proof of token, mask, provenance, and reward alignment."""

    schema_version: int
    trajectory_id: str
    session_id: str
    ledger_sha256: str
    policy: RolloutPolicyIdentity
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
            raise TrajectoryAlignmentError("unsupported alignment-audit schema_version")
        _sha(self.trajectory_id, "trajectory_id")
        _trimmed(self.session_id, "session_id")
        _sha(self.ledger_sha256, "ledger_sha256")
        if not isinstance(self.policy, RolloutPolicyIdentity):
            raise TrajectoryAlignmentError(
                "audit policy must be a RolloutPolicyIdentity"
            )
        expected_trajectory_id = _trajectory_id(
            self.ledger_sha256, self.policy.policy_sha256
        )
        if self.trajectory_id != expected_trajectory_id:
            raise TrajectoryAlignmentMismatchError(
                "audit trajectory_id does not match ledger and generating policy"
            )
        _positive_int(self.input_token_count, "input_token_count")
        if not _plain_int(self.action_token_count) or self.action_token_count <= 0:
            raise ZeroActionTrajectoryError(
                "action_token_count must be a positive integer"
            )
        if self.action_token_count > self.input_token_count:
            raise TrajectoryAlignmentError(
                "action_token_count cannot exceed input_token_count"
            )
        _sha(self.input_ids_sha256, "input_ids_sha256")
        _sha(self.action_token_ids_sha256, "action_token_ids_sha256")
        _sha(self.action_mask_sha256, "action_mask_sha256")
        _index(self.rollout_log_prob_count, "rollout_log_prob_count")
        if self.rollout_log_probs_sha256 is None:
            if self.rollout_log_prob_count != 0:
                raise TrajectoryAlignmentError(
                    "log-prob count must be zero when log-prob evidence is absent"
                )
        else:
            _sha(self.rollout_log_probs_sha256, "rollout_log_probs_sha256")
            if self.rollout_log_prob_count != self.action_token_count:
                raise TrajectoryAlignmentMismatchError(
                    "audit log-prob count must equal exact action-token count"
                )
        if not isinstance(self.reward, float) or not math.isfinite(self.reward):
            raise TrajectoryAlignmentError("audit reward must be a finite float")
        _sha(self.reward_evidence_sha256, "reward_evidence_sha256")
        _validate_audit_boundaries(self)

    @property
    def audit_sha256(self) -> str:
        return hashlib.sha256(_canonical(_audit_payload(self))).hexdigest()


def align_rollout_trajectory(
    ledger: TokenLedgerSnapshot,
    *,
    policy: RolloutPolicyIdentity,
    rollout_log_probs: RolloutLogProbEvidence | None = None,
) -> AlignedRolloutTrajectory:
    """Derive a training view solely from exact PVRL-202 ledger token IDs."""

    if not isinstance(ledger, TokenLedgerSnapshot):
        raise TrajectoryAlignmentError("ledger must be a TokenLedgerSnapshot")
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
        rollout_log_probs=rollout_log_probs,
    )


def bind_rollout_log_probs(
    alignment: AlignedRolloutTrajectory,
    log_probs: Sequence[float],
) -> AlignedRolloutTrajectory:
    """Bind sampled-token log probabilities to one already-proven exact trajectory."""

    if not isinstance(alignment, AlignedRolloutTrajectory):
        raise TrajectoryAlignmentError("alignment must be an AlignedRolloutTrajectory")
    if isinstance(log_probs, (str, bytes)):
        raise TrajectoryAlignmentError("log_probs must be a numeric sequence")
    exact_log_probs = tuple(log_probs)
    evidence = RolloutLogProbEvidence(
        trajectory_id=alignment.trajectory_id,
        policy=alignment.policy,
        action_token_ids=alignment.action_token_ids,
        log_probs=exact_log_probs,
    )
    return replace(alignment, rollout_log_probs=evidence)


def bind_trajectory_reward(
    alignment: AlignedRolloutTrajectory,
    *,
    reward: float,
    reward_evidence_sha256: str,
) -> AlignedTrainingTrajectory:
    """Bind externally scored reward evidence to the exact content-addressed trajectory."""

    if not isinstance(alignment, AlignedRolloutTrajectory):
        raise TrajectoryAlignmentError("alignment must be an AlignedRolloutTrajectory")
    attribution = TrajectoryRewardAttribution(
        trajectory_id=alignment.trajectory_id,
        reward=reward,
        reward_evidence_sha256=reward_evidence_sha256,
    )
    return AlignedTrainingTrajectory(alignment=alignment, reward=attribution)


def create_trajectory_alignment_audit(
    trajectory: AlignedTrainingTrajectory,
) -> TrajectoryAlignmentAuditEvidence:
    """Create compact evidence without raw prompt/action tokens or log probabilities."""

    if not isinstance(trajectory, AlignedTrainingTrajectory):
        raise TrajectoryAlignmentError(
            "trajectory must be an AlignedTrainingTrajectory"
        )
    alignment = trajectory.alignment
    log_probs = alignment.rollout_log_probs
    return TrajectoryAlignmentAuditEvidence(
        schema_version=_SCHEMA_VERSION,
        trajectory_id=alignment.trajectory_id,
        session_id=alignment.ledger.session_id,
        ledger_sha256=alignment.ledger_sha256,
        policy=alignment.policy,
        input_token_count=len(alignment.input_ids),
        input_ids_sha256=token_ids_sha256(alignment.input_ids),
        action_token_count=len(alignment.action_token_ids),
        action_token_ids_sha256=token_ids_sha256(alignment.action_token_ids),
        action_mask_sha256=_mask_sha256(alignment.action_mask),
        turns=alignment.turns,
        segments=alignment.segments,
        rollout_log_prob_count=0 if log_probs is None else len(log_probs.log_probs),
        rollout_log_probs_sha256=None
        if log_probs is None
        else log_probs.evidence_sha256,
        reward=trajectory.reward.reward,
        reward_evidence_sha256=trajectory.reward.reward_evidence_sha256,
    )


def trajectory_alignment_audit_json(audit: TrajectoryAlignmentAuditEvidence) -> str:
    """Serialize compact PVRL-203 evidence deterministically."""

    if not isinstance(audit, TrajectoryAlignmentAuditEvidence):
        raise TrajectoryAlignmentError("audit must be TrajectoryAlignmentAuditEvidence")
    return _canonical(_audit_payload(audit)).decode("ascii") + "\n"


def trajectory_alignment_audit_from_json(text: str) -> TrajectoryAlignmentAuditEvidence:
    """Strictly parse persisted PVRL-203 alignment evidence."""

    try:
        raw: object = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TrajectoryAlignmentError("alignment audit is not valid JSON") from exc
    row = _object(raw, "alignment audit")
    _keys(
        row,
        {
            "schema_version",
            "trajectory_id",
            "session_id",
            "ledger_sha256",
            "policy",
            "input_token_count",
            "input_ids_sha256",
            "action_token_count",
            "action_token_ids_sha256",
            "action_mask_sha256",
            "turns",
            "segments",
            "rollout_log_prob_count",
            "rollout_log_probs_sha256",
            "reward",
            "reward_evidence_sha256",
        },
        "alignment audit",
    )
    turns = tuple(_audit_turn(item) for item in _list(row.get("turns"), "audit.turns"))
    segments = tuple(
        _audit_segment(item) for item in _list(row.get("segments"), "audit.segments")
    )
    log_hash = row.get("rollout_log_probs_sha256")
    if log_hash is not None and not isinstance(log_hash, str):
        raise TrajectoryAlignmentError(
            "audit.rollout_log_probs_sha256 must be string or null"
        )
    return TrajectoryAlignmentAuditEvidence(
        schema_version=_integer(row, "schema_version", "alignment audit"),
        trajectory_id=_string(row, "trajectory_id", "alignment audit"),
        session_id=_string(row, "session_id", "alignment audit"),
        ledger_sha256=_string(row, "ledger_sha256", "alignment audit"),
        policy=_audit_policy(row.get("policy")),
        input_token_count=_integer(row, "input_token_count", "alignment audit"),
        input_ids_sha256=_string(row, "input_ids_sha256", "alignment audit"),
        action_token_count=_integer(row, "action_token_count", "alignment audit"),
        action_token_ids_sha256=_string(
            row, "action_token_ids_sha256", "alignment audit"
        ),
        action_mask_sha256=_string(row, "action_mask_sha256", "alignment audit"),
        turns=turns,
        segments=segments,
        rollout_log_prob_count=_integer(
            row, "rollout_log_prob_count", "alignment audit"
        ),
        rollout_log_probs_sha256=cast(str | None, log_hash),
        reward=_float(row, "reward", "alignment audit"),
        reward_evidence_sha256=_string(
            row, "reward_evidence_sha256", "alignment audit"
        ),
    )


def _validate_alignment(value: AlignedRolloutTrajectory) -> None:
    ledger = value.ledger
    if ledger.state is not TokenLedgerState.AWAITING_PROMPT_APPEND:
        raise TrajectoryAlignmentError(
            "training trajectory must end after a recorded assistant output"
        )
    if not ledger.outputs:
        raise ZeroActionTrajectoryError("training trajectory has no assistant outputs")
    if value.policy.base_model != ledger.identity.base_model:
        raise TrajectoryAlignmentMismatchError(
            "generating policy base model differs from token-ledger identity"
        )
    _tokens(value.input_ids, "trajectory input_ids")
    _tokens(value.action_token_ids, "trajectory action_token_ids")
    _mask(value.action_mask, "trajectory action mask")
    expected = _derive_alignment(ledger)
    actual = (
        value.input_ids,
        value.action_token_ids,
        value.action_mask,
        value.turns,
        value.segments,
    )
    if actual != expected:
        raise TrajectoryAlignmentMismatchError(
            "trajectory token IDs, turn boundaries, or action mask differ from exact ledger"
        )
    if not any(value.action_mask):
        raise ZeroActionTrajectoryError(
            "training trajectory has zero trainable action tokens"
        )
    masked_ids = tuple(
        token_id
        for token_id, receives_loss in zip(
            value.input_ids, value.action_mask, strict=True
        )
        if receives_loss
    )
    if masked_ids != value.action_token_ids:
        raise TrajectoryAlignmentMismatchError(
            "assistant/action loss mask is not 1:1 with exact sampled action tokens"
        )
    log_probs = value.rollout_log_probs
    if log_probs is not None:
        if not isinstance(log_probs, RolloutLogProbEvidence):
            raise TrajectoryAlignmentError(
                "rollout_log_probs must be RolloutLogProbEvidence or None"
            )
        if log_probs.trajectory_id != value.trajectory_id:
            raise TrajectoryAlignmentMismatchError(
                "rollout log probabilities came from a different prompt/history trajectory"
            )
        if log_probs.policy != value.policy:
            raise TrajectoryAlignmentMismatchError(
                "rollout log probabilities came from a different policy/checkpoint"
            )
        if log_probs.action_token_ids != value.action_token_ids:
            raise TrajectoryAlignmentMismatchError(
                "rollout log probabilities are bound to different sampled action tokens"
            )


def _derive_alignment(
    ledger: TokenLedgerSnapshot,
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[bool, ...],
    tuple[ActionTurnBoundary, ...],
    tuple[AlignedActionSegment, ...],
]:
    if (
        ledger.state is not TokenLedgerState.AWAITING_PROMPT_APPEND
        or not ledger.outputs
    ):
        raise ZeroActionTrajectoryError(
            "exact training alignment requires at least one completed assistant turn"
        )
    if len(ledger.outputs) != len(ledger.prompts):
        raise TrajectoryAlignmentMismatchError(
            "completed ledger must have one assistant output for every prompt turn"
        )
    input_ids = ledger.prompts[-1].prompt_ids + ledger.outputs[-1].output_ids
    _tokens(input_ids, "trajectory input_ids")
    action_ids: list[int] = []
    mask = [False] * len(input_ids)
    turns: list[ActionTurnBoundary] = []
    segments: list[AlignedActionSegment] = []
    previous_input_end = 0
    action_offset = 0
    for turn_index, (prompt, output) in enumerate(
        zip(ledger.prompts, ledger.outputs, strict=True)
    ):
        if prompt.turn_index != turn_index or output.turn_index != turn_index:
            raise TrajectoryAlignmentMismatchError(
                "ledger turn indices are not contiguous"
            )
        input_start = len(prompt.prompt_ids)
        input_end = input_start + len(output.output_ids)
        action_start = action_offset
        action_end = action_start + len(output.output_ids)
        if input_start < previous_input_end or input_end > len(input_ids):
            raise TrajectoryAlignmentMismatchError(
                "assistant turn boundaries overlap or escape"
            )
        if input_ids[input_start:input_end] != output.output_ids:
            raise TrajectoryAlignmentMismatchError(
                "final trajectory does not contain exact sampled output IDs at turn boundary"
            )
        turns.append(
            ActionTurnBoundary(
                turn_index=turn_index,
                prompt_token_count=input_start,
                prompt_ids_sha256=token_ids_sha256(prompt.prompt_ids),
                input_start=input_start,
                input_end=input_end,
                action_start=action_start,
                action_end=action_end,
                output_ids_sha256=token_ids_sha256(output.output_ids),
            )
        )
        action_ids.extend(output.output_ids)
        mask[input_start:input_end] = [True] * len(output.output_ids)
        for segment_index, segment in enumerate(output.segments):
            segment_input_start = input_start + segment.start
            segment_input_end = input_start + segment.end
            segment_action_start = action_start + segment.start
            segment_action_end = action_start + segment.end
            segment_ids = input_ids[segment_input_start:segment_input_end]
            exact_segment_ids = output.output_ids[segment.start : segment.end]
            if segment_ids != exact_segment_ids:
                raise TrajectoryAlignmentMismatchError(
                    "semantic assistant segment differs from exact sampled token IDs"
                )
            segments.append(
                AlignedActionSegment(
                    turn_index=turn_index,
                    segment_index=segment_index,
                    kind=segment.kind,
                    input_start=segment_input_start,
                    input_end=segment_input_end,
                    action_start=segment_action_start,
                    action_end=segment_action_end,
                    token_ids_sha256=token_ids_sha256(segment_ids),
                )
            )
        previous_input_end = input_end
        action_offset = action_end
    exact_actions = tuple(action_ids)
    if not exact_actions:
        raise ZeroActionTrajectoryError(
            "training trajectory has zero sampled action tokens"
        )
    if action_offset != len(exact_actions):
        raise TrajectoryAlignmentMismatchError("flat action-token accounting drifted")
    exact_mask = tuple(mask)
    masked_ids = tuple(
        token_id
        for token_id, receives_loss in zip(input_ids, exact_mask, strict=True)
        if receives_loss
    )
    if masked_ids != exact_actions:
        raise TrajectoryAlignmentMismatchError(
            "derived assistant/action mask does not recover exact sampled action tokens"
        )
    return input_ids, exact_actions, exact_mask, tuple(turns), tuple(segments)


def _trajectory_id(ledger_sha256: str, policy_sha256: str) -> str:
    _sha(ledger_sha256, "ledger_sha256")
    _sha(policy_sha256, "policy_sha256")
    return hashlib.sha256(
        _canonical(
            {
                "schema_version": _SCHEMA_VERSION,
                "ledger_sha256": ledger_sha256,
                "policy_sha256": policy_sha256,
            }
        )
    ).hexdigest()


def _validate_audit_boundaries(audit: TrajectoryAlignmentAuditEvidence) -> None:
    if not isinstance(audit.turns, tuple) or not audit.turns:
        raise TrajectoryAlignmentError("audit must contain tuple turn boundaries")
    if not isinstance(audit.segments, tuple) or not audit.segments:
        raise TrajectoryAlignmentError("audit must contain tuple segment boundaries")
    reconstructed_mask = [False] * audit.input_token_count
    previous_input_end = 0
    previous_action_end = 0
    for index, turn in enumerate(audit.turns):
        if not isinstance(turn, ActionTurnBoundary):
            raise TrajectoryAlignmentError(
                "audit turns must be ActionTurnBoundary values"
            )
        if turn.turn_index != index:
            raise TrajectoryAlignmentMismatchError(
                "audit turn indices must be contiguous"
            )
        if (
            turn.input_start < previous_input_end
            or turn.input_end > audit.input_token_count
        ):
            raise TrajectoryAlignmentMismatchError(
                "audit turn input boundaries overlap or escape"
            )
        if (
            turn.action_start != previous_action_end
            or turn.action_end > audit.action_token_count
        ):
            raise TrajectoryAlignmentMismatchError(
                "audit turn action boundaries are not contiguous"
            )
        reconstructed_mask[turn.input_start : turn.input_end] = [True] * (
            turn.input_end - turn.input_start
        )
        previous_input_end = turn.input_end
        previous_action_end = turn.action_end
    if previous_action_end != audit.action_token_count:
        raise TrajectoryAlignmentMismatchError(
            "audit turns do not cover every action token"
        )
    if _mask_sha256(tuple(reconstructed_mask)) != audit.action_mask_sha256:
        raise TrajectoryAlignmentMismatchError(
            "audit turn spans do not reproduce action mask hash"
        )

    by_turn: dict[int, list[AlignedActionSegment]] = {}
    for segment in audit.segments:
        if not isinstance(segment, AlignedActionSegment):
            raise TrajectoryAlignmentError(
                "audit segments must be AlignedActionSegment values"
            )
        by_turn.setdefault(segment.turn_index, []).append(segment)
    for turn in audit.turns:
        turn_segments = by_turn.get(turn.turn_index, [])
        if not turn_segments:
            raise TrajectoryAlignmentMismatchError(
                "audit turn has no semantic action segments"
            )
        input_position = turn.input_start
        action_position = turn.action_start
        for segment_index, segment in enumerate(turn_segments):
            if segment.segment_index != segment_index:
                raise TrajectoryAlignmentMismatchError(
                    "audit segment indices must be contiguous within each turn"
                )
            if (
                segment.input_start != input_position
                or segment.action_start != action_position
            ):
                raise TrajectoryAlignmentMismatchError(
                    "audit segments must partition turn input/action spans without gaps"
                )
            input_position = segment.input_end
            action_position = segment.action_end
        if input_position != turn.input_end or action_position != turn.action_end:
            raise TrajectoryAlignmentMismatchError(
                "audit segments must cover each assistant turn exactly"
            )
    if set(by_turn) != {turn.turn_index for turn in audit.turns}:
        raise TrajectoryAlignmentMismatchError(
            "audit segment references an unknown turn"
        )


def _policy_payload(policy: RolloutPolicyIdentity) -> dict[str, object]:
    return {
        "base_model": asdict(policy.base_model),
        "adapter": asdict(policy.adapter),
        "checkpoint_id": policy.checkpoint_id,
    }


def _audit_payload(audit: TrajectoryAlignmentAuditEvidence) -> dict[str, object]:
    turns = [asdict(turn) for turn in audit.turns]
    segments = []
    for segment in audit.segments:
        item = asdict(segment)
        item["kind"] = segment.kind.value
        segments.append(item)
    return {
        "schema_version": audit.schema_version,
        "trajectory_id": audit.trajectory_id,
        "session_id": audit.session_id,
        "ledger_sha256": audit.ledger_sha256,
        "policy": _policy_payload(audit.policy),
        "input_token_count": audit.input_token_count,
        "input_ids_sha256": audit.input_ids_sha256,
        "action_token_count": audit.action_token_count,
        "action_token_ids_sha256": audit.action_token_ids_sha256,
        "action_mask_sha256": audit.action_mask_sha256,
        "turns": turns,
        "segments": segments,
        "rollout_log_prob_count": audit.rollout_log_prob_count,
        "rollout_log_probs_sha256": audit.rollout_log_probs_sha256,
        "reward": audit.reward,
        "reward_evidence_sha256": audit.reward_evidence_sha256,
    }


def _mask(mask: object, name: str) -> None:
    if (
        not isinstance(mask, tuple)
        or not mask
        or any(not isinstance(value, bool) for value in mask)
    ):
        raise TrajectoryAlignmentError(
            f"{name} must be a non-empty exact boolean tuple"
        )


def _mask_sha256(mask: Sequence[bool]) -> str:
    if not isinstance(mask, tuple):
        mask = tuple(mask)
    _mask(mask, "action mask")
    return hashlib.sha256(_canonical([1 if value else 0 for value in mask])).hexdigest()


def _plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _positive_int(value: int, name: str) -> None:
    if not _plain_int(value) or value <= 0:
        raise TrajectoryAlignmentError(f"{name} must be a positive integer")


def _index(value: int, name: str) -> None:
    if not _plain_int(value) or value < 0:
        raise TrajectoryAlignmentError(f"{name} must be a non-negative integer")


def _tokens(values: object, name: str) -> None:
    if not isinstance(values, tuple) or not values:
        raise ZeroActionTrajectoryError(
            f"{name} must be a non-empty immutable token tuple"
        )
    for index, value in enumerate(values):
        if not _plain_int(value) or value < 0:
            raise TrajectoryAlignmentError(
                f"{name}[{index}] must be a non-negative integer"
            )


def _sha(value: str, name: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise TrajectoryAlignmentError(f"{name} must be a lowercase SHA-256 digest")


def _trimmed(value: str, name: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TrajectoryAlignmentError(f"{name} must be a non-empty trimmed string")


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _object(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TrajectoryAlignmentError(f"{context} must be a string-keyed JSON object")
    return cast(dict[str, object], dict(value))


def _list(value: object, context: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TrajectoryAlignmentError(f"{context} must be a list")
    return cast(Sequence[object], value)


def _keys(row: Mapping[str, object], expected: set[str], context: str) -> None:
    if set(row) != expected:
        raise TrajectoryAlignmentError(f"{context} keys do not match schema")


def _string(row: Mapping[str, object], key: str, context: str) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise TrajectoryAlignmentError(f"{context}.{key} must be a string")
    return value


def _integer(row: Mapping[str, object], key: str, context: str) -> int:
    value = row.get(key)
    if not _plain_int(value):
        raise TrajectoryAlignmentError(f"{context}.{key} must be an integer")
    return cast(int, value)


def _float(row: Mapping[str, object], key: str, context: str) -> float:
    value = row.get(key)
    if not isinstance(value, float) or not math.isfinite(value):
        raise TrajectoryAlignmentError(f"{context}.{key} must be a finite float")
    return value


def _audit_policy(value: object) -> RolloutPolicyIdentity:
    row = _object(value, "audit policy")
    _keys(row, {"base_model", "adapter", "checkpoint_id"}, "audit policy")
    base = _object(row.get("base_model"), "audit policy.base_model")
    _keys(
        base,
        {"repository", "revision", "tokenizer_repository", "tokenizer_revision"},
        "audit policy.base_model",
    )
    adapter = _object(row.get("adapter"), "audit policy.adapter")
    _keys(adapter, {"family", "adapter_id"}, "audit policy.adapter")
    family = adapter.get("family")
    adapter_id = adapter.get("adapter_id")
    if family is not None and not isinstance(family, str):
        raise TrajectoryAlignmentError(
            "audit policy.adapter.family must be string or null"
        )
    if adapter_id is not None and not isinstance(adapter_id, str):
        raise TrajectoryAlignmentError(
            "audit policy.adapter.adapter_id must be string or null"
        )
    try:
        base_model = BaseModelIdentity(
            repository=_string(base, "repository", "audit policy.base_model"),
            revision=_string(base, "revision", "audit policy.base_model"),
            tokenizer_repository=_string(
                base, "tokenizer_repository", "audit policy.base_model"
            ),
            tokenizer_revision=_string(
                base, "tokenizer_revision", "audit policy.base_model"
            ),
        )
        adapter_identity = AdapterIdentity(
            family=cast(str | None, family),
            adapter_id=cast(str | None, adapter_id),
        )
    except (TypeError, ValueError) as exc:
        raise TrajectoryAlignmentError("audit policy identity is invalid") from exc
    return RolloutPolicyIdentity(
        base_model=base_model,
        adapter=adapter_identity,
        checkpoint_id=_string(row, "checkpoint_id", "audit policy"),
    )


def _audit_turn(value: object) -> ActionTurnBoundary:
    row = _object(value, "audit turn")
    _keys(
        row,
        {
            "turn_index",
            "prompt_token_count",
            "prompt_ids_sha256",
            "input_start",
            "input_end",
            "action_start",
            "action_end",
            "output_ids_sha256",
        },
        "audit turn",
    )
    return ActionTurnBoundary(
        turn_index=_integer(row, "turn_index", "audit turn"),
        prompt_token_count=_integer(row, "prompt_token_count", "audit turn"),
        prompt_ids_sha256=_string(row, "prompt_ids_sha256", "audit turn"),
        input_start=_integer(row, "input_start", "audit turn"),
        input_end=_integer(row, "input_end", "audit turn"),
        action_start=_integer(row, "action_start", "audit turn"),
        action_end=_integer(row, "action_end", "audit turn"),
        output_ids_sha256=_string(row, "output_ids_sha256", "audit turn"),
    )


def _audit_segment(value: object) -> AlignedActionSegment:
    row = _object(value, "audit segment")
    _keys(
        row,
        {
            "turn_index",
            "segment_index",
            "kind",
            "input_start",
            "input_end",
            "action_start",
            "action_end",
            "token_ids_sha256",
        },
        "audit segment",
    )
    try:
        kind = AssistantSegmentKind(_string(row, "kind", "audit segment"))
    except ValueError as exc:
        raise TrajectoryAlignmentError("audit segment.kind is unknown") from exc
    return AlignedActionSegment(
        turn_index=_integer(row, "turn_index", "audit segment"),
        segment_index=_integer(row, "segment_index", "audit segment"),
        kind=kind,
        input_start=_integer(row, "input_start", "audit segment"),
        input_end=_integer(row, "input_end", "audit segment"),
        action_start=_integer(row, "action_start", "audit segment"),
        action_end=_integer(row, "action_end", "audit segment"),
        token_ids_sha256=_string(row, "token_ids_sha256", "audit segment"),
    )
