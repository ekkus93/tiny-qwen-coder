"""Winner-only P9-007E qualification for the repaired V4-2000 student study."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import NoReturn, cast

from tiny_qwen_coder.config import EvaluationConfig
from tiny_qwen_coder.evaluation._baseline_generation import BaselineGenerator
from tiny_qwen_coder.evaluation._baseline_runner import _generate_items
from tiny_qwen_coder.evaluation._baseline_types import BaselineGeneratedResponse
from tiny_qwen_coder.evaluation.execution import DirectExecutionHarness
from tiny_qwen_coder.evaluation.humaneval import (
    HumanEvalCompletion,
    HumanEvalEvaluator,
    HumanEvalProblem,
)
from tiny_qwen_coder.evaluation.mbpp import MBPPCompletion, MBPPEvaluator, MBPPProblem
from tiny_qwen_coder.evaluation.promotion import load_frozen_python_promotion_policy
from tiny_qwen_coder.evaluation.python_distilled_trajectory import (
    DistilledSnapshotGenerator,
    load_local_trajectory,
)
from tiny_qwen_coder.evaluation.python_minimum_intervention import (
    SnapshotIdentity,
    TrajectoryIdentity,
    _evaluation_context,
    load_development_manifest,
)
from tiny_qwen_coder.evaluation.repository_holdout import (
    RepositoryHoldoutCompletion,
    RepositoryHoldoutEvaluator,
    RepositoryHoldoutTask,
)
from tiny_qwen_coder.evaluation.settings import FrozenEvaluationSettings, evaluation_settings_sha256
from tiny_qwen_coder.identities import AdapterIdentity, BaseModelIdentity

_AUTHORIZATION_PATH = Path("docs/evidence/P9_007_DEVELOPMENT_RUN_34255423595.json")
_TRAJECTORY_EVIDENCE_PATH = Path("docs/evidence/P9_007_DISTILLED_V4_2000_TRAJECTORY.json")
_OUTPUT_ROOT = Path("artifacts/eval/python/p9-distilled-v4-2000-qualification-v1")
_SELECTED_STEP = 185
_EXPECTED_AUTH_RUN_ID = 34255423595
_EXPECTED_AUTH_RUN_ATTEMPT = 2
_EXPECTED_AUTH_JOB_ID = 102160760346
_EXPECTED_AUTH_SOURCE_SHA = "00ef8ad3e2edf2e8e97636026eb66bc2fb7fd8f5"
_EXPECTED_DEVELOPMENT_ARTIFACT_ID = 10070646552
_EXPECTED_DEVELOPMENT_ARTIFACT_DIGEST = (
    "sha256:514c6be5de40d881a42c31bf7574ae61730254a036051ddf7e8603a4206ec286"
)
_EXPECTED_TRAJECTORY_RUN_ID = 34230186622
_EXPECTED_TRAJECTORY_JOB_ID = 102074065676
_EXPECTED_TRAJECTORY_SOURCE_SHA = "d434ed274f2bcf56ca8e37863076c8937d30c83f"
_EXPECTED_TRAJECTORY_ARTIFACT_ID = 10059380380
_EXPECTED_TRAJECTORY_ARTIFACT_DIGEST = (
    "sha256:9bc106fa6d334119366b108f4225f71b69c88fb05bbf772467ebb064ebc00e5b"
)
_EXPECTED_DEVELOPMENT_MANIFEST_SHA256 = (
    "260682d773640b28673357c7a441474656bbaffc67f0b5fcbe45eca3a07283de"
)
_EXPECTED_MEMBERSHIP_SHA256 = "c8765b7a3a69f134be4066e274a64640a15f2d05858374035432207b3521498b"
_EXPECTED_SELECTED_ARTIFACT_SET_SHA256 = (
    "0462ed5e7dc08df76c1e72c58bf53d153ca9d74cc072a0724c62537307a476da"
)
_EXPECTED_SELECTED_CONFIG_SHA256 = (
    "4a7ad9b29cf752a7c93bfd6a67bd3227ee3b0d313436d408611b179d90e91ef1"
)
_EXPECTED_SELECTED_MODEL_SHA256 = (
    "2ee57b8c6fd10237e6e2faf11a9ff376fe7115f58062a8ed54eb962c3be6478a"
)
_EXPECTED_QUAL_HE = 119
_EXPECTED_QUAL_MBPP = 370
_EXPECTED_QUAL_HOLDOUT = 11
_EXPECTED_QUAL_TOTAL = 500
_DEVELOPMENT_HE_PASSED = 34
_DEVELOPMENT_MBPP_PASSED = 70
_DEVELOPMENT_COMBINED_PASSED = 104


class DistilledQualificationError(RuntimeError):
    """Raised when P9-007E authorization, execution, or evidence drifts."""


@dataclass(frozen=True, slots=True)
class QualificationAuthorization:
    """Immutable clean development replay authorization for exactly one checkpoint."""

    workflow_run_id: int
    workflow_run_attempt: int
    workflow_job_id: int
    source_git_sha: str
    selected_step: int
    selected_artifact_set_sha256: str
    selected_adapter_config_sha256: str
    selected_adapter_model_sha256: str
    development_artifact_id: int
    development_artifact_digest: str


@dataclass(frozen=True, slots=True)
class QualificationScore:
    """Protected one-shot score plus the reconstructed full protected score."""

    selected_step: int
    qualification_humaneval_passed: int
    qualification_humaneval_total: int
    qualification_mbpp_passed: int
    qualification_mbpp_total: int
    qualification_repository_holdout_passed: int
    qualification_repository_holdout_total: int
    qualification_combined_passed: int
    qualification_combined_total: int
    full_humaneval_passed: int
    full_humaneval_total: int
    full_mbpp_passed: int
    full_mbpp_total: int
    full_repository_holdout_passed: int
    full_repository_holdout_total: int
    full_combined_passed: int
    full_combined_total: int
    target_language_gate_passed: bool


def _mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise DistilledQualificationError(f"{context} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise DistilledQualificationError(f"{context} keys must be strings")
    return cast(dict[str, object], dict(value))


def _string_list(value: object, *, context: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise DistilledQualificationError(f"{context} must be a string list")
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise DistilledQualificationError(f"{context} must contain non-empty strings")
    return cast(tuple[str, ...], result)


def _load_json(path: Path, *, context: str) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DistilledQualificationError(f"could not read {context}: {path}") from exc
    return _mapping(value, context=context)


def _write_json(path: Path, value: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _source_git_sha(repo_root: Path) -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DistilledQualificationError("could not inspect P9-007E source tree") from exc
    if len(sha) != 40 or any(character not in "0123456789abcdef" for character in sha):
        raise DistilledQualificationError("P9-007E source Git SHA is invalid")
    if status.strip():
        raise DistilledQualificationError("P9-007E requires a clean source tree")
    return sha


def load_qualification_authorization(
    repo_root: Path = Path("."),
) -> QualificationAuthorization:
    """Validate the exact clean replay and prove that step 185 is the sole winner."""

    evidence = _load_json(repo_root / _AUTHORIZATION_PATH, context="P9-007D replay evidence")
    expected_scalars = {
        "schema_version": 1,
        "task_id": "P9-007D",
        "status": "passed",
        "workflow_run_id": _EXPECTED_AUTH_RUN_ID,
        "workflow_run_attempt": _EXPECTED_AUTH_RUN_ATTEMPT,
        "workflow_job_id": _EXPECTED_AUTH_JOB_ID,
        "source_git_sha": _EXPECTED_AUTH_SOURCE_SHA,
        "capture_hardening_merge_sha": _EXPECTED_AUTH_SOURCE_SHA,
        "development_manifest_sha256": _EXPECTED_DEVELOPMENT_MANIFEST_SHA256,
        "membership_sha256": _EXPECTED_MEMBERSHIP_SHA256,
        "repository_holdout_evaluated": False,
        "qualification_authorized": True,
    }
    for key, expected in expected_scalars.items():
        if evidence.get(key) != expected:
            raise DistilledQualificationError(f"P9-007D authorization {key} drifted")

    trajectory = _mapping(evidence.get("trajectory"), context="authorization.trajectory")
    trajectory_expected = {
        "workflow_run_id": _EXPECTED_TRAJECTORY_RUN_ID,
        "workflow_job_id": _EXPECTED_TRAJECTORY_JOB_ID,
        "source_git_sha": _EXPECTED_TRAJECTORY_SOURCE_SHA,
        "artifact_id": _EXPECTED_TRAJECTORY_ARTIFACT_ID,
        "artifact_digest": _EXPECTED_TRAJECTORY_ARTIFACT_DIGEST,
    }
    for key, expected in trajectory_expected.items():
        if trajectory.get(key) != expected:
            raise DistilledQualificationError(f"authorization trajectory {key} drifted")

    artifact = _mapping(
        evidence.get("development_artifact"), context="authorization.development_artifact"
    )
    if (
        artifact.get("artifact_id") != _EXPECTED_DEVELOPMENT_ARTIFACT_ID
        or artifact.get("artifact_digest") != _EXPECTED_DEVELOPMENT_ARTIFACT_DIGEST
        or artifact.get("expired") is not False
    ):
        raise DistilledQualificationError("clean development artifact identity drifted")

    score_rows = evidence.get("scores")
    if isinstance(score_rows, str) or not isinstance(score_rows, Sequence):
        raise DistilledQualificationError("authorization scores must be a list")
    expected_scores = (
        (25, 32, 63, 95, False),
        (50, 30, 66, 96, False),
        (100, 31, 70, 101, False),
        (185, 34, 70, 104, True),
    )
    observed: list[tuple[int, int, int, int, bool]] = []
    for index, raw in enumerate(score_rows):
        row = _mapping(raw, context=f"authorization.scores[{index}]")
        values = (
            row.get("step"),
            row.get("humaneval_passed"),
            row.get("mbpp_passed"),
            row.get("combined_passed"),
            row.get("eligible"),
        )
        if (
            any(isinstance(value, bool) for value in values[:4])
            or any(not isinstance(value, int) for value in values[:4])
            or not isinstance(values[4], bool)
        ):
            raise DistilledQualificationError("authorization score types are invalid")
        observed.append(cast(tuple[int, int, int, int, bool], values))
    if tuple(observed) != expected_scores:
        raise DistilledQualificationError("clean development score grid drifted")

    selected = _mapping(evidence.get("selected"), context="authorization.selected")
    selected_expected = {
        "step": _SELECTED_STEP,
        "humaneval_passed": _DEVELOPMENT_HE_PASSED,
        "mbpp_passed": _DEVELOPMENT_MBPP_PASSED,
        "combined_passed": _DEVELOPMENT_COMBINED_PASSED,
        "eligible": True,
        "artifact_set_sha256": _EXPECTED_SELECTED_ARTIFACT_SET_SHA256,
        "adapter_config_sha256": _EXPECTED_SELECTED_CONFIG_SHA256,
        "adapter_model_sha256": _EXPECTED_SELECTED_MODEL_SHA256,
    }
    for key, expected in selected_expected.items():
        if selected.get(key) != expected:
            raise DistilledQualificationError(f"selected development winner {key} drifted")

    frozen = _load_json(
        repo_root / _TRAJECTORY_EVIDENCE_PATH,
        context="P9-007C trajectory evidence",
    )
    if (
        frozen.get("workflow_run_id") != _EXPECTED_TRAJECTORY_RUN_ID
        or frozen.get("workflow_job_id") != _EXPECTED_TRAJECTORY_JOB_ID
        or frozen.get("source_git_sha") != _EXPECTED_TRAJECTORY_SOURCE_SHA
    ):
        raise DistilledQualificationError("frozen trajectory lineage drifted")
    snapshots = frozen.get("snapshots")
    if isinstance(snapshots, str) or not isinstance(snapshots, Sequence):
        raise DistilledQualificationError("frozen trajectory snapshots must be a list")
    selected_rows = [
        _mapping(item, context="trajectory snapshot")
        for item in snapshots
        if isinstance(item, Mapping) and item.get("step") == _SELECTED_STEP
    ]
    if len(selected_rows) != 1:
        raise DistilledQualificationError("frozen trajectory lacks one selected snapshot")
    frozen_selected = selected_rows[0]
    if (
        frozen_selected.get("artifact_set_sha256") != _EXPECTED_SELECTED_ARTIFACT_SET_SHA256
        or frozen_selected.get("adapter_config_sha256") != _EXPECTED_SELECTED_CONFIG_SHA256
        or frozen_selected.get("adapter_model_sha256") != _EXPECTED_SELECTED_MODEL_SHA256
    ):
        raise DistilledQualificationError("selected snapshot identity disagrees with trajectory")

    return QualificationAuthorization(
        workflow_run_id=_EXPECTED_AUTH_RUN_ID,
        workflow_run_attempt=_EXPECTED_AUTH_RUN_ATTEMPT,
        workflow_job_id=_EXPECTED_AUTH_JOB_ID,
        source_git_sha=_EXPECTED_AUTH_SOURCE_SHA,
        selected_step=_SELECTED_STEP,
        selected_artifact_set_sha256=_EXPECTED_SELECTED_ARTIFACT_SET_SHA256,
        selected_adapter_config_sha256=_EXPECTED_SELECTED_CONFIG_SHA256,
        selected_adapter_model_sha256=_EXPECTED_SELECTED_MODEL_SHA256,
        development_artifact_id=_EXPECTED_DEVELOPMENT_ARTIFACT_ID,
        development_artifact_digest=_EXPECTED_DEVELOPMENT_ARTIFACT_DIGEST,
    )


def qualification_membership(
    repo_root: Path = Path("."),
) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    """Return the untouched P9-004 qualification membership after full validation."""

    manifest = load_development_manifest(repo_root)
    membership = _mapping(manifest.get("membership"), context="qualification membership")
    he = _mapping(membership.get("humaneval"), context="qualification humaneval")
    mb = _mapping(membership.get("mbpp"), context="qualification mbpp")
    rh = _mapping(membership.get("repository_holdout"), context="qualification holdout")
    he_ids = frozenset(_string_list(he.get("qualification"), context="humaneval qualification"))
    mb_ids = frozenset(_string_list(mb.get("qualification"), context="mbpp qualification"))
    rh_ids = frozenset(
        _string_list(rh.get("qualification"), context="repository holdout qualification")
    )
    if (len(he_ids), len(mb_ids), len(rh_ids)) != (
        _EXPECTED_QUAL_HE,
        _EXPECTED_QUAL_MBPP,
        _EXPECTED_QUAL_HOLDOUT,
    ):
        raise DistilledQualificationError("qualification membership cardinality drifted")
    return he_ids, mb_ids, rh_ids


def _validate_selected_snapshot(
    trajectory: TrajectoryIdentity,
    snapshot: SnapshotIdentity,
    authorization: QualificationAuthorization,
) -> None:
    if snapshot.step != _SELECTED_STEP or authorization.selected_step != _SELECTED_STEP:
        raise DistilledQualificationError("P9-007E may evaluate only selected step 185")
    if trajectory.adapter_id != "language/python/p9-distilled-v4-2000-r8-lr1e5":
        raise DistilledQualificationError("selected adapter ID drifted")
    if (
        snapshot.artifact_set_sha256 != authorization.selected_artifact_set_sha256
        or snapshot.adapter_config_sha256 != authorization.selected_adapter_config_sha256
        or snapshot.adapter_model_sha256 != authorization.selected_adapter_model_sha256
    ):
        raise DistilledQualificationError("selected local snapshot does not match authorization")


def _qualification_context(
    training_output: Path,
    *,
    repo_root: Path,
    harness: DirectExecutionHarness | None = None,
) -> tuple[
    QualificationAuthorization,
    TrajectoryIdentity,
    dict[int, Path],
    SnapshotIdentity,
    EvaluationConfig,
    FrozenEvaluationSettings,
    BaseModelIdentity,
    str,
    HumanEvalEvaluator,
    tuple[HumanEvalProblem, ...],
    MBPPEvaluator,
    tuple[MBPPProblem, ...],
    RepositoryHoldoutEvaluator,
    tuple[RepositoryHoldoutTask, ...],
]:
    authorization = load_qualification_authorization(repo_root)
    trajectory, snapshot_dirs = load_local_trajectory(training_output, repo_root=repo_root)
    snapshot = trajectory.snapshot(_SELECTED_STEP)
    _validate_selected_snapshot(trajectory, snapshot, authorization)
    evaluation, settings, base_model, system_prompt = _evaluation_context(trajectory)
    evaluation = replace(
        evaluation,
        suites=("humaneval", "mbpp", "repository-holdout"),
        output_dir=_OUTPUT_ROOT.as_posix(),
    )
    adapter = AdapterIdentity(family="language", adapter_id=trajectory.adapter_id)
    humaneval = HumanEvalEvaluator(
        evaluation,
        base_model=base_model,
        adapter=adapter,
        settings=settings,
        harness=harness,
    )
    mbpp = MBPPEvaluator(
        evaluation,
        base_model=base_model,
        adapter=adapter,
        settings=settings,
        harness=harness,
    )
    holdout = RepositoryHoldoutEvaluator(
        evaluation,
        base_model=base_model,
        adapter=adapter,
        settings=settings,
        harness=harness,
    )
    he_ids, mb_ids, rh_ids = qualification_membership(repo_root)
    he_problems = tuple(
        problem for problem in humaneval.load_problems() if problem.task_id in he_ids
    )
    mb_problems = tuple(problem for problem in mbpp.load_problems() if problem.task_id in mb_ids)
    rh_tasks = tuple(task for task in holdout.suite.tasks if task.problem_id in rh_ids)
    if (
        {item.task_id for item in he_problems} != he_ids
        or {item.task_id for item in mb_problems} != mb_ids
        or {item.problem_id for item in rh_tasks} != rh_ids
    ):
        raise DistilledQualificationError("loaded qualification benchmark membership drifted")
    return (
        authorization,
        trajectory,
        snapshot_dirs,
        snapshot,
        evaluation,
        settings,
        base_model,
        system_prompt,
        humaneval,
        he_problems,
        mbpp,
        mb_problems,
        holdout,
        rh_tasks,
    )


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _generation_contract(
    *,
    authorization: QualificationAuthorization,
    trajectory: TrajectoryIdentity,
    snapshot: SnapshotIdentity,
    base_model: BaseModelIdentity,
    settings: FrozenEvaluationSettings,
    system_prompt: str,
) -> str:
    payload = {
        "schema_version": 1,
        "task_id": "P9-007E",
        "stage": "winner-only-qualification-v1",
        "authorization_run_id": authorization.workflow_run_id,
        "authorization_job_id": authorization.workflow_job_id,
        "selected_step": authorization.selected_step,
        "trajectory": trajectory.label,
        "adapter_id": trajectory.adapter_id,
        "adapter_model_sha256": snapshot.adapter_model_sha256,
        "artifact_set_sha256": snapshot.artifact_set_sha256,
        "base_model": asdict(base_model),
        "evaluation_settings_sha256": evaluation_settings_sha256(settings),
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
        "development_manifest_sha256": _EXPECTED_DEVELOPMENT_MANIFEST_SHA256,
        "membership_sha256": _EXPECTED_MEMBERSHIP_SHA256,
        "qualification_counts": {
            "humaneval": _EXPECTED_QUAL_HE,
            "mbpp": _EXPECTED_QUAL_MBPP,
            "repository_holdout": _EXPECTED_QUAL_HOLDOUT,
            "combined": _EXPECTED_QUAL_TOTAL,
        },
    }
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


class _CheckpointOnlyGenerator(BaselineGenerator):
    """Never regenerate missing protected responses during the scoring stage."""

    def generate(self, *, system_prompt: str, user_prompt: str) -> BaselineGeneratedResponse:
        del system_prompt, user_prompt
        raise DistilledQualificationError(
            "P9-007E scoring is missing a transported GPU response; refusing regeneration"
        )


def generate_qualification(
    *,
    training_output: Path,
    repo_root: Path = Path("."),
    device_index: int = 0,
) -> Path:
    """Consume the frozen 500-task qualification slice for selected step 185 only."""

    source_git_sha = _source_git_sha(repo_root)
    (
        authorization,
        trajectory,
        snapshot_dirs,
        snapshot,
        _evaluation,
        settings,
        base_model,
        system_prompt,
        humaneval,
        he_problems,
        mbpp,
        mb_problems,
        holdout,
        rh_tasks,
    ) = _qualification_context(training_output, repo_root=repo_root)
    generator = DistilledSnapshotGenerator(
        snapshot_dirs=snapshot_dirs,
        base_model=base_model,
        settings=settings,
        device_index=device_index,
    )
    generator.select_step(_SELECTED_STEP)
    contract = _generation_contract(
        authorization=authorization,
        trajectory=trajectory,
        snapshot=snapshot,
        base_model=base_model,
        settings=settings,
        system_prompt=system_prompt,
    )
    _OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    he_responses = _generate_items(
        suite_id="humaneval-qualification",
        prompts=tuple(
            (problem.task_id, humaneval.prompt_for(problem).user_content)
            for problem in he_problems
        ),
        generator=generator,
        system_prompt=system_prompt,
        generation_contract=contract,
        output_dir=_OUTPUT_ROOT,
    )
    mb_responses = _generate_items(
        suite_id="mbpp-qualification",
        prompts=tuple(
            (problem.task_id, mbpp.prompt_for(problem).user_content) for problem in mb_problems
        ),
        generator=generator,
        system_prompt=system_prompt,
        generation_contract=contract,
        output_dir=_OUTPUT_ROOT,
    )
    rh_responses = _generate_items(
        suite_id="repository-holdout-qualification",
        prompts=tuple(
            (task.problem_id, holdout.prompt_for(task).user_content) for task in rh_tasks
        ),
        generator=generator,
        system_prompt=system_prompt,
        generation_contract=contract,
        output_dir=_OUTPUT_ROOT,
    )
    if (len(he_responses), len(mb_responses), len(rh_responses)) != (
        _EXPECTED_QUAL_HE,
        _EXPECTED_QUAL_MBPP,
        _EXPECTED_QUAL_HOLDOUT,
    ):
        raise DistilledQualificationError("qualification generation cardinality is incomplete")
    stage = {
        "schema_version": 1,
        "task_id": "P9-007E",
        "stage": "qualification-generation",
        "source_git_sha": source_git_sha,
        "authorization": asdict(authorization),
        "selected_step": _SELECTED_STEP,
        "selected_artifact_set_sha256": snapshot.artifact_set_sha256,
        "selected_adapter_model_sha256": snapshot.adapter_model_sha256,
        "generation_contract_sha256": contract,
        "development_manifest_sha256": _EXPECTED_DEVELOPMENT_MANIFEST_SHA256,
        "membership_sha256": _EXPECTED_MEMBERSHIP_SHA256,
        "humaneval_requests": len(he_responses),
        "mbpp_requests": len(mb_responses),
        "repository_holdout_requests": len(rh_responses),
        "combined_requests": len(he_responses) + len(mb_responses) + len(rh_responses),
        "runner_up_requests": 0,
        "qualification_consumed": True,
    }
    return _write_json(_OUTPUT_ROOT / "qualification-generation-stage.json", stage)


def evaluate_target_language_gate(
    *, humaneval_passed: int, mbpp_passed: int, repository_holdout_passed: int
) -> QualificationScore:
    """Combine untouched qualification results with frozen development winner results."""

    for value, total, name in (
        (humaneval_passed, _EXPECTED_QUAL_HE, "humaneval"),
        (mbpp_passed, _EXPECTED_QUAL_MBPP, "mbpp"),
        (repository_holdout_passed, _EXPECTED_QUAL_HOLDOUT, "repository_holdout"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= total:
            raise DistilledQualificationError(f"invalid qualification {name} pass count")
    policy = load_frozen_python_promotion_policy()
    full_he = _DEVELOPMENT_HE_PASSED + humaneval_passed
    full_mb = _DEVELOPMENT_MBPP_PASSED + mbpp_passed
    full_rh = repository_holdout_passed
    qualification_combined = humaneval_passed + mbpp_passed + repository_holdout_passed
    full_combined = _DEVELOPMENT_COMBINED_PASSED + qualification_combined
    target_gate = (
        full_he >= policy.humaneval.passed
        and full_mb >= policy.mbpp.passed
        and full_rh >= policy.repository_holdout.passed
        and full_combined >= policy.minimum_combined_passed
    )
    return QualificationScore(
        selected_step=_SELECTED_STEP,
        qualification_humaneval_passed=humaneval_passed,
        qualification_humaneval_total=_EXPECTED_QUAL_HE,
        qualification_mbpp_passed=mbpp_passed,
        qualification_mbpp_total=_EXPECTED_QUAL_MBPP,
        qualification_repository_holdout_passed=repository_holdout_passed,
        qualification_repository_holdout_total=_EXPECTED_QUAL_HOLDOUT,
        qualification_combined_passed=qualification_combined,
        qualification_combined_total=_EXPECTED_QUAL_TOTAL,
        full_humaneval_passed=full_he,
        full_humaneval_total=policy.humaneval.total,
        full_mbpp_passed=full_mb,
        full_mbpp_total=policy.mbpp.total,
        full_repository_holdout_passed=full_rh,
        full_repository_holdout_total=policy.repository_holdout.total,
        full_combined_passed=full_combined,
        full_combined_total=policy.combined.total,
        target_language_gate_passed=target_gate,
    )


def score_qualification(
    *,
    training_output: Path,
    repo_root: Path = Path("."),
) -> Path:
    """Execute only persisted winner responses and score the protected qualification slice."""

    source_git_sha = _source_git_sha(repo_root)
    harness = DirectExecutionHarness(allow_reduced_isolation=True)
    (
        authorization,
        trajectory,
        _snapshot_dirs,
        snapshot,
        _evaluation,
        settings,
        base_model,
        system_prompt,
        humaneval,
        he_problems,
        mbpp,
        mb_problems,
        holdout,
        rh_tasks,
    ) = _qualification_context(training_output, repo_root=repo_root, harness=harness)
    contract = _generation_contract(
        authorization=authorization,
        trajectory=trajectory,
        snapshot=snapshot,
        base_model=base_model,
        settings=settings,
        system_prompt=system_prompt,
    )
    stage = _load_json(
        _OUTPUT_ROOT / "qualification-generation-stage.json",
        context="P9-007E generation stage",
    )
    expected_stage = {
        "task_id": "P9-007E",
        "stage": "qualification-generation",
        "source_git_sha": source_git_sha,
        "selected_step": _SELECTED_STEP,
        "selected_artifact_set_sha256": snapshot.artifact_set_sha256,
        "selected_adapter_model_sha256": snapshot.adapter_model_sha256,
        "generation_contract_sha256": contract,
        "development_manifest_sha256": _EXPECTED_DEVELOPMENT_MANIFEST_SHA256,
        "membership_sha256": _EXPECTED_MEMBERSHIP_SHA256,
        "humaneval_requests": _EXPECTED_QUAL_HE,
        "mbpp_requests": _EXPECTED_QUAL_MBPP,
        "repository_holdout_requests": _EXPECTED_QUAL_HOLDOUT,
        "combined_requests": _EXPECTED_QUAL_TOTAL,
        "runner_up_requests": 0,
        "qualification_consumed": True,
    }
    for key, expected in expected_stage.items():
        if stage.get(key) != expected:
            raise DistilledQualificationError(f"qualification generation stage {key} drifted")

    checkpoint_only = _CheckpointOnlyGenerator()
    he_responses = _generate_items(
        suite_id="humaneval-qualification",
        prompts=tuple(
            (problem.task_id, humaneval.prompt_for(problem).user_content)
            for problem in he_problems
        ),
        generator=checkpoint_only,
        system_prompt=system_prompt,
        generation_contract=contract,
        output_dir=_OUTPUT_ROOT,
    )
    mb_responses = _generate_items(
        suite_id="mbpp-qualification",
        prompts=tuple(
            (problem.task_id, mbpp.prompt_for(problem).user_content) for problem in mb_problems
        ),
        generator=checkpoint_only,
        system_prompt=system_prompt,
        generation_contract=contract,
        output_dir=_OUTPUT_ROOT,
    )
    rh_responses = _generate_items(
        suite_id="repository-holdout-qualification",
        prompts=tuple(
            (task.problem_id, holdout.prompt_for(task).user_content) for task in rh_tasks
        ),
        generator=checkpoint_only,
        system_prompt=system_prompt,
        generation_contract=contract,
        output_dir=_OUTPUT_ROOT,
    )

    he_result = humaneval.evaluate_suite(
        he_problems,
        tuple(
            HumanEvalCompletion(
                task_id=problem.task_id,
                generated_text=response.generated_text,
                generation=response.generation,
            )
            for problem, response in zip(he_problems, he_responses, strict=True)
        ),
    )
    mb_result = mbpp.evaluate_suite(
        mb_problems,
        tuple(
            MBPPCompletion(
                task_id=problem.task_id,
                generated_text=response.generated_text,
                generation=response.generation,
            )
            for problem, response in zip(mb_problems, mb_responses, strict=True)
        ),
    )
    rh_result = holdout.evaluate_suite(
        tuple(
            RepositoryHoldoutCompletion(
                problem_id=task.problem_id,
                generated_text=response.generated_text,
                generation=response.generation,
            )
            for task, response in zip(rh_tasks, rh_responses, strict=True)
        )
    )
    if (
        he_result.aggregate.harness_errors
        or mb_result.aggregate.harness_errors
        or rh_result.aggregate.harness_errors
    ):
        raise DistilledQualificationError("P9-007E scoring encountered harness errors")

    humaneval.write_artifacts(he_result, _OUTPUT_ROOT / "humaneval-qualification")
    mbpp.write_artifacts(mb_result, _OUTPUT_ROOT / "mbpp-qualification")
    holdout.write_artifacts(rh_result, _OUTPUT_ROOT / "repository-holdout-qualification")
    score = evaluate_target_language_gate(
        humaneval_passed=he_result.aggregate.passed,
        mbpp_passed=mb_result.aggregate.passed,
        repository_holdout_passed=rh_result.aggregate.passed,
    )
    policy = load_frozen_python_promotion_policy()
    payload = {
        "schema_version": 1,
        "task_id": "P9-007E",
        "stage": "winner-only-qualification-score",
        "source_git_sha": source_git_sha,
        "authorization": asdict(authorization),
        "winner_only": True,
        "runner_up_evaluated": False,
        "repository_holdout_evaluated": True,
        "selected_artifact_set_sha256": snapshot.artifact_set_sha256,
        "selected_adapter_model_sha256": snapshot.adapter_model_sha256,
        "generation_contract_sha256": contract,
        "score": asdict(score),
        "promotion_policy_id": policy.policy_id,
        "promotion_minimum_combined_passed": policy.minimum_combined_passed,
        "target_language_gate_passed": score.target_language_gate_passed,
        "next_action": (
            "run winner-only Phase-8 preservation gates"
            if score.target_language_gate_passed
            else "stop P9-007; do not evaluate any runner-up"
        ),
    }
    return _write_json(_OUTPUT_ROOT / "qualification-score.json", payload)


def verify_qualification(
    *,
    training_output: Path,
    repo_root: Path = Path("."),
) -> dict[str, object]:
    """Recompute all persisted P9-007E decision arithmetic without generation or execution."""

    authorization = load_qualification_authorization(repo_root)
    trajectory, _snapshot_dirs = load_local_trajectory(training_output, repo_root=repo_root)
    snapshot = trajectory.snapshot(_SELECTED_STEP)
    _validate_selected_snapshot(trajectory, snapshot, authorization)
    stage = _load_json(
        _OUTPUT_ROOT / "qualification-generation-stage.json",
        context="P9-007E generation stage",
    )
    score_payload = _load_json(
        _OUTPUT_ROOT / "qualification-score.json",
        context="P9-007E score",
    )
    if stage.get("runner_up_requests") != 0 or score_payload.get("runner_up_evaluated") is not False:
        raise DistilledQualificationError("P9-007E evidence indicates runner-up evaluation")
    if stage.get("combined_requests") != _EXPECTED_QUAL_TOTAL:
        raise DistilledQualificationError("P9-007E generation count drifted")
    persisted = _mapping(score_payload.get("score"), context="qualification score")
    recomputed = evaluate_target_language_gate(
        humaneval_passed=cast(int, persisted.get("qualification_humaneval_passed")),
        mbpp_passed=cast(int, persisted.get("qualification_mbpp_passed")),
        repository_holdout_passed=cast(
            int, persisted.get("qualification_repository_holdout_passed")
        ),
    )
    if persisted != asdict(recomputed):
        raise DistilledQualificationError("persisted qualification score arithmetic drifted")
    if score_payload.get("target_language_gate_passed") is not recomputed.target_language_gate_passed:
        raise DistilledQualificationError("persisted target-language decision drifted")
    return {
        "selected_step": _SELECTED_STEP,
        "qualification_requests": _EXPECTED_QUAL_TOTAL,
        "runner_up_evaluated": False,
        "target_language_gate_passed": recomputed.target_language_gate_passed,
        "full_humaneval_passed": recomputed.full_humaneval_passed,
        "full_mbpp_passed": recomputed.full_mbpp_passed,
        "full_repository_holdout_passed": recomputed.full_repository_holdout_passed,
        "full_combined_passed": recomputed.full_combined_passed,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run winner-only P9-007E qualification")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--repo-root", type=Path, default=Path("."))
    for name in ("generate", "score", "verify"):
        item = sub.add_parser(name)
        item.add_argument("--training-output", type=Path, required=True)
        item.add_argument("--repo-root", type=Path, default=Path("."))
        if name == "generate":
            item.add_argument("--device-index", type=int, default=0)
    return parser


def _fail(message: str) -> NoReturn:
    raise DistilledQualificationError(message)


def main() -> None:
    args = _parser().parse_args()
    if args.command == "validate":
        authorization = load_qualification_authorization(args.repo_root)
        he_ids, mb_ids, rh_ids = qualification_membership(args.repo_root)
        print(
            json.dumps(
                {
                    "authorization": asdict(authorization),
                    "qualification_counts": {
                        "humaneval": len(he_ids),
                        "mbpp": len(mb_ids),
                        "repository_holdout": len(rh_ids),
                        "combined": len(he_ids) + len(mb_ids) + len(rh_ids),
                    },
                    "winner_only": True,
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "generate":
        print(
            generate_qualification(
                training_output=args.training_output,
                repo_root=args.repo_root,
                device_index=args.device_index,
            )
        )
    elif args.command == "score":
        print(score_qualification(training_output=args.training_output, repo_root=args.repo_root))
    elif args.command == "verify":
        print(
            json.dumps(
                verify_qualification(
                    training_output=args.training_output,
                    repo_root=args.repo_root,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    else:  # pragma: no cover - argparse enforces the command choices
        _fail(f"unsupported command {args.command!r}")


if __name__ == "__main__":
    main()
