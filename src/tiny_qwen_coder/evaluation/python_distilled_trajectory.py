"""Development-only evaluation for the frozen P9-007C distilled trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, NoReturn, Protocol, cast

import torch
from torch import nn

from tiny_qwen_coder.evaluation._baseline_generation import BaselineGenerator
from tiny_qwen_coder.evaluation._baseline_runner import _generate_items
from tiny_qwen_coder.evaluation._baseline_types import BaselineGeneratedResponse
from tiny_qwen_coder.evaluation._python_p0_generation import _parameter_dtypes, _resolved_revision
from tiny_qwen_coder.evaluation.execution import DirectExecutionHarness
from tiny_qwen_coder.evaluation.humaneval import HumanEvalCompletion
from tiny_qwen_coder.evaluation.mbpp import MBPPCompletion
from tiny_qwen_coder.evaluation.python_minimum_intervention import (
    SnapshotIdentity,
    TrajectoryIdentity,
    _evaluation_context,
    _load_development_problems,
    load_development_manifest,
)
from tiny_qwen_coder.evaluation.results import GenerationStats
from tiny_qwen_coder.evaluation.settings import evaluation_settings_sha256
from tiny_qwen_coder.identities import BaseModelIdentity
from tiny_qwen_coder.reproducibility import seed_everything
from tiny_qwen_coder.training.distilled_trajectory import validate_distilled_trajectory

_LABEL = "distilled-v4-2000-r8-lr1e5"
_LEARNING_RATE = 0.00001
_EXPECTED_STEPS = (25, 50, 100, 185)
_EXPECTED_HE_TOTAL = 45
_EXPECTED_MBPP_TOTAL = 130
_EXPECTED_COMBINED_TOTAL = 175
_EXPECTED_BASE_HE = 33
_EXPECTED_BASE_MBPP = 70
_EXPECTED_BASE_COMBINED = 103
_EXPECTED_MIN_COMBINED = 104
_EXPECTED_DEVELOPMENT_SHA256 = "260682d773640b28673357c7a441474656bbaffc67f0b5fcbe45eca3a07283de"
_EXPECTED_MEMBERSHIP_SHA256 = "c8765b7a3a69f134be4066e274a64640a15f2d05858374035432207b3521498b"
_OUTPUT_ROOT = Path("artifacts/eval/python/p9-distilled-v4-2000-development-v1")
_ENABLE_THINKING = False


class DistilledTrajectoryEvaluationError(RuntimeError):
    """Raised when P9-007C development evidence violates the frozen contract."""


@dataclass(frozen=True, slots=True)
class DistilledDevelopmentScore:
    label: str
    learning_rate: float
    step: int
    humaneval_passed: int
    humaneval_total: int
    mbpp_passed: int
    mbpp_total: int
    combined_passed: int
    combined_total: int
    eligible: bool


class _GenerateCapable(Protocol):
    def generate(self, **kwargs: object) -> torch.Tensor: ...


class _Tokenizer(Protocol):
    chat_template: object

    def apply_chat_template(self, *args: object, **kwargs: object) -> object: ...

    def decode(self, token_ids: list[int], **kwargs: object) -> object: ...


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise DistilledTrajectoryEvaluationError(f"{context} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise DistilledTrajectoryEvaluationError(f"{context} keys must be strings")
    return cast(dict[str, object], dict(value))


def _integer(mapping: Mapping[str, object], key: str, *, context: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise DistilledTrajectoryEvaluationError(f"{context}.{key} must be an integer")
    return value


def _string(mapping: Mapping[str, object], key: str, *, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise DistilledTrajectoryEvaluationError(f"{context}.{key} must be a non-empty string")
    return value


def _load_json(path: Path, *, context: str) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DistilledTrajectoryEvaluationError(f"could not read {context}: {path}") from exc
    return _mapping(value, context=context)


def _snapshot_identity(
    *, training_output: Path, snapshot_row: Mapping[str, object], expected_step: int
) -> SnapshotIdentity:
    step = _integer(snapshot_row, "step", context="training snapshot")
    if step != expected_step:
        raise DistilledTrajectoryEvaluationError("P9-007C training snapshot step drifted")
    relative = Path(_string(snapshot_row, "directory", context="training snapshot"))
    directory = (training_output / relative).resolve()
    if not directory.is_relative_to(training_output.resolve()) or not directory.is_dir():
        raise DistilledTrajectoryEvaluationError("P9-007C snapshot directory is missing or escaping")

    files_value = snapshot_row.get("files")
    if isinstance(files_value, str) or not isinstance(files_value, Sequence):
        raise DistilledTrajectoryEvaluationError("P9-007C snapshot files must be a list")
    files = tuple(_mapping(item, context="snapshot file") for item in files_value)
    by_path = {_string(item, "path", context="snapshot file"): item for item in files}
    if len(by_path) != len(files):
        raise DistilledTrajectoryEvaluationError("P9-007C snapshot file inventory contains duplicates")
    config_row = by_path.get("adapter_config.json")
    model_row = by_path.get("adapter_model.safetensors")
    if config_row is None or model_row is None:
        raise DistilledTrajectoryEvaluationError("P9-007C snapshot lacks canonical adapter artifacts")

    canonical_files: list[dict[str, object]] = []
    for item in files:
        relative_file = Path(_string(item, "path", context="snapshot file"))
        path = (directory / relative_file).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise DistilledTrajectoryEvaluationError("P9-007C snapshot file is missing or escaping")
        size = _integer(item, "size_bytes", context="snapshot file")
        digest = _string(item, "sha256", context="snapshot file")
        if path.stat().st_size != size or _sha256_file(path) != digest:
            raise DistilledTrajectoryEvaluationError("P9-007C snapshot file integrity drifted")
        canonical_files.append(dict(item))
    artifact_set = hashlib.sha256(_canonical_json(canonical_files).encode()).hexdigest()
    if snapshot_row.get("artifact_set_sha256") != artifact_set:
        raise DistilledTrajectoryEvaluationError("P9-007C snapshot artifact-set SHA drifted")

    return SnapshotIdentity(
        step=step,
        adapter_config_sha256=_string(config_row, "sha256", context="adapter config"),
        adapter_config_size_bytes=_integer(config_row, "size_bytes", context="adapter config"),
        adapter_model_sha256=_string(model_row, "sha256", context="adapter model"),
        adapter_model_size_bytes=_integer(model_row, "size_bytes", context="adapter model"),
        artifact_set_sha256=artifact_set,
    )


def load_local_trajectory(
    training_output: Path, *, repo_root: Path = Path(".")
) -> tuple[TrajectoryIdentity, dict[int, Path]]:
    """Validate one completed local P9-007C training output and recover four snapshots."""

    validation = validate_distilled_trajectory(repo_root=repo_root)
    report = _load_json(training_output / "training-report.json", context="P9-007C training report")
    if report.get("task_id") != "P9-007C" or report.get("study_id") != validation.study_id:
        raise DistilledTrajectoryEvaluationError("P9-007C training report identity drifted")
    if report.get("adapter_id") != validation.adapter_id:
        raise DistilledTrajectoryEvaluationError("P9-007C adapter identity drifted")
    if report.get("dataset_manifest_sha256") != validation.dataset_manifest_sha256:
        raise DistilledTrajectoryEvaluationError("P9-007C dataset identity drifted")
    if report.get("source_output_sha256") != validation.source_output_sha256:
        raise DistilledTrajectoryEvaluationError("P9-007C teacher source identity drifted")
    if report.get("trajectory_max_steps") != 185 or report.get("global_steps") != 185:
        raise DistilledTrajectoryEvaluationError("P9-007C trajectory did not complete 185 steps")
    if report.get("checkpoint_steps") != list(_EXPECTED_STEPS) or report.get("snapshot_count") != 4:
        raise DistilledTrajectoryEvaluationError("P9-007C snapshot grid drifted")
    if report.get("promotable") is not False:
        raise DistilledTrajectoryEvaluationError("P9-007C training output must be development-only")

    rows_value = report.get("snapshots")
    if isinstance(rows_value, str) or not isinstance(rows_value, Sequence):
        raise DistilledTrajectoryEvaluationError("P9-007C training report snapshots must be a list")
    rows = tuple(_mapping(item, context="training snapshot") for item in rows_value)
    if len(rows) != len(_EXPECTED_STEPS):
        raise DistilledTrajectoryEvaluationError("P9-007C training report snapshot count drifted")
    snapshots = tuple(
        _snapshot_identity(training_output=training_output, snapshot_row=row, expected_step=step)
        for row, step in zip(rows, _EXPECTED_STEPS, strict=True)
    )
    snapshot_dirs = {
        step: training_output / _string(row, "directory", context="training snapshot")
        for row, step in zip(rows, _EXPECTED_STEPS, strict=True)
    }

    run_manifest = _load_json(training_output / "run-manifest.json", context="P9-007C run manifest")
    git = _mapping(run_manifest.get("git"), context="run manifest.git")
    source_sha = _string(git, "sha", context="run manifest.git")
    if len(source_sha) != 40:
        raise DistilledTrajectoryEvaluationError("P9-007C source Git SHA is invalid")
    run_id = _string(report, "run_id", context="P9-007C training report")

    trajectory = TrajectoryIdentity(
        label=_LABEL,
        learning_rate=_LEARNING_RATE,
        adapter_id=validation.adapter_id,
        training_run_id=run_id,
        training_workflow_run_id=0,
        training_source_git_sha=source_sha,
        training_artifact_id=0,
        training_artifact_name="local-p9-007c-trajectory",
        training_artifact_digest=f"sha256:{hashlib.sha256(_canonical_json([asdict(item) for item in snapshots]).encode()).hexdigest()}",
        snapshots=snapshots,
    )
    return trajectory, snapshot_dirs


class DistilledSnapshotGenerator:
    """Load one BF16 base model and the four frozen P9-007C adapter snapshots."""

    def __init__(
        self,
        *,
        snapshot_dirs: Mapping[int, Path],
        base_model: BaseModelIdentity,
        settings: Any,
        device_index: int = 0,
    ) -> None:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise DistilledTrajectoryEvaluationError("P9-007C generation requires BF16 CUDA")
        if not 0 <= device_index < torch.cuda.device_count():
            raise DistilledTrajectoryEvaluationError("invalid P9-007C CUDA device index")
        if settings.generation.decoding_strategy != "greedy":
            raise DistilledTrajectoryEvaluationError("P9-007C requires frozen greedy decoding")
        if tuple(sorted(snapshot_dirs)) != _EXPECTED_STEPS:
            raise DistilledTrajectoryEvaluationError("P9-007C snapshot directory grid drifted")

        from peft import PeftModel
        from transformers import AutoModelForMultimodalLM, AutoTokenizer, PreTrainedTokenizerBase

        self._device = torch.device("cuda", device_index)
        self._settings = settings
        torch.cuda.set_device(self._device)
        seed_everything(settings.seed)
        tokenizer_obj: object = AutoTokenizer.from_pretrained(
            base_model.tokenizer_repository,
            revision=base_model.tokenizer_revision,
        )
        if not isinstance(tokenizer_obj, PreTrainedTokenizerBase):
            raise DistilledTrajectoryEvaluationError("Transformers returned unexpected tokenizer")
        tokenizer = cast(_Tokenizer, tokenizer_obj)
        if not isinstance(tokenizer.chat_template, str) or not tokenizer.chat_template:
            raise DistilledTrajectoryEvaluationError("canonical tokenizer lacks chat template")
        self._tokenizer = tokenizer

        loaded: object = cast(Any, AutoModelForMultimodalLM).from_pretrained(
            base_model.repository,
            revision=base_model.revision,
            dtype=torch.bfloat16,
            device_map={"": device_index},
            low_cpu_mem_usage=True,
        )
        if not isinstance(loaded, nn.Module):
            raise DistilledTrajectoryEvaluationError("Transformers returned unexpected model")
        base = loaded
        base.eval()
        if _resolved_revision(base) != base_model.revision:
            raise DistilledTrajectoryEvaluationError("loaded base revision is not canonical")
        if _parameter_dtypes(base) != ("torch.bfloat16",):
            raise DistilledTrajectoryEvaluationError("P9-007C base model is not pure BF16")

        first = _EXPECTED_STEPS[0]
        adapted_obj: object = cast(Any, PeftModel).from_pretrained(
            base,
            str(snapshot_dirs[first]),
            adapter_name=self._adapter_name(first),
            is_trainable=False,
        )
        if not isinstance(adapted_obj, nn.Module):
            raise DistilledTrajectoryEvaluationError("PEFT returned unexpected model")
        self._model = adapted_obj
        load_adapter = getattr(self._model, "load_adapter", None)
        if not callable(load_adapter):
            raise DistilledTrajectoryEvaluationError("PEFT model cannot load multiple adapters")
        for step in _EXPECTED_STEPS[1:]:
            cast(Any, self._model).load_adapter(
                str(snapshot_dirs[step]),
                adapter_name=self._adapter_name(step),
                is_trainable=False,
            )
        self._model.eval()
        self._model.requires_grad_(False)
        self._active_step: int | None = None

    @staticmethod
    def _adapter_name(step: int) -> str:
        return f"step-{step:04d}"

    def select_step(self, step: int) -> None:
        if step not in _EXPECTED_STEPS:
            raise DistilledTrajectoryEvaluationError(f"non-frozen P9-007C step {step}")
        setter = getattr(self._model, "set_adapter", None)
        if not callable(setter):
            raise DistilledTrajectoryEvaluationError("PEFT model cannot switch adapters")
        adapter_name = self._adapter_name(step)
        setter(adapter_name, inference_mode=True)
        status_getter = getattr(self._model, "get_model_status", None)
        if not callable(status_getter):
            raise DistilledTrajectoryEvaluationError("PEFT model lacks status reporting")
        status = status_getter()
        active = tuple(getattr(status, "active_adapters", ()))
        requires_grad = getattr(status, "requires_grad", None)
        if active != (adapter_name,):
            raise DistilledTrajectoryEvaluationError("PEFT active adapter does not match step")
        if (
            getattr(status, "trainable_params", None) != 0
            or not isinstance(requires_grad, Mapping)
            or requires_grad.get(adapter_name) is not False
        ):
            raise DistilledTrajectoryEvaluationError("P9-007C adapter unexpectedly trainable")
        self._active_step = step
        seed_everything(self._settings.seed)

    def _prepare_inputs(self, system_prompt: str, user_prompt: str) -> dict[str, torch.Tensor]:
        encoded = self._tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=_ENABLE_THINKING,
        )
        if not isinstance(encoded, Mapping):
            raise DistilledTrajectoryEvaluationError("chat template returned non-mapping")
        inputs = {
            str(key): value.to(self._device)
            for key, value in encoded.items()
            if isinstance(value, torch.Tensor)
        }
        ids = inputs.get("input_ids")
        if ids is None or ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] <= 0:
            raise DistilledTrajectoryEvaluationError("P9-007C prompt tokenization is invalid")
        return inputs

    def generate(self, *, system_prompt: str, user_prompt: str) -> BaselineGeneratedResponse:
        if self._active_step is None:
            raise DistilledTrajectoryEvaluationError("P9-007C adapter step was not selected")
        inputs = self._prepare_inputs(system_prompt, user_prompt)
        prompt_tokens = int(inputs["input_ids"].shape[1])
        kwargs: dict[str, object] = dict(inputs)
        kwargs.update(
            {
                "max_new_tokens": self._settings.generation.max_new_tokens,
                "do_sample": False,
                "num_beams": 1,
                "use_cache": True,
            }
        )
        torch.cuda.synchronize(self._device)
        started = time.perf_counter()
        with torch.inference_mode():
            output = cast(_GenerateCapable, self._model).generate(**kwargs)
        torch.cuda.synchronize(self._device)
        latency = time.perf_counter() - started
        if not isinstance(output, torch.Tensor) or output.ndim != 2 or output.shape[0] != 1:
            raise DistilledTrajectoryEvaluationError("model.generate returned invalid tensor")
        if output.shape[1] <= prompt_tokens or latency <= 0:
            raise DistilledTrajectoryEvaluationError("P9-007C generation produced no completion")
        token_ids = [int(item) for item in output[0, prompt_tokens:].detach().cpu().tolist()]
        decoded = self._tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not isinstance(decoded, str) or not token_ids:
            raise DistilledTrajectoryEvaluationError("tokenizer returned invalid completion")
        return BaselineGeneratedResponse(
            generated_text=decoded,
            generation=GenerationStats(
                prompt_tokens=prompt_tokens,
                generated_tokens=len(token_ids),
                latency_seconds=latency,
                tokens_per_second=len(token_ids) / latency,
            ),
        )


class _CheckpointOnlyGenerator(BaselineGenerator):
    def generate(self, *, system_prompt: str, user_prompt: str) -> BaselineGeneratedResponse:
        del system_prompt, user_prompt
        raise DistilledTrajectoryEvaluationError(
            "P9-007C scoring is missing transported GPU responses; refusing regeneration"
        )


def _generation_contract(
    *, trajectory: TrajectoryIdentity, snapshot: SnapshotIdentity, base_model: BaseModelIdentity, settings: Any, system_prompt: str
) -> str:
    payload = {
        "schema_version": 1,
        "study_id": "python-p9-distilled-v4-2000-trajectory-v1",
        "trajectory": trajectory.label,
        "learning_rate": trajectory.learning_rate,
        "step": snapshot.step,
        "adapter_id": trajectory.adapter_id,
        "adapter_model_sha256": snapshot.adapter_model_sha256,
        "base_model": asdict(base_model),
        "evaluation_settings_sha256": evaluation_settings_sha256(settings),
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
        "development_manifest_sha256": _EXPECTED_DEVELOPMENT_SHA256,
        "membership_sha256": _EXPECTED_MEMBERSHIP_SHA256,
    }
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def _evaluation_inputs(
    trajectory: TrajectoryIdentity, *, repo_root: Path
) -> tuple[Any, Any, BaseModelIdentity, str, dict[str, object]]:
    manifest = load_development_manifest(repo_root)
    evaluation, settings, base_model, system_prompt = _evaluation_context(trajectory)
    evaluation = replace(evaluation, output_dir=_OUTPUT_ROOT.as_posix())
    return evaluation, settings, base_model, system_prompt, manifest


def generate_trajectory(
    *, training_output: Path, repo_root: Path = Path("."), device_index: int = 0
) -> tuple[Path, ...]:
    """Generate development responses for all four frozen P9-007C snapshots."""

    trajectory, snapshot_dirs = load_local_trajectory(training_output, repo_root=repo_root)
    evaluation, settings, base_model, system_prompt, manifest = _evaluation_inputs(
        trajectory, repo_root=repo_root
    )
    humaneval, he, mbpp, mb = _load_development_problems(
        evaluation,
        base_model,
        trajectory,
        settings,
        manifest,
    )
    generator = DistilledSnapshotGenerator(
        snapshot_dirs=snapshot_dirs,
        base_model=base_model,
        settings=settings,
        device_index=device_index,
    )
    stage_paths: list[Path] = []
    for snapshot in trajectory.snapshots:
        generator.select_step(snapshot.step)
        output_dir = _OUTPUT_ROOT / f"step-{snapshot.step:04d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        contract = _generation_contract(
            trajectory=trajectory,
            snapshot=snapshot,
            base_model=base_model,
            settings=settings,
            system_prompt=system_prompt,
        )
        he_responses = _generate_items(
            suite_id="humaneval-development",
            prompts=tuple(
                (problem.task_id, humaneval.prompt_for(problem).user_content) for problem in he
            ),
            generator=generator,
            system_prompt=system_prompt,
            generation_contract=contract,
            output_dir=output_dir,
        )
        mb_responses = _generate_items(
            suite_id="mbpp-development",
            prompts=tuple((problem.task_id, mbpp.prompt_for(problem).user_content) for problem in mb),
            generator=generator,
            system_prompt=system_prompt,
            generation_contract=contract,
            output_dir=output_dir,
        )
        if len(he_responses) + len(mb_responses) != _EXPECTED_COMBINED_TOTAL:
            raise DistilledTrajectoryEvaluationError("P9-007C generation cardinality is incomplete")
        stage = {
            "schema_version": 1,
            "task_id": "P9-007C",
            "stage": "development-generation",
            "development_manifest_sha256": _EXPECTED_DEVELOPMENT_SHA256,
            "membership_sha256": _EXPECTED_MEMBERSHIP_SHA256,
            "trajectory": trajectory.label,
            "learning_rate": trajectory.learning_rate,
            "step": snapshot.step,
            "adapter_id": trajectory.adapter_id,
            "adapter_model_sha256": snapshot.adapter_model_sha256,
            "generation_contract_sha256": contract,
            "humaneval_requests": len(he_responses),
            "mbpp_requests": len(mb_responses),
            "repository_holdout_requests": 0,
        }
        path = output_dir / "generation-stage.json"
        path.write_text(json.dumps(stage, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        stage_paths.append(path)
    return tuple(stage_paths)


def score_checkpoint(
    *, training_output: Path, step: int, repo_root: Path = Path(".")
) -> DistilledDevelopmentScore:
    """Score one transported P9-007C snapshot on development HumanEval+MBPP only."""

    if step not in _EXPECTED_STEPS:
        raise DistilledTrajectoryEvaluationError(f"non-frozen P9-007C step {step}")
    trajectory, _ = load_local_trajectory(training_output, repo_root=repo_root)
    snapshot = trajectory.snapshot(step)
    evaluation, settings, base_model, system_prompt, manifest = _evaluation_inputs(
        trajectory, repo_root=repo_root
    )
    output_dir = _OUTPUT_ROOT / f"step-{step:04d}"
    stage = _load_json(output_dir / "generation-stage.json", context="P9-007C generation stage")
    if stage.get("task_id") != "P9-007C" or stage.get("stage") != "development-generation":
        raise DistilledTrajectoryEvaluationError("P9-007C generation stage identity drifted")
    if stage.get("step") != step or stage.get("adapter_model_sha256") != snapshot.adapter_model_sha256:
        raise DistilledTrajectoryEvaluationError("P9-007C generation stage snapshot drifted")
    if stage.get("repository_holdout_requests") != 0:
        raise DistilledTrajectoryEvaluationError("P9-007C generation touched qualification holdout")
    contract = _generation_contract(
        trajectory=trajectory,
        snapshot=snapshot,
        base_model=base_model,
        settings=settings,
        system_prompt=system_prompt,
    )
    if stage.get("generation_contract_sha256") != contract:
        raise DistilledTrajectoryEvaluationError("P9-007C generation contract drifted")

    harness = DirectExecutionHarness(allow_reduced_isolation=True)
    humaneval, he, mbpp, mb = _load_development_problems(
        evaluation,
        base_model,
        trajectory,
        settings,
        manifest,
        harness=harness,
    )
    checkpoint_only = _CheckpointOnlyGenerator()
    he_responses = _generate_items(
        suite_id="humaneval-development",
        prompts=tuple(
            (problem.task_id, humaneval.prompt_for(problem).user_content) for problem in he
        ),
        generator=checkpoint_only,
        system_prompt=system_prompt,
        generation_contract=contract,
        output_dir=output_dir,
    )
    mb_responses = _generate_items(
        suite_id="mbpp-development",
        prompts=tuple((problem.task_id, mbpp.prompt_for(problem).user_content) for problem in mb),
        generator=checkpoint_only,
        system_prompt=system_prompt,
        generation_contract=contract,
        output_dir=output_dir,
    )
    he_result = humaneval.evaluate_suite(
        he,
        tuple(
            HumanEvalCompletion(
                task_id=problem.task_id,
                generated_text=response.generated_text,
                generation=response.generation,
            )
            for problem, response in zip(he, he_responses, strict=True)
        ),
    )
    mb_result = mbpp.evaluate_suite(
        mb,
        tuple(
            MBPPCompletion(
                task_id=problem.task_id,
                generated_text=response.generated_text,
                generation=response.generation,
            )
            for problem, response in zip(mb, mb_responses, strict=True)
        ),
    )
    if he_result.aggregate.harness_errors or mb_result.aggregate.harness_errors:
        raise DistilledTrajectoryEvaluationError("P9-007C scoring encountered harness errors")
    humaneval.write_artifacts(he_result, output_dir / "humaneval-development")
    mbpp.write_artifacts(mb_result, output_dir / "mbpp-development")
    he_passed = he_result.aggregate.passed
    mb_passed = mb_result.aggregate.passed
    combined = he_passed + mb_passed
    eligible = (
        combined >= _EXPECTED_MIN_COMBINED
        and he_passed >= _EXPECTED_BASE_HE
        and mb_passed >= _EXPECTED_BASE_MBPP
    )
    score = DistilledDevelopmentScore(
        label=_LABEL,
        learning_rate=_LEARNING_RATE,
        step=step,
        humaneval_passed=he_passed,
        humaneval_total=_EXPECTED_HE_TOTAL,
        mbpp_passed=mb_passed,
        mbpp_total=_EXPECTED_MBPP_TOTAL,
        combined_passed=combined,
        combined_total=_EXPECTED_COMBINED_TOTAL,
        eligible=eligible,
    )
    payload = {
        "schema_version": 1,
        "task_id": "P9-007C",
        "stage": "development-score",
        **asdict(score),
        "base": {
            "humaneval_passed": _EXPECTED_BASE_HE,
            "mbpp_passed": _EXPECTED_BASE_MBPP,
            "combined_passed": _EXPECTED_BASE_COMBINED,
        },
        "delta": {
            "humaneval_passed": he_passed - _EXPECTED_BASE_HE,
            "mbpp_passed": mb_passed - _EXPECTED_BASE_MBPP,
            "combined_passed": combined - _EXPECTED_BASE_COMBINED,
        },
        "development_manifest_sha256": _EXPECTED_DEVELOPMENT_SHA256,
        "membership_sha256": _EXPECTED_MEMBERSHIP_SHA256,
        "adapter_model_sha256": snapshot.adapter_model_sha256,
        "repository_holdout_evaluated": False,
    }
    (output_dir / "development-score.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return score


def select_development_candidate(
    scores: Sequence[DistilledDevelopmentScore],
) -> DistilledDevelopmentScore | None:
    """Apply the frozen four-way development gate and fewer-steps tie breaker."""

    items = tuple(scores)
    if len(items) != 4 or {item.step for item in items} != set(_EXPECTED_STEPS):
        raise DistilledTrajectoryEvaluationError("P9-007C selection requires exactly four scores")
    if any(item.label != _LABEL or item.learning_rate != _LEARNING_RATE for item in items):
        raise DistilledTrajectoryEvaluationError("P9-007C score trajectory identity drifted")
    eligible = tuple(item for item in items if item.eligible)
    if not eligible:
        return None
    return sorted(eligible, key=lambda item: (-item.combined_passed, item.step))[0]


def _score_from_payload(value: object, *, context: str) -> DistilledDevelopmentScore:
    row = _mapping(value, context=context)
    if row.get("task_id") != "P9-007C" or row.get("stage") != "development-score":
        raise DistilledTrajectoryEvaluationError(f"{context} identity drifted")
    if row.get("repository_holdout_evaluated") is not False:
        raise DistilledTrajectoryEvaluationError(f"{context} touched qualification holdout")
    eligible = row.get("eligible")
    if not isinstance(eligible, bool):
        raise DistilledTrajectoryEvaluationError(f"{context}.eligible must be boolean")
    score = DistilledDevelopmentScore(
        label=_string(row, "label", context=context),
        learning_rate=float(row.get("learning_rate", -1.0)),
        step=_integer(row, "step", context=context),
        humaneval_passed=_integer(row, "humaneval_passed", context=context),
        humaneval_total=_integer(row, "humaneval_total", context=context),
        mbpp_passed=_integer(row, "mbpp_passed", context=context),
        mbpp_total=_integer(row, "mbpp_total", context=context),
        combined_passed=_integer(row, "combined_passed", context=context),
        combined_total=_integer(row, "combined_total", context=context),
        eligible=eligible,
    )
    expected_eligible = (
        score.combined_passed >= _EXPECTED_MIN_COMBINED
        and score.humaneval_passed >= _EXPECTED_BASE_HE
        and score.mbpp_passed >= _EXPECTED_BASE_MBPP
    )
    if (
        score.humaneval_total != _EXPECTED_HE_TOTAL
        or score.mbpp_total != _EXPECTED_MBPP_TOTAL
        or score.combined_total != _EXPECTED_COMBINED_TOTAL
        or score.combined_passed != score.humaneval_passed + score.mbpp_passed
        or score.eligible is not expected_eligible
    ):
        raise DistilledTrajectoryEvaluationError(f"{context} score arithmetic/policy is invalid")
    return score


def select_from_root(scores_root: Path, *, repo_root: Path = Path(".")) -> dict[str, object]:
    """Read exactly four development scores and authorize at most one qualification candidate."""

    validate_distilled_trajectory(repo_root=repo_root)
    load_development_manifest(repo_root)
    files = sorted(scores_root.rglob("development-score.json"))
    if len(files) != 4:
        raise DistilledTrajectoryEvaluationError(
            f"P9-007C selection expected 4 development scores, found {len(files)}"
        )
    scores = tuple(
        _score_from_payload(_load_json(path, context=f"score[{index}]"), context=f"score[{index}]")
        for index, path in enumerate(files)
    )
    selected = select_development_candidate(scores)
    payload: dict[str, object] = {
        "schema_version": 1,
        "task_id": "P9-007C",
        "stage": "development-selection",
        "development_manifest_sha256": _EXPECTED_DEVELOPMENT_SHA256,
        "membership_sha256": _EXPECTED_MEMBERSHIP_SHA256,
        "base": {
            "humaneval_passed": _EXPECTED_BASE_HE,
            "mbpp_passed": _EXPECTED_BASE_MBPP,
            "combined_passed": _EXPECTED_BASE_COMBINED,
        },
        "scores": [asdict(item) for item in sorted(scores, key=lambda item: item.step)],
        "selected": None if selected is None else asdict(selected),
        "qualification_authorized": selected is not None,
        "repository_holdout_evaluated": False,
        "no_candidate_action": (
            "stop before qualification; diagnose distillation data/objective before scaling"
            if selected is None
            else None
        ),
    }
    output = scores_root / "p9-007c-development-selection.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("--repo-root", type=Path, default=Path("."))
    generate_parser = sub.add_parser("generate-trajectory")
    generate_parser.add_argument("--training-output", type=Path, required=True)
    generate_parser.add_argument("--repo-root", type=Path, default=Path("."))
    generate_parser.add_argument("--device-index", type=int, default=0)
    score_parser = sub.add_parser("score")
    score_parser.add_argument("--training-output", type=Path, required=True)
    score_parser.add_argument("--step", type=int, required=True)
    score_parser.add_argument("--repo-root", type=Path, default=Path("."))
    select_parser = sub.add_parser("select")
    select_parser.add_argument("--scores-root", type=Path, required=True)
    select_parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args(argv)

    if args.command == "validate":
        validation = validate_distilled_trajectory(repo_root=args.repo_root)
        manifest = load_development_manifest(args.repo_root)
        print(
            json.dumps(
                {
                    "task_id": validation.task_id,
                    "checkpoint_steps": list(validation.checkpoint_steps),
                    "development_manifest_sha256": _EXPECTED_DEVELOPMENT_SHA256,
                    "membership_sha256": manifest["membership_sha256"],
                    "repository_holdout_qualification_only": True,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "generate-trajectory":
        paths = generate_trajectory(
            training_output=args.training_output,
            repo_root=args.repo_root,
            device_index=args.device_index,
        )
        print(json.dumps([path.as_posix() for path in paths], indent=2))
        return 0
    if args.command == "score":
        score = score_checkpoint(
            training_output=args.training_output,
            step=args.step,
            repo_root=args.repo_root,
        )
        print(json.dumps(asdict(score), indent=2, sort_keys=True))
        return 0
    if args.command == "select":
        print(
            json.dumps(
                select_from_root(args.scores_root, repo_root=args.repo_root),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    raise DistilledTrajectoryEvaluationError("unsupported P9-007C command")


def main() -> NoReturn:
    raise SystemExit(_main())


if __name__ == "__main__":
    main()
