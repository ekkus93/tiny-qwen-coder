"""P6-005 tests for the canonical unchanged-base Python baseline pipeline."""

from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from tiny_qwen_coder.config import EvaluationConfig, load_evaluation_config
from tiny_qwen_coder.evaluation._baseline_artifacts import (
    freeze_python_baseline,
    regression_baseline_results_sha256,
    system_prompt_sha256,
    validate_python_baseline_artifacts,
    write_regression_baseline_artifacts,
)
from tiny_qwen_coder.evaluation._baseline_generation import (
    BaselineGenerator,
    HuggingFaceBaselineGenerator,
    generation_contract_sha256,
)
from tiny_qwen_coder.evaluation._baseline_provenance import (
    BaselineGpuProvenance,
    BaselineProvenance,
    baseline_provenance_json,
    load_baseline_base_model_identity,
)
from tiny_qwen_coder.evaluation._baseline_runner import (
    _generate_items,
    _preflight_execution_images,
    _regression_aggregate,
    _regression_results,
    _validate_baseline_contract,
    _validate_runtime_metadata,
)
from tiny_qwen_coder.evaluation._baseline_types import (
    BaselineGeneratedResponse,
    BaselineRuntimeMetadata,
    BaselineSuitePerformance,
    PythonBaselineError,
)
from tiny_qwen_coder.evaluation.execution import OciRuntime, OciRuntimeSpec
from tiny_qwen_coder.evaluation.regression import (
    RegressionExpectationKind,
    load_frozen_general_tool_regression_suite,
)
from tiny_qwen_coder.evaluation.results import GenerationStats
from tiny_qwen_coder.evaluation.settings import load_frozen_evaluation_settings
from tiny_qwen_coder.identities import AdapterIdentity
from tiny_qwen_coder.languages.python import load_python_plugin

_BASELINE_CONFIG = Path("configs/eval/python/base_baseline_v1.yaml")
_BASE_CONFIG = Path("configs/base/qwen35-4b.yaml")
_EXPECTED_SUITES = (
    "humaneval",
    "mbpp",
    "repository-holdout",
    "general-tool-regression",
)
_REQUIRED_ARTIFACTS = (
    "provenance.json",
    "runtime-metadata.json",
    "humaneval/humaneval-results.jsonl",
    "humaneval/humaneval-aggregate.json",
    "mbpp/mbpp-results.jsonl",
    "mbpp/mbpp-aggregate.json",
    "repository-holdout/repository-holdout-results.jsonl",
    "repository-holdout/repository-holdout-aggregate.json",
    "general-tool-regression/general-tool-regression-results.jsonl",
    "general-tool-regression/general-tool-regression-aggregate.json",
)


def _generation(*, prompt_tokens: int = 8, generated_tokens: int = 2) -> GenerationStats:
    return GenerationStats(
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        latency_seconds=0.1,
        tokens_per_second=generated_tokens / 0.1,
    )


class _CountingGenerator(BaselineGenerator):
    def __init__(self, text: str = "OK") -> None:
        self.text = text
        self.calls: list[tuple[str, str]] = []

    def generate(self, *, system_prompt: str, user_prompt: str) -> BaselineGeneratedResponse:
        self.calls.append((system_prompt, user_prompt))
        return BaselineGeneratedResponse(generated_text=self.text, generation=_generation())


def _runtime_metadata() -> BaselineRuntimeMetadata:
    suites = (
        BaselineSuitePerformance("humaneval", 1, 8, 2, 0.1, 20.0),
        BaselineSuitePerformance("mbpp", 1, 8, 2, 0.1, 20.0),
        BaselineSuitePerformance("repository-holdout", 1, 8, 2, 0.1, 20.0),
        BaselineSuitePerformance("general-tool-regression", 1, 8, 2, 0.1, 20.0),
    )
    return BaselineRuntimeMetadata(
        schema_version=1,
        device="cuda:0",
        gpu_name="Unit Test GPU",
        gpu_compute_capability="8.9",
        torch_version="2.test",
        transformers_version="5.test",
        model_class="test.Model",
        parameter_dtypes=("torch.bfloat16",),
        resolved_model_revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        cuda_total_bytes=16_000,
        cuda_free_before_load_bytes=15_000,
        cuda_free_after_load_bytes=8_000,
        torch_allocated_after_load_bytes=7_000,
        torch_reserved_after_load_bytes=7_500,
        load_peak_allocated_bytes=7_100,
        load_peak_reserved_bytes=7_600,
        generation_peak_allocated_bytes=7_200,
        generation_peak_reserved_bytes=7_700,
        model_load_seconds=1.0,
        total_wall_seconds=2.0,
        total_requests=4,
        total_prompt_tokens=32,
        total_generated_tokens=8,
        total_generation_latency_seconds=0.4,
        overall_tokens_per_second=20.0,
        suites=suites,
    )


def _provenance() -> BaselineProvenance:
    base = load_baseline_base_model_identity(_BASE_CONFIG)
    return BaselineProvenance(
        schema_version=1,
        created_at_utc="2026-08-30T00:00:00.000000Z",
        source_git_sha="1" * 40,
        source_git_dirty=False,
        base_model=base,
        adapter=AdapterIdentity(family=None, adapter_id=None),
        language="python",
        seed=1729,
        hostname="test-host",
        system="Linux",
        release="test",
        machine="x86_64",
        python_version="3.11.0",
        cuda_available=True,
        cuda_runtime="13.0",
        gpus=(
            BaselineGpuProvenance(
                index=0,
                name="Unit Test GPU",
                total_memory_bytes=16_000,
                compute_capability="8.9",
            ),
        ),
        dependencies=(("torch", "1"), ("transformers", "1")),
    )


def _correct_regression_response(case_id: str) -> str:
    suite = load_frozen_general_tool_regression_suite()
    case = next(item for item in suite.cases if item.id == case_id)
    if case.expectation.kind is RegressionExpectationKind.EXACT_TEXT:
        return case.expectation.value
    if case.expectation.kind is RegressionExpectationKind.JSON:
        return case.expectation.value
    return f"<tool_call>{case.expectation.value}</tool_call>"


def test_canonical_baseline_config_is_base_only_and_complete() -> None:
    evaluation = load_evaluation_config(_BASELINE_CONFIG)
    settings = load_frozen_evaluation_settings()
    base = load_baseline_base_model_identity(_BASE_CONFIG)

    assert evaluation.adapter_id is None
    assert evaluation.language == "python"
    assert evaluation.suites == _EXPECTED_SUITES
    assert evaluation.output_dir == "artifacts/eval/python/base-baseline-v1"
    assert evaluation.seed == 1729
    assert not evaluation.execution.network_enabled
    assert evaluation.execution.timeout_seconds == 10.0
    version, text = _validate_baseline_contract(evaluation, settings, base)
    assert version == "python-v1"
    assert system_prompt_sha256(text) == (
        "ed10dcc67116e4b3633eed7413228bb083a797fc6917132df03890aa7e05497e"
    )


@pytest.mark.parametrize(
    "mutated",
    [
        lambda value: replace(value, adapter_id="language/python/test"),
        lambda value: replace(value, language="rust"),
        lambda value: replace(value, suites=("humaneval", "mbpp")),
        lambda value: replace(value, suites=tuple(reversed(value.suites))),
    ],
)
def test_baseline_contract_fails_closed_on_comparison_drift(mutated: object) -> None:
    evaluation = load_evaluation_config(_BASELINE_CONFIG)
    assert callable(mutated)
    changed = mutated(evaluation)
    assert isinstance(changed, EvaluationConfig)
    with pytest.raises(PythonBaselineError):
        _validate_baseline_contract(
            changed,
            load_frozen_evaluation_settings(),
            load_baseline_base_model_identity(_BASE_CONFIG),
        )


def test_generation_contract_binds_model_settings_and_system_prompt() -> None:
    settings = load_frozen_evaluation_settings()
    base = load_baseline_base_model_identity(_BASE_CONFIG)
    system_prompt = load_python_plugin().spec.config.system_prompt
    first = generation_contract_sha256(
        base_model=base,
        settings=settings,
        system_prompt_version=system_prompt.version,
        system_prompt=system_prompt.text,
    )
    second = generation_contract_sha256(
        base_model=base,
        settings=settings,
        system_prompt_version=system_prompt.version,
        system_prompt=system_prompt.text + " changed",
    )
    assert len(first) == 64
    assert first != second


def test_resume_checkpoint_reuses_exact_generated_responses(tmp_path: Path) -> None:
    prompts = (("one", "first prompt"), ("two", "second prompt"))
    first = _CountingGenerator("generated")
    responses = _generate_items(
        suite_id="test-suite",
        prompts=prompts,
        generator=first,
        system_prompt="system",
        generation_contract="a" * 64,
        output_dir=tmp_path,
    )
    assert len(first.calls) == 2
    assert tuple(item.generated_text for item in responses) == ("generated", "generated")

    second = _CountingGenerator("must-not-run")
    resumed = _generate_items(
        suite_id="test-suite",
        prompts=prompts,
        generator=second,
        system_prompt="system",
        generation_contract="a" * 64,
        output_dir=tmp_path,
    )
    assert second.calls == []
    assert resumed == responses

    with pytest.raises(PythonBaselineError, match="prompt drift"):
        _generate_items(
            suite_id="test-suite",
            prompts=(("one", "changed"), ("two", "second prompt")),
            generator=second,
            system_prompt="system",
            generation_contract="a" * 64,
            output_dir=tmp_path,
        )


def test_general_tool_regression_artifacts_are_complete_and_deterministic(tmp_path: Path) -> None:
    suite = load_frozen_general_tool_regression_suite()
    responses = tuple(
        BaselineGeneratedResponse(
            generated_text=_correct_regression_response(case.id),
            generation=_generation(),
        )
        for case in suite.cases
    )
    results = _regression_results(suite, responses)
    aggregate = _regression_aggregate(
        suite=suite,
        results=results,
        settings=load_frozen_evaluation_settings(),
        system_prompt_version="python-v1",
        system_prompt=load_python_plugin().spec.config.system_prompt.text,
        base_model=load_baseline_base_model_identity(_BASE_CONFIG),
    )

    assert all(result.passed for result in results)
    assert aggregate.passed == aggregate.total_cases == len(suite.cases)
    assert aggregate.pass_rate == 1.0
    assert aggregate.results_sha256 == regression_baseline_results_sha256(results)
    first_paths = write_regression_baseline_artifacts(
        results=results,
        aggregate=aggregate,
        output_dir=tmp_path / "first",
    )
    second_paths = write_regression_baseline_artifacts(
        results=results,
        aggregate=aggregate,
        output_dir=tmp_path / "second",
    )
    assert first_paths[0].read_bytes() == second_paths[0].read_bytes()
    assert first_paths[1].read_bytes() == second_paths[1].read_bytes()


def test_freeze_requires_complete_artifact_set_and_detects_tampering(tmp_path: Path) -> None:
    evaluation = load_evaluation_config(_BASELINE_CONFIG)
    settings = load_frozen_evaluation_settings()
    system_prompt = load_python_plugin().spec.config.system_prompt
    provenance = _provenance()
    output = tmp_path / "baseline"
    for relative_path in _REQUIRED_ARTIFACTS:
        path = output / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"artifact:{relative_path}\n", encoding="utf-8")
    valid_coding_aggregate = '{"harness_errors":0,"pass_at_1":0.0}\n'
    for relative_path in (
        "humaneval/humaneval-aggregate.json",
        "mbpp/mbpp-aggregate.json",
        "repository-holdout/repository-holdout-aggregate.json",
    ):
        (output / relative_path).write_text(valid_coding_aggregate, encoding="utf-8")
    (output / "provenance.json").write_text(baseline_provenance_json(provenance), encoding="utf-8")

    manifest = freeze_python_baseline(
        output_dir=output,
        evaluation=evaluation,
        settings=settings,
        system_prompt_version=system_prompt.version,
        system_prompt=system_prompt.text,
        generation_contract_sha256=generation_contract_sha256(
            base_model=provenance.base_model,
            settings=settings,
            system_prompt_version=system_prompt.version,
            system_prompt=system_prompt.text,
        ),
        provenance=provenance,
    )
    assert manifest.frozen
    assert len(manifest.artifacts) == len(_REQUIRED_ARTIFACTS)
    assert validate_python_baseline_artifacts(output) == manifest

    target = output / "mbpp/mbpp-aggregate.json"
    target.write_text(target.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
    with pytest.raises(PythonBaselineError, match="digest mismatch"):
        validate_python_baseline_artifacts(output)


def test_freeze_rejects_provenance_file_mismatch(tmp_path: Path) -> None:
    evaluation = load_evaluation_config(_BASELINE_CONFIG)
    settings = load_frozen_evaluation_settings()
    system_prompt = load_python_plugin().spec.config.system_prompt
    provenance = _provenance()
    for relative_path in _REQUIRED_ARTIFACTS:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x\n", encoding="utf-8")
    (tmp_path / "provenance.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(PythonBaselineError, match="provenance artifact does not match"):
        freeze_python_baseline(
            output_dir=tmp_path,
            evaluation=evaluation,
            settings=settings,
            system_prompt_version=system_prompt.version,
            system_prompt=system_prompt.text,
            generation_contract_sha256=generation_contract_sha256(
                base_model=provenance.base_model,
                settings=settings,
                system_prompt_version=system_prompt.version,
                system_prompt=system_prompt.text,
            ),
            provenance=provenance,
        )


def test_freeze_rejects_dirty_or_cpu_only_run_manifest(tmp_path: Path) -> None:
    evaluation = load_evaluation_config(_BASELINE_CONFIG)
    settings = load_frozen_evaluation_settings()
    system_prompt = load_python_plugin().spec.config.system_prompt
    for relative_path in _REQUIRED_ARTIFACTS:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x\n", encoding="utf-8")

    clean = _provenance()
    contract = generation_contract_sha256(
        base_model=clean.base_model,
        settings=settings,
        system_prompt_version=system_prompt.version,
        system_prompt=system_prompt.text,
    )
    with pytest.raises(PythonBaselineError, match="dirty source tree"):
        freeze_python_baseline(
            output_dir=tmp_path,
            evaluation=evaluation,
            settings=settings,
            system_prompt_version=system_prompt.version,
            system_prompt=system_prompt.text,
            generation_contract_sha256=contract,
            provenance=replace(clean, source_git_dirty=True),
        )
    with pytest.raises(PythonBaselineError, match="CUDA GPU"):
        freeze_python_baseline(
            output_dir=tmp_path,
            evaluation=evaluation,
            settings=settings,
            system_prompt_version=system_prompt.version,
            system_prompt=system_prompt.text,
            generation_contract_sha256=contract,
            provenance=replace(
                clean,
                cuda_available=False,
                cuda_runtime=None,
                gpus=(),
            ),
        )


def test_runtime_metadata_requires_consistent_suite_totals() -> None:
    metadata = _runtime_metadata()
    assert metadata.total_requests == 4
    with pytest.raises(PythonBaselineError, match="total_requests"):
        replace(metadata, total_requests=5)


def test_runtime_metadata_must_match_generated_suite_measurements() -> None:
    metadata = _runtime_metadata()
    changed = (
        replace(metadata.suites[0], generated_tokens=3, tokens_per_second=30.0),
        *metadata.suites[1:],
    )
    with pytest.raises(PythonBaselineError, match="suite measurements"):
        _validate_runtime_metadata(metadata, changed)


def test_huggingface_input_preparation_rejects_multiple_batches() -> None:
    import torch

    class Tokenizer:
        def apply_chat_template(self, *args: object, **kwargs: object) -> object:
            return {"input_ids": torch.tensor([[1, 2], [3, 4]])}

        def decode(self, token_ids: list[int], **kwargs: object) -> object:
            return ""

    generator = object.__new__(HuggingFaceBaselineGenerator)
    generator._tokenizer = Tokenizer()
    generator._device = torch.device("cpu")
    with pytest.raises(PythonBaselineError, match="one batch"):
        generator._prepare_inputs("system", "user")


def test_execution_image_preflight_deduplicates_local_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "docker"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    runtime = OciRuntimeSpec(kind=OciRuntime.DOCKER, executable=executable)
    calls: list[list[str]] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "tiny_qwen_coder.evaluation._baseline_runner.subprocess.run",
        fake_run,
    )
    image = "python:3.11@sha256:" + "a" * 64
    _preflight_execution_images(runtime, (image, image))
    assert calls == [[str(executable), "image", "inspect", image]]


def test_execution_image_preflight_reports_missing_pinned_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "podman"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    runtime = OciRuntimeSpec(kind=OciRuntime.PODMAN, executable=executable)

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=args,
            returncode=1,
            stdout="",
            stderr="image not known",
        )

    monkeypatch.setattr(
        "tiny_qwen_coder.evaluation._baseline_runner.subprocess.run",
        fake_run,
    )
    with pytest.raises(PythonBaselineError, match="pull the pinned image"):
        _preflight_execution_images(runtime, ("python:test",))


def test_huggingface_baseline_generator_fails_before_model_load_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(PythonBaselineError, match="requires CUDA"):
        HuggingFaceBaselineGenerator(
            base_model=load_baseline_base_model_identity(_BASE_CONFIG),
            settings=load_frozen_evaluation_settings(),
        )
