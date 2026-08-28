"""Tests for the standalone environment report."""

from __future__ import annotations

import json
import platform
from datetime import UTC, datetime

import torch

from tiny_qwen_coder.reporting.environment import (
    EnvironmentReport,
    GpuEnvironment,
    HostEnvironment,
    PythonEnvironment,
    PyTorchEnvironment,
    collect_environment_report,
    environment_report_json,
    main,
    write_environment_report,
)
from tiny_qwen_coder.reporting.manifest import DependencyVersions

_FIXED_TIME = datetime(2026, 8, 28, 9, 0, tzinfo=UTC)


def _dependencies() -> DependencyVersions:
    return DependencyVersions(
        accelerate="1.0",
        bitsandbytes=None,
        datasets="2.0",
        numpy="3.0",
        peft="4.0",
        pyyaml="5.0",
        tiny_qwen_coder="0.1.0",
        torch="6.0",
        transformers="7.0",
        trl="8.0",
    )


def _synthetic_report() -> EnvironmentReport:
    return EnvironmentReport(
        schema_version=1,
        generated_at_utc="2026-08-28T09:00:00.000000Z",
        python=PythonEnvironment(version="3.11.16", implementation="CPython"),
        pytorch=PyTorchEnvironment(
            version="2.13.0",
            cuda_available=True,
            cuda_runtime="13.0",
            cudnn_version=92000,
            deterministic_algorithms=True,
        ),
        dependencies=_dependencies(),
        host=HostEnvironment(
            hostname="test-host",
            system="Linux",
            release="test-release",
            machine="x86_64",
        ),
        gpus=(
            GpuEnvironment(
                index=0,
                name="Test GPU",
                compute_capability="8.9",
                total_vram_bytes=16 * 1024**3,
                free_vram_bytes=15 * 1024**3,
            ),
        ),
    )


def test_collect_environment_report_contains_required_metadata() -> None:
    report = collect_environment_report(now=_FIXED_TIME)

    assert report.schema_version == 1
    assert report.generated_at_utc == "2026-08-28T09:00:00.000000Z"
    assert report.python.version == platform.python_version()
    assert report.python.implementation == platform.python_implementation()
    assert report.pytorch.version == str(torch.__version__)
    assert report.pytorch.cuda_available is torch.cuda.is_available()
    assert report.pytorch.cuda_runtime == torch.version.cuda
    assert report.dependencies.torch
    assert report.dependencies.transformers
    assert report.dependencies.trl
    assert report.dependencies.peft
    assert report.dependencies.datasets
    assert report.dependencies.accelerate
    assert report.dependencies.bitsandbytes is None or report.dependencies.bitsandbytes

    if torch.cuda.is_available():
        assert report.gpus
        for gpu in report.gpus:
            assert gpu.name
            assert gpu.compute_capability
            assert gpu.total_vram_bytes > 0
            assert 0 <= gpu.free_vram_bytes <= gpu.total_vram_bytes
    else:
        assert report.gpus == ()


def test_environment_report_json_is_stable_for_same_report() -> None:
    report = _synthetic_report()

    first = environment_report_json(report)
    second = environment_report_json(report)

    assert first == second
    payload = json.loads(first)
    assert payload["python"]["version"] == "3.11.16"
    assert payload["pytorch"]["cuda_runtime"] == "13.0"
    assert payload["gpus"][0]["total_vram_bytes"] == 16 * 1024**3


def test_write_environment_report_round_trips(tmp_path) -> None:  # type: ignore[no-untyped-def]
    destination = tmp_path / "reports" / "environment.json"

    result = write_environment_report(_synthetic_report(), destination)

    assert result == destination
    assert json.loads(destination.read_text(encoding="utf-8"))["host"]["hostname"] == "test-host"
    assert not (destination.parent / ".environment.json.tmp").exists()


def test_cli_prints_machine_readable_report_without_training(capsys) -> None:  # type: ignore[no-untyped-def]
    main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["python"]["version"] == platform.python_version()
    assert payload["pytorch"]["version"] == str(torch.__version__)
    assert payload["dependencies"]["transformers"]
