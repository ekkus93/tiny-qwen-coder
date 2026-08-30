"""Strict frozen runner configuration loading for MBPP."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import yaml

from tiny_qwen_coder.evaluation._mbpp_common import (
    _FROZEN_MBPP_RUNNER_SHA256,
    _MBPP_RUNNER_CONFIG_PATH,
    MBPPError,
    _expect_bool,
    _expect_float,
    _expect_int,
    _expect_str,
    _expect_str_tuple,
    _strict_mapping,
    _validate_keys,
)
from tiny_qwen_coder.evaluation._mbpp_types import MBPPRunnerConfig
from tiny_qwen_coder.evaluation.execution import ExecutionLimits


def _parse_limits(value: object) -> ExecutionLimits:
    context = "MBPP runner.limits"
    mapping = _strict_mapping(value, context=context)
    _validate_keys(
        mapping,
        required=frozenset(
            {
                "cpus",
                "memory_mebibytes",
                "pids",
                "workspace_mebibytes",
                "temp_mebibytes",
                "max_output_bytes",
                "max_input_bytes",
                "open_files",
                "cleanup_timeout_seconds",
            }
        ),
        context=context,
    )
    return ExecutionLimits(
        cpus=_expect_float(mapping, "cpus", context=context),
        memory_mebibytes=_expect_int(mapping, "memory_mebibytes", context=context),
        pids=_expect_int(mapping, "pids", context=context),
        workspace_mebibytes=_expect_int(mapping, "workspace_mebibytes", context=context),
        temp_mebibytes=_expect_int(mapping, "temp_mebibytes", context=context),
        max_output_bytes=_expect_int(mapping, "max_output_bytes", context=context),
        max_input_bytes=_expect_int(mapping, "max_input_bytes", context=context),
        open_files=_expect_int(mapping, "open_files", context=context),
        cleanup_timeout_seconds=_expect_float(
            mapping,
            "cleanup_timeout_seconds",
            context=context,
        ),
    )


def load_mbpp_runner_config(path: Path) -> MBPPRunnerConfig:
    """Load one strict MBPP runner definition."""

    try:
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise MBPPError(f"could not read MBPP runner config {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise MBPPError(f"invalid YAML in MBPP runner config {path}: {exc}") from exc
    context = "MBPP runner"
    mapping = _strict_mapping(raw, context=context)
    _validate_keys(
        mapping,
        required=frozenset(
            {
                "schema_version",
                "runner_id",
                "runner_version",
                "frozen",
                "benchmark_id",
                "reference_repository",
                "reference_revision",
                "dataset_config",
                "dataset_split",
                "task_id_start",
                "task_id_end",
                "expected_problem_count",
                "tests_per_problem",
                "prompt_version",
                "completion_normalizer_version",
                "stop_words",
                "execution_image",
                "execution_timeout_seconds",
                "limits",
            }
        ),
        context=context,
    )
    return MBPPRunnerConfig(
        schema_version=_expect_int(mapping, "schema_version", context=context),
        runner_id=_expect_str(mapping, "runner_id", context=context),
        runner_version=_expect_int(mapping, "runner_version", context=context),
        frozen=_expect_bool(mapping, "frozen", context=context),
        benchmark_id=_expect_str(mapping, "benchmark_id", context=context),
        reference_repository=_expect_str(mapping, "reference_repository", context=context),
        reference_revision=_expect_str(mapping, "reference_revision", context=context),
        dataset_config=_expect_str(mapping, "dataset_config", context=context),
        dataset_split=_expect_str(mapping, "dataset_split", context=context),
        task_id_start=_expect_int(mapping, "task_id_start", context=context),
        task_id_end=_expect_int(mapping, "task_id_end", context=context),
        expected_problem_count=_expect_int(mapping, "expected_problem_count", context=context),
        tests_per_problem=_expect_int(mapping, "tests_per_problem", context=context),
        prompt_version=_expect_str(mapping, "prompt_version", context=context),
        completion_normalizer_version=_expect_str(
            mapping,
            "completion_normalizer_version",
            context=context,
        ),
        stop_words=_expect_str_tuple(mapping, "stop_words", context=context),
        execution_image=_expect_str(mapping, "execution_image", context=context),
        execution_timeout_seconds=_expect_float(
            mapping,
            "execution_timeout_seconds",
            context=context,
        ),
        limits=_parse_limits(mapping["limits"]),
    )


def mbpp_runner_config_json(config: MBPPRunnerConfig) -> str:
    return json.dumps(asdict(config), indent=2, sort_keys=True) + "\n"


def mbpp_runner_config_sha256(config: MBPPRunnerConfig) -> str:
    return hashlib.sha256(mbpp_runner_config_json(config).encode("utf-8")).hexdigest()


def load_frozen_mbpp_runner_config(
    path: Path = _MBPP_RUNNER_CONFIG_PATH,
) -> MBPPRunnerConfig:
    """Load the frozen P6-003 MBPP runner and fail closed on drift."""

    config = load_mbpp_runner_config(path)
    if mbpp_runner_config_sha256(config) != _FROZEN_MBPP_RUNNER_SHA256:
        raise MBPPError(
            "frozen MBPP runner fingerprint mismatch; increment runner_version and explicitly "
            "update the frozen fingerprint before evaluation"
        )
    return config
