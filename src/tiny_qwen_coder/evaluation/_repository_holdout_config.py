"""Strict loading and semantic freezing for the repository-owned Python holdout."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import yaml

from tiny_qwen_coder.evaluation._repository_holdout_common import (
    _FROZEN_REPOSITORY_HOLDOUT_SUITE_SHA256,
    _REPOSITORY_HOLDOUT_SUITE_CONFIG_PATH,
    RepositoryHoldoutError,
    _expect_bool,
    _expect_float,
    _expect_int,
    _expect_optional_str,
    _expect_str,
    _expect_text,
    _parse_limits,
    _read_asset,
    _strict_mapping,
    _validate_keys,
)
from tiny_qwen_coder.evaluation._repository_holdout_types import (
    RepositoryHoldoutSuiteConfig,
    RepositoryHoldoutTask,
)


def _parse_task(value: object, *, index: int) -> RepositoryHoldoutTask:
    context = f"repository holdout.tasks[{index}]"
    mapping = _strict_mapping(value, context=context)
    _validate_keys(
        mapping,
        required=frozenset(
            {
                "id",
                "category",
                "prompt",
                "test_path",
                "setup_path",
                "expected_tests",
            }
        ),
        context=context,
    )
    test_path = _expect_str(mapping, "test_path", context=context)
    test_source, test_sha256 = _read_asset(test_path, field_name=f"{context}.test_path")
    setup_path = _expect_optional_str(mapping, "setup_path", context=context)
    setup_source = ""
    setup_sha256: str | None = None
    if setup_path is not None:
        setup_source, setup_sha256 = _read_asset(
            setup_path,
            field_name=f"{context}.setup_path",
        )
    return RepositoryHoldoutTask(
        task_id=_expect_str(mapping, "id", context=context),
        category=_expect_str(mapping, "category", context=context),
        prompt=_expect_text(mapping, "prompt", context=context),
        test_path=test_path,
        setup_path=setup_path,
        expected_tests=_expect_int(mapping, "expected_tests", context=context),
        test_source=test_source,
        setup_source=setup_source,
        test_sha256=test_sha256,
        setup_sha256=setup_sha256,
    )


def load_repository_holdout_suite(
    path: Path = _REPOSITORY_HOLDOUT_SUITE_CONFIG_PATH,
) -> RepositoryHoldoutSuiteConfig:
    """Load the strict repo-owned P6-004 suite definition and executable assets."""

    try:
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RepositoryHoldoutError(
            f"could not read repository holdout config {path}: {exc}"
        ) from exc
    except yaml.YAMLError as exc:
        raise RepositoryHoldoutError(f"invalid repository holdout YAML {path}: {exc}") from exc
    context = "repository holdout"
    mapping = _strict_mapping(raw, context=context)
    _validate_keys(
        mapping,
        required=frozenset(
            {
                "schema_version",
                "suite_id",
                "suite_version",
                "frozen",
                "benchmark_id",
                "dataset_revision",
                "prompt_version",
                "completion_normalizer_version",
                "instruction",
                "execution_image",
                "execution_timeout_seconds",
                "limits",
                "tasks",
            }
        ),
        context=context,
    )
    task_values = mapping["tasks"]
    if not isinstance(task_values, Sequence) or isinstance(task_values, str | bytes):
        raise RepositoryHoldoutError("repository holdout.tasks must be a sequence")
    tasks = tuple(_parse_task(value, index=index) for index, value in enumerate(task_values))
    return RepositoryHoldoutSuiteConfig(
        schema_version=_expect_int(mapping, "schema_version", context=context),
        suite_id=_expect_str(mapping, "suite_id", context=context),
        suite_version=_expect_int(mapping, "suite_version", context=context),
        frozen=_expect_bool(mapping, "frozen", context=context),
        benchmark_id=_expect_str(mapping, "benchmark_id", context=context),
        dataset_revision=_expect_str(mapping, "dataset_revision", context=context),
        prompt_version=_expect_str(mapping, "prompt_version", context=context),
        completion_normalizer_version=_expect_str(
            mapping,
            "completion_normalizer_version",
            context=context,
        ),
        instruction=_expect_text(mapping, "instruction", context=context),
        execution_image=_expect_str(mapping, "execution_image", context=context),
        execution_timeout_seconds=_expect_float(
            mapping,
            "execution_timeout_seconds",
            context=context,
        ),
        limits=_parse_limits(mapping["limits"]),
        tasks=tasks,
    )


def repository_holdout_suite_json(suite: RepositoryHoldoutSuiteConfig) -> str:
    """Serialize suite identity without copying protected test source into output."""

    value = {
        "schema_version": suite.schema_version,
        "suite_id": suite.suite_id,
        "suite_version": suite.suite_version,
        "frozen": suite.frozen,
        "benchmark_id": suite.benchmark_id,
        "dataset_revision": suite.dataset_revision,
        "prompt_version": suite.prompt_version,
        "completion_normalizer_version": suite.completion_normalizer_version,
        "instruction": suite.instruction,
        "execution_image": suite.execution_image,
        "execution_timeout_seconds": suite.execution_timeout_seconds,
        "limits": asdict(suite.limits),
        "tasks": [
            {
                "id": task.task_id,
                "category": task.category,
                "prompt": task.prompt,
                "test_path": task.test_path,
                "setup_path": task.setup_path,
                "expected_tests": task.expected_tests,
                "test_sha256": task.test_sha256,
                "setup_sha256": task.setup_sha256,
            }
            for task in suite.tasks
        ],
    }
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def repository_holdout_suite_sha256(suite: RepositoryHoldoutSuiteConfig) -> str:
    """Return the semantic SHA-256 of prompts, metadata, and executable test bytes."""

    return hashlib.sha256(repository_holdout_suite_json(suite).encode("utf-8")).hexdigest()


def load_frozen_repository_holdout_suite(
    path: Path = _REPOSITORY_HOLDOUT_SUITE_CONFIG_PATH,
) -> RepositoryHoldoutSuiteConfig:
    """Load P6-004 and fail closed if any prompt/test/runtime byte drifts."""

    suite = load_repository_holdout_suite(path)
    fingerprint = repository_holdout_suite_sha256(suite)
    if fingerprint != _FROZEN_REPOSITORY_HOLDOUT_SUITE_SHA256:
        raise RepositoryHoldoutError(
            "frozen repository holdout fingerprint mismatch; increment suite_version and "
            "explicitly update the frozen fingerprint before evaluation"
        )
    return suite
