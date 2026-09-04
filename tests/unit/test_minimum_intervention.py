"""P9-004 minimum-intervention protocol and partition regression tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from tiny_qwen_coder.training.minimum_intervention import (
    MinimumInterventionError,
    load_minimum_intervention_protocol,
    partition_task,
    validate_minimum_intervention,
)

_PROTOCOL = Path("configs/train/python/p9_minimum_intervention_v1.yaml")


def test_frozen_minimum_intervention_protocol_is_valid() -> None:
    report = validate_minimum_intervention()

    assert report.task_id == "P9-004"
    assert report.study_id == "python-p9-minimum-intervention-v1"
    assert report.control_training_config == "configs/train/python/p9_rank_r8.yaml"
    assert report.trajectory_max_steps == 1000
    assert report.checkpoint_steps == (50, 100, 250, 500, 1000)
    assert tuple(item.learning_rate for item in report.candidates) == (
        0.00001,
        0.00002,
        0.00005,
        0.0001,
        0.0002,
    )
    assert report.snapshot_count == 25
    assert report.development_partition.source_suites == ("humaneval", "mbpp")
    assert report.development_partition.qualification_only_suites == ("repository_holdout",)
    assert report.selection.require_combined_improvement_over_base is True
    assert report.selection.require_no_suite_regression is True
    assert report.qualification.one_shot is True
    assert report.qualification.evaluate_only_selected_checkpoint is True


def test_partition_is_deterministic_and_keeps_holdout_untouched() -> None:
    protocol = load_minimum_intervention_protocol(_PROTOCOL)
    partition = protocol.development_partition

    assert partition_task(partition, suite="humaneval", task_id="HumanEval/0") == "development"
    assert partition_task(partition, suite="humaneval", task_id="HumanEval/1") == "qualification"
    assert partition_task(partition, suite="mbpp", task_id="11") == "development"
    assert partition_task(partition, suite="mbpp", task_id="12") == "qualification"
    assert (
        partition_task(partition, suite="repository_holdout", task_id="any-holdout-task")
        == "qualification"
    )


def test_partition_rejects_unknown_suite_and_empty_task_id() -> None:
    partition = load_minimum_intervention_protocol(_PROTOCOL).development_partition

    with pytest.raises(MinimumInterventionError, match="not part of the P9-004 partition"):
        partition_task(partition, suite="unknown", task_id="1")
    with pytest.raises(MinimumInterventionError, match="task_id must not be empty"):
        partition_task(partition, suite="humaneval", task_id="")


def test_protocol_sha_fails_closed_on_post_hoc_change(tmp_path: Path) -> None:
    text = _PROTOCOL.read_text(encoding="utf-8")
    mutated = tmp_path / "protocol.yaml"
    mutated.write_text(text.replace("trajectory_max_steps: 1000", "trajectory_max_steps: 999"), encoding="utf-8")

    with pytest.raises(MinimumInterventionError, match="protocol SHA-256"):
        validate_minimum_intervention(mutated)
