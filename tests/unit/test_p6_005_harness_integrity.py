"""Regression tests for P6-005 baseline execution integrity."""

from tiny_qwen_coder.evaluation.execution import _COPY_AND_EXEC_SCRIPT


def test_workspace_bootstrap_never_chmods_the_tmpfs_mount_root() -> None:
    assert "chmod -R u+rwX /workspace;" not in _COPY_AND_EXEC_SCRIPT
    assert 'chmod -R u+rwX "$path"' in _COPY_AND_EXEC_SCRIPT
