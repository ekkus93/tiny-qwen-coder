"""CPU-only tests for the P4-003 constrained OCI execution harness."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import cast

import pytest

from tiny_qwen_coder.config import ExecutionConfig
from tiny_qwen_coder.evaluation.execution import (
    ConstrainedExecutionHarness,
    ExecutionCleanupError,
    ExecutionFile,
    ExecutionHarnessUnavailableError,
    ExecutionLimits,
    ExecutionRequest,
    ExecutionRequestError,
    ExecutionStatus,
    OciRuntime,
    OciRuntimeSpec,
    discover_oci_runtime,
)


def _fake_runtime(
    tmp_path: Path,
    *,
    cleanup_succeeds: bool = True,
) -> tuple[OciRuntimeSpec, Path, Path]:
    record_path = tmp_path / "runtime-record.json"
    descendant_marker = tmp_path / "descendant-survived.txt"
    executable = tmp_path / "fake-docker"
    executable.write_text(
        f"""#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import cast

record_path = Path({str(record_path)!r})
descendant_marker = Path({str(descendant_marker)!r})
args = sys.argv[1:]
if not args:
    raise SystemExit(2)
if args[0] == "rm":
    raise SystemExit({0 if cleanup_succeeds else 9})
if args[0] != "run":
    raise SystemExit(3)

mount_value = args[args.index("--mount") + 1]
source = next(
    part.removeprefix("src=")
    for part in mount_value.split(",")
    if part.startswith("src=")
)
metadata = {{
    "args": args,
    "aws_secret": os.environ.get("AWS_SECRET_ACCESS_KEY"),
    "home": os.environ.get("HOME"),
    "input_source": source,
    "input_text": (
        (Path(source) / "nested" / "fixture.txt").read_text()
        if (Path(source) / "nested" / "fixture.txt").exists()
        else None
    ),
}}
record_path.write_text(json.dumps(metadata, sort_keys=True))
marker = args[-1]
if marker == "fixture:success":
    print("candidate stdout")
    print("candidate stderr", file=sys.stderr)
    raise SystemExit(0)
if marker == "fixture:fail":
    print("failed stdout")
    print("failed stderr", file=sys.stderr)
    raise SystemExit(7)
if marker == "fixture:flood":
    print("X" * 5000)
    print("Y" * 5000, file=sys.stderr)
    raise SystemExit(0)
if marker == "fixture:timeout":
    code = (
        "import time; from pathlib import Path; time.sleep(0.3); Path("
        + repr(str(descendant_marker))
        + ").write_text('survived')"
    )
    subprocess.Popen([sys.executable, "-c", code])
    time.sleep(60)
    raise SystemExit(0)
raise SystemExit(4)
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return (
        OciRuntimeSpec(kind=OciRuntime.DOCKER, executable=executable.resolve()),
        record_path,
        descendant_marker,
    )


def _execution(*, network_enabled: bool = False, timeout_seconds: float = 2.0) -> ExecutionConfig:
    return ExecutionConfig(
        timeout_seconds=timeout_seconds,
        network_enabled=network_enabled,
    )


def _request(marker: str = "fixture:success") -> ExecutionRequest:
    return ExecutionRequest(
        image="fixtures/sandbox@sha256:" + "a" * 64,
        command=(marker,),
        files=(ExecutionFile.from_text("nested/fixture.txt", "fixture payload\n"),),
    )


def _metadata(record_path: Path) -> dict[str, object]:
    return cast(
        dict[str, object],
        json.loads(record_path.read_text(encoding="utf-8")),
    )


def _metadata_args(metadata: dict[str, object]) -> list[str]:
    return cast(list[str], metadata["args"])


def _option_value(args: list[str], option: str) -> str:
    return args[args.index(option) + 1]


def test_success_captures_stdout_stderr_status_and_disposes_workspace(tmp_path: Path) -> None:
    runtime, record_path, _ = _fake_runtime(tmp_path)
    result = ConstrainedExecutionHarness(runtime=runtime).run(_request(), _execution())
    metadata = _metadata(record_path)

    assert result.status is ExecutionStatus.SUCCEEDED
    assert result.exit_code == 0
    assert result.succeeded is True
    assert result.stdout == "candidate stdout\n"
    assert result.stderr == "candidate stderr\n"
    assert result.stdout_truncated is False
    assert result.stderr_truncated is False
    assert metadata["input_text"] == "fixture payload\n"
    assert not Path(str(metadata["input_source"])).exists()


def test_nonzero_exit_is_captured_without_raising(tmp_path: Path) -> None:
    runtime, _, _ = _fake_runtime(tmp_path)
    result = ConstrainedExecutionHarness(runtime=runtime).run(
        _request("fixture:fail"),
        _execution(),
    )

    assert result.status is ExecutionStatus.FAILED
    assert result.exit_code == 7
    assert result.stdout == "failed stdout\n"
    assert result.stderr == "failed stderr\n"


def test_default_command_enforces_container_security_and_resource_bounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, record_path, _ = _fake_runtime(tmp_path)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-not-cross-boundary")
    monkeypatch.setenv("HOME", "/host/home/with-credentials")
    limits = ExecutionLimits(
        cpus=0.5,
        memory_mebibytes=192,
        pids=23,
        workspace_mebibytes=17,
        temp_mebibytes=19,
        open_files=77,
    )

    ConstrainedExecutionHarness(runtime=runtime).run(
        _request(),
        _execution(),
        limits=limits,
    )
    metadata = _metadata(record_path)
    args = _metadata_args(metadata)

    assert _option_value(args, "--pull") == "never"
    assert _option_value(args, "--log-driver") == "none"
    assert "--no-healthcheck" in args
    assert "--read-only" in args
    assert _option_value(args, "--cap-drop") == "ALL"
    assert _option_value(args, "--security-opt") == "no-new-privileges"
    assert _option_value(args, "--ipc") == "none"
    assert _option_value(args, "--entrypoint") == "/bin/sh"
    assert _option_value(args, "--network") == "none"
    assert _option_value(args, "--pids-limit") == "23"
    assert _option_value(args, "--memory") == "192m"
    assert _option_value(args, "--memory-swap") == "384m"
    assert _option_value(args, "--cpus") == "0.5"
    assert _option_value(args, "--user") == "65534:65534"
    assert "nofile=77:77" in args
    assert "core=0:0" in args
    assert any(value.startswith("/workspace:rw,nosuid,nodev,size=17m") for value in args)
    assert any(value.startswith("/tmp:rw,noexec,nosuid,nodev,size=19m") for value in args)
    mount_value = _option_value(args, "--mount")
    assert "dst=/input" in mount_value
    assert mount_value.endswith("readonly")
    assert metadata["aws_secret"] is None
    assert metadata["home"] != "/host/home/with-credentials"
    assert "must-not-cross-boundary" not in "\0".join(args)


def test_network_must_be_explicitly_enabled(tmp_path: Path) -> None:
    runtime, record_path, _ = _fake_runtime(tmp_path)

    ConstrainedExecutionHarness(runtime=runtime).run(
        _request(),
        _execution(network_enabled=True),
    )
    args = _metadata_args(_metadata(record_path))

    assert "--network" not in args


def test_timeout_force_removes_container_and_kills_runtime_process_group(tmp_path: Path) -> None:
    runtime, _, descendant_marker = _fake_runtime(tmp_path)
    started = time.monotonic()

    result = ConstrainedExecutionHarness(runtime=runtime).run(
        _request("fixture:timeout"),
        _execution(timeout_seconds=0.1),
        limits=ExecutionLimits(cleanup_timeout_seconds=1.0),
    )

    assert result.status is ExecutionStatus.TIMED_OUT
    assert result.exit_code is None
    assert time.monotonic() - started < 2.0
    time.sleep(0.5)
    assert not descendant_marker.exists()


def test_cleanup_failure_is_fail_closed(tmp_path: Path) -> None:
    runtime, _, _ = _fake_runtime(tmp_path, cleanup_succeeds=False)

    with pytest.raises(ExecutionCleanupError, match="force-remove timed-out container"):
        ConstrainedExecutionHarness(runtime=runtime).run(
            _request("fixture:timeout"),
            _execution(timeout_seconds=0.1),
            limits=ExecutionLimits(cleanup_timeout_seconds=1.0),
        )


def test_output_capture_is_bounded_while_streams_are_fully_drained(tmp_path: Path) -> None:
    runtime, _, _ = _fake_runtime(tmp_path)

    result = ConstrainedExecutionHarness(runtime=runtime).run(
        _request("fixture:flood"),
        _execution(),
        limits=ExecutionLimits(max_output_bytes=128),
    )

    assert result.status is ExecutionStatus.SUCCEEDED
    assert len(result.stdout.encode("utf-8")) <= 128
    assert len(result.stderr.encode("utf-8")) <= 128
    assert result.stdout_truncated is True
    assert result.stderr_truncated is True


def test_input_size_and_path_traversal_are_rejected_before_runtime_launch(tmp_path: Path) -> None:
    runtime, record_path, _ = _fake_runtime(tmp_path)
    harness = ConstrainedExecutionHarness(runtime=runtime)

    with pytest.raises(ExecutionRequestError, match="parent components"):
        ExecutionFile.from_text("../escape.py", "pass")

    with pytest.raises(ExecutionRequestError, match="limit is 4"):
        harness.run(
            ExecutionRequest(
                image="fixtures/sandbox:local",
                command=("fixture:success",),
                files=(ExecutionFile.from_text("main.py", "12345"),),
            ),
            _execution(),
            limits=ExecutionLimits(max_input_bytes=4),
        )

    assert not record_path.exists()


def test_image_cannot_inject_oci_runtime_options() -> None:
    with pytest.raises(ExecutionRequestError, match="OCI runtime option"):
        ExecutionRequest(image="--privileged", command=("python",))


def test_podman_disables_implicit_writable_tmpfs_mounts(tmp_path: Path) -> None:
    runtime, record_path, _ = _fake_runtime(tmp_path)
    podman_runtime = OciRuntimeSpec(
        kind=OciRuntime.PODMAN,
        executable=runtime.executable,
    )

    ConstrainedExecutionHarness(runtime=podman_runtime).run(_request(), _execution())
    args = _metadata_args(_metadata(record_path))

    assert "--read-only-tmpfs=false" in args


def test_runtime_discovery_fails_closed_without_supported_oci_runtime(tmp_path: Path) -> None:
    with pytest.raises(ExecutionHarnessUnavailableError, match="refusing untrusted execution"):
        discover_oci_runtime(search_path=str(tmp_path))
