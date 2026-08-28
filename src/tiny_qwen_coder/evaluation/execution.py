"""Constrained OCI execution harness for untrusted generated code.

P4-003 deliberately uses an OCI runtime as the security boundary rather than
pretending a plain host subprocess can disable networking or protect host
credentials.  The harness builds a disposable, read-only-root container with
bounded tmpfs work areas, drops capabilities, runs as an unprivileged numeric
user, never inherits host credentials into the container, and disables network
access unless evaluation configuration explicitly enables it.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from tiny_qwen_coder.config import ExecutionConfig

_CONTAINER_WORKSPACE = "/workspace"
_CONTAINER_INPUT = "/input"
_CONTAINER_TMP = "/tmp"
_SANDBOX_UID = 65534
_SANDBOX_GID = 65534
_COPY_AND_EXEC_SCRIPT = (
    "set -eu; "
    "cp -R /input/. /workspace/; "
    "chmod -R u+rwX /workspace; "
    "cd /workspace; "
    'exec "$@"'
)


class ExecutionHarnessError(RuntimeError):
    """Base class for constrained execution harness failures."""


class ExecutionHarnessUnavailableError(ExecutionHarnessError):
    """Raised when no supported OCI isolation runtime is available."""


class ExecutionCleanupError(ExecutionHarnessError):
    """Raised when a timed-out container cannot be forcefully removed."""


class ExecutionRequestError(ValueError):
    """Raised when an execution request or resource policy is unsafe/invalid."""


class OciRuntime(StrEnum):
    """Supported OCI-compatible command-line runtimes."""

    PODMAN = "podman"
    DOCKER = "docker"


class ExecutionStatus(StrEnum):
    """High-level outcome of one sandboxed command."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


@dataclass(frozen=True, slots=True)
class OciRuntimeSpec:
    """Resolved OCI runtime executable used as the host-side isolation boundary."""

    kind: OciRuntime
    executable: Path

    def __post_init__(self) -> None:
        if not self.executable.is_absolute():
            raise ExecutionRequestError("OCI runtime executable must use an absolute path")
        if not self.executable.is_file() or not os.access(self.executable, os.X_OK):
            raise ExecutionRequestError(
                f"OCI runtime executable is not an executable file: {self.executable}"
            )


@dataclass(frozen=True, slots=True)
class ExecutionFile:
    """One trusted evaluator-provided input file copied into the disposable workspace."""

    path: str
    content: bytes
    executable: bool = False

    def __post_init__(self) -> None:
        candidate = PurePosixPath(self.path)
        if candidate.is_absolute() or not candidate.parts:
            raise ExecutionRequestError("execution file path must be a non-empty relative path")
        if any(part in {"", ".", ".."} for part in candidate.parts):
            raise ExecutionRequestError(
                "execution file path must not contain empty, current, or parent components"
            )
        if "\x00" in self.path:
            raise ExecutionRequestError("execution file path must not contain NUL bytes")
        if not isinstance(self.content, bytes):
            raise TypeError("execution file content must be bytes")

    @classmethod
    def from_text(
        cls,
        path: str,
        content: str,
        *,
        executable: bool = False,
    ) -> ExecutionFile:
        """Create one UTF-8 execution file without platform newline rewriting."""

        return cls(path=path, content=content.encode("utf-8"), executable=executable)


@dataclass(frozen=True, slots=True)
class ExecutionRequest:
    """Trusted evaluator description of one command to run against generated files."""

    image: str
    command: tuple[str, ...]
    files: tuple[ExecutionFile, ...] = ()

    def __post_init__(self) -> None:
        if not self.image.strip() or self.image != self.image.strip():
            raise ExecutionRequestError("execution image must be a non-empty exact reference")
        if self.image.startswith("-"):
            raise ExecutionRequestError(
                "execution image must not be parsed as an OCI runtime option"
            )
        if not self.command:
            raise ExecutionRequestError("execution command must contain at least one argument")
        if any(not argument or "\x00" in argument for argument in self.command):
            raise ExecutionRequestError(
                "execution command arguments must be non-empty and NUL-free"
            )
        paths = tuple(item.path for item in self.files)
        if len(paths) != len(set(paths)):
            raise ExecutionRequestError("execution files must not contain duplicate paths")


@dataclass(frozen=True, slots=True)
class ExecutionLimits:
    """Generic resource bounds applied to every untrusted candidate container."""

    cpus: float = 1.0
    memory_mebibytes: int = 512
    pids: int = 64
    workspace_mebibytes: int = 64
    temp_mebibytes: int = 64
    max_output_bytes: int = 1_048_576
    max_input_bytes: int = 1_048_576
    open_files: int = 256
    cleanup_timeout_seconds: float = 5.0

    def __post_init__(self) -> None:
        if self.cpus <= 0:
            raise ExecutionRequestError("execution cpus must be greater than zero")
        for field_name, value in (
            ("memory_mebibytes", self.memory_mebibytes),
            ("pids", self.pids),
            ("workspace_mebibytes", self.workspace_mebibytes),
            ("temp_mebibytes", self.temp_mebibytes),
            ("max_output_bytes", self.max_output_bytes),
            ("max_input_bytes", self.max_input_bytes),
            ("open_files", self.open_files),
        ):
            if value <= 0:
                raise ExecutionRequestError(f"execution {field_name} must be greater than zero")
        if self.cleanup_timeout_seconds <= 0:
            raise ExecutionRequestError(
                "execution cleanup_timeout_seconds must be greater than zero"
            )


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Complete bounded process evidence from one sandboxed execution."""

    status: ExecutionStatus
    runtime: OciRuntime
    exit_code: int | None
    duration_seconds: float
    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool

    def __post_init__(self) -> None:
        if self.duration_seconds < 0:
            raise ExecutionRequestError("execution duration_seconds must not be negative")
        if self.status is ExecutionStatus.TIMED_OUT:
            if self.exit_code is not None:
                raise ExecutionRequestError("timed-out execution must not expose an exit code")
        elif self.exit_code is None:
            raise ExecutionRequestError("completed execution must expose an exit code")
        if self.status is ExecutionStatus.SUCCEEDED and self.exit_code != 0:
            raise ExecutionRequestError("successful execution must have exit code zero")
        if self.status is ExecutionStatus.FAILED and self.exit_code == 0:
            raise ExecutionRequestError("failed execution must have a non-zero exit code")

    @property
    def succeeded(self) -> bool:
        """Return whether the sandboxed command completed successfully."""

        return self.status is ExecutionStatus.SUCCEEDED


@dataclass(slots=True)
class _BoundedCapture:
    limit: int
    data: bytearray
    truncated: bool = False

    @classmethod
    def create(cls, limit: int) -> _BoundedCapture:
        return cls(limit=limit, data=bytearray())

    def consume(self, stream: BinaryIO) -> None:
        while True:
            chunk = stream.read(64 * 1024)
            if not chunk:
                return
            remaining = self.limit - len(self.data)
            if remaining > 0:
                self.data.extend(chunk[:remaining])
            if len(chunk) > max(remaining, 0):
                self.truncated = True

    def text(self) -> str:
        return bytes(self.data).decode("utf-8", errors="replace")


def discover_oci_runtime(*, search_path: str | None = None) -> OciRuntimeSpec:
    """Resolve a supported local OCI runtime, preferring rootless-friendly Podman."""

    path = search_path if search_path is not None else os.defpath
    for kind in (OciRuntime.PODMAN, OciRuntime.DOCKER):
        resolved = shutil.which(str(kind), path=path)
        if resolved is not None:
            return OciRuntimeSpec(kind=kind, executable=Path(resolved).resolve())
    raise ExecutionHarnessUnavailableError(
        "no supported OCI runtime found; refusing untrusted execution because "
        "network and filesystem isolation cannot be guaranteed"
    )


def _runtime_environment(runtime_home: Path) -> dict[str, str]:
    """Return a credential-free host environment for invoking the trusted OCI CLI."""

    environment = {
        "HOME": str(runtime_home),
        "PATH": os.defpath,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }
    # Rootless Podman commonly needs the runtime directory to find its user socket/state.
    xdg_runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if xdg_runtime_dir:
        environment["XDG_RUNTIME_DIR"] = xdg_runtime_dir
    return environment


def _write_input_files(
    root: Path,
    files: tuple[ExecutionFile, ...],
    *,
    max_input_bytes: int,
) -> None:
    total_bytes = sum(len(item.content) for item in files)
    if total_bytes > max_input_bytes:
        raise ExecutionRequestError(
            f"execution input contains {total_bytes} bytes; limit is {max_input_bytes}"
        )

    root.chmod(0o755)
    for item in files:
        relative = PurePosixPath(item.path)
        destination = root.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(item.content)
        destination.chmod(0o555 if item.executable else 0o444)
    for directory in sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        directory.chmod(0o555)


def _container_name() -> str:
    return f"tiny-qwen-coder-{uuid.uuid4().hex}"


def _oci_run_command(
    *,
    runtime: OciRuntimeSpec,
    request: ExecutionRequest,
    execution: ExecutionConfig,
    limits: ExecutionLimits,
    input_directory: Path,
    container_name: str,
) -> tuple[str, ...]:
    command: list[str] = [
        str(runtime.executable),
        "run",
        "--name",
        container_name,
        "--rm",
        "--pull",
        "never",
        "--log-driver",
        "none",
        "--no-healthcheck",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--ipc",
        "none",
        "--pids-limit",
        str(limits.pids),
        "--memory",
        f"{limits.memory_mebibytes}m",
        "--memory-swap",
        f"{limits.memory_mebibytes * 2}m",
        "--cpus",
        f"{limits.cpus:g}",
        "--ulimit",
        f"nofile={limits.open_files}:{limits.open_files}",
        "--ulimit",
        "core=0:0",
        "--user",
        f"{_SANDBOX_UID}:{_SANDBOX_GID}",
        "--workdir",
        _CONTAINER_WORKSPACE,
        "--entrypoint",
        "/bin/sh",
        "--mount",
        f"type=bind,src={input_directory},dst={_CONTAINER_INPUT},readonly",
        "--tmpfs",
        (
            f"{_CONTAINER_WORKSPACE}:rw,nosuid,nodev,size="
            f"{limits.workspace_mebibytes}m,mode=1777"
        ),
        "--tmpfs",
        f"{_CONTAINER_TMP}:rw,noexec,nosuid,nodev,size={limits.temp_mebibytes}m,mode=1777",
        "--env",
        "HOME=/tmp/home",
        "--env",
        "TMPDIR=/tmp",
        "--env",
        "LANG=C.UTF-8",
        "--env",
        "LC_ALL=C.UTF-8",
        "--env",
        "PYTHONNOUSERSITE=1",
    ]
    if runtime.kind is OciRuntime.PODMAN:
        command.append("--read-only-tmpfs=false")
    if not execution.network_enabled:
        command.extend(("--network", "none"))
    command.extend(
        (
            request.image,
            "-c",
            _COPY_AND_EXEC_SCRIPT,
            "tiny-qwen-coder-sandbox",
            *request.command,
        )
    )
    return tuple(command)


def _start_capture_thread(stream: BinaryIO, capture: _BoundedCapture) -> threading.Thread:
    thread = threading.Thread(target=capture.consume, args=(stream,), daemon=True)
    thread.start()
    return thread


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait()


def _force_remove_container(
    *,
    runtime: OciRuntimeSpec,
    container_name: str,
    environment: Mapping[str, str],
    timeout_seconds: float,
) -> None:
    try:
        cleanup = subprocess.run(
            [str(runtime.executable), "rm", "--force", container_name],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=dict(environment),
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ExecutionCleanupError(
            f"failed to force-remove timed-out container {container_name!r}"
        ) from exc
    if cleanup.returncode != 0:
        detail = (cleanup.stderr or b"").decode("utf-8", errors="replace").strip()
        raise ExecutionCleanupError(
            f"failed to force-remove timed-out container {container_name!r}: "
            f"{detail or f'exit code {cleanup.returncode}'}"
        )


class ConstrainedExecutionHarness:
    """Run generated code inside one disposable constrained OCI container."""

    def __init__(
        self,
        *,
        runtime: OciRuntimeSpec | None = None,
        temp_root: Path | None = None,
    ) -> None:
        self._runtime = runtime
        self._temp_root = temp_root

    def run(
        self,
        request: ExecutionRequest,
        execution: ExecutionConfig,
        *,
        limits: ExecutionLimits | None = None,
    ) -> ExecutionResult:
        """Execute one request with bounded output and deterministic isolation flags."""

        resolved_limits = limits if limits is not None else ExecutionLimits()
        runtime = self._runtime if self._runtime is not None else discover_oci_runtime()

        with tempfile.TemporaryDirectory(
            prefix="tiny-qwen-coder-exec-",
            dir=self._temp_root,
        ) as temporary_root_text:
            temporary_root = Path(temporary_root_text)
            input_directory = temporary_root / "input"
            runtime_home = temporary_root / "runtime-home"
            input_directory.mkdir()
            runtime_home.mkdir(mode=0o700)
            _write_input_files(
                input_directory,
                request.files,
                max_input_bytes=resolved_limits.max_input_bytes,
            )

            container_name = _container_name()
            host_environment = _runtime_environment(runtime_home)
            command = _oci_run_command(
                runtime=runtime,
                request=request,
                execution=execution,
                limits=resolved_limits,
                input_directory=input_directory,
                container_name=container_name,
            )

            try:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=host_environment,
                    start_new_session=True,
                )
            except OSError as exc:
                raise ExecutionHarnessUnavailableError(
                    f"could not launch OCI runtime {runtime.executable}"
                ) from exc

            if (  # pragma: no cover - defensive Popen contract check
                process.stdout is None or process.stderr is None
            ):
                _kill_process_group(process)
                raise ExecutionHarnessError("OCI runtime did not provide output pipes")

            stdout_capture = _BoundedCapture.create(resolved_limits.max_output_bytes)
            stderr_capture = _BoundedCapture.create(resolved_limits.max_output_bytes)
            stdout_thread = _start_capture_thread(process.stdout, stdout_capture)
            stderr_thread = _start_capture_thread(process.stderr, stderr_capture)
            started = time.monotonic()
            timed_out = False
            cleanup_error: ExecutionCleanupError | None = None

            try:
                process.wait(timeout=execution.timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    _force_remove_container(
                        runtime=runtime,
                        container_name=container_name,
                        environment=host_environment,
                        timeout_seconds=resolved_limits.cleanup_timeout_seconds,
                    )
                except ExecutionCleanupError as exc:
                    cleanup_error = exc
                finally:
                    _kill_process_group(process)
            finally:
                process.stdout.close()
                process.stderr.close()
                stdout_thread.join(timeout=resolved_limits.cleanup_timeout_seconds)
                stderr_thread.join(timeout=resolved_limits.cleanup_timeout_seconds)

            duration = time.monotonic() - started
            if stdout_thread.is_alive() or stderr_thread.is_alive():
                raise ExecutionHarnessError("output capture threads did not terminate cleanly")
            if cleanup_error is not None:
                raise cleanup_error

            if timed_out:
                status = ExecutionStatus.TIMED_OUT
                exit_code: int | None = None
            else:
                return_code = process.returncode
                if return_code is None:  # pragma: no cover - process.wait() contract
                    raise ExecutionHarnessError(
                        "OCI runtime returned without a process exit code"
                    )
                exit_code = return_code
                status = (
                    ExecutionStatus.SUCCEEDED
                    if exit_code == 0
                    else ExecutionStatus.FAILED
                )

            return ExecutionResult(
                status=status,
                runtime=runtime.kind,
                exit_code=exit_code,
                duration_seconds=duration,
                stdout=stdout_capture.text(),
                stderr=stderr_capture.text(),
                stdout_truncated=stdout_capture.truncated,
                stderr_truncated=stderr_capture.truncated,
            )
