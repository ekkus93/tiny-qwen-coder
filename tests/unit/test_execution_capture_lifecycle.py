"""Regression tests for execution output-capture lifecycle hardening."""

from __future__ import annotations

import io
import time

import pytest

from tiny_qwen_coder.evaluation import execution


class _SlowEofStream(io.BytesIO):
    """Expose whether the parent closes a stream before the reader reaches EOF."""

    def __init__(self) -> None:
        super().__init__(b"payload")
        self.read_calls = 0
        self.closed_before_eof = False

    def read(self, size: int | None = -1) -> bytes:
        self.read_calls += 1
        if self.read_calls == 2:
            time.sleep(0.05)
            self.closed_before_eof = self.closed
        return super().read(size)


class _FailingStream(io.BytesIO):
    def read(self, size: int | None = -1) -> bytes:
        del size
        raise ValueError("synthetic capture failure")


def test_finalize_capture_waits_for_reader_eof_before_closing_streams() -> None:
    stdout = _SlowEofStream()
    stderr = io.BytesIO(b"")
    stdout_capture = execution._BoundedCapture.create(128)
    stderr_capture = execution._BoundedCapture.create(128)
    stdout_thread = execution._start_capture_thread(stdout, stdout_capture)
    stderr_thread = execution._start_capture_thread(stderr, stderr_capture)

    execution._finalize_capture_threads(
        stdout_stream=stdout,
        stderr_stream=stderr,
        stdout_capture=stdout_capture,
        stderr_capture=stderr_capture,
        stdout_thread=stdout_thread,
        stderr_thread=stderr_thread,
        timeout_seconds=1.0,
        context="test",
    )

    assert stdout_capture.text() == "payload"
    assert stdout.closed_before_eof is False
    assert stdout.closed is True
    assert stderr.closed is True


def test_capture_thread_exception_is_propagated_to_the_caller() -> None:
    stdout = _FailingStream()
    stderr = io.BytesIO(b"")
    stdout_capture = execution._BoundedCapture.create(128)
    stderr_capture = execution._BoundedCapture.create(128)
    stdout_thread = execution._start_capture_thread(stdout, stdout_capture)
    stderr_thread = execution._start_capture_thread(stderr, stderr_capture)

    with pytest.raises(execution.ExecutionHarnessError, match="test stdout capture failed"):
        execution._finalize_capture_threads(
            stdout_stream=stdout,
            stderr_stream=stderr,
            stdout_capture=stdout_capture,
            stderr_capture=stderr_capture,
            stdout_thread=stdout_thread,
            stderr_thread=stderr_thread,
            timeout_seconds=1.0,
            context="test",
        )

    assert stdout.closed is True
    assert stderr.closed is True
