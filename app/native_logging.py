from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from functools import wraps
from threading import Thread
from typing import Callable, Iterator, ParamSpec, TypeVar


_KNOWN_NATIVE_LOG_FRAGMENTS = (
    b"All log messages before absl::InitializeLog() is called are written to STDERR",
    b"] GL version:",
    b"Created TensorFlow Lite XNNPACK delegate for CPU",
    b"Feedback manager requires a model with a single signature inference",
    b"Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI",
)

P = ParamSpec("P")
R = TypeVar("R")


def _is_known_native_diagnostic(line: bytes) -> bool:
    return any(fragment in line for fragment in _KNOWN_NATIVE_LOG_FRAGMENTS)


def _forward_filtered_stderr(read_fd: int, destination_fd: int) -> None:
    pending = b""
    try:
        while chunk := os.read(read_fd, 4096):
            pending += chunk
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                line += b"\n"
                if not _is_known_native_diagnostic(line):
                    os.write(destination_fd, line)
        if pending and not _is_known_native_diagnostic(pending):
            os.write(destination_fd, pending)
    finally:
        os.close(read_fd)


@contextmanager
def filter_known_native_diagnostics() -> Iterator[None]:
    """Hide known MediaPipe startup noise while preserving all other stderr."""
    if os.name != "posix":
        yield
        return

    sys.stderr.flush()
    destination_fd = os.dup(2)
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, 2)
    os.close(write_fd)
    forwarding_thread = Thread(
        target=_forward_filtered_stderr,
        args=(read_fd, destination_fd),
        name="native-stderr-filter",
        daemon=True,
    )
    forwarding_thread.start()
    try:
        yield
    finally:
        sys.stderr.flush()
        os.dup2(destination_fd, 2)
        forwarding_thread.join()
        os.close(destination_fd)


def with_filtered_native_diagnostics(
    function: Callable[P, R],
) -> Callable[P, R]:
    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        with filter_known_native_diagnostics():
            return function(*args, **kwargs)

    return wrapped
