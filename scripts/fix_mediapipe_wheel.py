"""Correct an upstream metadata typo in MediaPipe's macOS universal2 wheel.

MediaPipe 0.10.21's official universal2 archive contains native arm64 binaries but
its internal WHEEL file says x86_64. uv therefore treats an otherwise working
installation as incompatible. This changes metadata only; it does not alter code.
"""

from __future__ import annotations

import platform
import sys
from importlib.metadata import distribution
from pathlib import Path


def main() -> int:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        print("MediaPipe wheel metadata fix is not needed on this platform.")
        return 0

    dist = distribution("mediapipe")
    wheel_entries = [
        entry
        for entry in (dist.files or [])
        if entry.name == "WHEEL" and ".dist-info" in str(entry.parent)
    ]
    if len(wheel_entries) != 1:
        raise RuntimeError("Could not locate MediaPipe's installed WHEEL metadata file.")
    wheel = Path(dist.locate_file(wheel_entries[0]))
    content = wheel.read_text(encoding="utf-8")
    incorrect = "Tag: cp311-cp311-macosx_14_0_x86_64"
    corrected = "Tag: cp311-cp311-macosx_11_0_universal2"
    if corrected in content:
        print("MediaPipe wheel metadata is already correct.")
        return 0
    if incorrect not in content:
        raise RuntimeError(f"Unexpected MediaPipe wheel tag in {wheel}:\n{content}")
    wheel.write_text(content.replace(incorrect, corrected), encoding="utf-8")
    print(f"Corrected MediaPipe universal2 metadata: {wheel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
