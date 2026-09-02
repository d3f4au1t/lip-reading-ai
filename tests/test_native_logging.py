from __future__ import annotations

import os

from app.native_logging import filter_known_native_diagnostics


def test_known_native_diagnostics_are_filtered_without_hiding_errors(capfd) -> None:
    with filter_known_native_diagnostics():
        os.write(
            2,
            b"W0000 inference_feedback_manager.cc:114] Feedback manager requires "
            b"a model with a single signature inference.\n",
        )
        os.write(2, b"Unexpected native error that must remain visible.\n")

    captured = capfd.readouterr()
    assert "Feedback manager" not in captured.err
    assert "Unexpected native error" in captured.err
