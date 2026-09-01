from __future__ import annotations

import numpy as np
import pytest

from app.video import resample_frames


def test_resample_30_to_25_fps() -> None:
    frames = np.zeros((30, 8, 8, 3), dtype=np.uint8)
    frames[:, 0, 0, 0] = np.arange(30)
    output = resample_frames(frames, source_fps=30, target_fps=25)
    assert output.shape == (25, 8, 8, 3)
    assert output[0, 0, 0, 0] == 0
    assert output[-1, 0, 0, 0] <= 29


def test_resample_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="RGB video"):
        resample_frames(np.zeros((10, 8, 8), dtype=np.uint8), 25, 25)

