from __future__ import annotations

import os

import pytest

from app.config import DEFAULT_CHECKPOINT, DEFAULT_SAMPLE
from app.pipeline import VisualSpeechPipeline


@pytest.mark.integration
@pytest.mark.skipif(
    not (DEFAULT_CHECKPOINT.is_file() and DEFAULT_SAMPLE.is_file()),
    reason="Run scripts/download_assets.py first",
)
def test_visual_only_pipeline_produces_text() -> None:
    pipeline = VisualSpeechPipeline(
        checkpoint=DEFAULT_CHECKPOINT,
        device=os.environ.get("VSR_TEST_DEVICE", "auto"),
        beam_size=1,
    )
    result = pipeline.transcribe_file(DEFAULT_SAMPLE)
    assert result.transcription.strip()
    assert result.frames >= 3
    assert 0 < result.face_detection_rate <= 1
    assert result.recognition.video_seconds > 0
    assert result.recognition.inference_seconds > 0

