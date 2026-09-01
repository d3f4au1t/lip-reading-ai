from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from app.config import DEFAULT_CHECKPOINT, TARGET_FPS
from app.model import AutoAVSRRecognizer, RecognitionResult
from app.video import MouthPreprocessor, PreprocessedVideo


@dataclass(frozen=True)
class PipelineResult:
    transcription: str
    recognition: RecognitionResult
    preprocessing_seconds: float
    face_detection_rate: float
    frames: int

    def to_dict(self) -> dict:
        result = asdict(self)
        result["recognition"] = self.recognition.to_dict()
        return result


class VisualSpeechPipeline:
    def __init__(
        self,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        device: str = "auto",
        beam_size: int = 10,
    ) -> None:
        self.preprocessor = MouthPreprocessor()
        self.recognizer = AutoAVSRRecognizer(checkpoint, device, beam_size)

    def transcribe_file(self, path: str | Path) -> PipelineResult:
        started = time.perf_counter()
        video = self.preprocessor.process_file(path)
        preprocessing_seconds = time.perf_counter() - started
        return self._recognize(video, preprocessing_seconds)

    def transcribe_frames(self, frames: Sequence[np.ndarray]) -> PipelineResult:
        started = time.perf_counter()
        video = self.preprocessor.process_frames(frames, source_fps=TARGET_FPS)
        preprocessing_seconds = time.perf_counter() - started
        return self._recognize(video, preprocessing_seconds)

    def _recognize(
        self, video: PreprocessedVideo, preprocessing_seconds: float
    ) -> PipelineResult:
        recognition = self.recognizer.transcribe(video.tensor, video.processed_fps)
        return PipelineResult(
            transcription=recognition.text,
            recognition=recognition,
            preprocessing_seconds=preprocessing_seconds,
            face_detection_rate=video.face_detection_rate,
            frames=video.processed_frame_count,
        )

