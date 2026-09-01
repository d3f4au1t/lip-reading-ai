from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import av
import numpy as np
import torch

from app.config import TARGET_FPS, THIRD_PARTY_ROOT, validate_model_source


class FaceNotFoundError(RuntimeError):
    """Raised when no usable face is visible in a segment."""


@dataclass(frozen=True)
class DecodedVideo:
    frames: np.ndarray
    source_fps: float
    duration_seconds: float


@dataclass(frozen=True)
class PreprocessedVideo:
    tensor: torch.Tensor
    source_fps: float
    processed_fps: float
    source_frame_count: int
    processed_frame_count: int
    face_detection_rate: float

    @property
    def duration_seconds(self) -> float:
        return self.processed_frame_count / self.processed_fps


def _add_autoavsr_to_path() -> None:
    validate_model_source()
    source = str(THIRD_PARTY_ROOT)
    if source not in sys.path:
        sys.path.insert(0, source)


def read_video_frames(path: str | Path) -> DecodedVideo:
    video_path = Path(path).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    frames: list[np.ndarray] = []
    with av.open(str(video_path)) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream found in: {video_path}")
        stream = container.streams.video[0]
        source_fps = float(stream.average_rate) if stream.average_rate else TARGET_FPS
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))

    if not frames:
        raise ValueError(f"Video contains no decodable frames: {video_path}")
    if not np.isfinite(source_fps) or source_fps <= 0:
        source_fps = TARGET_FPS

    array = np.stack(frames)
    return DecodedVideo(
        frames=array,
        source_fps=source_fps,
        duration_seconds=len(array) / source_fps,
    )


def resample_frames(
    frames: np.ndarray, source_fps: float, target_fps: float = TARGET_FPS
) -> np.ndarray:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("Expected RGB video shaped [frames, height, width, 3].")
    if len(frames) == 0:
        raise ValueError("Cannot resample an empty video.")
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("Frame rates must be positive.")
    if abs(source_fps - target_fps) < 1e-3:
        return frames

    target_count = max(1, int(round(len(frames) * target_fps / source_fps)))
    source_indices = np.rint(
        np.arange(target_count, dtype=np.float64) * source_fps / target_fps
    ).astype(np.int64)
    source_indices = np.clip(source_indices, 0, len(frames) - 1)
    return frames[source_indices]


class MouthPreprocessor:
    """Run Auto-AVSR's official MediaPipe alignment and test transform."""

    def __init__(self) -> None:
        _add_autoavsr_to_path()
        from datamodule.transforms import VideoTransform
        from preparation.detectors.mediapipe.detector import LandmarksDetector
        from preparation.detectors.mediapipe.video_process import VideoProcess

        self._landmarks_detector = LandmarksDetector()
        self._video_process = VideoProcess(convert_gray=False)
        self._video_transform = VideoTransform(subset="test")

    def process_file(self, path: str | Path) -> PreprocessedVideo:
        decoded = read_video_frames(path)
        frames = resample_frames(decoded.frames, decoded.source_fps, TARGET_FPS)
        return self.process_frames(frames, source_fps=decoded.source_fps)

    def process_frames(
        self, frames: Sequence[np.ndarray] | np.ndarray, source_fps: float = TARGET_FPS
    ) -> PreprocessedVideo:
        frame_array = np.asarray(frames, dtype=np.uint8)
        if frame_array.ndim != 4 or frame_array.shape[-1] != 3:
            raise ValueError("Expected RGB frames shaped [frames, height, width, 3].")
        if len(frame_array) < 3:
            raise ValueError("At least three video frames are required for lip reading.")

        # This follows the official detector's full-range-first behavior while allowing
        # us to report how many frames were detected before interpolation.
        detector = self._landmarks_detector
        landmarks = detector.detect(frame_array, detector.full_range_detector)
        if all(item is None for item in landmarks):
            landmarks = detector.detect(frame_array, detector.short_range_detector)
        detected = sum(item is not None for item in landmarks)
        if detected == 0:
            raise FaceNotFoundError(
                "No face was detected. Use a front-facing, well-lit video with the "
                "speaker's full mouth visible."
            )

        mouth = self._video_process(frame_array, landmarks)
        if mouth is None or len(mouth) == 0:
            raise FaceNotFoundError("A face was found, but a stable mouth crop could not be produced.")
        if mouth.shape[1:3] != (96, 96):
            raise RuntimeError(f"Unexpected mouth crop shape: {mouth.shape}; expected [T, 96, 96, 3].")

        tensor = torch.from_numpy(np.ascontiguousarray(mouth)).permute(0, 3, 1, 2)
        tensor = self._video_transform(tensor).float().contiguous()
        return PreprocessedVideo(
            tensor=tensor,
            source_fps=source_fps,
            processed_fps=TARGET_FPS,
            source_frame_count=len(frame_array),
            processed_frame_count=len(tensor),
            face_detection_rate=detected / len(frame_array),
        )

