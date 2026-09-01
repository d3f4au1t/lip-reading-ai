from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Sequence

import cv2
import mediapipe as mp
import numpy as np


LIP_LANDMARK_INDICES = (
    61,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    291,
    308,
    324,
    318,
    402,
    317,
    14,
    87,
    178,
    88,
    95,
    78,
    191,
    80,
    81,
    82,
    13,
    312,
    311,
    310,
    415,
)


@dataclass(frozen=True)
class LipMotionObservation:
    face_visible: bool
    motion_score: float
    threshold: float
    active: bool


def normalized_lip_shape(landmarks: np.ndarray) -> np.ndarray:
    """Normalize lip points for face translation, scale, and in-plane rotation."""
    points = np.asarray(landmarks, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) <= max(
        LIP_LANDMARK_INDICES
    ):
        raise ValueError("Expected at least 416 two-dimensional face landmarks.")

    left_corner = points[61]
    right_corner = points[291]
    mouth_axis = right_corner - left_corner
    mouth_width = float(np.linalg.norm(mouth_axis))
    if mouth_width <= 1e-6:
        raise ValueError("Mouth corners are too close to measure lip motion.")

    center = (left_corner + right_corner) / 2
    angle = math.atan2(float(mouth_axis[1]), float(mouth_axis[0]))
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float32)
    lip_points = points[list(LIP_LANDMARK_INDICES)]
    return ((lip_points - center) @ rotation) / mouth_width


class LipMotionDetector:
    """Detect sustained lip-shape changes with MediaPipe Face Mesh."""

    def __init__(
        self,
        min_motion_score: float = 0.01,
        vote_window: int = 5,
        required_motion_frames: int = 2,
        maximum_input_width: int = 640,
    ) -> None:
        if min_motion_score <= 0:
            raise ValueError("The mouth-motion threshold must be positive.")
        if not 1 <= required_motion_frames <= vote_window:
            raise ValueError("Required motion frames must fit inside the vote window.")
        self.min_motion_score = min_motion_score
        self.required_motion_frames = required_motion_frames
        self.maximum_input_width = maximum_input_width
        self._recent_motion: deque[bool] = deque(maxlen=vote_window)
        self._previous_shape: np.ndarray | None = None
        self._face_frames = 0
        self._noise_floor = min_motion_score / 5
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def observe(self, rgb_frame: np.ndarray) -> LipMotionObservation:
        frame = np.asarray(rgb_frame, dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected an RGB camera frame shaped [height, width, 3].")
        height, width = frame.shape[:2]
        if width > self.maximum_input_width:
            resized_height = max(1, round(height * self.maximum_input_width / width))
            frame = cv2.resize(
                frame,
                (self.maximum_input_width, resized_height),
                interpolation=cv2.INTER_AREA,
            )

        result = self._face_mesh.process(frame)
        threshold = max(self.min_motion_score, self._noise_floor * 4)
        if not result.multi_face_landmarks:
            self._previous_shape = None
            self._recent_motion.clear()
            self._face_frames = 0
            return LipMotionObservation(False, 0.0, threshold, False)

        frame_height, frame_width = frame.shape[:2]
        landmarks = result.multi_face_landmarks[0].landmark
        points = np.array(
            [
                (landmark.x * frame_width, landmark.y * frame_height)
                for landmark in landmarks
            ],
            dtype=np.float32,
        )
        shape = normalized_lip_shape(points)
        self._face_frames += 1
        if self._previous_shape is None:
            score = 0.0
        else:
            score = float(
                np.mean(np.linalg.norm(shape - self._previous_shape, axis=1))
            )
        self._previous_shape = shape

        threshold = max(self.min_motion_score, self._noise_floor * 4)
        moving = self._face_frames >= 3 and score >= threshold
        if 0 < score < threshold:
            self._noise_floor = 0.97 * self._noise_floor + 0.03 * score
        self._recent_motion.append(moving)
        active = sum(self._recent_motion) >= self.required_motion_frames
        return LipMotionObservation(True, score, threshold, active)

    def close(self) -> None:
        self._face_mesh.close()


@dataclass(frozen=True)
class SpeechWindowUpdate:
    started: bool = False
    completed_frames: tuple[np.ndarray, ...] | None = None


class SpeechWindowCollector:
    """Collect a bounded speech window with preroll and silence endpointing."""

    def __init__(
        self,
        fps: float,
        maximum_seconds: float,
        preroll_seconds: float = 0.35,
        ending_silence_seconds: float = 0.7,
        minimum_seconds: float = 0.8,
    ) -> None:
        if fps <= 0 or maximum_seconds <= 0:
            raise ValueError("Frame rate and maximum window length must be positive.")
        if minimum_seconds > maximum_seconds:
            raise ValueError("Minimum window length cannot exceed the maximum.")
        self.maximum_frames = max(3, round(fps * maximum_seconds))
        self.minimum_frames = max(3, round(fps * minimum_seconds))
        self.ending_silence_frames = max(1, round(fps * ending_silence_seconds))
        self._preroll: deque[np.ndarray] = deque(
            maxlen=max(1, round(fps * preroll_seconds))
        )
        self._frames: list[np.ndarray] | None = None
        self._silent_frames = 0

    @property
    def capturing(self) -> bool:
        return self._frames is not None

    def update(self, frame: np.ndarray, mouth_active: bool) -> SpeechWindowUpdate:
        if self._frames is None:
            self._preroll.append(frame)
            if not mouth_active:
                return SpeechWindowUpdate()
            self._frames = list(self._preroll)
            self._preroll.clear()
            self._silent_frames = 0
            return SpeechWindowUpdate(started=True)

        self._frames.append(frame)
        self._silent_frames = 0 if mouth_active else self._silent_frames + 1
        reached_maximum = len(self._frames) >= self.maximum_frames
        reached_silence = (
            len(self._frames) >= self.minimum_frames
            and self._silent_frames >= self.ending_silence_frames
        )
        if not reached_maximum and not reached_silence:
            return SpeechWindowUpdate()

        completed = tuple(self._frames)
        self._frames = None
        self._silent_frames = 0
        self._preroll.extend(completed[-self._preroll.maxlen :])
        return SpeechWindowUpdate(completed_frames=completed)

