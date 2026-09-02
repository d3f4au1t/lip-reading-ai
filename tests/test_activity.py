from __future__ import annotations

import math

import numpy as np

from app.activity import (
    LIP_LANDMARK_INDICES,
    MotionActivationGate,
    SpeechWindowCollector,
    accumulated_lip_motion_score,
    adaptive_motion_threshold,
    normalized_lip_shape,
)


def _frame(value: int) -> np.ndarray:
    return np.full((4, 4, 3), value, dtype=np.uint8)


def test_idle_frames_do_not_start_or_complete_a_speech_window() -> None:
    collector = SpeechWindowCollector(
        fps=10,
        maximum_seconds=2,
        preroll_seconds=0.2,
        ending_silence_seconds=0.3,
        minimum_seconds=0.5,
    )

    updates = [collector.update(_frame(index), False) for index in range(20)]

    assert not collector.capturing
    assert not any(update.started for update in updates)
    assert not any(update.completed_frames is not None for update in updates)


def test_motion_starts_window_and_silence_completes_it_with_preroll() -> None:
    collector = SpeechWindowCollector(
        fps=10,
        maximum_seconds=2,
        preroll_seconds=0.2,
        ending_silence_seconds=0.3,
        minimum_seconds=0.5,
    )
    collector.update(_frame(1), False)
    collector.update(_frame(2), False)

    started = collector.update(_frame(3), True)
    collector.update(_frame(4), True)
    collector.update(_frame(5), False)
    collector.update(_frame(6), False)
    completed = collector.update(_frame(7), False)

    assert started.started
    assert completed.completed_frames is not None
    assert [int(frame[0, 0, 0]) for frame in completed.completed_frames] == [
        2,
        3,
        4,
        5,
        6,
        7,
    ]
    assert not collector.capturing


def test_speech_window_stops_at_maximum_length() -> None:
    collector = SpeechWindowCollector(
        fps=10,
        maximum_seconds=1,
        preroll_seconds=0.1,
        ending_silence_seconds=0.3,
        minimum_seconds=0.5,
    )

    collector.update(_frame(0), True)
    completed = None
    for index in range(1, 10):
        update = collector.update(_frame(index), True)
        completed = update.completed_frames or completed

    assert completed is not None
    assert len(completed) == 10


def test_brief_pause_does_not_split_a_sentence() -> None:
    collector = SpeechWindowCollector(
        fps=10,
        maximum_seconds=12,
        preroll_seconds=0.1,
        minimum_seconds=0.5,
    )

    collector.update(_frame(0), True)
    brief_pause = [collector.update(_frame(index), False) for index in range(1, 6)]
    resumed = collector.update(_frame(6), True)
    ending_pause = [
        collector.update(_frame(index), False) for index in range(7, 17)
    ]

    assert not any(update.completed_frames is not None for update in brief_pause)
    assert resumed.completed_frames is None
    assert all(update.completed_frames is None for update in ending_pause[:-1])
    assert ending_pause[-1].completed_frames is not None
    assert not collector.capturing


def test_lip_shape_normalization_ignores_translation_scale_and_rotation() -> None:
    landmarks = np.zeros((468, 2), dtype=np.float32)
    angles = np.linspace(0, 2 * math.pi, len(LIP_LANDMARK_INDICES), endpoint=False)
    for index, angle in zip(LIP_LANDMARK_INDICES, angles):
        landmarks[index] = (2.0 * math.cos(angle), math.sin(angle))
    landmarks[61] = (-2.0, 0.0)
    landmarks[291] = (2.0, 0.0)

    angle = math.radians(31)
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.float32,
    )
    transformed = landmarks @ rotation.T * 3.5 + np.array([20.0, -7.0])

    assert np.allclose(
        normalized_lip_shape(landmarks),
        normalized_lip_shape(transformed),
        atol=1e-5,
    )


def test_slow_lip_motion_accumulates_across_recent_frames() -> None:
    base = np.zeros((len(LIP_LANDMARK_INDICES), 2), dtype=np.float32)
    history = [base + index * 0.002 for index in range(6)]
    current = base + 0.012

    immediate_score = float(
        np.mean(np.linalg.norm(current - history[-1], axis=1))
    )
    accumulated_score = accumulated_lip_motion_score(current, history)

    assert immediate_score < 0.006
    assert accumulated_score > 0.006


def test_adaptive_threshold_cannot_become_desensitized() -> None:
    assert adaptive_motion_threshold(0.006, 0.1) == 0.006 * 1.35


def test_motion_gate_ignores_warmup_then_accepts_repeated_motion() -> None:
    gate = MotionActivationGate(
        minimum_score=0.0055,
        vote_window=6,
        required_motion_frames=2,
        warmup_frames=4,
    )

    warmup = [gate.update(0.05, True) for _ in range(4)]
    first_motion = gate.update(0.007, True)
    second_motion = gate.update(0.007, True)

    assert not any(state.active for state in warmup)
    assert first_motion.moving
    assert not first_motion.active
    assert second_motion.active
