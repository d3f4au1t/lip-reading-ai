from __future__ import annotations

import numpy as np

from app import webcam
from app.model import WordCertainty
from app.webcam import (
    CaptionHistory,
    _build_display_canvas,
    _certainty_color,
    _draw_word_certainties,
    _processing_placeholder,
)


def test_camera_and_caption_panel_do_not_overlap() -> None:
    camera_frame = np.full((90, 160, 3), 220, dtype=np.uint8)

    display, camera_top, caption_panel_top = _build_display_canvas(
        camera_frame,
        display_width=320,
        header_height=20,
        caption_panel_height=60,
    )

    assert display.shape == (260, 320, 3)
    assert camera_top == 20
    assert caption_panel_top == 200
    assert np.all(display[:camera_top] == 15)
    assert np.all(display[camera_top:caption_panel_top] == 220)
    assert np.all(display[caption_panel_top:] == 15)


def test_certainty_color_runs_from_red_through_yellow_to_green() -> None:
    assert _certainty_color(0.0) == (0, 0, 255)
    assert _certainty_color(0.5) == (0, 255, 255)
    assert _certainty_color(1.0) == (0, 255, 0)


def test_certainty_color_clamps_values_to_probability_range() -> None:
    assert _certainty_color(-1.0) == _certainty_color(0.0)
    assert _certainty_color(2.0) == _certainty_color(1.0)


def test_percentage_is_smaller_and_drawn_below_its_word(monkeypatch) -> None:
    calls: list[tuple[str, tuple[int, int], float, tuple[int, int, int]]] = []

    def record_text(frame, text, origin, scale, color, thickness=2) -> None:
        calls.append((text, origin, scale, color))

    monkeypatch.setattr(webcam, "_draw_text", record_text)
    frame = np.zeros((80, 400, 3), dtype=np.uint8)
    certainty = WordCertainty("HELLO", 0.8, 1)

    _draw_word_certainties(
        frame,
        (certainty,),
        left=10,
        word_baseline=28,
        percentage_baseline=55,
        available_width=380,
    )

    word_call, percentage_call = calls
    assert word_call[0] == "HELLO"
    assert percentage_call[0] == "80%"
    assert percentage_call[1][1] > word_call[1][1]
    assert percentage_call[2] < word_call[2]
    assert word_call[3] == (255, 255, 255)
    assert percentage_call[3] == _certainty_color(0.8)


def test_processing_placeholder_cycles_between_one_and_three_dots() -> None:
    assert _processing_placeholder(10.0, 10.0) == "."
    assert _processing_placeholder(10.0, 10.5) == ".."
    assert _processing_placeholder(10.0, 11.0) == "..."
    assert _processing_placeholder(10.0, 11.5) == "."


def test_caption_history_scrolls_when_fourth_placeholder_starts() -> None:
    history = CaptionHistory(limit=3)
    first = history.start(started_at=1.0)
    history.complete(first, "FIRST")
    second = history.start(started_at=2.0)
    history.complete(second, "SECOND")
    third = history.start(started_at=3.0)

    fourth = history.start(started_at=4.0)

    assert [entry.entry_id for entry in history.entries] == [second, third, fourth]
    assert [entry.text for entry in history.entries] == ["SECOND", "", ""]
    assert [entry.pending for entry in history.entries] == [False, True, True]


def test_caption_rows_are_bottom_anchored_and_shift_up_each_time() -> None:
    history = CaptionHistory(limit=3)
    first = history.start(started_at=1.0)
    assert [entry.entry_id if entry else None for entry in history.display_rows()] == [
        None,
        None,
        first,
    ]

    second = history.start(started_at=2.0)
    assert [entry.entry_id if entry else None for entry in history.display_rows()] == [
        None,
        first,
        second,
    ]

    third = history.start(started_at=3.0)
    assert [entry.entry_id if entry else None for entry in history.display_rows()] == [
        first,
        second,
        third,
    ]

    fourth = history.start(started_at=4.0)
    assert [entry.entry_id if entry else None for entry in history.display_rows()] == [
        second,
        third,
        fourth,
    ]


def test_caption_result_replaces_its_placeholder() -> None:
    history = CaptionHistory(limit=3)
    entry_id = history.start(started_at=1.0)
    certainties = (WordCertainty("HELLO", 0.8, 1),)

    history.complete(entry_id, "HELLO", certainties)

    assert history.entries[0].text == "HELLO"
    assert history.entries[0].word_certainties == certainties
    assert history.entries[0].pending is False
