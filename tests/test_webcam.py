from __future__ import annotations

from app.model import WordCertainty
from app.webcam import CaptionHistory, _certainty_color, _processing_placeholder


def test_certainty_color_runs_from_red_through_yellow_to_green() -> None:
    assert _certainty_color(0.0) == (0, 0, 255)
    assert _certainty_color(0.5) == (0, 255, 255)
    assert _certainty_color(1.0) == (0, 255, 0)


def test_certainty_color_clamps_values_to_probability_range() -> None:
    assert _certainty_color(-1.0) == _certainty_color(0.0)
    assert _certainty_color(2.0) == _certainty_color(1.0)


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


def test_caption_result_replaces_its_placeholder() -> None:
    history = CaptionHistory(limit=3)
    entry_id = history.start(started_at=1.0)
    certainties = (WordCertainty("HELLO", 0.8, 1),)

    history.complete(entry_id, "HELLO", certainties)

    assert history.entries[0].text == "HELLO"
    assert history.entries[0].word_certainties == certainties
    assert history.entries[0].pending is False
