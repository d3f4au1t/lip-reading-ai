from __future__ import annotations

from app.webcam import _certainty_color


def test_certainty_color_runs_from_red_through_yellow_to_green() -> None:
    assert _certainty_color(0.0) == (0, 0, 255)
    assert _certainty_color(0.5) == (0, 255, 255)
    assert _certainty_color(1.0) == (0, 255, 0)


def test_certainty_color_clamps_values_to_probability_range() -> None:
    assert _certainty_color(-1.0) == _certainty_color(0.0)
    assert _certainty_color(2.0) == _certainty_color(1.0)
