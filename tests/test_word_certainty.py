from __future__ import annotations

import math

import pytest

from app.model import group_word_certainties


def test_sentencepiece_tokens_are_grouped_by_word() -> None:
    token_list = ["<blank>", "▁HEL", "LO", "▁WORLD", "<eos>"]
    result = group_word_certainties(
        [1, 2, 3, 4],
        [math.log(0.81), math.log(0.49), math.log(0.64), math.log(0.99)],
        token_list,
    )
    assert [item.word for item in result] == ["HELLO", "WORLD"]
    assert result[0].certainty == pytest.approx(math.sqrt(0.81 * 0.49))
    assert result[0].token_count == 2
    assert result[1].certainty == pytest.approx(0.64)


def test_grouping_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        group_word_certainties([1], [], ["<blank>", "▁WORD"])

