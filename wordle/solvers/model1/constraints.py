from __future__ import annotations

from collections.abc import Iterable

from wordle.game import GuessResult, score_guess


def filter_candidates(
    candidates: Iterable[str],
    history: list[GuessResult],
) -> list[str]:
    """Keep only words consistent with every prior guess and its feedback."""
    remaining = list(candidates)
    for result in history:
        remaining = [
            word
            for word in remaining
            if score_guess(result.word, word).states == result.states
        ]
    return remaining
