from __future__ import annotations

from wordle.game import GuessResult

from .constraints import filter_candidates
from .frequency import best_scoring_word


class FrequencySolver:
    """Constraint filtering + positional letter-frequency scoring (Model 1)."""

    def __init__(self, words: list[str]) -> None:
        self._words = [w.lower() for w in words]

    def next_guess(self, history: list[GuessResult]) -> str:
        candidates = filter_candidates(self._words, history)
        if len(candidates) == 1:
            return candidates[0]
        return best_scoring_word(candidates)
