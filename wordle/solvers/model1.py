"""Model 1: filter by Wordle feedback, then pick the best positional-frequency guess."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from wordle.game import GuessResult, score_guess

REPEAT_PENALTY = 0.5


def filter_candidates(candidates: Iterable[str], history: list[GuessResult]) -> list[str]:
    remaining = list(candidates)
    for result in history:
        remaining = [
            w for w in remaining if score_guess(result.word, w).states == result.states
        ]
    return remaining


def _best_guess(candidates: list[str]) -> str:
    freqs = [Counter() for _ in range(5)]
    for word in candidates:
        for i, ch in enumerate(word):
            freqs[i][ch] += 1

    def score(word: str) -> float:
        total, seen = 0.0, set()
        for i, ch in enumerate(word):
            w = REPEAT_PENALTY if ch in seen else 1.0
            total += freqs[i][ch] * w
            seen.add(ch)
        return total

    return max(candidates, key=lambda w: (score(w), w))


class FrequencySolver:
    def __init__(self, words: list[str]) -> None:
        self._words = [w.lower() for w in words]

    def next_guess(self, history: list[GuessResult]) -> str:
        candidates = filter_candidates(self._words, history)
        if len(candidates) == 1:
            return candidates[0]
        return _best_guess(candidates)
