"""Model 1 — constraint filter + positional letter-frequency scoring."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

import pygame

from wordle.game import GuessResult, WordleGame, score_guess
from wordle.players import Player

REPEAT_PENALTY = 0.5


def filter_candidates(candidates: Iterable[str], history: list[GuessResult]) -> list[str]:
    remaining = list(candidates)

    for result in history:
        consistent = []
        for word in remaining:
            simulated = score_guess(result.word, word)
            if simulated.states == result.states:
                consistent.append(word)
        remaining = consistent

    return remaining


def _best_guess(candidates: list[str]) -> str:
    # Build a frequency table: freqs[i][ch] = how many candidates have ch at position i
    freqs = [Counter() for _ in range(5)]
    for word in candidates:
        for i, ch in enumerate(word):
            freqs[i][ch] += 1

    def score(word: str) -> float:
        total = 0.0
        seen = set()
        for i, ch in enumerate(word):
            if ch in seen:
                weight = REPEAT_PENALTY
            else:
                weight = 1.0
            total += freqs[i][ch] * weight
            seen.add(ch)
        return total

    best = candidates[0]
    for word in candidates:
        word_score = score(word)
        best_score = score(best)
        if word_score > best_score or (word_score == best_score and word > best):
            best = word

    return best


class BaseModelSolver:
    def __init__(self, words: list[str]) -> None:
        self._words = [w.lower() for w in words]

    def next_guess(self, history: list[GuessResult]) -> str:
        candidates = filter_candidates(self._words, history)
        if len(candidates) == 1:
            return candidates[0]
        return _best_guess(candidates)


class BaseModelPlayer(Player):
    name = "Model 1: Base"

    def __init__(self, words: list[str]) -> None:
        self._solver = BaseModelSolver(words)

    def handle_event(self, event: pygame.event.Event, game: WordleGame) -> str | None:
        return None

    def tick(self, game: WordleGame) -> str | None:
        if game.is_over:
            return None
        return self._solver.next_guess(game.history)


__all__ = ["BaseModelSolver", "BaseModelPlayer", "filter_candidates"]
