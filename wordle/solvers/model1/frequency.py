from __future__ import annotations

from collections import Counter

REPEAT_PENALTY = 0.5


def positional_frequencies(candidates: list[str]) -> list[Counter[str]]:
    """Per-position letter counts across the candidate set."""
    freqs = [Counter() for _ in range(5)]
    for word in candidates:
        for i, letter in enumerate(word):
            freqs[i][letter] += 1
    return freqs


def score_word(word: str, pos_freqs: list[Counter[str]]) -> float:
    """Sum positional frequencies; down-weight repeated letters in the guess."""
    total = 0.0
    seen: set[str] = set()
    for i, letter in enumerate(word):
        weight = REPEAT_PENALTY if letter in seen else 1.0
        total += pos_freqs[i][letter] * weight
        seen.add(letter)
    return total


def best_scoring_word(candidates: list[str]) -> str:
    """Pick the highest-scoring candidate; ties break alphabetically."""
    pos_freqs = positional_frequencies(candidates)
    return max(candidates, key=lambda w: (score_word(w, pos_freqs), w))
