"""Guess-selection strategies used as *concrete actions* for tabular Q-learning.

The RL agent does not pick among 12,000 words directly (that state space is
enormous). Instead it picks a *strategy* (frequency, entropy, probe, …) and
these functions turn that choice into an actual 5-letter guess.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from enum import IntEnum

from wordle.game import GuessResult, score_guess

from .constraints import ConstraintState, matches_history


class GuessStrategy(IntEnum):
    """Abstract actions for tabular Q-learning."""

    FREQUENCY = 0  # positional letter frequency over candidates
    ENTROPY = 1  # one-step entropy maximization over candidates
    RANDOM_CANDIDATE = 2  # uniform random remaining answer
    PROBE = 3  # entropy over full guess list (easy mode)


STRATEGY_NAMES = {s: s.name for s in GuessStrategy}


def pattern_key(guess: str, answer: str) -> tuple[str, ...]:
    return tuple(s.value for s in score_guess(guess, answer).states)


def partition_by_pattern(guess: str, candidates: list[str]) -> dict[tuple[str, ...], list[str]]:
    groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for word in candidates:
        groups[pattern_key(guess, word)].append(word)
    return groups


def entropy_bits(groups: dict[tuple[str, ...], list[str]], total: int) -> float:
    if total == 0:
        return 0.0
    h = 0.0
    for words in groups.values():
        p = len(words) / total
        if p > 0:
            h -= p * math.log2(p)
    return h


def score_frequency(word: str, candidates: list[str]) -> float:
    """Sum positional letter frequencies; penalize repeated letters."""
    if not candidates:
        return 0.0
    n = len(candidates)
    pos_counts: list[Counter[str]] = [Counter() for _ in range(5)]
    for w in candidates:
        for i, ch in enumerate(w):
            pos_counts[i][ch] += 1

    score = 0.0
    seen: set[str] = set()
    for i, ch in enumerate(word):
        score += pos_counts[i][ch] / n
        if ch in seen:
            score *= 0.85
        seen.add(ch)
    return score


def best_by_frequency(candidates: list[str]) -> str:
    return max(candidates, key=lambda w: score_frequency(w, candidates))


def best_by_entropy(guess_pool: list[str], candidates: list[str]) -> str:
    if len(candidates) <= 1:
        return candidates[0]
    total = len(candidates)
    best_word = guess_pool[0]
    best_h = -1.0
    for guess in guess_pool:
        groups = partition_by_pattern(guess, candidates)
        h = entropy_bits(groups, total)
        if h > best_h:
            best_h = h
            best_word = guess
    return best_word


def opening_guess(hard_mode: bool) -> str:
    """Fixed strong opener — must exist in data/allowed.txt (e.g. salet, soare)."""
    return "salet"


def pick_guess(
    strategy: GuessStrategy,
    state: ConstraintState,
    all_words: list[str],
    *,
    hard_mode: bool = True,
    rng: random.Random | None = None,
) -> str:
    """Map an abstract RL action to a concrete guess string."""
    rng = rng or random.Random()
    candidates = state.candidates
    history = state.history

    if not history:
        return opening_guess(hard_mode)

    if len(candidates) == 1:
        return candidates[0]

    if strategy is GuessStrategy.RANDOM_CANDIDATE:
        return rng.choice(candidates)

    if strategy is GuessStrategy.FREQUENCY:
        return best_by_frequency(candidates)

    if strategy is GuessStrategy.ENTROPY:
        pool = candidates if hard_mode else all_words
        return best_by_entropy(pool, candidates)

    if strategy is GuessStrategy.PROBE:
        if hard_mode:
            # In hard mode, probing non-candidates is illegal; fall back to entropy.
            return best_by_entropy(candidates, candidates)
        return best_by_entropy(all_words, candidates)

    raise ValueError(f"unknown strategy: {strategy}")


def legal_guess_pool(
    candidates: list[str],
    all_words: list[str],
    hard_mode: bool,
) -> list[str]:
    """Words the agent is allowed to submit under hard/easy rules."""
    if hard_mode:
        return candidates
    return all_words


def is_hard_consistent(guess: str, state: ConstraintState) -> bool:
    """Hard mode: guess must be a possible answer given feedback so far."""
    return matches_history(guess, state.history)
