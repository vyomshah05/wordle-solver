"""State representations for reinforcement learning.

Two encodings live here:

1. **TabularStateKey** — small discrete key for Q-learning (Option A).
2. **StateVector** — fixed-length float vector for a neural scorer (Option B).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from wordle.game import GuessResult

from .constraints import (
    ConstraintState,
    count_absent_letters,
    count_locked_greens,
    count_yellow_letters,
)


def bucket_candidate_count(n: int) -> int:
    """Coarse buckets so the Q-table stays finite."""
    if n <= 1:
        return 0
    if n <= 5:
        return 1
    if n <= 20:
        return 2
    if n <= 100:
        return 3
    if n <= 500:
        return 4
    if n <= 2000:
        return 5
    return 6


def bucket_absent_count(n: int) -> int:
    if n <= 5:
        return 0
    if n <= 10:
        return 1
    if n <= 15:
        return 2
    return 3


@dataclass(frozen=True)
class TabularStateKey:
    """Hashable state for tabular Q-learning."""

    turn: int  # 0..5 guesses already played
    candidate_bucket: int
    green_count: int  # 0..5
    yellow_count: int  # 0..26, bucketed in __post_init__ via min
    absent_bucket: int

    @classmethod
    def from_game(
        cls,
        history: list[GuessResult],
        constraint: ConstraintState,
    ) -> TabularStateKey:
        yc = count_yellow_letters(history)
        return cls(
            turn=min(len(history), 5),
            candidate_bucket=bucket_candidate_count(constraint.num_candidates),
            green_count=count_locked_greens(history),
            yellow_count=min(yc, 10),  # cap distinct yellows for table size
            absent_bucket=bucket_absent_count(count_absent_letters(history)),
        )

    def as_key(self) -> str:
        return (
            f"t{self.turn}|c{self.candidate_bucket}|g{self.green_count}|"
            f"y{self.yellow_count}|a{self.absent_bucket}"
        )


# --- Option B: vector state (26 letters + 5 positions) ---

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
LETTER_TO_IDX = {ch: i for i, ch in enumerate(ALPHABET)}
VECTOR_DIM = 5 * 26 + 26 + 26 + 2  # greens + yellow + gray + (turn, log candidates)


@dataclass
class StateVector:
    """Dense features for neural policies."""

    data: list[float]

    @classmethod
    def from_game(
        cls,
        history: list[GuessResult],
        constraint: ConstraintState,
    ) -> StateVector:
        from wordle.game import LetterState, keyboard_state

        greens = [0.0] * (5 * 26)
        for result in history:
            for i, (ch, st) in enumerate(zip(result.word, result.states)):
                if st is LetterState.CORRECT:
                    greens[i * 26 + LETTER_TO_IDX[ch]] = 1.0

        kb = keyboard_state(history)
        yellow = [0.0] * 26
        gray = [0.0] * 26
        for ch, st in kb.items():
            idx = LETTER_TO_IDX[ch]
            if st.value == "present":
                yellow[idx] = 1.0
            elif st.value == "absent":
                gray[idx] = 1.0

        turn_norm = min(len(history), 5) / 5.0
        n = max(constraint.num_candidates, 1)
        log_cand = math.log10(n) / math.log10(13000)

        return cls(data=greens + yellow + gray + [turn_norm, log_cand])

    def to_list(self) -> list[float]:
        return list(self.data)
