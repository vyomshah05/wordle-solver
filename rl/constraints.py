"""Feedback-aware candidate filtering.

Every RL and heuristic solver needs the same primitive: given everything Wordle
has revealed so far (greens, yellows, grays), which words could still be the
answer? The RL agent never sees the secret word during play — only this shrinking
set plus summary features we derive from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from wordle.game import GuessResult, LetterState, score_guess


@dataclass
class ConstraintState:
    """Snapshot of solver knowledge at one point in a game."""

    candidates: list[str]
    history: list[GuessResult] = field(default_factory=list)

    @property
    def num_candidates(self) -> int:
        return len(self.candidates)


def matches_history(word: str, history: list[GuessResult]) -> bool:
    """True if *word* could still be the answer given past guesses."""
    word = word.lower()
    for result in history:
        simulated = score_guess(result.word, word)
        if simulated.states != result.states:
            return False
    return True


def filter_candidates(all_words: list[str], history: list[GuessResult]) -> list[str]:
    """Keep only words consistent with every row of feedback."""
    if not history:
        return list(all_words)
    return [w for w in all_words if matches_history(w, history)]


def build_constraint_state(
    all_words: list[str],
    history: list[GuessResult],
) -> ConstraintState:
    return ConstraintState(
        candidates=filter_candidates(all_words, history),
        history=list(history),
    )


def count_locked_greens(history: list[GuessResult]) -> int:
    """Letters known to be in the correct position (best per column)."""
    locked = [None] * 5
    for result in history:
        for i, state in enumerate(result.states):
            if state is LetterState.CORRECT:
                locked[i] = result.word[i]
    return sum(1 for x in locked if x is not None)


def count_yellow_letters(history: list[GuessResult]) -> int:
    """Distinct letters seen yellow at least once (still in word somewhere)."""
    yellows: set[str] = set()
    for result in history:
        for letter, state in zip(result.word, result.states):
            if state is LetterState.PRESENT:
                yellows.add(letter)
    return len(yellows)


def count_absent_letters(history: list[GuessResult]) -> int:
    """Distinct letters ruled absent (gray with no green/yellow for that letter)."""
    from wordle.game import keyboard_state

    best = keyboard_state(history)
    return sum(1 for state in best.values() if state is LetterState.ABSENT)
