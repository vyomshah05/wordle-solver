from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum


class LetterState(Enum):
    CORRECT = "correct"
    PRESENT = "present"
    ABSENT = "absent"


@dataclass(frozen=True)
class GuessResult:
    word: str
    states: tuple[LetterState, ...]


class InvalidGuess(ValueError):
    pass


class WordleGame:
    WORD_LENGTH = 5

    def __init__(
        self,
        answer: str,
        max_guesses: int = 6,
        allowed_words: set[str] | None = None,
    ) -> None:
        answer = answer.lower()
        if len(answer) != self.WORD_LENGTH or not answer.isalpha():
            raise ValueError(f"answer must be {self.WORD_LENGTH} letters: {answer!r}")
        self.answer = answer
        self.max_guesses = max_guesses
        self.allowed_words = allowed_words
        self._history: list[GuessResult] = []

    @property
    def history(self) -> list[GuessResult]:
        return list(self._history)

    @property
    def guesses_left(self) -> int:
        return self.max_guesses - len(self._history)

    @property
    def is_won(self) -> bool:
        return bool(self._history) and self._history[-1].word == self.answer

    @property
    def is_lost(self) -> bool:
        return not self.is_won and len(self._history) >= self.max_guesses

    @property
    def is_over(self) -> bool:
        return self.is_won or self.is_lost

    def submit(self, guess: str) -> GuessResult:
        if self.is_over:
            raise InvalidGuess("game is already over")
        guess = guess.lower()
        if len(guess) != self.WORD_LENGTH or not guess.isalpha():
            raise InvalidGuess(f"guess must be {self.WORD_LENGTH} letters")
        if self.allowed_words is not None and guess not in self.allowed_words:
            raise InvalidGuess(f"{guess!r} is not in the word list")

        result = score_guess(guess, self.answer)
        self._history.append(result)
        return result


def score_guess(guess: str, answer: str) -> GuessResult:
    """Two-pass scoring that handles duplicate letters correctly."""
    guess = guess.lower()
    answer = answer.lower()
    states: list[LetterState] = [LetterState.ABSENT] * len(guess)
    remaining = Counter(answer)

    # Pass 1: greens
    for i, (g, a) in enumerate(zip(guess, answer)):
        if g == a:
            states[i] = LetterState.CORRECT
            remaining[g] -= 1

    # Pass 2: yellows
    for i, g in enumerate(guess):
        if states[i] is LetterState.CORRECT:
            continue
        if remaining[g] > 0:
            states[i] = LetterState.PRESENT
            remaining[g] -= 1

    return GuessResult(word=guess, states=tuple(states))


def keyboard_state(history: list[GuessResult]) -> dict[str, LetterState]:
    """Best-known state for each letter so far. CORRECT > PRESENT > ABSENT."""
    priority = {
        LetterState.CORRECT: 3,
        LetterState.PRESENT: 2,
        LetterState.ABSENT: 1,
    }
    best: dict[str, LetterState] = {}
    for result in history:
        for letter, state in zip(result.word, result.states):
            current = best.get(letter)
            if current is None or priority[state] > priority[current]:
                best[letter] = state
    return best
