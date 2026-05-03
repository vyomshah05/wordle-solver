from __future__ import annotations

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ANSWERS_PATH = DATA_DIR / "answers.txt"
ALLOWED_PATH = DATA_DIR / "allowed.txt"


def _load(path: Path) -> list[str]:
    words = []
    for line in path.read_text(encoding="utf-8").splitlines():
        word = line.strip().lower()
        if len(word) == 5 and word.isalpha():
            words.append(word)
    return words


def load_answers() -> list[str]:
    return _load(ANSWERS_PATH)


def load_valid_guesses() -> set[str]:
    """Valid guesses = answer pool ∪ extra-allowed pool.

    The official Wordle lists are disjoint: answers.txt holds answer-eligible
    words and allowed.txt holds the remaining valid (but never-an-answer)
    guesses. Both must be accepted from the keyboard.
    """
    guesses = set(load_answers())
    if ALLOWED_PATH.exists():
        guesses.update(_load(ALLOWED_PATH))
    return guesses
