from __future__ import annotations

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ALLOWED_PATH = DATA_DIR / "allowed.txt"


def load_words() -> list[str]:
    """Load every valid 5-letter word. Any word can be the secret answer."""
    words = []
    for line in ALLOWED_PATH.read_text(encoding="utf-8").splitlines():
        word = line.strip().lower()
        if len(word) == 5 and word.isalpha():
            words.append(word)
    return words
