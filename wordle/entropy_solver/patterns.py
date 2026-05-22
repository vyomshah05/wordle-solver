from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

from wordle.game import LetterState, score_guess

ABSENT = 0
PRESENT = 1
CORRECT = 2

_STATE_MAP = {
    LetterState.ABSENT: ABSENT,
    LetterState.PRESENT: PRESENT,
    LetterState.CORRECT: CORRECT,
}

CACHE_DIR = Path(__file__).resolve().parent / "cache"
MATRIX_PATH = CACHE_DIR / "pattern_matrix.npy"
HASH_PATH = CACHE_DIR / "pattern_matrix.hash"


def encode_states(states: tuple[LetterState, ...]) -> int:
    result = 0
    for s in states:
        result = result * 3 + _STATE_MAP[s]
    return result


def decode_pattern(p: int) -> tuple[int, ...]:
    digits = []
    for _ in range(5):
        digits.append(p % 3)
        p //= 3
    return tuple(reversed(digits))


def pattern_int(guess: str, answer: str) -> int:
    return encode_states(score_guess(guess, answer).states)


def _word_list_hash(words: list[str]) -> str:
    joined = "\n".join(sorted(words))
    return hashlib.sha256(joined.encode()).hexdigest()


def build_pattern_matrix(words: list[str]) -> np.ndarray:
    N = len(words)
    # Encode words as (N, 5) int8 letter-index arrays
    letters = np.array([[ord(c) - ord('a') for c in w] for w in words], dtype=np.uint8)

    # Start with all-absent pattern (0)
    matrix = np.zeros((N, N), dtype=np.uint8)

    # Track which positions are green for each (guess, answer) pair
    green = np.zeros((N, N, 5), dtype=bool)
    for p in range(5):
        eq = letters[:, p:p+1] == letters[None, :, p]  # (N, N)
        green[:, :, p] = eq
        # Green contributes CORRECT=2, positional weight 3^(4-p)
        matrix += (2 * eq * (3 ** (4 - p))).astype(np.uint8)

    # Yellow pass: for each letter c, count unmatched occurrences
    for c in range(26):
        # For each word, positions where letter==c and NOT green
        guess_c = (letters == c)  # (N, 5)
        ans_c = (letters == c)    # (N, 5)

        # Unmatched = letter is c AND not green
        # We process guess positions left-to-right greedily
        # answer_unmatched_count[j] = #positions in answer j where letter==c and not green with any specific guess
        # This needs to be per (i,j) pair

        # answer_unmatched[i,j,p] = ans_c[j,p] AND NOT green[i,j,p]
        # We want answer_pool[i,j] = sum over p of (ans_c[j,p] AND NOT green[i,j,p])
        # guess positions not green with c: guess_unmatched_mask[i,p] = guess_c[i,p] AND NOT green[i,j,p] -- but depends on j

        # Vectorized: shape (N,N) answer pool for letter c
        ans_pool = np.sum(
            ans_c[np.newaxis, :, :] & ~green, axis=2  # (N, N, 5) -> (N, N)
        )  # ans_pool[i,j] = #answer-j positions with letter c not matched green with guess i

        # For each guess position p, check if guess has letter c and not green
        for p in range(5):
            # guess i has letter c at position p and it's not green with answer j
            guess_has_c_at_p = guess_c[:, p]  # (N,)
            not_green_p = ~green[:, :, p]  # (N, N)

            # Only process positions where guess has c and is not green
            # yellow credit: min(remaining_ans_pool, 1) per position, left to right
            can_yellow = guess_has_c_at_p[:, np.newaxis] & not_green_p  # (N, N)

            # Yellow credit available: min(ans_pool, 1) where can_yellow
            yellow = can_yellow & (ans_pool > 0)  # (N, N)

            matrix += (yellow * (3 ** (4 - p))).astype(np.uint8)

            # Consume one from pool where we issued a yellow
            ans_pool -= yellow.astype(np.uint8)

    return matrix


def load_or_build_matrix(words: list[str]) -> np.ndarray:
    h = _word_list_hash(words)
    if MATRIX_PATH.exists() and HASH_PATH.exists():
        if HASH_PATH.read_text().strip() == h:
            return np.load(str(MATRIX_PATH))

    print("Building pattern matrix (first run, ~30–90 s)...", flush=True)
    matrix = build_pattern_matrix(words)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(MATRIX_PATH), matrix)
    HASH_PATH.write_text(h)
    print("Pattern matrix built and cached.", flush=True)
    return matrix
