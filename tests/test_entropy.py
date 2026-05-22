from __future__ import annotations

import random

import numpy as np
import pytest

from wordle.game import LetterState, score_guess
from wordle.entropy_solver.patterns import (
    encode_states,
    decode_pattern,
    pattern_int,
    build_pattern_matrix,
    ABSENT,
    PRESENT,
    CORRECT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode(*codes: int) -> int:
    result = 0
    for c in codes:
        result = result * 3 + c
    return result


# ---------------------------------------------------------------------------
# test_repeated_letters
# ---------------------------------------------------------------------------

class TestRepeatedLetters:
    def _check(self, guess: str, answer: str) -> None:
        oracle = encode_states(score_guess(guess, answer).states)
        got = pattern_int(guess, answer)
        assert got == oracle, (
            f"pattern_int({guess!r}, {answer!r}) = {got} != oracle {oracle}\n"
            f"  oracle decoded: {decode_pattern(oracle)}\n"
            f"  got decoded:    {decode_pattern(got)}"
        )

    def test_bobby_abbey(self):
        # B-O-B-B-Y vs A-B-B-E-Y
        # b@0: absent(b not in abbey at 0; abbey has b@1,b@2 but not at 0 — present? abbey=a,b,b,e,y — b exists → present)
        # Actually let oracle decide; just verify they match.
        self._check("bobby", "abbey")

    def test_alley_llama(self):
        self._check("alley", "llama")

    def test_speed_erase(self):
        self._check("speed", "erase")

    def test_batch_match(self):
        self._check("batch", "match")

    def test_catch_hatch(self):
        self._check("catch", "hatch")

    def test_exact_match(self):
        self._check("crane", "crane")
        p = pattern_int("crane", "crane")
        assert p == _encode(CORRECT, CORRECT, CORRECT, CORRECT, CORRECT)

    def test_all_absent(self):
        self._check("bbbbb", "aaaaa")
        p = pattern_int("bbbbb", "aaaaa")
        assert p == _encode(ABSENT, ABSENT, ABSENT, ABSENT, ABSENT)


# ---------------------------------------------------------------------------
# test_pattern_matrix_matches_oracle
# ---------------------------------------------------------------------------

def test_pattern_matrix_matches_oracle():
    rng = random.Random(0)
    from wordle.words import load_words
    all_words = load_words()
    words = rng.sample(all_words, 50)
    matrix = build_pattern_matrix(words)
    for i, g in enumerate(words):
        for j, a in enumerate(words):
            expected = pattern_int(g, a)
            got = int(matrix[i, j])
            assert got == expected, (
                f"matrix[{i},{j}] ({g!r} vs {a!r}): got {got}, expected {expected}"
            )


# ---------------------------------------------------------------------------
# test_filter_keeps_answer
# ---------------------------------------------------------------------------

def test_filter_keeps_answer():
    from wordle.words import load_words
    from wordle.entropy_solver import EntropySolver
    from wordle.game import WordleGame

    rng = random.Random(7)
    words = load_words()
    solver = EntropySolver(words)
    answer = rng.choice(words)
    game = WordleGame(answer, allowed_words=set(words))

    for _ in range(6):
        if game.is_over:
            break
        guess = solver.choose_guess(game.history)
        game.submit(guess)
        ans_idx = solver._index[answer]
        # After filtering via choose_guess, answer must still be a candidate
        assert solver._candidates_mask[ans_idx], (
            f"Answer {answer!r} was filtered out after guess {guess!r}"
        )


# ---------------------------------------------------------------------------
# test_entropy_reference
# ---------------------------------------------------------------------------

def test_entropy_reference():
    words = ["crane", "slate", "raise", "arise", "stare",
             "snare", "share", "spare", "flare", "glare",
             "blare", "flute", "brute", "crude", "prude",
             "grade", "trade", "blade", "shade", "spade"]
    matrix = build_pattern_matrix(words)
    N = len(words)

    def entropy(i: int) -> float:
        counts = np.bincount(matrix[i].astype(np.int64), minlength=243)
        total = counts.sum()
        p = counts[counts > 0] / total
        return float(-np.sum(p * np.log2(p)))

    entropies = [entropy(i) for i in range(N)]
    best_idx = int(np.argmax(entropies))
    # "crane" and "slate" are well-known strong openers; just verify best entropy
    # is above a reasonable floor (> 3 bits for 20 words).
    assert entropies[best_idx] > 3.0, f"Best entropy {entropies[best_idx]:.3f} seems too low"


# ---------------------------------------------------------------------------
# test_solver_smoke (slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_solver_smoke():
    from wordle.words import load_words
    from wordle.entropy_solver import EntropySolver
    from wordle.game import WordleGame

    rng = random.Random(42)
    words = load_words()
    solver = EntropySolver(words)
    word_set = set(words)

    sample = rng.sample(words, 100)
    solved = 0
    for answer in sample:
        solver.reset()
        game = WordleGame(answer, allowed_words=word_set)
        for _ in range(6):
            if game.is_over:
                break
            guess = solver.choose_guess(game.history)
            game.submit(guess)
        if game.is_won:
            solved += 1

    assert solved >= 95, f"Only solved {solved}/100 games (need ≥ 95)"
