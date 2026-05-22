from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wordle.game import GuessResult
from wordle.entropy_solver.patterns import (
    encode_states,
    load_or_build_matrix,
    CACHE_DIR,
)

OPENER_PATH = CACHE_DIR / "opener.json"


class EntropySolver:
    def __init__(self, word_list: list[str]) -> None:
        self._words = word_list
        self._index = {w: i for i, w in enumerate(word_list)}
        self._matrix = load_or_build_matrix(word_list)
        self._N = len(word_list)
        self._candidates_mask = np.ones(self._N, dtype=bool)
        self._opener = self._load_or_compute_opener()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._candidates_mask[:] = True

    # ------------------------------------------------------------------
    def choose_guess(self, history: list[GuessResult]) -> str:
        # Apply all history to filter candidates
        self._candidates_mask[:] = True
        for gr in history:
            if gr.word not in self._index:
                continue
            g_idx = self._index[gr.word]
            pat = encode_states(gr.states)
            self._candidates_mask &= self._matrix[g_idx] == pat

        n_cands = int(self._candidates_mask.sum())

        if not history:
            return self._opener

        if n_cands == 1:
            return self._words[int(np.argmax(self._candidates_mask))]

        if n_cands == 0:
            # Fallback: shouldn't happen with valid game state
            return self._opener

        if n_cands <= 10:
            return self._two_step_guess()

        return self._entropy_guess()

    # ------------------------------------------------------------------
    def _entropy_for_row(self, row_idx: int) -> float:
        counts = np.bincount(
            self._matrix[row_idx, self._candidates_mask].astype(np.int64),
            minlength=243,
        )
        total = counts.sum()
        probs = counts[counts > 0] / total
        return float(-np.sum(probs * np.log2(probs)))

    def _entropy_guess(self) -> str:
        scores = np.array([self._entropy_for_row(i) for i in range(self._N)])
        # Tie-break: prefer candidates
        in_cands = self._candidates_mask.astype(float) * 1e-6
        best = int(np.argmax(scores + in_cands))
        return self._words[best]

    # ------------------------------------------------------------------
    def _two_step_guess(self) -> str:
        """Expected-guesses-remaining lookahead over all guesses."""
        cand_indices = np.where(self._candidates_mask)[0]
        best_word = self._words[cand_indices[0]]
        best_score = float("inf")

        for g_idx in range(self._N):
            row = self._matrix[g_idx, self._candidates_mask]
            counts = np.bincount(row.astype(np.int64), minlength=243)
            buckets = counts[counts > 0]
            total = buckets.sum()

            # E[guesses] = sum over buckets: (bucket/total) * (1 if solved else 2)
            # Simplification: solved bucket has size 1, others need another guess
            exp = float(np.sum(buckets * np.where(buckets == 1, 1, 2))) / total
            if exp < best_score or (exp == best_score and self._candidates_mask[g_idx]):
                best_score = exp
                best_word = self._words[g_idx]

        return best_word

    # ------------------------------------------------------------------
    def _load_or_compute_opener(self) -> str:
        if OPENER_PATH.exists():
            data = json.loads(OPENER_PATH.read_text())
            return data["opener"]

        print("Computing best opener (one-time)...", flush=True)
        # Temporarily set all candidates for opener calculation
        mask_backup = self._candidates_mask.copy()
        self._candidates_mask[:] = True

        scores = np.array([self._entropy_for_row(i) for i in range(self._N)])
        top10_idx = np.argsort(scores)[::-1][:10]
        top10 = [(self._words[i], float(scores[i])) for i in top10_idx]

        print("Top 10 openers by entropy:")
        for word, h in top10:
            print(f"  {word}: {h:.4f} bits")

        opener = top10[0][0]
        OPENER_PATH.parent.mkdir(parents=True, exist_ok=True)
        OPENER_PATH.write_text(json.dumps({"opener": opener, "top10": top10}, indent=2))

        self._candidates_mask = mask_backup
        return opener

    # ------------------------------------------------------------------
    @property
    def candidates_mask(self) -> np.ndarray:
        return self._candidates_mask

    @property
    def words(self) -> list[str]:
        return self._words
