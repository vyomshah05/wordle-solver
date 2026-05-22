#!/usr/bin/env python3
"""Run a headless simulation of the entropy solver and report statistics."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wordle.words import load_words
from wordle.entropy_solver import EntropySolver
from wordle.entropy_solver.patterns import encode_states, decode_pattern
from wordle.game import WordleGame


def run_simulation(
    words: list[str],
    sample: list[str],
) -> list[dict]:
    solver = EntropySolver(words)
    word_set = set(words)
    records = []

    for i, answer in enumerate(sample):
        solver.reset()
        game = WordleGame(answer, allowed_words=word_set)
        turns = []

        while not game.is_over:
            cands_before = int(solver.candidates_mask.sum()) if game.history else len(words)
            guess = solver.choose_guess(game.history)

            # Entropy of this guess over current candidates
            import numpy as np
            if guess in solver._index:
                g_idx = solver._index[guess]
                mask = solver.candidates_mask
                counts = np.bincount(
                    solver._matrix[g_idx, mask].astype(np.int64), minlength=243
                )
                total = counts.sum()
                p = counts[counts > 0] / total
                entropy = float(-np.sum(p * np.log2(p))) if total > 0 else 0.0
            else:
                entropy = 0.0

            result = game.submit(guess)
            pat = decode_pattern(encode_states(result.states))
            turns.append({
                "guess": guess,
                "pattern": list(pat),
                "entropy": round(entropy, 4),
                "candidates_remaining": cands_before,
            })

        records.append({
            "answer": answer,
            "num_guesses": len(game.history),
            "solved": game.is_won,
            "turns": turns,
        })

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(sample)} done...", flush=True)

    return records


def print_stats(records: list[dict]) -> None:
    total = len(records)
    solved = sum(1 for r in records if r["solved"])
    guesses = [r["num_guesses"] for r in records if r["solved"]]

    hist: dict[str, int] = {str(k): 0 for k in range(1, 7)}
    hist["fail"] = 0
    for r in records:
        if r["solved"] and r["num_guesses"] <= 6:
            hist[str(r["num_guesses"])] += 1
        elif not r["solved"]:
            hist["fail"] += 1

    mean = sum(guesses) / len(guesses) if guesses else float("nan")
    guesses_sorted = sorted(guesses)
    median = guesses_sorted[len(guesses_sorted) // 2] if guesses_sorted else float("nan")
    worst = max(guesses) if guesses else 0

    print(f"\n{'='*40}")
    print(f"Total games:       {total}")
    print(f"Solved:            {solved}/{total} ({100*solved/total:.1f}%)")
    print(f"Win rate (≤6):     {100*solved/total:.1f}%")
    print(f"Mean guesses:      {mean:.3f}")
    print(f"Median guesses:    {median}")
    print(f"Worst case:        {worst}")
    print(f"Histogram:         {hist}")
    print(f"{'='*40}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Entropy solver benchmark")
    parser.add_argument("--sample", type=int, default=0, help="Number of words to sample (0 = all)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    words = load_words()
    rng = random.Random(args.seed)

    if args.sample and args.sample < len(words):
        sample = rng.sample(words, args.sample)
    else:
        sample = list(words)

    out_path = args.out or f"results/entropy_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"Running entropy solver on {len(sample)} words (seed={args.seed})...")
    t0 = time.time()
    records = run_simulation(words, sample)
    elapsed = time.time() - t0

    print_stats(records)
    print(f"Elapsed: {elapsed:.1f}s")

    Path(out_path).write_text(json.dumps({"records": records}, indent=2))
    print(f"Results written to: {out_path}")


if __name__ == "__main__":
    main()
