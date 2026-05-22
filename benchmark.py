from __future__ import annotations

import argparse
import statistics
import time

from wordle.game import score_guess
from wordle.solvers.model1 import FrequencySolver
from wordle.words import load_words


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Model 1")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    words = load_words()
    targets = words[: args.limit] if args.limit else words
    solver = FrequencySolver(words)

    attempts, wins = [], 0
    start = time.perf_counter()
    for answer in targets:
        history = []
        for n in range(1, 7):
            guess = solver.next_guess(history)
            history.append(score_guess(guess, answer))
            if guess == answer:
                attempts.append(n)
                wins += 1
                break
        else:
            attempts.append(6)

    elapsed = time.perf_counter() - start
    print(f"Games: {len(targets)}  Win rate: {100 * wins / len(targets):.1f}%")
    print(f"Mean guesses: {statistics.mean(attempts):.2f}  Median: {statistics.median(attempts):.1f}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
