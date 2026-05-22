from __future__ import annotations

import argparse
import statistics
import time

from wordle.game import GuessResult, score_guess
from wordle.solvers.model1.solver import FrequencySolver
from wordle.words import load_words


def play_game(solver: FrequencySolver, answer: str, max_guesses: int = 6) -> tuple[int, bool]:
    history: list[GuessResult] = []
    for attempt in range(1, max_guesses + 1):
        guess = solver.next_guess(history)
        result = score_guess(guess, answer)
        history.append(result)
        if guess == answer:
            return attempt, True
    return max_guesses, False


def run_benchmark(
    words: list[str],
    *,
    limit: int | None = None,
    max_guesses: int = 6,
) -> None:
    targets = words[:limit] if limit is not None else words
    solver = FrequencySolver(words)
    attempts: list[int] = []
    win_attempts: list[int] = []
    wins = 0
    start = time.perf_counter()

    for answer in targets:
        n, won = play_game(solver, answer, max_guesses=max_guesses)
        attempts.append(n)
        if won:
            wins += 1
            win_attempts.append(n)

    elapsed = time.perf_counter() - start
    n_games = len(targets)
    win_rate = 100.0 * wins / n_games
    mean_all = statistics.mean(attempts)
    mean_wins = statistics.mean(win_attempts) if win_attempts else float("nan")

    print("Model: frequency (model1)")
    print(f"Games: {n_games}")
    print(f"Win rate: {win_rate:.2f}% ({wins}/{n_games})")
    print(f"Mean guesses (all games): {mean_all:.3f}")
    print(f"Mean guesses (wins only): {mean_wins:.3f}")
    print(f"Median guesses: {statistics.median(attempts):.1f}")
    print(f"Elapsed: {elapsed:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Wordle solvers")
    parser.add_argument(
        "--model",
        choices=("model1", "frequency"),
        default="model1",
        help="Solver to benchmark (only model1 implemented)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap number of answers tested")
    parser.add_argument("--max-guesses", type=int, default=6)
    args = parser.parse_args()

    if args.model not in ("model1", "frequency"):
        raise SystemExit(f"Unknown model: {args.model}")

    words = load_words()
    if not words:
        raise SystemExit("No words found in data/allowed.txt")

    run_benchmark(words, limit=args.limit, max_guesses=args.max_guesses)


if __name__ == "__main__":
    main()
