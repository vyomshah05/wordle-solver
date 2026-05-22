"""Benchmark a trained RL checkpoint without pygame."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from wordle.words import load_words

from .agent_tabular import QLearningAgent
from .environment import WordleRLEnv
from .heuristics import GuessStrategy, pick_guess
from .state import TabularStateKey


def evaluate(
    checkpoint: Path,
    games: int = 1000,
    *,
    hard_mode: bool = True,
    seed: int = 0,
) -> dict[str, float]:
    words = load_words()
    agent = QLearningAgent.load(checkpoint)
    rng = random.Random(seed)
    env = WordleRLEnv(all_words=words, allowed_set=set(words), hard_mode=hard_mode)
    env._rng = rng

    wins = 0
    guess_counts: list[int] = []
    strategy_use: dict[str, int] = {s.name: 0 for s in GuessStrategy}

    for _ in range(games):
        state = env.reset()
        done = False
        while not done:
            action = agent.best_action(state)
            strategy_use[action.name] += 1
            step = env.step_strategy(action)
            state = step.tabular_state
            done = step.done
        if step.won:
            wins += 1
            guess_counts.append(len(env.game.history))  # type: ignore[union-attr]

    return {
        "games": games,
        "win_rate": wins / games,
        "mean_guesses_when_won": sum(guess_counts) / max(len(guess_counts), 1),
        "strategy_use": strategy_use,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RL Wordle agent")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).resolve().parent / "checkpoints" / "q_table.json",
    )
    parser.add_argument("--games", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--easy-mode", action="store_true")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(f"Missing checkpoint: {args.checkpoint}")

    stats = evaluate(
        args.checkpoint,
        games=args.games,
        hard_mode=not args.easy_mode,
        seed=args.seed,
    )
    print(f"Games: {stats['games']}")
    print(f"Win rate: {stats['win_rate']:.4f}")
    print(f"Mean guesses (wins only): {stats['mean_guesses_when_won']:.3f}")
    print("Strategy uses:", stats["strategy_use"])


if __name__ == "__main__":
    main()
