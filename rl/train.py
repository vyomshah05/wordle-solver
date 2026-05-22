"""Train tabular Q-learning via self-play.

Example::

    python -m rl.train --episodes 50000
    python -m rl.train --episodes 100000 --hard-mode --out rl/checkpoints/q_table.json
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from wordle.words import load_words

from .agent_tabular import QLearningAgent
from .environment import WordleRLEnv


def train(
    episodes: int,
    *,
    hard_mode: bool = True,
    seed: int = 42,
    out_path: Path,
    log_every: int = 5000,
) -> QLearningAgent:
    words = load_words()
    if not words:
        raise SystemExit("No words in data/allowed.txt")

    rng = random.Random(seed)
    env = WordleRLEnv(all_words=words, allowed_set=set(words), hard_mode=hard_mode)
    agent = QLearningAgent()
    agent._rng = rng

    wins = 0
    total_guesses = 0

    for ep in range(1, episodes + 1):
        state = env.reset()
        done = False

        while not done:
            action = agent.select_action(state, explore=True)
            step = env.step_strategy(action)
            agent.update(
                state,
                action,
                step.reward,
                step.tabular_state if not step.done else None,
                step.done,
            )
            state = step.tabular_state
            done = step.done

        agent.on_episode_end()
        if step.won:
            wins += 1
            total_guesses += len(env.game.history)  # type: ignore[union-attr]

        if ep % log_every == 0:
            win_rate = wins / log_every
            mean_g = total_guesses / max(wins, 1)
            print(
                f"episode {ep}/{episodes}  "
                f"win_rate={win_rate:.3f}  mean_guesses(wins)={mean_g:.2f}  "
                f"epsilon={agent.epsilon():.3f}  q_states={len(agent.q)}",
                flush=True,
            )
            wins = 0
            total_guesses = 0

    agent.save(out_path)
    print(f"Saved checkpoint to {out_path} ({len(agent.q)} states)", flush=True)
    return agent


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train RL Wordle agent (tabular Q-learning)")
    parser.add_argument("--episodes", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hard-mode", action="store_true", default=True)
    parser.add_argument("--easy-mode", action="store_true", help="Allow probe guesses")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "checkpoints" / "q_table.json",
    )
    parser.add_argument("--log-every", type=int, default=5000)
    args = parser.parse_args(argv)

    hard = not args.easy_mode
    train(
        args.episodes,
        hard_mode=hard,
        seed=args.seed,
        out_path=args.out,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
