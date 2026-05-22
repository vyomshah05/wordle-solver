"""Tabular Q-learning agent (Model 3 — Option A).

The agent learns *which heuristic to use* in each situation, not which exact
word to type. That keeps the state space small enough for a Q-table while still
learning a feedback-aware policy.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from .heuristics import GuessStrategy
from .state import TabularStateKey


@dataclass
class QLearningConfig:
    alpha: float = 0.15  # learning rate
    gamma: float = 0.95  # discount factor
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 30_000
    num_actions: int = len(GuessStrategy)


@dataclass
class QLearningAgent:
    config: QLearningConfig = field(default_factory=QLearningConfig)
    q: dict[str, list[float]] = field(default_factory=dict)
    _rng: random.Random = field(default_factory=random.Random)
    _episodes_seen: int = 0

    def epsilon(self) -> float:
        c = self.config
        t = min(self._episodes_seen, c.epsilon_decay_episodes)
        frac = t / max(c.epsilon_decay_episodes, 1)
        return c.epsilon_start + (c.epsilon_end - c.epsilon_start) * frac

    def _ensure(self, key: str) -> list[float]:
        if key not in self.q:
            self.q[key] = [0.0] * self.config.num_actions
        return self.q[key]

    def select_action(self, state: TabularStateKey, explore: bool = True) -> GuessStrategy:
        key = state.as_key()
        values = self._ensure(key)
        if explore and self._rng.random() < self.epsilon():
            return GuessStrategy(self._rng.randrange(self.config.num_actions))
        best = max(range(len(values)), key=lambda i: values[i])
        return GuessStrategy(best)

    def best_action(self, state: TabularStateKey) -> GuessStrategy:
        return self.select_action(state, explore=False)

    def update(
        self,
        state: TabularStateKey,
        action: GuessStrategy,
        reward: float,
        next_state: TabularStateKey | None,
        done: bool,
    ) -> None:
        c = self.config
        key = state.as_key()
        values = self._ensure(key)
        a = int(action)

        if done or next_state is None:
            target = reward
        else:
            next_vals = self._ensure(next_state.as_key())
            target = reward + c.gamma * max(next_vals)

        values[a] += c.alpha * (target - values[a])

    def on_episode_end(self) -> None:
        self._episodes_seen += 1

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "alpha": self.config.alpha,
                "gamma": self.config.gamma,
                "epsilon_start": self.config.epsilon_start,
                "epsilon_end": self.config.epsilon_end,
                "epsilon_decay_episodes": self.config.epsilon_decay_episodes,
                "num_actions": self.config.num_actions,
            },
            "episodes_seen": self._episodes_seen,
            "q": self.q,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> QLearningAgent:
        data = json.loads(path.read_text(encoding="utf-8"))
        cfg = QLearningConfig(**data["config"])
        agent = cls(config=cfg, q={k: list(v) for k, v in data["q"].items()})
        agent._episodes_seen = data.get("episodes_seen", 0)
        return agent
