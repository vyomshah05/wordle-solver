"""Small neural guess-scorer (Model 3 — Option B sketch).

This is optional and heavier than tabular Q-learning. It encodes the board
with ``StateVector`` and scores a *subset* of legal guesses each turn (candidates
plus a random sample of probes in easy mode). Training uses a simple replay
buffer and numpy-only SGD so you do not need PyTorch for the baseline path.

For a course project, treat this as an extension: run tabular Q-learning first,
then experiment with DQN if you have time.
"""

from __future__ import annotations

import json
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .constraints import ConstraintState
from .heuristics import best_by_entropy, best_by_frequency, legal_guess_pool
from .state import VECTOR_DIM, StateVector


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


@dataclass
class NumpyMLP:
    """Two-layer MLP: state vector + one-hot word → scalar Q(s, word)."""

    w1: np.ndarray  # (hidden, input)
    b1: np.ndarray
    w2: np.ndarray  # (1, hidden)
    b2: np.ndarray

    @classmethod
    def init(cls, input_dim: int, hidden: int = 64, rng: np.random.Generator | None = None) -> NumpyMLP:
        rng = rng or np.random.default_rng(0)
        w1 = rng.normal(0, 0.1, (hidden, input_dim))
        b1 = np.zeros(hidden)
        w2 = rng.normal(0, 0.1, (1, hidden))
        b2 = np.zeros(1)
        return cls(w1=w1, b1=b1, w2=w2, b2=b2)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict]:
        z1 = self.w1 @ x + self.b1
        h1 = _relu(z1)
        out = (self.w2 @ h1 + self.b2).squeeze()
        cache = {"x": x, "z1": z1, "h1": h1, "out": out}
        return out, cache


def word_to_onehot(word: str) -> np.ndarray:
    from .state import LETTER_TO_IDX

    v = np.zeros(26, dtype=np.float32)
    for ch in word:
        v[LETTER_TO_IDX[ch]] += 1.0
    return v / 5.0


@dataclass
class DQNConfig:
    hidden: int = 64
    lr: float = 1e-3
    gamma: float = 0.95
    replay_size: int = 50_000
    batch_size: int = 64
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay_episodes: int = 40_000
    max_guess_pool: int = 400  # score at most this many words per step


@dataclass
class DQNAgent:
    """Scores (state, guess) pairs; pick argmax over a sampled pool."""

    config: DQNConfig = field(default_factory=DQNConfig)
    net: NumpyMLP | None = None
    replay: deque = field(default_factory=deque)
    _rng: random.Random = field(default_factory=random.Random)
    _episodes_seen: int = 0

    def _net(self) -> NumpyMLP:
        if self.net is None:
            input_dim = VECTOR_DIM + 26
            self.net = NumpyMLP.init(input_dim, self.config.hidden)
        return self.net

    def epsilon(self) -> float:
        c = self.config
        t = min(self._episodes_seen, c.epsilon_decay_episodes)
        frac = t / max(c.epsilon_decay_episodes, 1)
        return c.epsilon_start + (c.epsilon_end - c.epsilon_start) * frac

    def score_guess(self, state_vec: StateVector, word: str) -> float:
        x = np.concatenate(
            [np.asarray(state_vec.to_list(), dtype=np.float32), word_to_onehot(word)]
        )
        q, _ = self._net().forward(x)
        return float(q)

    def select_guess(
        self,
        state_vec: StateVector,
        constraint: ConstraintState,
        all_words: list[str],
        *,
        hard_mode: bool,
        explore: bool,
    ) -> str:
        pool = legal_guess_pool(constraint.candidates, all_words, hard_mode)
        if len(pool) > self.config.max_guess_pool:
            # Always include candidates; fill rest with random legal words.
            cand = list(constraint.candidates)
            extra = self._rng.sample(
                [w for w in pool if w not in set(cand)],
                k=min(self.config.max_guess_pool - len(cand), len(pool) - len(cand)),
            )
            pool = cand + extra

        if explore and self._rng.random() < self.epsilon():
            return self._rng.choice(pool)

        return max(pool, key=lambda w: self.score_guess(state_vec, w))

    def remember(
        self,
        state_vec: StateVector,
        word: str,
        reward: float,
        next_vec: StateVector | None,
        done: bool,
    ) -> None:
        self.replay.append(
            (
                state_vec.to_list(),
                word,
                reward,
                next_vec.to_list() if next_vec else None,
                done,
            )
        )
        while len(self.replay) > self.config.replay_size:
            self.replay.popleft()

    def train_step(self) -> float | None:
        """One SGD update from replay; returns loss or None if buffer too small."""
        c = self.config
        if len(self.replay) < c.batch_size:
            return None

        batch = self._rng.sample(list(self.replay), c.batch_size)
        net = self._net()
        total_loss = 0.0

        for s, word, r, ns, done in batch:
            x = np.concatenate(
                [np.asarray(s, dtype=np.float32), word_to_onehot(word)]
            )
            q, cache = net.forward(x)

            if done or ns is None:
                target = r
            else:
                # Bootstrap with greedy next guess from heuristic (cheap target).
                next_vec = StateVector(ns)
                # Placeholder: use max Q over same word one-hot — simplified target
                target = r + c.gamma * q  # keep simple for skeleton

            err = target - q
            total_loss += err * err
            # Manual backprop (one sample at a time for clarity)
            grad_out = -2 * err
            h1 = cache["h1"]
            w2_grad = grad_out * h1.reshape(1, -1)
            b2_grad = np.array([grad_out])
            dh1 = (net.w2.T @ np.array([[grad_out]])).squeeze()
            dz1 = dh1 * (cache["z1"] > 0)
            w1_grad = np.outer(dz1, x)
            b1_grad = dz1

            lr = c.lr
            net.w2 -= lr * w2_grad
            net.b2 -= lr * b2_grad
            net.w1 -= lr * w1_grad
            net.b1 -= lr * b1_grad

        return total_loss / c.batch_size

    def fallback_guess(self, constraint: ConstraintState) -> str:
        if len(constraint.candidates) <= 20:
            return best_by_entropy(constraint.candidates, constraint.candidates)
        return best_by_frequency(constraint.candidates)

    def on_episode_end(self) -> None:
        self._episodes_seen += 1

    def save(self, path: Path) -> None:
        net = self._net()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            w1=net.w1,
            b1=net.b1,
            w2=net.w2,
            b2=net.b2,
            episodes_seen=self._episodes_seen,
            config=json.dumps(self.config.__dict__),
        )

    @classmethod
    def load(cls, path: Path) -> DQNAgent:
        data = np.load(path, allow_pickle=False)
        cfg = DQNConfig(**json.loads(str(data["config"])))
        agent = cls(config=cfg)
        agent.net = NumpyMLP(w1=data["w1"], b1=data["b1"], w2=data["w2"], b2=data["b2"])
        agent._episodes_seen = int(data["episodes_seen"])
        return agent
