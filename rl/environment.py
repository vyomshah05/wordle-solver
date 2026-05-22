"""Wordle as a reinforcement-learning environment.

Each *episode* is one game. The agent chooses abstract actions (strategies);
this module applies them, talks to ``WordleGame`` scoring, and returns rewards.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from wordle.game import GuessResult, WordleGame

from .constraints import ConstraintState, build_constraint_state
from .heuristics import GuessStrategy, pick_guess
from .state import TabularStateKey


# Rewards (tunable hyperparameters)
REWARD_STEP = -1.0
REWARD_WIN = 10.0
REWARD_LOSS = -20.0


@dataclass
class StepResult:
    tabular_state: TabularStateKey
    constraint: ConstraintState
    reward: float
    done: bool
    won: bool
    guess: str
    feedback: GuessResult | None = None


@dataclass
class WordleRLEnv:
    """Self-contained training environment (no pygame)."""

    all_words: list[str]
    allowed_set: set[str]
    answers: list[str] | None = None
    hard_mode: bool = True
    max_guesses: int = 6

    game: WordleGame | None = None
    constraint: ConstraintState | None = None
    _rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self) -> None:
        self.answers = self.answers or self.all_words

    def reset(self, answer: str | None = None) -> TabularStateKey:
        if answer is None:
            answer = self._rng.choice(self.answers)
        self.game = WordleGame(
            answer,
            max_guesses=self.max_guesses,
            allowed_words=self.allowed_set,
        )
        self.constraint = build_constraint_state(self.all_words, [])
        return TabularStateKey.from_game([], self.constraint)

    def get_tabular_state(self) -> TabularStateKey:
        assert self.game is not None and self.constraint is not None
        return TabularStateKey.from_game(self.game.history, self.constraint)

    def step_strategy(self, strategy: GuessStrategy) -> StepResult:
        """Take one turn: map strategy → guess → Wordle feedback → reward."""
        assert self.game is not None and self.constraint is not None

        if self.game.is_over:
            return StepResult(
                tabular_state=self.get_tabular_state(),
                constraint=self.constraint,
                reward=0.0,
                done=True,
                won=self.game.is_won,
                guess="",
            )

        guess = pick_guess(
            strategy,
            self.constraint,
            self.all_words,
            hard_mode=self.hard_mode,
            rng=self._rng,
        )

        result = self.game.submit(guess)
        self.constraint = build_constraint_state(self.all_words, self.game.history)

        done = self.game.is_over
        reward = REWARD_STEP
        if self.game.is_won:
            reward += REWARD_WIN
        elif self.game.is_lost:
            reward += REWARD_LOSS

        return StepResult(
            tabular_state=self.get_tabular_state(),
            constraint=self.constraint,
            reward=reward,
            done=done,
            won=self.game.is_won,
            guess=guess,
            feedback=result,
        )
