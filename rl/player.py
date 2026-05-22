"""Pygame ``Player`` adapter for the trained RL policy."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from wordle.game import WordleGame
from wordle.players import Player

if TYPE_CHECKING:
    import pygame

from .agent_tabular import QLearningAgent
from .constraints import build_constraint_state
from .environment import WordleRLEnv
from .heuristics import pick_guess
from .state import TabularStateKey


DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "q_table.json"


class RLPlayer(Player):
    """Uses tabular Q-learning to pick a heuristic, then a concrete guess."""

    name = "RL"

    def __init__(
        self,
        all_words: list[str],
        checkpoint: Path | str | None = None,
        *,
        hard_mode: bool = True,
    ) -> None:
        self.all_words = all_words
        self.allowed_set = set(all_words)
        self.hard_mode = hard_mode
        path = Path(checkpoint) if checkpoint else DEFAULT_CHECKPOINT
        if not path.exists():
            raise FileNotFoundError(
                f"No RL checkpoint at {path}. Train first:\n"
                f"  python -m rl.train --episodes 50000 --out {path}"
            )
        self.agent = QLearningAgent.load(path)
        self._submitted_for_turn: int = -1

    def reset(self) -> None:
        self._submitted_for_turn = -1

    def tick(self, game: WordleGame) -> Optional[str]:
        if game.is_over:
            return None

        turn = len(game.history)
        if turn == self._submitted_for_turn:
            return None

        constraint = build_constraint_state(self.all_words, game.history)
        tabular = TabularStateKey.from_game(game.history, constraint)
        action = self.agent.best_action(tabular)
        guess = pick_guess(
            action,
            constraint,
            self.all_words,
            hard_mode=self.hard_mode,
        )

        if self.hard_mode and guess not in constraint.candidates:
            # Safety: never submit illegal hard-mode guesses.
            guess = constraint.candidates[0]

        self._submitted_for_turn = turn
        return guess

    def handle_event(self, event: Any, game: WordleGame) -> Optional[str]:
        return None


def make_rl_env(all_words: list[str], **kwargs) -> WordleRLEnv:
    return WordleRLEnv(all_words=all_words, allowed_set=set(all_words), **kwargs)
