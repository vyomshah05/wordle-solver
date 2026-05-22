from __future__ import annotations

from abc import ABC, abstractmethod

import pygame

from wordle.game import WordleGame

# id, label, available — matches project: Model 1 / 2 / 3 (+ human for the UI)
PLAYER_MODES: list[tuple[str, str, bool]] = [
    ("human", "Human", True),
    ("base_model", "Model 1: Base", True),
    ("entropy", "Model 2: Entropy", False),
    ("rl", "Model 3: RL", True),
]


class Player(ABC):
    name: str = "player"

    @abstractmethod
    def handle_event(self, event: pygame.event.Event, game: WordleGame) -> str | None:
        ...

    def tick(self, game: WordleGame) -> str | None:
        return None

    @property
    def current_input(self) -> str:
        return ""

    def reset(self) -> None:
        pass


class HumanPlayer(Player):
    name = "Human"

    def __init__(self) -> None:
        self._buffer = ""

    @property
    def current_input(self) -> str:
        return self._buffer

    def reset(self) -> None:
        self._buffer = ""

    def handle_event(self, event: pygame.event.Event, game: WordleGame) -> str | None:
        if event.type != pygame.KEYDOWN:
            return None
        if event.key == pygame.K_BACKSPACE:
            return self.press_key("BACK")
        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            return self.press_key("ENTER")
        if event.unicode and event.unicode.isalpha() and len(event.unicode) == 1:
            return self.press_key(event.unicode)
        return None

    def press_key(self, label: str) -> str | None:
        if label == "BACK":
            self._buffer = self._buffer[:-1]
        elif label == "ENTER" and len(self._buffer) == WordleGame.WORD_LENGTH:
            guess, self._buffer = self._buffer, ""
            return guess
        elif len(label) == 1 and label.isalpha() and len(self._buffer) < WordleGame.WORD_LENGTH:
            self._buffer += label.lower()
        return None


class ModelPlayer(Player):
    name = "Model"

    def __init__(self) -> None:
        from wordle.words import load_words
        from wordle.entropy_solver import EntropySolver
        print("Building/loading pattern matrix (~1 min on first run)...", flush=True)
        self._solver = EntropySolver(load_words())
        self._last_handled_history_len = -1

    def handle_event(self, event: pygame.event.Event, game: WordleGame) -> Optional[str]:
        return None

    def tick(self, game: WordleGame) -> Optional[str]:
        if game.is_over:
            return None
        n = len(game.history)
        if n < self._last_handled_history_len:
            self._solver.reset()
            self._last_handled_history_len = -1
        if n > self._last_handled_history_len:
            guess = self._solver.choose_guess(game.history)
            self._last_handled_history_len = n
            return guess
        return None
def make_player(mode_id: str, words: list[str], *, rl_checkpoint: str | None = None) -> Player:
    if mode_id == "human":
        return HumanPlayer()
    if mode_id == "base_model":
        from base_model.base_model import BaseModelPlayer

        return BaseModelPlayer(words)
    if mode_id == "rl":
        from rl.player import RLPlayer

        return RLPlayer(words, checkpoint=rl_checkpoint)
    raise ValueError(f"Unknown player: {mode_id!r}")
