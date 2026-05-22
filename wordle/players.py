from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import pygame

from .game import WordleGame


class Player(ABC):
    """Pluggable guess source. Returns a complete 5-letter guess when ready."""

    name: str = "player"

    @abstractmethod
    def handle_event(self, event: pygame.event.Event, game: WordleGame) -> Optional[str]:
        ...

    def tick(self, game: WordleGame) -> Optional[str]:
        """Called once per frame. Useful for autonomous players (model)."""
        return None

    @property
    def current_input(self) -> str:
        """In-progress letters to render in the active row."""
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

    def handle_event(self, event: pygame.event.Event, game: WordleGame) -> Optional[str]:
        if event.type != pygame.KEYDOWN:
            return None

        if event.key == pygame.K_BACKSPACE:
            return self.press_key("BACK")

        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            return self.press_key("ENTER")

        char = event.unicode
        if char and char.isalpha() and len(char) == 1:
            return self.press_key(char)
        return None

    def press_key(self, label: str) -> Optional[str]:
        """Inject input from the on-screen keyboard or a synthesized source."""
        if label == "BACK":
            self._buffer = self._buffer[:-1]
            return None
        if label == "ENTER":
            if len(self._buffer) == WordleGame.WORD_LENGTH:
                guess = self._buffer
                self._buffer = ""
                return guess
            return None
        if len(label) == 1 and label.isalpha() and len(self._buffer) < WordleGame.WORD_LENGTH:
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
