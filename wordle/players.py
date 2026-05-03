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
    """Stub for the future LLM-driven player.

    Constructor intentionally raises so a half-wired model can't ship.
    Once the LLM is connected, override `tick(game)` to return a guess
    string when the model is ready to commit one.
    """

    name = "Model"

    def __init__(self) -> None:
        raise NotImplementedError(
            "ModelPlayer is not wired up yet. The UI exposes a toggle button "
            "for it, but the LLM integration lands in a follow-up step."
        )

    def handle_event(self, event: pygame.event.Event, game: WordleGame) -> Optional[str]:
        return None
