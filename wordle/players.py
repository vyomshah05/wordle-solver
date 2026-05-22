from __future__ import annotations

from abc import ABC, abstractmethod

import pygame

from wordle.game import WordleGame

# id, label, available — flip available when model2/3 are ready
PLAYER_MODES: list[tuple[str, str, bool]] = [
    ("human", "Human", True),
    ("model1", "Model 1", True),
    ("model2", "Model 2", False),
    ("model3", "Model 3", False),
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


def make_player(mode_id: str, words: list[str]) -> Player:
    if mode_id == "human":
        return HumanPlayer()
    if mode_id == "model1":
        from wordle.solvers.model1 import FrequencySolver

        solver = FrequencySolver(words)

        class Model1Player(Player):
            name = "Model 1"

            def handle_event(self, event: pygame.event.Event, game: WordleGame) -> str | None:
                return None

            def tick(self, game: WordleGame) -> str | None:
                if game.is_over:
                    return None
                return solver.next_guess(game.history)

        return Model1Player()
    raise ValueError(f"Unknown player: {mode_id!r}")
