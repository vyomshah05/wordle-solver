from __future__ import annotations

from typing import Optional

import pygame

from wordle.game import WordleGame
from wordle.players import Player

from .solver import FrequencySolver


class FrequencyPlayer(Player):
    """Autonomous player using Model 1 (constraint + frequency baseline)."""

    name = "Model 1"

    def __init__(self, words: list[str]) -> None:
        self._solver = FrequencySolver(words)

    def handle_event(self, event: pygame.event.Event, game: WordleGame) -> Optional[str]:
        return None

    def tick(self, game: WordleGame) -> Optional[str]:
        if game.is_over:
            return None
        return self._solver.next_guess(game.history)
