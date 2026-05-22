from __future__ import annotations

from dataclasses import dataclass

from wordle.players import HumanPlayer, Player


@dataclass(frozen=True)
class PlayerChoice:
    id: str
    label: str
    available: bool


def list_choices() -> list[PlayerChoice]:
    return [
        PlayerChoice("human", "Human", True),
        PlayerChoice("model1", "Model 1", True),
        PlayerChoice("model2", "Model 2", False),
        PlayerChoice("model3", "Model 3", False),
    ]


def create_player(choice_id: str, words: list[str]) -> Player:
    if choice_id == "human":
        return HumanPlayer()
    if choice_id == "model1":
        from wordle.solvers.model1.player import FrequencyPlayer

        return FrequencyPlayer(words)
    raise ValueError(f"Unknown or unavailable player: {choice_id!r}")


def choice_by_id(choice_id: str) -> PlayerChoice | None:
    for choice in list_choices():
        if choice.id == choice_id:
            return choice
    return None
