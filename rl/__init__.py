"""Reinforcement-learning Wordle solver (Model 3).

Train with::

    python -m rl.train --episodes 50000

Play in the pygame UI::

    python main.py --player rl
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["RLPlayer"]

if TYPE_CHECKING:
    from .player import RLPlayer


def __getattr__(name: str):
    if name == "RLPlayer":
        from .player import RLPlayer

        return RLPlayer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
