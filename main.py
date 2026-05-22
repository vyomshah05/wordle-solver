from __future__ import annotations

import argparse

from wordle.players import HumanPlayer
from wordle.solvers.model1.player import FrequencyPlayer
from wordle.ui import WordleUI
from wordle.words import load_words


def main() -> None:
    parser = argparse.ArgumentParser(description="Wordle visual simulator")
    parser.add_argument(
        "--player",
        choices=("human", "model1"),
        default="human",
        help="Initial player. The on-screen button toggles Human ↔ Model 1 at runtime.",
    )
    args = parser.parse_args()

    words = load_words()
    if not words:
        raise SystemExit("No words found in data/allowed.txt")

    ui = WordleUI(answers=words, allowed_words=set(words))

    if args.player == "model1":
        ui.active_player = FrequencyPlayer(words)
    else:
        ui.active_player = HumanPlayer()
        ui.human = ui.active_player  # type: ignore[assignment]

    ui.run()


if __name__ == "__main__":
    main()
