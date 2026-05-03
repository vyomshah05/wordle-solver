from __future__ import annotations

import argparse

from wordle.players import HumanPlayer, ModelPlayer
from wordle.ui import WordleUI
from wordle.words import load_answers, load_valid_guesses


def main() -> None:
    parser = argparse.ArgumentParser(description="Wordle visual simulator")
    parser.add_argument(
        "--player",
        choices=("human", "model"),
        default="human",
        help="Initial player. The on-screen button toggles between them at runtime.",
    )
    args = parser.parse_args()

    answers = load_answers()
    allowed = load_valid_guesses()
    if not answers:
        raise SystemExit("No answer words found in data/answers.txt")

    ui = WordleUI(answers=answers, allowed_words=allowed)

    if args.player == "model":
        # Surface the not-implemented error early instead of waiting for a click.
        ui.active_player = ModelPlayer()
    else:
        ui.active_player = HumanPlayer()
        ui.human = ui.active_player  # type: ignore[assignment]

    ui.run()


if __name__ == "__main__":
    main()
