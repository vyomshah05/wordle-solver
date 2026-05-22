from __future__ import annotations

import argparse

from wordle.players import HumanPlayer, ModelPlayer
from wordle.ui import WordleUI
from wordle.words import load_words


def main() -> None:
    parser = argparse.ArgumentParser(description="Wordle visual simulator")
    parser.add_argument(
        "--player",
        choices=("human", "model", "rl"),
        default="human",
        help="Initial player. The on-screen button toggles between them at runtime.",
    )
    parser.add_argument(
        "--rl-checkpoint",
        type=str,
        default=None,
        help="Path to rl/checkpoints/q_table.json (for --player rl)",
    )
    args = parser.parse_args()

    words = load_words()
    if not words:
        raise SystemExit("No words found in data/allowed.txt")

    ui = WordleUI(answers=words, allowed_words=set(words))

    if args.player == "rl":
        from rl.player import RLPlayer

        ui.active_player = RLPlayer(words, checkpoint=args.rl_checkpoint)
    elif args.player == "model":
        # Surface the not-implemented error early instead of waiting for a click.
        ui.active_player = ModelPlayer()
    else:
        ui.active_player = HumanPlayer()
        ui.human = ui.active_player  # type: ignore[assignment]

    ui.run()


if __name__ == "__main__":
    main()
