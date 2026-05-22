from __future__ import annotations

import argparse

from wordle.players import PLAYER_MODES, make_player
from wordle.ui import WordleUI
from wordle.words import load_words


def main() -> None:
    available = [m[0] for m in PLAYER_MODES if m[2]]
    parser = argparse.ArgumentParser(description="Wordle visual simulator")
    parser.add_argument(
        "--player",
        choices=available,
        default="human",
        help="Initial player. Use the dropdown to switch at runtime.",
    )
    parser.add_argument(
        "--rl-checkpoint",
        type=str,
        default=None,
        help="Path to rl/checkpoints/q_table.json (only for --player rl)",
    )
    args = parser.parse_args()

    words = load_words()
    if not words:
        raise SystemExit("No words found in data/allowed.txt")

    ui = WordleUI(answers=words, allowed_words=set(words))
    ui.set_player(args.player, rl_checkpoint=args.rl_checkpoint)
    ui.run()


if __name__ == "__main__":
    main()
