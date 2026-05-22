from __future__ import annotations

import argparse

from wordle.player_select import choice_by_id, list_choices
from wordle.ui import WordleUI
from wordle.words import load_words


def main() -> None:
    parser = argparse.ArgumentParser(description="Wordle visual simulator")
    parser.add_argument(
        "--player",
        choices=tuple(c.id for c in list_choices() if c.available),
        default="human",
        help="Initial player. Use the header dropdown to switch at runtime.",
    )
    args = parser.parse_args()

    words = load_words()
    if not words:
        raise SystemExit("No words found in data/allowed.txt")

    ui = WordleUI(answers=words, allowed_words=set(words))

    initial = choice_by_id(args.player)
    if initial is None or not initial.available:
        raise SystemExit(f"Player {args.player!r} is not available.")
    ui.set_player(args.player)
    if args.player == "human":
        ui.human = ui.active_player  # type: ignore[assignment]

    ui.run()


if __name__ == "__main__":
    main()
