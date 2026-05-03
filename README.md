# wordle-solver

A Python + pygame Wordle simulator with a pluggable Player seam, built as the
foundation for an LLM-driven Wordle solver.

## Run

```
pip install -r requirements.txt
python main.py
```

## Controls

- Type a 5-letter word, press **Enter** to submit.
- **Backspace** to delete a letter.
- **R** to start a new game with a new random answer.
- **Esc** to quit.
- Click the **Player: Human / Model** button in the header to toggle who is
  guessing. Model mode currently shows a "not wired yet" toast — the LLM is the
  next step.

## Layout

```
wordle/
├── game.py      # rules, scoring, history (pure logic, easy to test/feed a solver)
├── players.py   # Player ABC + HumanPlayer + ModelPlayer stub
├── ui.py        # pygame rendering and event loop
└── words.py     # word-list loading
data/
├── answers.txt  # ~2,300 answer-eligible Wordle words (the secret-word pool)
└── allowed.txt  # ~10,600 additional valid-but-never-an-answer guesses
```

## Word list

The official NYT Wordle word lists are bundled. `answers.txt` and `allowed.txt`
are disjoint — `wordle.words.load_valid_guesses()` returns their union, so any
of the ~13,000 valid Wordle guesses is accepted from the keyboard, while the
secret word is always drawn from the smaller answer pool.

## Plugging in a model

The solver lands by:

1. Implementing `ModelPlayer` in [wordle/players.py](wordle/players.py) so its
   constructor stops raising and `tick(game)` returns a guess string when the
   model is ready to commit one.
2. The UI already calls `active_player.tick(...)` every frame and submits any
   non-`None` return, so no UI changes should be needed.
