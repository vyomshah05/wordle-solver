# wordle-solver

Wordle simulator (pygame) with pluggable solvers.

## Run

```bash
pip install -r requirements.txt
python main.py                  # human play
python main.py --player model1  # start as Model 1
python benchmark.py --limit 500 # benchmark Model 1
```

Use the **Player** dropdown (top-right) to switch between Human, Model 1, and future models.

## Layout

```
main.py              # entry point
benchmark.py         # benchmark Model 1
wordle/
  game.py            # Wordle rules + scoring
  words.py           # load data/allowed.txt
  players.py         # Human player + dropdown list + make_player()
  ui.py              # pygame UI
  solvers/
    model1.py        # Model 1 solver + player (all in one file)
    model2.py        # (your teammate adds this)
    model3.py        # (your teammate adds this)
data/allowed.txt
```

## Adding Model 2 or 3

1. Add `wordle/solvers/model2.py` with a `Player` class and solver logic.
2. In `players.py`, set `("model2", "Model 2", True)` and add one line to `make_player()`.
